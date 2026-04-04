"""DDSP timbre transfer.

Provides :class:`TimbreTransfer` — a stateful wrapper around a DDSP
Autoencoder checkpoint that loads the model once and exposes a
:meth:`synthesize` method for inference.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import gin
import numpy as np
import tensorflow.compat.v2 as tf  # type: ignore

import ddsp
import ddsp.training

from model_loading import (
    find_gin_file,
    find_model_dir,
    restore_autoencoder,
)
from feature_utils import compute_features, trim_features
from utils import DEFAULT_SAMPLE_RATE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class TimbreTransfer:
    """DDSP timbre transfer engine.

    Loads a DDSP Autoencoder checkpoint and its gin configuration once,
    then exposes :meth:`synthesize` for repeated inference.

    Parameters
    ----------
    model_dir : str or Path
        Directory containing the DDSP checkpoint files.
    gin_file : str or Path
        Path to the operative gin config file.
    sr : int
        Sample rate (Hz).
    """

    @classmethod
    def from_path(
        cls,
        model_path: str | Path,
        gin_file: str | Path | None = None,
        sr: int = DEFAULT_SAMPLE_RATE,
        gin_overrides: Sequence[str] | None = None,
    ) -> "TimbreTransfer":
        """Create a :class:`TimbreTransfer` from a model path.

        Automatically discovers the model directory and gin config file.
        """
        model_dir = find_model_dir(str(model_path))
        gin_path = str(gin_file) if gin_file else find_gin_file(model_dir)
        return cls(model_dir, gin_path, sr=sr, gin_overrides=gin_overrides)

    def __init__(
        self,
        model_dir: str | Path,
        gin_file: str | Path,
        sr: int = DEFAULT_SAMPLE_RATE,
        gin_overrides: Sequence[str] | None = None,
    ) -> None:
        self.model_dir = str(model_dir)
        self.gin_file = str(gin_file)
        self.sr = sr
        self.gin_overrides: List[str] = list(gin_overrides) if gin_overrides else []
        self._model: Any = None
        self._time_steps: int = 0
        self._n_samples: int = 0
        self._hop_size: int = 0

    @property
    def is_loaded(self) -> bool:
        """Whether the model has been loaded."""
        return self._model is not None

    @property
    def time_steps(self) -> int:
        """Number of feature frames per chunk (set after :meth:`load`)."""
        return self._time_steps

    @property
    def n_samples(self) -> int:
        """Number of audio samples per chunk (set after :meth:`load`)."""
        return self._n_samples

    def load(self) -> None:
        """Load the model checkpoint and parse the gin configuration.

        After this call, :attr:`time_steps`, :attr:`n_samples`, and
        :attr:`is_loaded` are available.
        """
        with gin.unlock_config():
            gin.parse_config_file(self.gin_file, skip_unknown=True)
            if self.gin_overrides:
                gin.parse_config(self.gin_overrides)

        self._time_steps = int(
            gin.query_parameter("F0LoudnessPreprocessor.time_steps")
        )
        self._n_samples = int(gin.query_parameter("Harmonic.n_samples"))
        self._hop_size = self._n_samples // self._time_steps

        # Warm up the model with a silent reference
        ref_audio = np.zeros((1, self._n_samples), dtype=np.float32)
        ref_features = compute_features(ref_audio)

        gin_params = [
            f"Harmonic.n_samples = {self._n_samples}",
            f"FilteredNoise.n_samples = {self._n_samples}",
            f"F0LoudnessPreprocessor.time_steps = {self._time_steps}",
            "oscillator_bank.use_angular_cumsum = True",
        ]
        with gin.unlock_config():
            gin.parse_config(gin_params)

        ref_features = trim_features(
            ref_features, self._time_steps, self._n_samples
        )

        self._model = restore_autoencoder(self.model_dir)

        start = time.time()
        _ = self._model(ref_features, training=False)
        logger.info("Model restored in %.1f s", time.time() - start)

    def synthesize(self, audio_features: Dict[str, Any]) -> np.ndarray:
        """Synthesize audio from features of arbitrary duration.

        If the features span more than one model chunk (``time_steps``
        frames / ``n_samples`` samples), the input is automatically split
        into chunks, each synthesized independently, then concatenated.

        Parameters
        ----------
        audio_features : dict
            Feature dictionary with keys ``f0_hz``, ``f0_confidence``,
            ``loudness_db``, ``audio`` — as returned by
            :func:`feature_utils.compute_features`.

        Returns
        -------
        np.ndarray
            Synthesized audio (1-D float32).
        """
        if not self.is_loaded:
            raise RuntimeError(
                "Model not loaded. Call .load() before .synthesize()."
            )

        n_frames = len(audio_features["f0_hz"])
        n_chunks = int(np.ceil(n_frames / self._time_steps))

        chunks = []
        for i in range(n_chunks):
            chunk = self._slice_chunk(audio_features, i)
            actual_samples = self._chunk_sample_len(audio_features, i)

            gen = self._synthesize_single(chunk)

            # Trim padding from the last chunk
            if actual_samples < self._n_samples:
                gen = gen[:actual_samples]

            chunks.append(gen)
            if n_chunks > 1:
                logger.info("Chunk %d/%d synthesized", i + 1, n_chunks)

        return np.concatenate(chunks)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _synthesize_single(self, chunk: Dict[str, Any]) -> np.ndarray:
        """Run the model on a single, correctly-sized chunk."""
        start = time.time()
        outputs = self._model(chunk, training=False)
        audio_gen = self._model.get_audio_from_outputs(outputs)
        logger.debug("Synthesis took %.1f s", time.time() - start)
        return np.array(audio_gen)[0]

    def _slice_chunk(
        self, audio_features: Dict[str, Any], idx: int
    ) -> Dict[str, Any]:
        """Extract chunk *idx* from full-length features, padded to model size."""
        n_frames = len(audio_features["f0_hz"])
        n_audio = audio_features["audio"].shape[-1]

        f_start = idx * self._time_steps
        f_end = min(f_start + self._time_steps, n_frames)
        s_start = idx * self._n_samples
        s_end = min(s_start + self._n_samples, n_audio)

        chunk: Dict[str, Any] = {}

        # Feature arrays (1-D)
        for key in ("f0_hz", "f0_confidence", "loudness_db"):
            feat = audio_features[key][f_start:f_end]
            pad_needed = self._time_steps - len(feat)
            if pad_needed > 0:
                pad_val = (
                    float(audio_features["loudness_db"].min())
                    if key == "loudness_db"
                    else 0.0
                )
                feat = np.pad(feat, (0, pad_needed), constant_values=pad_val)
            chunk[key] = feat

        # Audio array (2-D: batch × samples)
        audio_slice = audio_features["audio"][:, s_start:s_end]
        pad_needed = self._n_samples - audio_slice.shape[-1]
        if pad_needed > 0:
            audio_slice = np.pad(
                audio_slice, ((0, 0), (0, pad_needed)), constant_values=0.0
            )
        chunk["audio"] = audio_slice

        return chunk

    def _chunk_sample_len(
        self, audio_features: Dict[str, Any], idx: int
    ) -> int:
        """Return the number of *real* (unpadded) audio samples in chunk *idx*."""
        n_audio = audio_features["audio"].shape[-1]
        s_start = idx * self._n_samples
        return min(self._n_samples, n_audio - s_start)


__all__ = [
    "TimbreTransfer",
]
