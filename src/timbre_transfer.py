"""DDSP timbre transfer.

Provides :class:`TimbreTransfer` — a stateful wrapper around a DDSP
Autoencoder checkpoint that loads the model once and exposes a
:meth:`synthesize` method for inference.

Low-level helpers (:func:`load_model`, :func:`resynthesize`) are kept as
module-level functions for backward compatibility with existing notebooks.
"""

from __future__ import annotations

import copy
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Optional

import gin
import numpy as np
import tensorflow.compat.v2 as tf  # type: ignore

import ddsp
import ddsp.training

from model_loading import (
    find_model_dir,
    load_pretrained_model,
    restore_autoencoder,
    PRETRAINED_MODELS,
)
from feature_utils import compute_features, trim_features
from utils import DEFAULT_SAMPLE_RATE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stateful class API (recommended)
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

    def __init__(
        self,
        model_dir: str | Path,
        gin_file: str | Path,
        sr: int = DEFAULT_SAMPLE_RATE,
    ) -> None:
        self.model_dir = str(model_dir)
        self.gin_file = str(gin_file)
        self.sr = sr
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
        """Run inference on a single chunk of audio features.

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
        start = time.time()
        outputs = self._model(audio_features, training=False)
        audio_gen = self._model.get_audio_from_outputs(outputs)
        logger.debug("Synthesis took %.1f s", time.time() - start)
        return np.array(audio_gen)[0]


# ---------------------------------------------------------------------------
# Function API (backward-compatible wrappers)
# ---------------------------------------------------------------------------


def load_model(
    model_dir: str,
    gin_file: str,
    audio: np.ndarray,
    audio_features: Dict[str, Any],
) -> tuple:
    """Load a DDSP model and prepare features for inference.

    Parameters
    ----------
    model_dir : str
        Path to the checkpoint directory.
    gin_file : str
        Path to the operative gin config.
    audio : np.ndarray
        Reference audio used to infer alignment dimensions (shape ``[1, N]``).
    audio_features : dict
        Reference features (output of :func:`compute_features`).

    Returns
    -------
    tuple
        ``(model, trimmed_audio_features)``
    """
    with gin.unlock_config():
        gin.parse_config_file(gin_file, skip_unknown=True)

    time_steps_train = gin.query_parameter("F0LoudnessPreprocessor.time_steps")
    n_samples_train = gin.query_parameter("Harmonic.n_samples")
    hop_size = int(n_samples_train / time_steps_train)

    time_steps = int(audio.shape[1] / hop_size)
    n_samples = time_steps * hop_size

    gin_params = [
        f"Harmonic.n_samples = {n_samples}",
        f"FilteredNoise.n_samples = {n_samples}",
        f"F0LoudnessPreprocessor.time_steps = {time_steps}",
        "oscillator_bank.use_angular_cumsum = True",
    ]
    with gin.unlock_config():
        gin.parse_config(gin_params)

    audio_features = trim_features(audio_features, time_steps, n_samples)

    model = restore_autoencoder(model_dir)

    start_time = time.time()
    _ = model(audio_features, training=False)  # type: ignore
    logger.info("Model restored in %.1f s", time.time() - start_time)
    return model, audio_features


def resynthesize(model: Any, audio_features: Dict[str, Any]) -> np.ndarray:
    """Run a forward pass through *model* and return the generated audio.

    Parameters
    ----------
    model : ddsp.training.models.Autoencoder
        A restored DDSP model.
    audio_features : dict
        Feature dictionary for one chunk.

    Returns
    -------
    np.ndarray
        Synthesized audio (1-D).
    """
    start_time = time.time()
    outputs = model(audio_features, training=False)
    audio_gen = model.get_audio_from_outputs(outputs)
    logger.debug("Synthesis took %.1f s", time.time() - start_time)
    audio_gen = np.array(audio_gen)[0]
    return audio_gen


__all__ = [
    "DEFAULT_SAMPLE_RATE",
    "TimbreTransfer",
    "load_model",
    "resynthesize",
]
