"""Timbre metrics extraction utilities.

Extracts spectral, level, harmonic, and temporal timbre descriptors from mono
audio signals using the ``pytimbre`` library.

For **spectral / level / harmonic** features the audio is split into
overlapping frames via ``FrameBuilder`` → ``SpectralTimeHistory``, each
frame's ``Spectrum`` is analysed, and the per-frame results are either
returned as time-series arrays or averaged into single scalars.

For **temporal** features the full waveform is analysed via
``TemporalMetrics`` (envelope-based globals such as attack / decay /
temporal centroid, plus per-frame ZCR and autocorrelation computed with
the library's own internal framing).

Usage
-----
>>> tm = TimbreMetrics(sample_rate=16000, frame_width_sec=0.25, overlap_pct=0.5)

Spectral features (default):
>>> scalars = tm.extract_from_array(signal)
>>> series  = tm.extract_series_from_array(signal)

Include level and harmonic features too:
>>> series = tm.extract_series_from_array(
...     signal, feature_types=["spectral", "level", "harmonic"]
... )

Temporal features (full-signal globals):
>>> temporal = tm.extract_temporal_features(signal)

Temporal per-frame texture (ZCR, autocorrelation):
>>> temporal_series = tm.extract_temporal_series(signal)

All features at once via pytimbre's built-in aggregator (returns DataFrame):
>>> df = tm.extract_all_from_time_history(signal)

Batch extraction over a directory of WAV files:
>>> concatenated = tm.extract_from_dir("data/wav_folder", n_workers=4)
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pytimbre.audio import Waveform
from pytimbre.spectral.spectral_frame_builder import FrameBuilder
from pytimbre.spectral.spectra import Spectrum
from pytimbre.spectral.time_histories import SpectralTimeHistory
from pytimbre.timbre_features.features import TimbreFeatures
from pytimbre.timbre_features.metrics.harmonic import HarmonicMetrics
from pytimbre.timbre_features.metrics.level import LevelMetrics
from pytimbre.timbre_features.metrics.spectral import SpectralMetrics
from pytimbre.timbre_features.metrics.temporal import TemporalMetrics

from utils import load_audio

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Module-level worker for ProcessPoolExecutor (must be picklable)
# ----------------------------------------------------------------------

def _extract_worker(
    file_path: Path,
    sr: int,
    frame_width_sec: float,
    overlap_pct: float,
    metrics: Optional[List[str]],
    feature_types: Optional[List[str]],
) -> Optional[Dict[str, np.ndarray]]:
    """Extract per-frame features from a single WAV file.

    Returns ``None`` on error so the caller can skip failures.
    """
    try:
        audio, _ = load_audio(file_path, sr=sr, mono=True)
        tm = TimbreMetrics(
            sample_rate=sr,
            frame_width_sec=frame_width_sec,
            overlap_pct=overlap_pct,
        )
        series = tm.extract_series_from_array(
            audio, metrics=metrics, feature_types=feature_types,
        )
        if not series:
            logger.warning("Empty features: %s", file_path)
            return None
        return series
    except Exception:
        logger.warning("Feature extraction failed: %s", file_path, exc_info=True)
        return None


class TimbreMetrics:
    """Configurable timbre feature extractor.

    Parameters
    ----------
    sample_rate : int
        Expected sample rate of the input audio (Hz).
    frame_width_sec : float
        Duration of each analysis frame in seconds.
    overlap_pct : float
        Fraction of overlap between consecutive frames (0.0 – 1.0).
    """

    SPECTRAL = "spectral"
    LEVEL = "level"
    HARMONIC = "harmonic"
    ALL_FRAME_TYPES = (SPECTRAL, LEVEL, HARMONIC)
    DEFAULT_FEATURE_TYPES = (SPECTRAL,)

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_width_sec: float = 0.25,#240bpm
        overlap_pct: float = 0.0,
    ):
        self.sample_rate = sample_rate
        self.frame_width_sec = frame_width_sec
        self.overlap_pct = overlap_pct

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def from_array(signal: np.ndarray, sr: int) -> Waveform:
        """Wrap a raw numpy array as a pytimbre ``Waveform``.

        Validates that *signal* is a non-empty 1-D array and casts it to
        float64 (required by pytimbre internals).
        """
        if signal is None:
            raise ValueError("signal cannot be None")

        arr = np.asarray(signal)
        if arr.ndim != 1:
            raise ValueError("signal must be a 1D mono numpy array")
        if arr.size == 0:
            raise ValueError("signal cannot be empty")

        return Waveform(arr.astype(np.float64), sr, 0.0)

    @staticmethod
    def _filter_metrics(features: dict, metrics: Optional[List[str]] = None) -> dict:
        """Keep only the requested metric keys, or all if *metrics* is None."""
        if metrics is None:
            return features
        return {k: v for k, v in features.items() if k in metrics}

    def _build_time_history(self, wfm: Waveform) -> Optional[SpectralTimeHistory]:
        """Slice *wfm* into overlapping frames and compute the FFT for each.

        Returns ``None`` if the waveform is too short to fill even one frame.
        """
        if not isinstance(wfm, Waveform):
            return None

        frame_builder = FrameBuilder.from_waveform(
            wfm,
            overlap_pct=self.overlap_pct,
            frame_width_sec=self.frame_width_sec,
        )

        if frame_builder.complete_frame_count <= 0:
            return None

        return SpectralTimeHistory.from_fourier_transform(wfm, frame_builder)

    def _extract_frame_features(
        self, spec: Spectrum, feature_types: set[str]
    ) -> Dict[str, float]:
        """Extract selected metric types from a single ``Spectrum`` frame."""
        if not isinstance(spec, Spectrum):
            return {}

        features: Dict[str, float] = {}
        try:
            if self.SPECTRAL in feature_types:
                features.update(SpectralMetrics.from_spectrum(spec).get_features())
            if self.LEVEL in feature_types:
                features.update(LevelMetrics.from_spectrum(spec).get_features())
            if self.HARMONIC in feature_types:
                features.update(HarmonicMetrics(spec).get_features())
        except Exception:
            pass

        return features

    # ------------------------------------------------------------------
    # Per-frame spectral / level / harmonic extraction
    # ------------------------------------------------------------------

    def extract_time_history_features(
        self,
        wfm: Waveform,
        metrics: Optional[List[str]] = None,
        feature_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Extract per-frame features then average across frames.

        Parameters
        ----------
        wfm : Waveform
            Input waveform.
        metrics : list of str, optional
            Metric names to include.  If *None*, all metrics are returned.
        feature_types : list of str, optional
            Which metric classes to run (``"spectral"``, ``"level"``,
            ``"harmonic"``).  Defaults to ``["spectral"]``.
        """
        types = set(feature_types or self.DEFAULT_FEATURE_TYPES)

        sth = self._build_time_history(wfm)
        if sth is None:
            return {}

        spectra = sth.spectra
        if spectra is None or len(spectra) == 0:
            return {}

        aggregated: Dict[str, list[float]] = {}
        for spec in spectra:
            for k, v in self._extract_frame_features(spec, types).items():
                aggregated.setdefault(k, []).append(float(v))

        if not aggregated:
            return {}

        features: Dict[str, float] = {}
        for key, values in aggregated.items():
            numeric = np.asarray(values, dtype=np.float64)
            features[key] = float(np.nanmean(numeric)) if numeric.size else np.nan

        return self._filter_metrics(features, metrics)

    def extract_time_history_series(
        self,
        wfm: Waveform,
        metrics: Optional[List[str]] = None,
        feature_types: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """Extract per-frame features without averaging.

        Parameters
        ----------
        wfm : Waveform
            Input waveform.
        metrics : list of str, optional
            Metric names to include.  If *None*, all metrics are returned.
        feature_types : list of str, optional
            Which metric classes to run (``"spectral"``, ``"level"``,
            ``"harmonic"``).  Defaults to ``["spectral"]``.
        """
        types = set(feature_types or self.DEFAULT_FEATURE_TYPES)

        sth = self._build_time_history(wfm)
        if sth is None:
            return {}

        spectra = sth.spectra
        if spectra is None or len(spectra) == 0:
            return {}

        series: Dict[str, list[float]] = {}
        for spec in spectra:
            for k, v in self._extract_frame_features(spec, types).items():
                series.setdefault(k, []).append(float(v))

        out: Dict[str, np.ndarray] = {}
        for key, values in series.items():
            out[key] = np.asarray(values, dtype=np.float64)

        return self._filter_metrics(out, metrics)

    # ------------------------------------------------------------------
    # Temporal features (full-signal, via TemporalMetrics)
    # ------------------------------------------------------------------

    def extract_temporal_features(
        self,
        wfm_or_signal: Waveform | np.ndarray,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Extract global temporal features from the full waveform.

        Returns averaged autocorrelation coefficients, mean ZCR, attack/decay
        envelope parameters, modulation metrics, etc.  For the raw per-frame
        arrays use :meth:`extract_temporal_series`.
        """
        if isinstance(wfm_or_signal, np.ndarray):
            wfm_or_signal = self.from_array(wfm_or_signal, self.sample_rate)
        tm = TemporalMetrics.from_waveform(wfm_or_signal)
        return self._filter_metrics(tm.get_features(), metrics)

    def extract_temporal_series(
        self, wfm_or_signal: Waveform | np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Extract per-frame temporal texture arrays (ZCR and autocorrelation).

        These use ``TemporalMetrics``' own internal framing (hop ~ 2.9 ms,
        window ~ 23.2 ms), which is independent of this class's
        ``frame_width_sec`` / ``overlap_pct`` settings.
        """
        if isinstance(wfm_or_signal, np.ndarray):
            wfm_or_signal = self.from_array(wfm_or_signal, self.sample_rate)
        tm = TemporalMetrics.from_waveform(wfm_or_signal)

        series: Dict[str, np.ndarray] = {
            "zero_crossing_rate": np.asarray(tm.zero_crossing_rate, dtype=np.float64),
        }
        auto = np.asarray(tm.auto_correlation, dtype=np.float64)
        for i in range(auto.shape[1]):
            series[f"auto_correlation_{i + 1:02d}"] = auto[:, i]

        return series

    # ------------------------------------------------------------------
    # Built-in all-features extraction (pytimbre TimbreFeatures)
    # ------------------------------------------------------------------

    def extract_all_from_time_history(
        self, wfm_or_signal: Waveform | np.ndarray
    ) -> pd.DataFrame:
        """Use pytimbre's ``TimbreFeatures.from_time_history`` for full extraction.

        Returns a pandas DataFrame with one row per frame and columns for
        every available feature type (spectral, level, harmonic, temporal,
        and sound quality).
        """
        if isinstance(wfm_or_signal, np.ndarray):
            wfm_or_signal = self.from_array(wfm_or_signal, self.sample_rate)
        sth = self._build_time_history(wfm_or_signal)
        if sth is None:
            return pd.DataFrame()
        return TimbreFeatures.from_time_history(sth)

    # ------------------------------------------------------------------
    # Convenience wrappers — single inputs
    # ------------------------------------------------------------------

    def extract_series_from_array(
        self,
        signal: np.ndarray,
        metrics: Optional[List[str]] = None,
        feature_types: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """Numpy array in, per-frame metric arrays out."""
        wfm = self.from_array(signal, self.sample_rate)
        return self.extract_time_history_series(wfm, metrics=metrics, feature_types=feature_types)

    def extract_from_array(
        self,
        signal: np.ndarray,
        metrics: Optional[List[str]] = None,
        feature_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Numpy array in, one averaged scalar per metric out."""
        wfm = self.from_array(signal, self.sample_rate)
        return self.extract_time_history_features(wfm, metrics=metrics, feature_types=feature_types)

    def extract_series_from_file(
        self,
        file_path: str | Path,
        metrics: Optional[List[str]] = None,
        feature_types: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """Load a WAV file (with resampling to ``self.sample_rate``), return
        per-frame metric arrays."""
        audio, _ = load_audio(file_path, sr=self.sample_rate, mono=True)
        return self.extract_series_from_array(
            audio, metrics=metrics, feature_types=feature_types,
        )

    # ------------------------------------------------------------------
    # Batch extraction over a directory
    # ------------------------------------------------------------------

    def extract_from_dir(
        self,
        wav_dir: str | Path,
        metrics: Optional[List[str]] = None,
        feature_types: Optional[List[str]] = None,
        n_workers: int = 1,
    ) -> Dict[str, np.ndarray]:
        """Extract per-frame features from every WAV under *wav_dir* and concatenate.

        Returns a dict mapping each feature name to a single 1-D array formed
        by concatenating frames across all files.  Only feature keys present
        in **every** successfully extracted file are kept.

        Parameters
        ----------
        wav_dir : path
            Root directory to search recursively for ``.wav`` files.
        metrics, feature_types :
            Forwarded to :meth:`extract_series_from_array`.
        n_workers : int
            Number of parallel workers.  ``1`` runs sequentially.
        """
        paths = sorted(Path(wav_dir).rglob("*.wav"))
        if not paths:
            raise FileNotFoundError(f"No .wav files found in {wav_dir}")

        logger.info("Extracting features from %d files in %s", len(paths), wav_dir)

        all_series: List[Dict[str, np.ndarray]] = []

        if n_workers <= 1:
            items = tqdm(paths, desc="Feature extraction") if tqdm else paths
            for p in items:
                result = _extract_worker(
                    p, self.sample_rate, self.frame_width_sec,
                    self.overlap_pct, metrics, feature_types,
                )
                if result is not None:
                    all_series.append(result)
        else:
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = {
                    pool.submit(
                        _extract_worker,
                        p, self.sample_rate, self.frame_width_sec,
                        self.overlap_pct, metrics, feature_types,
                    ): p
                    for p in paths
                }
                items = (
                    tqdm(as_completed(futures), desc="Feature extraction",
                         total=len(futures))
                    if tqdm else as_completed(futures)
                )
                for future in items:
                    result = future.result()
                    if result is not None:
                        all_series.append(result)

        if not all_series:
            raise ValueError(f"No features extracted from any file in {wav_dir}")

        # Intersection of feature keys across all files
        common_keys = set(all_series[0].keys())
        for s in all_series[1:]:
            common_keys &= set(s.keys())
        feat_keys = sorted(common_keys)

        if not feat_keys:
            raise ValueError("No common feature keys across files")

        concatenated: Dict[str, np.ndarray] = {
            k: np.concatenate([s[k] for s in all_series]) for k in feat_keys
        }

        logger.info(
            "Concatenated %d files -> %d frames, %d features",
            len(all_series), len(concatenated[feat_keys[0]]), len(feat_keys),
        )
        return concatenated
