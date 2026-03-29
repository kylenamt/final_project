"""Timbre metrics extraction utilities.

Extracts spectral timbre descriptors (e.g. spectral centroid, spread,
loudness levels) from mono audio signals. Audio is split into overlapping
frames, each frame's spectrum is analysed, and the per-frame results are
either returned as time-series arrays or averaged into single scalars.

Usage
-----
>>> tm = TimbreMetrics(sample_rate=16000, frame_width_sec=0.25, overlap_pct=0.5)

From a numpy array (preferred):
>>> scalars = tm.extract_from_array(signal)        # dict of metric -> float
>>> series  = tm.extract_series_from_array(signal)  # dict of metric -> 1-D array

From a .wav file on disk:
>>> series  = tm.extract_series_from_file("clip.wav")

Notes
-----
- All array inputs must be 1-D mono float/int numpy arrays.
- Level metrics ``la``, ``lc``, ``lz`` are renamed to ``spec_la``, ``spec_lc``,
  ``spec_lz`` for consistency with the rest of the feature set.
- If a metric fails on a particular frame it is silently skipped so that the
  remaining metrics are still returned.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np
from pytimbre.audio import Waveform
from pytimbre.spectral.spectral_frame_builder import FrameBuilder
from pytimbre.spectral.spectra import Spectrum
from pytimbre.spectral.time_histories import SpectralTimeHistory
from pytimbre.timbre_features.metrics.spectral import SpectralMetrics



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

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_width_sec: float = 0.25,
        overlap_pct: float = 0.5,
    ):
        self.sample_rate = sample_rate
        self.frame_width_sec = frame_width_sec
        self.overlap_pct = overlap_pct

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
    def _to_scalar(value: Any) -> float:
        """Coerce a feature value to a single float.

        Handles ints, floats, numpy scalars, lists/tuples, and ndarrays.
        Single-element containers return that element; multi-element ones
        return the mean.  Returns ``np.nan`` for empty or unconvertible inputs.
        """
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)

        if isinstance(value, (list, tuple)):
            arr = np.asarray(value, dtype=np.float64)
            if arr.size == 0:
                return np.nan
            return float(arr.ravel()[0]) if arr.size == 1 else float(np.nanmean(arr))

        if isinstance(value, np.ndarray):
            if value.size == 0:
                return np.nan
            return float(value.ravel()[0]) if value.size == 1 else float(np.nanmean(value))

        try:
            return float(value)
        except (TypeError, ValueError):
            return np.nan


    @staticmethod
    def _normalize_feature_name(name: str) -> str:
        """Prefix bare level names (``la``, ``lc``, ``lz``) with ``spec_`` so
        all feature keys share a consistent naming scheme."""
        if name in {"la", "lc", "lz"}:
            return f"spec_{name}"
        return name

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

    def _extract_single_spectrum_features(self, spec: Spectrum) -> Dict[str, Any]:
        """Compute all available spectral metrics for one frame.

        Catches errors from ``SpectralMetrics`` so that a failure in one
        metric does not prevent the others from being returned.
        """
        if not isinstance(spec, Spectrum):
            return {}

        features: Dict[str, Any] = {}

        try:
            for k, v in SpectralMetrics.from_spectrum(spec).get_features().items():
                features[self._normalize_feature_name(k)] = v
        except Exception:
            pass

        return features

    def extract_time_history_features(self, wfm: Waveform) -> Dict[str, Any]:
        """Extract timbre metrics per frame, then average across frames.

        Returns a dict mapping each metric name to a single float (the mean
        over all frames, ignoring NaNs).
        """
        time_history = self._build_time_history(wfm)
        if time_history is None:
            return {}

        spectra = time_history.spectra
        if spectra is None or len(spectra) == 0:
            return {}

        aggregated: Dict[str, list[float]] = {}
        for spec in spectra:
            row = self._extract_single_spectrum_features(spec)
            for key, value in row.items():
                scalar = self._to_scalar(value)
                if key not in aggregated:
                    aggregated[key] = []
                aggregated[key].append(scalar)

        if not aggregated:
            return {}

        features: Dict[str, Any] = {}
        for key, values in aggregated.items():
            numeric = np.asarray(values, dtype=np.float64)
            features[key] = float(np.nanmean(numeric)) if numeric.size else np.nan

        return features

    def extract_time_history_series(self, wfm: Waveform) -> Dict[str, np.ndarray]:
        """Extract per-frame timbre metrics without averaging.

        Returns a dict mapping each metric name to a 1-D numpy array whose
        length equals the number of frames.
        """
        time_history = self._build_time_history(wfm)
        if time_history is None:
            return {}

        spectra = time_history.spectra
        if spectra is None or len(spectra) == 0:
            return {}

        series: Dict[str, list[float]] = {}
        for spec in spectra:
            row = self._extract_single_spectrum_features(spec)
            for key, value in row.items():
                scalar = self._to_scalar(value)
                if key not in series:
                    series[key] = []
                series[key].append(scalar)

        out: Dict[str, np.ndarray] = {}
        for key, values in series.items():
            out[key] = np.asarray(values, dtype=np.float64)

        return out

    def extract_series_from_array(self, signal: np.ndarray) -> Dict[str, np.ndarray]:
        """Convenience wrapper: numpy array in, per-frame metric arrays out."""
        wfm = self.from_array(signal, self.sample_rate)
        return self.extract_time_history_series(wfm)

    def extract_from_array(self, signal: np.ndarray) -> Dict[str, Any]:
        """Convenience wrapper: numpy array in, one averaged scalar per metric out."""
        wfm = self.from_array(signal, self.sample_rate)
        return self.extract_time_history_features(wfm)
    
    def extract_series_from_file(self, file_path: str) -> Dict[str, np.ndarray]:
        """Load a ``.wav`` file from disk and return per-frame metric arrays.

        Raises ``ValueError`` if the file is missing or contains only a WAV
        header with no actual audio samples (≤ 44 bytes).
        """
        if not os.path.exists(file_path) or os.path.getsize(file_path) <= 44:
            raise ValueError(
                f"File {file_path} is missing or headers only (no audio data)."
            )

        wfm = Waveform.from_wave_file(file_path)
        if not isinstance(wfm, Waveform):
            raise ValueError(f"Failed to load waveform from {file_path}")

        return self.extract_time_history_series(wfm)
