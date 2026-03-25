"""Timbre metrics extraction utilities.

This module provides the ``TimbreMetrics`` class, which extracts frame-based
spectral descriptors from mono audio signals using ``SpectralTimeHistory``.

Quick usage guide:
- ``TimbreMetrics(...)``: configure sample rate, frame width, and frame overlap.
- ``extract_series_from_array(signal)``: returns per-frame arrays for each
    metric (time-history trajectories).
- ``extract_from_array(signal)``: returns one scalar per metric by averaging
    the frame-wise trajectories.
- ``_extract_features(file_path)``: compatibility helper for ``.wav`` file
    paths (loads file then calls array-based extraction).

Notes:
- Input must be a 1D mono numpy array for array-based methods.
- Level metrics are normalized to ``spec_lz``, ``spec_la``, and ``spec_lc``.
- Feature extraction is best-effort per frame; failing metric paths are
    skipped so extraction can continue.
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
        """Create a pytimbre Waveform from a mono numpy array."""
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
        """Convert feature outputs to scalar floats for stable aggregation."""
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
        """Keep compatibility for level keys that were previously prefixed as spectrum metrics."""
        if name in {"la", "lc", "lz"}:
            return f"spec_{name}"
        return name

    def _build_time_history(self, wfm: Waveform) -> Optional[SpectralTimeHistory]:
        """Create a spectral time-history representation from a waveform."""
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
        """Extract robust per-frame metrics while tolerating optional pytimbre failures."""
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
        """Extract and aggregate frame-wise timbre features from SpectralTimeHistory."""
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
        """Extract frame-wise feature trajectories from SpectralTimeHistory."""
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
        """Return per-frame feature arrays from an in-memory mono signal."""
        wfm = self.from_array(signal, self.sample_rate)
        return self.extract_time_history_series(wfm)

    def extract_from_array(self, signal: np.ndarray) -> Dict[str, Any]:
        """Primary extraction API for in-memory segment arrays using frame-wise time history."""
        wfm = self.from_array(signal, self.sample_rate)
        return self.extract_time_history_features(wfm)
    
    def extract_series_from_file(self, file_path: str) -> Dict[str, np.ndarray]:
        """Extract per-frame feature arrays from a mono .wav file."""
        if not os.path.exists(file_path) or os.path.getsize(file_path) <= 44:
            raise ValueError(
                f"File {file_path} is missing or headers only (no audio data)."
            )

        wfm = Waveform.from_wave_file(file_path)
        if not isinstance(wfm, Waveform):
            raise ValueError(f"Failed to load waveform from {file_path}")

        return self.extract_time_history_series(wfm)
