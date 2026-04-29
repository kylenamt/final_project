"""Clip-level noisiness metrics for synthesized audio.

This module provides a small, modular set of metrics focused on audible
noise artifacts (hiss, buzz, roughness) in generated audio.

Implemented metrics
-------------------
- ``hnr_db``: harmonic-to-noise ratio from frame autocorrelation (higher is cleaner).
- ``hf_noise_ratio``: ratio of high-frequency energy to full-band energy.
- ``spectral_flatness_median``: median short-time spectral flatness.
- ``roughness_flux``: high-band spectral flux (frame-to-frame instability).

The class is designed to let callers choose any subset of these metrics.

Example
-------
>>> nm = NoiseMetrics(sample_rate=16000)
>>> nm.evaluate(signal, metrics=["hnr_db", "roughness_flux"])
>>> nm.evaluate_file("example.wav", metrics=NoiseMetrics.METRIC_NAMES)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import librosa
import numpy as np
from scipy.signal import stft as scipy_stft

__all__ = ["NoiseMetrics"]


class NoiseMetrics:
    """Noisiness evaluator with four modular, clip-level metrics.

    Parameters
    ----------
    sample_rate : int
        Audio sample rate used for analysis.
    n_fft : int
        STFT FFT size.
    hop_length : int
        STFT hop size in samples.
    win_length : int or None
        STFT window size; defaults to ``n_fft``.
    f0_min_hz, f0_max_hz : float
        F0 lag search range for HNR estimation.
    hf_cutoff_hz : float
        Cutoff for high-frequency noise ratio.
    flux_band_hz : float
        Lower frequency bound for roughness-flux band.
    rms_floor : float
        Absolute floor to reject near-silent frames.
    eps : float
        Numerical stability epsilon.
    """

    METRIC_HNR_DB = "hnr_db"
    METRIC_HF_NOISE_RATIO = "hf_noise_ratio"
    METRIC_SPECTRAL_FLATNESS = "spectral_flatness_median"
    METRIC_SPECTRAL_FLATNESS_HF = "spectral_flatness_highband_median"
    METRIC_ROUGHNESS_FLUX = "roughness_flux"

    METRIC_NAMES: Tuple[str, ...] = (
        METRIC_HNR_DB,
        METRIC_HF_NOISE_RATIO,
        METRIC_SPECTRAL_FLATNESS,
        METRIC_SPECTRAL_FLATNESS_HF,
        METRIC_ROUGHNESS_FLUX,
    )

    _DEFAULT_WEIGHTS: Mapping[str, float] = {
        METRIC_HNR_DB: 0.40,
        METRIC_HF_NOISE_RATIO: 0.25,
        METRIC_SPECTRAL_FLATNESS: 0.20,
        METRIC_ROUGHNESS_FLUX: 0.15,
    }

    # Normalization ranges for composite noisiness index.
    _RANGES: Mapping[str, Tuple[float, float]] = {
        METRIC_HNR_DB: (-5.0, 25.0),
        METRIC_HF_NOISE_RATIO: (0.05, 0.45),
        METRIC_SPECTRAL_FLATNESS: (0.02, 0.40),
        METRIC_ROUGHNESS_FLUX: (0.02, 0.35),
    }

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: Optional[int] = None,
        f0_min_hz: float = 80.0,
        f0_max_hz: float = 1200.0,
        hf_cutoff_hz: float = 3500.0,
        flux_band_hz: float = 2500.0,
        rms_floor: float = 1e-4,
        eps: float = 1e-10,
    ):
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if n_fft <= 0:
            raise ValueError("n_fft must be positive")
        if hop_length <= 0:
            raise ValueError("hop_length must be positive")
        if win_length is not None and hop_length > win_length:
            raise ValueError("hop_length must be <= win_length")
        if f0_min_hz <= 0 or f0_max_hz <= 0 or f0_min_hz >= f0_max_hz:
            raise ValueError("Require 0 < f0_min_hz < f0_max_hz")

        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length) if win_length is not None else int(n_fft)
        self.f0_min_hz = float(f0_min_hz)
        self.f0_max_hz = float(f0_max_hz)
        self.hf_cutoff_hz = float(hf_cutoff_hz)
        self.flux_band_hz = float(flux_band_hz)
        self.rms_floor = float(rms_floor)
        self.eps = float(eps)

        self._registry = {
            self.METRIC_HNR_DB: self._metric_hnr_db,
            self.METRIC_HF_NOISE_RATIO: self._metric_hf_noise_ratio,
            self.METRIC_SPECTRAL_FLATNESS: self._metric_spectral_flatness_median,
            self.METRIC_SPECTRAL_FLATNESS_HF: self._metric_spectral_flatness_highband_median,
            self.METRIC_ROUGHNESS_FLUX: self._metric_roughness_flux,
        }

    @classmethod
    def available_metrics(cls) -> Tuple[str, ...]:
        """Return supported metric names."""
        return cls.METRIC_NAMES

    def evaluate(
        self,
        signal: np.ndarray,
        metrics: Optional[Sequence[str]] = None,
    ) -> Dict[str, float]:
        """Compute selected noise metrics for a mono waveform.

        Parameters
        ----------
        signal : np.ndarray
            1-D waveform.
        metrics : sequence of str or None
            Which metrics to compute. ``None`` means all four.
        """
        names = self._resolve_metrics(metrics)
        y = self._prepare_signal(signal)

        if y.size == 0:
            return {name: float("nan") for name in names}

        cache: Dict[str, np.ndarray] = {}
        out: Dict[str, float] = {}
        for name in names:
            out[name] = float(self._registry[name](y, cache))
        return out

    def evaluate_file(
        self,
        file_path: str | Path,
        metrics: Optional[Sequence[str]] = None,
    ) -> Dict[str, float]:
        """Load an audio file and evaluate selected metrics.

        Supports any format librosa/audioread can decode (wav, mp3, opus,
        flac, ogg, ...). Audio is downmixed to mono and resampled to
        ``self.sample_rate``.
        """
        audio, _ = librosa.load(str(file_path), sr=self.sample_rate, mono=True)
        return self.evaluate(np.asarray(audio, dtype=np.float64), metrics=metrics)

    @classmethod
    def noisiness_index(
        cls,
        values: Mapping[str, float],
        weights: Optional[Mapping[str, float]] = None,
    ) -> float:
        """Build a single [0, 1] noisiness score from metric outputs.

        Lower HNR means noisier, so HNR is inverted before aggregation.
        Other metrics map directly (higher means noisier).

        This method uses only keys present in *values*.
        """
        w_map = dict(cls._DEFAULT_WEIGHTS)
        if weights is not None:
            w_map.update(weights)

        total_w = 0.0
        total = 0.0
        for name, raw in values.items():
            if name not in cls._RANGES:
                continue
            if not np.isfinite(raw):
                continue
            lo, hi = cls._RANGES[name]
            span = max(hi - lo, 1e-12)
            norm = float(np.clip((raw - lo) / span, 0.0, 1.0))
            if name == cls.METRIC_HNR_DB:
                norm = 1.0 - norm
            weight = float(w_map.get(name, 0.0))
            if weight <= 0:
                continue
            total += weight * norm
            total_w += weight

        if total_w <= 0:
            return float("nan")
        return float(total / total_w)

    # ------------------------------------------------------------------
    # Metric implementations
    # ------------------------------------------------------------------

    def _metric_hnr_db(self, signal: np.ndarray, cache: Dict[str, np.ndarray]) -> float:
        """Harmonic-to-noise ratio in dB (median over valid frames)."""
        frames = self._get_frames(signal, cache)
        if frames.size == 0:
            return float("nan")

        rms = np.sqrt(np.mean(frames * frames, axis=1))
        rms_thr = max(self.rms_floor, float(np.percentile(rms, 20.0)) * 0.5)
        valid = np.where(rms > rms_thr)[0]
        if valid.size == 0:
            return float("nan")

        lag_min = max(1, int(self.sample_rate / self.f0_max_hz))
        lag_max = max(lag_min + 1, int(self.sample_rate / self.f0_min_hz))

        win = np.hanning(self.win_length)
        vals = []
        for idx in valid:
            x = frames[idx] * win
            x = x - np.mean(x)
            energy = float(np.dot(x, x))
            if energy <= self.eps:
                continue

            ac = np.correlate(x, x, mode="full")[self.win_length - 1 :]
            ac = ac / (ac[0] + self.eps)

            upper = min(lag_max, ac.shape[0] - 1)
            if upper <= lag_min:
                continue

            peak = float(np.max(ac[lag_min : upper + 1]))
            if not np.isfinite(peak) or peak <= 0.0:
                continue

            peak = float(np.clip(peak, self.eps, 1.0 - self.eps))
            vals.append(10.0 * np.log10(peak / (1.0 - peak)))

        if not vals:
            return float("nan")
        return float(np.median(vals))

    def _metric_hf_noise_ratio(
        self,
        signal: np.ndarray,
        cache: Dict[str, np.ndarray],
    ) -> float:
        """Median high-frequency energy ratio per frame."""
        power, freqs = self._get_power_spectrogram(signal, cache)
        if power.size == 0:
            return float("nan")

        hf_mask = freqs >= self.hf_cutoff_hz
        if not np.any(hf_mask):
            return float("nan")

        total = np.sum(power, axis=0)
        hf = np.sum(power[hf_mask], axis=0)

        floor = self._energy_floor(total)
        valid = total > floor
        if not np.any(valid):
            return float("nan")

        ratios = hf[valid] / (total[valid] + self.eps)
        return float(np.median(ratios))

    def _metric_spectral_flatness_median(
        self,
        signal: np.ndarray,
        cache: Dict[str, np.ndarray],
    ) -> float:
        """Median short-time spectral flatness."""
        power, _ = self._get_power_spectrogram(signal, cache)
        if power.size == 0:
            return float("nan")

        spec = power + self.eps
        geo = np.exp(np.mean(np.log(spec), axis=0))
        arith = np.mean(spec, axis=0)
        flat = geo / (arith + self.eps)

        total = np.sum(power, axis=0)
        floor = self._energy_floor(total)
        valid = total > floor
        if flat.shape[0] != valid.shape[0]:
            n = min(flat.shape[0], valid.shape[0])
            flat = flat[:n]
            valid = valid[:n]

        if not np.any(valid):
            return float("nan")
        return float(np.median(flat[valid]))

    def _metric_spectral_flatness_highband_median(
        self,
        signal: np.ndarray,
        cache: Dict[str, np.ndarray],
    ) -> float:
        """Median spectral flatness restricted to high-frequency band."""

        power, freqs = self._get_power_spectrogram(signal, cache)
        if power.size == 0:
            return float("nan")

        # Select high-frequency bins
        hf_mask = freqs >= self.hf_cutoff_hz
        if not np.any(hf_mask):
            return float("nan")

        band_power = power[hf_mask, :]
        if band_power.size == 0:
            return float("nan")

        # Same flatness formula: geometric mean / arithmetic mean
        spec = band_power + self.eps
        geo = np.exp(np.mean(np.log(spec), axis=0))
        arith = np.mean(spec, axis=0)
        flat = geo / (arith + self.eps)

        # Exclude low-energy frames based on full-band energy,
        # so the energy floor logic is shared with other metrics.
        total = np.sum(power, axis=0)
        floor = self._energy_floor(total)
        valid = total > floor

        if flat.shape[0] != valid.shape[0]:
            n = min(flat.shape[0], valid.shape[0])
            flat = flat[:n]
            valid = valid[:n]

        if not np.any(valid):
            return float("nan")

        return float(np.median(flat[valid]))

    def _metric_roughness_flux(
        self,
        signal: np.ndarray,
        cache: Dict[str, np.ndarray],
    ) -> float:
        """Median high-band spectral flux on log-power spectra."""
        power, freqs = self._get_power_spectrogram(signal, cache)
        if power.size == 0:
            return float("nan")

        band_mask = freqs >= self.flux_band_hz
        if not np.any(band_mask):
            return float("nan")

        band = np.log1p(power[band_mask])
        if band.shape[1] < 2:
            return float("nan")

        delta = np.diff(band, axis=1)
        flux = np.sqrt(np.mean(delta * delta, axis=0))

        total = np.sum(power, axis=0)
        floor = self._energy_floor(total)
        valid = total[1:] > floor
        if not np.any(valid):
            return float("nan")

        return float(np.median(flux[valid]))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_metrics(self, metrics: Optional[Sequence[str]]) -> Tuple[str, ...]:
        if metrics is None:
            return self.METRIC_NAMES

        # Keep order and remove duplicates.
        names = tuple(dict.fromkeys(metrics))
        unknown = [name for name in names if name not in self._registry]
        if unknown:
            raise ValueError(
                f"Unknown metrics: {unknown}. Available: {list(self.METRIC_NAMES)}"
            )
        if not names:
            raise ValueError("metrics must not be empty")
        return names

    def _prepare_signal(self, signal: np.ndarray) -> np.ndarray:
        y = np.asarray(signal, dtype=np.float64)
        if y.ndim != 1:
            raise ValueError("signal must be a 1D mono waveform")
        if y.size == 0:
            return y
        return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    def _get_power_spectrogram(
        self,
        signal: np.ndarray,
        cache: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        if "power" in cache and "freqs" in cache:
            return cache["power"], cache["freqs"]

        noverlap = self.win_length - self.hop_length
        freqs, _, zxx = scipy_stft(
            signal,
            fs=self.sample_rate,
            window="hann",
            nperseg=self.win_length,
            noverlap=noverlap,
            nfft=self.n_fft,
            boundary="zeros",
            padded=True,
        )
        power = np.abs(zxx) ** 2

        cache["power"] = power
        cache["freqs"] = freqs
        return power, freqs

    def _get_frames(self, signal: np.ndarray, cache: Dict[str, np.ndarray]) -> np.ndarray:
        if "frames" in cache:
            return cache["frames"]

        if signal.size < self.win_length:
            padded = np.pad(signal, (0, self.win_length - signal.size), mode="constant")
        else:
            extra = (signal.size - self.win_length) % self.hop_length
            pad_right = (self.hop_length - extra) % self.hop_length
            padded = np.pad(signal, (0, pad_right), mode="constant")

        n_frames = 1 + (padded.size - self.win_length) // self.hop_length
        frames = np.empty((n_frames, self.win_length), dtype=np.float64)
        for i in range(n_frames):
            start = i * self.hop_length
            frames[i] = padded[start : start + self.win_length]

        cache["frames"] = frames
        return frames

    def _energy_floor(self, frame_energy: np.ndarray) -> float:
        if frame_energy.size == 0:
            return self.eps
        return max(self.eps, float(np.percentile(frame_energy, 20.0)) * 0.1)
