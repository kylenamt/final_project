"""Unified audio evaluation metrics and loss runner.

Metrics
-------
- **SNR** – signal-to-noise ratio (dB)
- **SpectralLoss** – DDSP multi-scale spectral loss (L1/L2, magnitude + log-magnitude)
- **Pitch_centRMSE** – CREPE-based pitch RMSE in cents
- **FAD** – Fréchet Audio Distance (directory-level)

Quick start
-----------
>>> from loss import run_evaluation
>>> results = run_evaluation("data/eval", ref_subdir="original", est_subdir="transferred")
>>> print(results["averages"])
"""

import os
from math import gcd
from typing import Any, Dict, Iterable, List, Optional, Tuple

import crepe
import librosa
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

try:
    import tensorflow.compat.v2 as tf
    import ddsp.losses as _ddsp_losses
except ImportError:  # pragma: no cover
    tf = None  # type: ignore
    _ddsp_losses = None  # type: ignore

try:
    from frechet_audio_distance import FrechetAudioDistance
except Exception:  # pragma: no cover
    FrechetAudioDistance = None  # type: ignore


# ---------------------------------------------------------------------------
# Audio I/O helpers
# ---------------------------------------------------------------------------

AUDIO_EXTS = (".wav", ".flac", ".ogg", ".mp3", ".aif", ".aiff")


def _to_numpy(x: np.ndarray) -> np.ndarray:
    if not isinstance(x, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(x)}")
    return x.astype(np.float64, copy=False)


def resample_to_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    """Resample mono audio to 16 kHz using polyphase resampling."""
    if sr == 16000:
        return audio.astype(np.float32, copy=False)
    g = gcd(sr, 16000)
    return resample_poly(audio, 16000 // g, sr // g).astype(np.float32, copy=False)


def read_audio_mono(
    path: str, target_sr: Optional[int] = None
) -> Tuple[np.ndarray, int]:
    """Read an audio file as mono float32, optionally resampling."""
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32, copy=False)
    if target_sr is not None and sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return audio, sr


def _list_audio_files(directory: str) -> Dict[str, str]:
    """Return ``{stem: path}`` for audio files in *directory*."""
    files: Dict[str, str] = {}
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if os.path.isfile(path) and name.lower().endswith(AUDIO_EXTS):
            files[os.path.splitext(name)[0]] = path
    return files


def _resolve_dir(data_dir: str, subdir: str) -> str:
    d = subdir if os.path.isabs(subdir) else os.path.join(data_dir, subdir)
    if not os.path.isdir(d):
        raise FileNotFoundError(f"Directory not found: {d}")
    return d


def _load_pair(
    ref_path: str, est_path: str, sample_rate: Optional[int]
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Load a matched ref/est pair, align sample rate & length."""
    ref, ref_sr = read_audio_mono(ref_path, sample_rate)
    est, est_sr = read_audio_mono(est_path, sample_rate)
    if ref_sr != est_sr:
        est = librosa.resample(est, orig_sr=est_sr, target_sr=ref_sr)
    n = min(len(ref), len(est))
    return ref[:n], est[:n], ref_sr


# ---------------------------------------------------------------------------
# F0 extraction (CREPE)
# ---------------------------------------------------------------------------

def crepe_f0_hz(
    audio: np.ndarray,
    sr: int,
    step_size_ms: int = 10,
    model_capacity: str = "tiny",
    viterbi: bool = True,
    conf_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract F0 with CREPE.

    Returns ``(time_s, f0_hz, confidence)`` – unvoiced frames are set to 0 Hz.
    """
    if audio.ndim != 1:
        raise ValueError("audio must be mono 1-D waveform")

    x16 = np.clip(resample_to_16k(audio, sr), -1.0, 1.0).astype(np.float32)

    time_s, f0_hz, conf, _ = crepe.predict(
        x16, 16000,
        step_size=step_size_ms,
        model_capacity=model_capacity,
        viterbi=viterbi,
        verbose=0,
    )

    f0_hz = f0_hz.astype(np.float64, copy=False)
    conf = conf.astype(np.float64, copy=False)
    f0_hz = np.where(conf >= conf_threshold, f0_hz, 0.0)
    return time_s, f0_hz, conf


# ---------------------------------------------------------------------------
# Metric classes
# ---------------------------------------------------------------------------

class SNR:
    """Signal-to-noise ratio (dB)."""

    def __call__(
        self, ref_audio: np.ndarray, est_audio: np.ndarray, eps: float = 1e-12
    ) -> float:
        r = _to_numpy(ref_audio)
        e = _to_numpy(est_audio)
        if r.shape != e.shape:
            raise ValueError(f"Shape mismatch: {r.shape} vs {e.shape}")
        return float(10.0 * np.log10(
            (np.sum(r ** 2) + eps) / (np.sum((r - e) ** 2) + eps)
        ))


class SpectralLoss:
    """DDSP multi-scale spectral loss wrapper."""

    def __init__(
        self,
        loss_type: str = "L1",
        mag_weight: float = 1.0,
        logmag_weight: float = 1.0,
    ) -> None:
        if _ddsp_losses is None:
            raise ImportError("ddsp is required for SpectralLoss")
        self._loss = _ddsp_losses.SpectralLoss(
            loss_type=loss_type,
            mag_weight=mag_weight,
            logmag_weight=logmag_weight,
        )

    def __call__(
        self, ref_audio: np.ndarray, est_audio: np.ndarray
    ) -> float:
        ref_t = tf.convert_to_tensor(ref_audio[None, :], dtype=tf.float32)
        est_t = tf.convert_to_tensor(est_audio[None, :], dtype=tf.float32)
        out = self._loss(ref_t, est_t)
        if isinstance(out, dict):
            return float(np.asarray(
                out.get("loss", sum(np.asarray(v) for v in out.values()))
            ))
        return float(np.asarray(out))


class Pitch_centRMSE:
    """Frame-wise pitch RMSE in cents (via CREPE).

    Frames where either ref or est is unvoiced are excluded.
    """

    def __call__(
        self,
        ref_audio: np.ndarray,
        est_audio: np.ndarray,
        sr: int,
        voiced_threshold_hz: float = 1.0,
        conf_threshold: float = 0.5,
        step_size_ms: int = 10,
        model_capacity: str = "tiny",
        viterbi: bool = True,
        eps: float = 1e-12,
    ) -> float:
        _, f0_ref, _ = crepe_f0_hz(
            _to_numpy(ref_audio).ravel(), sr,
            step_size_ms=step_size_ms, model_capacity=model_capacity,
            viterbi=viterbi, conf_threshold=conf_threshold,
        )
        _, f0_est, _ = crepe_f0_hz(
            _to_numpy(est_audio).ravel(), sr,
            step_size_ms=step_size_ms, model_capacity=model_capacity,
            viterbi=viterbi, conf_threshold=conf_threshold,
        )

        n = min(len(f0_ref), len(f0_est))
        r, e = f0_ref[:n], f0_est[:n]

        voiced = (r > voiced_threshold_hz) & (e > voiced_threshold_hz)
        if not np.any(voiced):
            return 0.0

        err = 1200.0 * np.log2((e[voiced] + eps) / (r[voiced] + eps))
        return float(np.sqrt(np.mean(err ** 2)))


class FAD:
    """Fréchet Audio Distance (directory-level)."""

    def __init__(
        self,
        model_name: str = "vggish",
        sample_rate: int = 16000,
        use_pca: bool = True,
        use_activation: bool = True,
    ) -> None:
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.use_pca = use_pca
        self.use_activation = use_activation

    def __call__(self, real_dir: str, gen_dir: str) -> float:
        if FrechetAudioDistance is None:
            raise ImportError(
                "frechet-audio-distance is not installed. "
                "Install with: pip install frechet-audio-distance"
            )
        fad = FrechetAudioDistance(
            model_name=self.model_name,
            sample_rate=self.sample_rate,
            use_pca=self.use_pca,
            use_activation=self.use_activation,
        )
        return float(fad.score(background_dir=real_dir, eval_dir=gen_dir))


# ---------------------------------------------------------------------------
# Unified evaluation runner
# ---------------------------------------------------------------------------

_METRIC_KEYS = {
    "snr": "snr",
    "spectral": "spectral_loss",
    "pitch": "pitch_cent_rmse",
    "fad": "fad",
}


def run_evaluation(
    data_dir: str,
    ref_subdir: str = "ref",
    est_subdir: str = "est",
    metrics: Iterable[str] = ("snr", "spectral", "pitch"),
    sample_rate: Optional[int] = None,
    # SpectralLoss params
    loss_type: str = "L1",
    mag_weight: float = 1.0,
    logmag_weight: float = 1.0,
    # Pitch params
    conf_threshold: float = 0.5,
    voiced_threshold_hz: float = 1.0,
    # FAD params
    fad_sample_rate: int = 16000,
) -> Dict[str, object]:
    """Evaluate audio quality between reference and estimated directories.

    Parameters
    ----------
    metrics : iterable of ``{"snr", "spectral", "pitch", "fad"}``
        Which metrics to compute.  ``"fad"`` is directory-level; the rest
        are per-file.

    Returns
    -------
    dict with keys ``reference_dir``, ``estimated_dir``, ``num_files``,
    ``averages``, ``per_file``, and optionally ``fad``.
    """
    metrics_set = set(m.lower() for m in metrics)
    ref_dir = _resolve_dir(data_dir, ref_subdir)
    est_dir = _resolve_dir(data_dir, est_subdir)

    result: Dict[str, object] = {
        "reference_dir": ref_dir,
        "estimated_dir": est_dir,
    }

    # FAD is directory-level (not per-file)
    if "fad" in metrics_set:
        result["fad"] = FAD(sample_rate=fad_sample_rate)(ref_dir, est_dir)
        metrics_set.discard("fad")

    if not metrics_set:
        return result

    # Discover matched file pairs
    ref_files = _list_audio_files(ref_dir)
    est_files = _list_audio_files(est_dir)
    common = sorted(set(ref_files) & set(est_files))
    if not common:
        raise FileNotFoundError(
            "No matching audio filenames between ref and est directories"
        )

    # Instantiate requested per-file metrics
    runners: Dict[str, Any] = {}
    if "snr" in metrics_set:
        runners["snr"] = SNR()
    if "spectral" in metrics_set:
        runners["spectral"] = SpectralLoss(loss_type, mag_weight, logmag_weight)
    if "pitch" in metrics_set:
        runners["pitch"] = Pitch_centRMSE()

    per_file: List[Dict[str, object]] = []
    accum: Dict[str, List[float]] = {k: [] for k in runners}

    for stem in common:
        ref_audio, est_audio, sr = _load_pair(
            ref_files[stem], est_files[stem], sample_rate
        )
        row: Dict[str, object] = {"file": stem}

        if "snr" in runners:
            v = runners["snr"](ref_audio, est_audio)
            row["snr"] = v
            accum["snr"].append(v)

        if "spectral" in runners:
            v = runners["spectral"](ref_audio, est_audio)
            row["spectral_loss"] = v
            accum["spectral"].append(v)

        if "pitch" in runners:
            v = runners["pitch"](
                ref_audio, est_audio, sr,
                conf_threshold=conf_threshold,
                voiced_threshold_hz=voiced_threshold_hz,
            )
            row["pitch_cent_rmse"] = v
            accum["pitch"].append(v)

        per_file.append(row)

    result["num_files"] = len(common)
    result["averages"] = {
        _METRIC_KEYS[k]: float(np.mean(vals)) if vals else 0.0
        for k, vals in accum.items()
    }
    result["per_file"] = per_file
    return result


