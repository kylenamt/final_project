import numpy as np
import crepe
from scipy.signal import resample_poly
from math import gcd

try:
    # pip install frechet-audio-distance
    from frechet_audio_distance import FrechetAudioDistance
except Exception:  # pragma: no cover
    FrechetAudioDistance = None  # type: ignore

def _to_numpy(x: np.ndarray) -> np.ndarray:
    if not isinstance(x, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(x)}")
    return x.astype(np.float64, copy=False)

def resample_to_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    """Resample mono audio to 16 kHz using polyphase resampling."""
    if sr == 16000:
        return audio.astype(np.float32, copy=False)
    g = gcd(sr, 16000)
    up = 16000 // g
    down = sr // g
    return resample_poly(audio, up, down).astype(np.float32, copy=False)

def crepe_f0_hz(
    audio: np.ndarray,
    sr: int,
    step_size_ms: int = 10,     # 10 ms hop (common)
    model_capacity: str = "tiny",  # "tiny" | "small" | "medium" | "large" | "full"
    viterbi: bool = True,
    conf_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      time_s: (T,)
      f0_hz:  (T,) with unvoiced set to 0.0 based on conf_threshold
      conf:   (T,) CREPE confidence
    """
    if audio.ndim != 1:
        raise ValueError("audio must be mono 1D waveform")

    x16 = resample_to_16k(audio, sr)

    # CREPE expects float32 in [-1, 1] ideally
    x16 = np.clip(x16, -1.0, 1.0).astype(np.float32, copy=False)

    time_s, f0_hz, conf, _ = crepe.predict(
        x16,
        16000,
        step_size=step_size_ms,
        model_capacity=model_capacity,
        viterbi=viterbi,
        verbose=0,
    )

    f0_hz = f0_hz.astype(np.float64, copy=False)
    conf = conf.astype(np.float64, copy=False)

    # Voicing decision from confidence
    voiced = conf >= conf_threshold
    f0_hz = np.where(voiced, f0_hz, 0.0)

    return time_s, f0_hz, conf


class SNR:
    """Signal-to-noise ratio metric.
    Input: ref_audio, reference audio as numpy array
        est_audio, reconstructed audio as numpy array
    """
    def __call__(self, ref_audio: np.ndarray, est_audio: np.ndarray, eps: float = 1e-12) -> float:
        r = _to_numpy(ref_audio)
        e = _to_numpy(est_audio)
        if r.shape != e.shape:
            raise ValueError(f"ref and est must have same shape: {r.shape} vs {e.shape}")

        num = np.sum(r ** 2)
        den = np.sum((r - e) ** 2)
        return float(10.0 * np.log10((num + eps) / (den + eps)))


class Pitch_centRMSE:
    """Frame-wise pitch RMSE in cents from audio via CREPE.

    Input: ref_audio and est_audio (mono waveforms).
    Voicing decision: frames with CREPE confidence < conf_threshold
    or f0 <= voiced_threshold_hz are excluded.
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
        r_audio = _to_numpy(ref_audio).reshape(-1)
        e_audio = _to_numpy(est_audio).reshape(-1)

        _, f0_ref, _ = crepe_f0_hz(
            r_audio,
            sr,
            step_size_ms=step_size_ms,
            model_capacity=model_capacity,
            viterbi=viterbi,
            conf_threshold=conf_threshold,
        )
        _, f0_est, _ = crepe_f0_hz(
            e_audio,
            sr,
            step_size_ms=step_size_ms,
            model_capacity=model_capacity,
            viterbi=viterbi,
            conf_threshold=conf_threshold,
        )

        r = _to_numpy(f0_ref).reshape(-1)
        e = _to_numpy(f0_est).reshape(-1)
        n_frames = min(len(r), len(e))
        r = r[:n_frames]
        e = e[:n_frames]

        voiced = (r > voiced_threshold_hz) & (e > voiced_threshold_hz)
        if not np.any(voiced):
            return 0.0

        err_cents = 1200.0 * np.log2((e[voiced] + eps) / (r[voiced] + eps))
        rmse_cents = np.sqrt(np.mean(err_cents ** 2))
        return float(rmse_cents)


class FAD:
    """Frechet Audio Distance for directories of audio clips."""

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
                "frechet-audio-distance is not installed. Install with: pip install frechet-audio-distance"
            )
        fad = FrechetAudioDistance(
            model_name=self.model_name,
            sample_rate=self.sample_rate,
            use_pca=self.use_pca,
            use_activation=self.use_activation,
        )
        return float(fad.score(background_dir=real_dir, eval_dir=gen_dir))


