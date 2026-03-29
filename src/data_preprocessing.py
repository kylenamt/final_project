"""Audio preprocessing: silence trimming, resampling, and fixed-length splitting."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

__all__ = [
    "trim_silence",
    "clip_silence",
    "downsample_audio",
    "split_audio",
]


def trim_silence(
    audio: np.ndarray,
    threshold_db: float = -40.0,
    frame_length: int = 1024,
    hop_length: int = 256,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Trim leading and trailing silence from a 1-D waveform.

    Silence is detected using frame RMS in dB.  Frames with loudness
    below ``threshold_db`` are considered silent.

    Parameters
    ----------
    audio:
        1-D mono waveform.
    threshold_db:
        Silence threshold in dBFS.
    frame_length:
        Number of samples per analysis frame.
    hop_length:
        Hop size between analysis frames.

    Returns
    -------
    Tuple[np.ndarray, Tuple[int, int]]
        ``(trimmed_audio, (start_sample, end_sample))`` where sample
        indices refer to the original waveform.
    """
    n = len(audio)
    if n == 0:
        return audio, (0, 0)
    if frame_length <= 0 or hop_length <= 0:
        raise ValueError("frame_length and hop_length must be > 0.")

    starts = np.arange(0, max(1, n - frame_length + 1), hop_length)
    if starts.size == 0:
        starts = np.array([0])

    rms = np.empty(starts.size, dtype=np.float64)
    for i, s in enumerate(starts):
        end = min(s + frame_length, n)
        frame = audio[s:end]
        rms[i] = np.sqrt(np.mean(frame * frame) + 1e-12)

    db = 20.0 * np.log10(np.maximum(rms, 1e-12))
    non_silent = np.where(db >= threshold_db)[0]

    if non_silent.size == 0:
        return audio, (0, n)

    first = int(non_silent[0])
    last = int(non_silent[-1])
    start_sample = int(starts[first])
    end_sample = int(min(n, starts[last] + frame_length))
    return audio[start_sample:end_sample], (start_sample, end_sample)


def clip_silence(
    in_path: str | Path,
    out_path: Optional[str | Path] = None,
    threshold_db: float = -40.0,
    frame_length: int = 1024,
    hop_length: int = 256,
    mono: bool = True,
    subtype: Optional[str] = "PCM_16",
) -> Tuple[np.ndarray, int, Tuple[int, int]]:
    """Trim leading and trailing silence from an audio file.

    Delegates to :func:`trim_silence` for the core detection logic.

    Parameters
    ----------
    in_path:
        Input audio path.
    out_path:
        Optional output path for the trimmed waveform. If ``None``, audio is
        only returned and not written.
    threshold_db:
        Silence threshold in dBFS.
    frame_length:
        Number of samples per analysis frame.
    hop_length:
        Hop size between analysis frames.
    mono:
        Convert to mono before trimming.
    subtype:
        Output encoding subtype when writing WAV.

    Returns
    -------
    Tuple[np.ndarray, int, Tuple[int, int]]
        ``(trimmed_audio, sample_rate, (start_sample, end_sample))`` where
        sample indices refer to the original waveform.
    """
    in_path = Path(in_path)
    audio, sr = sf.read(in_path, always_2d=True)  # (T, C)

    if mono:
        audio = audio.mean(axis=1, keepdims=True)

    # Work on a mono view for silence detection.
    y = audio[:, 0] if audio.shape[1] == 1 else audio.mean(axis=1)

    _, (start_sample, end_sample) = trim_silence(
        y, threshold_db=threshold_db, frame_length=frame_length,
        hop_length=hop_length,
    )

    trimmed = audio[start_sample:end_sample, :]
    to_write = trimmed[:, 0] if trimmed.shape[1] == 1 else trimmed

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path, to_write, sr, subtype=subtype)

    return to_write, int(sr), (start_sample, end_sample)


def downsample_audio(
    in_path: str | Path,
    out_path: str | Path,
    target_sr: int = 16000,
    mono: bool = True,
    subtype: Optional[str] = "PCM_16",
) -> Tuple[Path, int]:
    """
    Load an audio file, optionally convert to mono, resample to target_sr (default 16 kHz),
    and write to out_path.

    Returns: (out_path_resolved, target_sr)
    """
    in_path = Path(in_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    audio, sr = sf.read(in_path, always_2d=True)  # shape (T, C)

    if mono:
        audio = audio.mean(axis=1, keepdims=True)  # (T, 1)

    if sr != target_sr:
        # Resample each channel with polyphase filtering (high quality, efficient)
        # resample_poly expects shape (T,) so do channel-wise.
        resampled = []
        for ch in range(audio.shape[1]):
            y = audio[:, ch]
            # Use rational approximation sr -> target_sr: up=target_sr, down=sr
            # resample_poly handles gcd internally reasonably, but we can reduce:
            g = np.gcd(sr, target_sr)
            up = target_sr // g
            down = sr // g
            y_rs = resample_poly(y, up=up, down=down)
            resampled.append(y_rs)

        # Stack back to (T_new, C)
        min_len = min(len(r) for r in resampled)
        audio = np.stack([r[:min_len] for r in resampled], axis=1)
        sr = target_sr

    # If mono requested, write 1D to keep common WAV convention
    to_write = audio[:, 0] if audio.shape[1] == 1 else audio
    sf.write(out_path, to_write, sr, subtype=subtype)

    return out_path.resolve(), sr


def split_audio(
    in_path: str | Path,
    out_dir: str | Path,
    n_seconds: float,
    prefix: Optional[str] = None,
    force_sr: Optional[int] = None,
    mono: bool = True,
    subtype: Optional[str] = "PCM_16",
) -> List[Path]:
    """
    Split audio from in_path into fixed-length segments of n_seconds and save as WAV files in out_dir.
    Drops the last remainder if it's shorter than n_seconds.

    - If force_sr is provided, the audio is resampled to that sample rate before splitting.
    - If mono is True, audio is converted to mono before splitting.

    Returns: list of written file paths.
    """
    in_path = Path(in_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    audio, sr = sf.read(in_path, always_2d=True)  # (T, C)

    if mono:
        audio = audio.mean(axis=1, keepdims=True)  # (T, 1)

    if force_sr is not None and sr != force_sr:
        # Resample to force_sr
        resampled = []
        for ch in range(audio.shape[1]):
            y = audio[:, ch]
            g = np.gcd(sr, force_sr)
            up = force_sr // g
            down = sr // g
            y_rs = resample_poly(y, up=up, down=down)
            resampled.append(y_rs)

        min_len = min(len(r) for r in resampled)
        audio = np.stack([r[:min_len] for r in resampled], axis=1)
        sr = force_sr

    seg_len = int(round(n_seconds * sr))
    if seg_len <= 0:
        raise ValueError("n_seconds must be > 0.")

    total_len = audio.shape[0]
    n_full = total_len // seg_len  # drops remainder automatically

    if prefix is None:
        prefix = in_path.stem

    written: List[Path] = []
    for i in range(n_full):
        start = i * seg_len
        end = start + seg_len
        seg = audio[start:end, :]

        out_path = out_dir / f"{prefix}_{i:05d}_{sr}Hz_{n_seconds:g}s.wav"
        to_write = seg[:, 0] if seg.shape[1] == 1 else seg
        sf.write(out_path, to_write, sr, subtype=subtype)
        written.append(out_path.resolve())

    return written

