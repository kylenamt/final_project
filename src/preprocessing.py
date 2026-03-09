from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


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