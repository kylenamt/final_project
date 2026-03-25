"""Basic audio utilities.

This module centralizes common audio operations used in experiments and
pipelines: loading, saving, channel conversion, normalization, resampling,
segmentation, and simple signal statistics.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf


def load_audio(
	path: str | Path,
	sr: Optional[int] = None,
	mono: bool = True,
	normalize: bool = False,
	dtype: type = np.float32,
) -> Tuple[np.ndarray, int]:
	"""Load audio from disk.

	Parameters
	----------
	path:
		Path to audio file.
	sr:
		Target sampling rate. ``None`` keeps original sample rate.
	mono:
		Convert to mono if True.
	normalize:
		Apply peak normalization if True.
	dtype:
		Output dtype.
	"""
	audio, sample_rate = librosa.load(str(path), sr=sr, mono=mono)
	audio = np.asarray(audio, dtype=dtype)
	if normalize:
		audio = normalize_audio(audio)
	return audio, int(sample_rate)


def save_audio(
	path: str | Path,
	audio: np.ndarray,
	sr: int,
	normalize: bool = False,
) -> None:
	"""Save audio to disk as a waveform file supported by soundfile."""
	out = np.asarray(audio, dtype=np.float32)
	if normalize:
		out = normalize_audio(out)
	Path(path).parent.mkdir(parents=True, exist_ok=True)
	sf.write(str(path), out, sr)


def to_mono(audio: np.ndarray) -> np.ndarray:
	"""Convert audio to mono by averaging channels when needed."""
	arr = np.asarray(audio)
	if arr.ndim == 1:
		return arr
	if arr.ndim == 2:
		return np.mean(arr, axis=0 if arr.shape[0] < arr.shape[1] else 1)
	raise ValueError("audio must be 1D or 2D")


def normalize_audio(audio: np.ndarray, peak: float = 0.99, eps: float = 1e-8) -> np.ndarray:
	"""Peak-normalize audio to a target absolute peak value."""
	arr = np.asarray(audio, dtype=np.float32)
	max_val = np.max(np.abs(arr)) if arr.size else 0.0
	if max_val < eps:
		return arr
	return arr * (peak / max_val)

def clip_audio(
		audio: np.ndarray,
		end_time: float,
		start_time: float = 0.0,
		sr: int = 16000,
	) -> np.ndarray:
		"""Clip audio to a specified time range.

		Parameters
		----------
		audio:
			Audio waveform.
		end_time:
			End time in seconds.
		start_time:
			Start time in seconds. Defaults to 0.0.
		sr:
			Sampling rate used to convert time to samples.

		Returns
		-------
		Clipped audio array.
		"""
		start_sample = int(start_time * sr)
		end_sample = int(end_time * sr)
		return audio[start_sample:end_sample]

__all__ = [
	"load_audio",
	"save_audio",
	"to_mono",
	"normalize_audio",
	"clip_audio",
]
