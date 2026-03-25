"""Non-overlapping audio segmentation utilities."""

from __future__ import annotations

from typing import Optional

import librosa
import numpy as np


class AudioSegmenter:
    def __init__(
        self,
        sr: int,
        bpm: Optional[float] = None,
        beats: int = 1,
        L: Optional[int] = None,
    ):
        """Create a non-overlapping audio segmenter.

        Default workflow is BPM-driven: BPM is estimated from the signal in ``fit``
        when not provided, then segment length is derived from ``beats``.

        Set ``L`` to force fixed-length segmentation in samples.
        """
        if beats < 1:
            raise ValueError("beats must be >= 1")

        self.sr = sr
        self.beats = beats
        self.bpm = bpm
        self.L = L

        if self.bpm is not None and self.bpm <= 0:
            raise ValueError("bpm must be > 0")

        if self.L is not None and self.L < 1:
            raise ValueError("L must be >= 1")

        self.N = None
        self.M = None
        self.n_tail = None

    @staticmethod
    def estimate_bpm(signal: np.ndarray, sr: int = 16000) -> float:
        """Estimate tempo in beats per minute from a mono signal."""
        tempo, _ = librosa.beat.beat_track(y=signal, sr=sr)
        return float(tempo)

    def _resolve_segment_length(self, signal: np.ndarray) -> int:
        """Resolve ``L`` using fixed length or BPM-derived beat length."""
        if self.L is not None:
            return self.L

        if self.bpm is None:
            self.bpm = self.estimate_bpm(signal, sr=self.sr)

        if self.bpm is None or self.bpm <= 0:
            raise ValueError("Unable to derive valid bpm from signal")

        L = int(np.floor(60 * self.sr * self.beats / self.bpm))
        if L < 1:
            raise ValueError("Derived segment length L must be >= 1")

        self.L = L
        return self.L

    def fit(self, signal: np.ndarray) -> "AudioSegmenter":
        """Store segmentation metadata after resolving BPM and segment length."""
        L = self._resolve_segment_length(signal)
        self.N = len(signal)
        self.M = self.N // L
        self.n_tail = self.N - self.M * L
        return self

    def transform(self, signal: np.ndarray) -> list[np.ndarray]:
        """Segment a signal into non-overlapping windows of exact length L."""
        if self.N is None or self.M is None or self.n_tail is None:
            raise RuntimeError("fit() must be called before transform()")
        if self.L is None:
            raise RuntimeError("segment length L is not initialized; call fit() first")
        L = self.L
        M = self.M
        return [signal[m * L : m * L + L] for m in range(M)]

    def fit_transform(self, signal: np.ndarray) -> list[np.ndarray]:
        """Estimate BPM if needed, fit metadata, and return segments."""
        self.fit(signal)
        return self.transform(signal)

    @property
    def n_segments(self) -> int:
        """Number of complete, non-overlapping segments."""
        if self.M is None:
            raise RuntimeError("fit() must be called before accessing n_segments")
        return self.M

    @property
    def tail_samples(self) -> int:
        """Number of discarded samples at the signal tail."""
        if self.n_tail is None:
            raise RuntimeError("fit() must be called before accessing tail_samples")
        return self.n_tail
