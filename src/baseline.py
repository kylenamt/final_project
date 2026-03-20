"""Deterministic timbre-transfer baselines using the WORLD vocoder.

Methods
-------
- **F0 transfer** – keep source pitch, replace spectral envelope & aperiodicity
  with the target's.
- **F0 + AP transfer** – keep source pitch *and* aperiodicity, replace only the
  spectral envelope.

All methods operate on WORLD's three-way decomposition:
F0  (pitch), spectral envelope (SP), and aperiodicity (AP).
"""

from typing import Optional, Tuple

import librosa
import numpy as np
import pyworld as pw
import soundfile as sf


class Baseline:
    """WORLD-vocoder baseline for timbre transfer.

    Parameters
    ----------
    sample_rate : int
        Working sample rate.  All audio loaded / synthesised uses this rate.
    """

    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def load_audio(self, path: str, sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """Load an audio file as mono float64.

        Parameters
        ----------
        path : str
            Path to the audio file.
        sr : int, optional
            Target sample rate.  Defaults to ``self.sample_rate``.

        Returns
        -------
        (audio, sample_rate)
        """
        sr = sr or self.sample_rate
        y, _ = librosa.load(path, sr=sr, mono=True)
        return y.astype(np.float64), sr

    @staticmethod
    def match_length(y: np.ndarray, target_len: int) -> np.ndarray:
        """Pad (wrapping) or truncate *y* to exactly *target_len* samples."""
        if len(y) >= target_len:
            return y[:target_len]
        return np.pad(y, (0, target_len - len(y)), mode="wrap")

    # ------------------------------------------------------------------
    # WORLD analysis / synthesis
    # ------------------------------------------------------------------

    def decompose(
        self, y: np.ndarray, fs: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """WORLD analysis: returns ``(f0, sp, ap)``.

        Parameters
        ----------
        y : ndarray
            Mono waveform (float64).
        fs : int, optional
            Sample rate.  Defaults to ``self.sample_rate``.
        """
        fs = fs or self.sample_rate
        f0, timeaxis = pw.dio(y, fs)  # type: ignore[attr-defined]
        f0 = pw.stonemask(y, f0, timeaxis, fs)  # type: ignore[attr-defined]
        sp = pw.cheaptrick(y, f0, timeaxis, fs)  # type: ignore[attr-defined]
        ap = pw.d4c(y, f0, timeaxis, fs)  # type: ignore[attr-defined]
        return f0, sp, ap

    @staticmethod
    def synthesize(
        f0: np.ndarray,
        sp: np.ndarray,
        ap: np.ndarray,
        fs: int,
    ) -> np.ndarray:
        """WORLD synthesis from ``(f0, sp, ap)``."""
        return pw.synthesize(f0, sp, ap, fs)  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Transfer methods
    # ------------------------------------------------------------------

    def _validate_pair(self, source_y: np.ndarray, target_y: np.ndarray) -> None:
        if len(target_y) != len(source_y):
            raise ValueError(
                f"Source and target must have the same length "
                f"({len(source_y)} vs {len(target_y)}). "
                f"Use match_length() first."
            )

    def f0_transfer(
        self,
        source_y: np.ndarray,
        target_y: np.ndarray,
        output_path: Optional[str] = None,
    ) -> np.ndarray:
        """Transfer: source F0 + target SP + target AP.

        Keeps the *pitch contour* of the source while adopting the
        *timbre* (spectral envelope + aperiodicity) of the target.
        """
        self._validate_pair(source_y, target_y)

        source_f0, _, _ = self.decompose(source_y)
        _, target_sp, target_ap = self.decompose(target_y)

        out = self.synthesize(source_f0, target_sp, target_ap, self.sample_rate)

        if output_path is not None:
            sf.write(output_path, out, self.sample_rate)
        return out

    def f0_and_ap_transfer(
        self,
        source_y: np.ndarray,
        target_y: np.ndarray,
        output_path: Optional[str] = None,
    ) -> np.ndarray:
        """Transfer: source F0 + target SP + source AP.

        Keeps the *pitch contour* and *breathiness / noise character* of
        the source while adopting the *spectral envelope* of the target.
        """
        self._validate_pair(source_y, target_y)

        source_f0, _, source_ap = self.decompose(source_y)
        _, target_sp, _ = self.decompose(target_y)

        out = self.synthesize(source_f0, target_sp, source_ap, self.sample_rate)

        if output_path is not None:
            sf.write(output_path, out, self.sample_rate)
        return out


    

