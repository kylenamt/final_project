"""Deterministic timbre-transfer baselines using the WORLD vocoder.

Methods
-------
- **F0 transfer** – keep target pitch, replace spectral envelope & aperiodicity
  with the source's.
- **F0 + AP transfer** – keep target pitch *and* aperiodicity, replace only the
  spectral envelope.

All methods operate on WORLD's three-way decomposition:
F0  (pitch), spectral envelope (SP), and aperiodicity (AP).
"""
from __future__ import annotations

from typing import Optional, Tuple

import librosa
import numpy as np
import pyworld as pw
import soundfile as sf
from ddsp.spectral_ops import compute_f0
from scipy.signal import get_window
try:
    from smstools.models import hpsModel as _HPS
    _HAS_SMS_TOOLS = True
except ImportError:
    _HAS_SMS_TOOLS = False

__all__ = [
    "Baseline",
    "BaselineSMS",
]


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

    @staticmethod
    def match_length(target_y: np.ndarray, source_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Pad (wrapping) or truncate *y* to exactly *source_len* samples."""
        if len(source_y) >= len(target_y):
            return target_y,source_y[:len(target_y)]
        # Pad target by wrapping until it's long enough, then truncate to exact length.        
        return target_y,np.pad(source_y, (0, len(target_y) - len(source_y)), mode="wrap")
    
    # ------------------------------------------------------------------
    

    # Frame period (ms) used for both WORLD analysis and CREPE (frame_rate=200 → 5 ms).
    _FRAME_PERIOD_MS = 5.0

    def _estimate_f0(self, y: np.ndarray, use_crepe: bool) -> np.ndarray:
        """Return an f0 contour at 5 ms hop, using CREPE or Harvest+Stonemask."""
        if use_crepe:
            f0 = self.crepe_f0(y)[0]
        else:
            f0, t = pw.harvest(y, self.sample_rate, frame_period=self._FRAME_PERIOD_MS)  # type: ignore[attr-defined]
            f0 = pw.stonemask(y, f0, t, self.sample_rate)  # type: ignore[attr-defined]
        return np.asarray(f0, dtype=np.float64)

    def _compute_sp_ap(
        self, y: np.ndarray, f0: np.ndarray, fs: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute WORLD spectral envelope and aperiodicity for a given f0."""
        fs = fs or self.sample_rate
        f0 = np.asarray(f0, dtype=np.float64)
        t = np.arange(len(f0)) * (self._FRAME_PERIOD_MS / 1000.0)
        sp = pw.cheaptrick(y, f0, t, fs)  # type: ignore[attr-defined]
        ap = pw.d4c(y, f0, t, fs)  # type: ignore[attr-defined]
        return sp, ap

    @staticmethod
    def synthesize(
        f0: np.ndarray,
        sp: np.ndarray,
        ap: np.ndarray,
        fs: int,
    ) -> np.ndarray:
        """WORLD synthesis from ``(f0, sp, ap)``."""
        f0 = f0.astype('float64', copy=False)
        sp = sp.astype('float64', copy=False)
        ap = ap.astype('float64', copy=False)
        return pw.synthesize(f0, sp, ap, fs)  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Transfer methods
    # ------------------------------------------------------------------

    def _validate_pair(self, target_y: np.ndarray, source_y: np.ndarray) -> None:
        if len(source_y) != len(target_y):
            raise ValueError(
                f"Source and target must have the same length "
                f"({len(target_y)} vs {len(source_y)}). "
                f"Use match_length() first."
            )

    def crepe_f0(self, y: np.ndarray, fs: Optional[int] = None):
        """Compute F0 using CREPE, which may be more robust than WORLD's built-in F0 estimator."""
        return compute_f0(y, frame_rate=200)

    def f0_transfer(
        self,
        target_y: np.ndarray,
        source_y: np.ndarray,
        output_path: Optional[str] = None,
        use_crepe: bool = True,
        target_f0: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Transfer: target F0 + source SP + source AP.

        Keeps the *pitch contour* of the target while adopting the
        *timbre* (spectral envelope + aperiodicity) of the source.

        If *target_f0* is provided it is used directly and both CREPE and the
        target-side WORLD decomposition are skipped.  This allows callers to
        precompute CREPE F0 contours in a GPU-bound main process and hand
        them off to CPU workers.
        """
        self._validate_pair(target_y, source_y)

        if target_f0 is None:
            target_f0 = self._estimate_f0(target_y, use_crepe)
        source_f0 = self._estimate_f0(source_y, use_crepe)
        source_sp, source_ap = self._compute_sp_ap(source_y, source_f0)

        out = self.synthesize(target_f0, source_sp, source_ap, self.sample_rate)

        if output_path is not None:
            sf.write(output_path, out, self.sample_rate)
        return out

    def f0_and_ap_transfer(
        self,
        target_y: np.ndarray,
        source_y: np.ndarray,
        output_path: Optional[str] = None,
        use_crepe: bool = False,
        target_f0: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Transfer: target F0 + source SP + target AP.

        Keeps the *pitch contour* and *breathiness / noise character* of
        the target while adopting the *spectral envelope* of the source.

        If *target_f0* is provided it replaces both the WORLD-estimated F0
        and the CREPE path, letting callers precompute F0 on a GPU main
        process and hand it off to CPU workers.
        """
        self._validate_pair(target_y, source_y)

        if target_f0 is None:
            target_f0 = self._estimate_f0(target_y, use_crepe)
        _, target_ap = self._compute_sp_ap(target_y, target_f0)

        source_f0 = self._estimate_f0(source_y, use_crepe)
        source_sp, _ = self._compute_sp_ap(source_y, source_f0)

        out = self.synthesize(target_f0, source_sp, target_ap, self.sample_rate)

        if output_path is not None:
            sf.write(output_path, out, self.sample_rate)
        return out



# ─────────────────────────────────────────────────────────────────────────────
# SMS decomposition result
# ─────────────────────────────────────────────────────────────────────────────

class SMSDecomposition:
    """Container for SMS/HPS decomposition components.

    Attributes
    ----------
    f0       : ndarray, shape (n_frames,)
        Fundamental frequency per frame in Hz (0 = unvoiced).
    hfreq    : ndarray, shape (n_frames, nH)
        Harmonic track frequencies in Hz (0 = inactive track).
    hmag     : ndarray, shape (n_frames, nH)
        Harmonic track magnitudes in dB.
    hphase   : ndarray, shape (n_frames, nH)
        Harmonic track phases in radians.
    stocEnv  : ndarray, shape (n_frames, K)
        Stochastic residual envelope per frame.
    H, Ns, fs : int   Analysis / synthesis parameters.
    """

    def __init__(self, raw: dict) -> None:
        self.hfreq   = raw['hfreq']       # (F, nH)
        self.hmag    = raw['hmag']        # (F, nH)  dB
        self.hphase  = raw['hphase']      # (F, nH)  rad
        self.stocEnv = raw['stocEnv']     # (F, K)
        self.H       = raw['H']
        self.Ns      = raw['Ns']
        self.fs      = raw['fs']
        self.f0      = self.hfreq[:, 0].copy()

    def __iter__(self):
        """Unpack as ``f0, hmag, stocEnv = decomposition``."""
        yield self.f0
        yield self.hmag
        yield self.stocEnv


# ─────────────────────────────────────────────────────────────────────────────
# BaselineSMS
# ─────────────────────────────────────────────────────────────────────────────

class BaselineSMS:
    """SMS/HPS-vocoder baseline for timbre transfer.

    Uses ``smstools.models.hpsModel`` (Harmonic Plus Stochastic) for
    analysis and synthesis.  Drop-in API replacement for the WORLD-based
    ``Baseline`` class.

    Parameters
    ----------
    sample_rate : int
        Working sample rate.
    window : str
        scipy window name for analysis frames (default ``'blackmanharris'``).
    M : int
        Analysis window length in samples.
    N : int
        Analysis FFT size.
    H : int
        Hop size in samples.
    Ns : int
        Synthesis FFT size (default 512).
    n_harm : int
        Maximum number of harmonics tracked per frame.
    min_f0, max_f0 : float
        Search range for the f0 estimator in Hz.
    t : float
        Peak-detection threshold in negative dB (default -80).
    f0et : float
        F0 error threshold for TWM algorithm (default 5.0).
    harmDevSlope : float
        Slope of harmonic deviation (default 0.01).
    minSineDur : float
        Minimum duration of harmonic tracks in seconds (default 0.02).
    stocf : float
        Decimation factor for stochastic envelope, 0 < stocf <= 1
        (default 0.1).
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        *,
        window:       str   = 'blackmanharris',
        M:            int   = 1001,
        N:            int   = 2048,
        H:            int   = 256,
        Ns:           int   = 512,
        n_harm:       int   = 20,
        min_f0:       float = 80.0,
        max_f0:       float = 1200.0,
        t:            float = -80,
        f0et:         float = 5.0,
        harmDevSlope: float = 0.01,
        minSineDur:   float = 0.02,
        stocf:        float = 0.1,
    ) -> None:
        if not _HAS_SMS_TOOLS:
            raise ImportError(
                "BaselineSMS requires the 'sms-tools' package.  "
                "Install it with `pip install sms-tools` or use the "
                "WORLD-based Baseline class instead."
            )
        self.sample_rate   = sample_rate
        self._window       = window
        self._M            = M
        self._N            = N
        self._H            = H
        self._Ns           = Ns
        self._n_harm       = n_harm
        self._min_f0       = min_f0
        self._max_f0       = max_f0
        self._t            = t
        self._f0et         = f0et
        self._harmDevSlope = harmDevSlope
        self._minSineDur   = minSineDur
        self._stocf        = stocf

    # ------------------------------------------------------------------
    # I/O helpers  (mirrors Baseline)
    # ------------------------------------------------------------------

    @staticmethod
    def match_length(
        target_y: np.ndarray,
        source_y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(target_y, source_y)`` with lengths matched.

        Truncates if source is longer, wraps (pads) if shorter.
        """
        if len(source_y) >= len(target_y):
            return target_y, source_y[:len(target_y)]
        return target_y, np.pad(
            source_y,
            (0, len(target_y) - len(source_y)),
            mode='wrap',
        )

    # ------------------------------------------------------------------
    # Decompose / synthesize
    # ------------------------------------------------------------------

    def decompose(
        self,
        y:  np.ndarray,
        fs: Optional[int] = None,
    ) -> SMSDecomposition:
        """HPS analysis → :class:`SMSDecomposition`.

        Iterable as ``f0, hmag, stocEnv = baseline.decompose(y)``.
        """
        fs = fs or self.sample_rate
        w  = get_window(self._window, self._M)
        w  = w / w.sum()

        hfreq, hmag, hphase, stocEnv = _HPS.hpsModelAnal(
            y, fs, w, self._N, self._H, self._t, self._n_harm,
            self._min_f0, self._max_f0, self._f0et,
            self._harmDevSlope, self._minSineDur, self._Ns, self._stocf,
        )
        return SMSDecomposition({
            'hfreq': hfreq, 'hmag': hmag, 'hphase': hphase,
            'stocEnv': stocEnv,
            'H': self._H, 'Ns': self._Ns, 'fs': fs,
        })

    def synthesize_hps(
        self,
        hfreq:   np.ndarray,
        hmag:    np.ndarray,
        hphase:  np.ndarray,
        stocEnv: np.ndarray,
    ) -> np.ndarray:
        """HPS synthesis wrapper.  Returns the combined waveform."""
        y, _yh, _yst = _HPS.hpsModelSynth(
            hfreq, hmag, hphase, stocEnv,
            self._Ns, self._H, self.sample_rate,
        )
        return y

    # ------------------------------------------------------------------
    # CREPE f0 override
    # ------------------------------------------------------------------

    def _crepe_f0(self, y: np.ndarray, n_hps_frames: int) -> np.ndarray:
        """Compute CREPE f0 and resample to *n_hps_frames* HPS frames."""
        f0_crepe = compute_f0(y, frame_rate=200)[0]
        # Nearest-neighbour resample to avoid interpolation artefacts
        # around unvoiced (f0=0) boundaries.
        idx = np.round(
            np.linspace(0, len(f0_crepe) - 1, n_hps_frames)
        ).astype(int)
        return np.asarray(f0_crepe[idx], dtype=np.float64)

    @staticmethod
    def _replace_f0(dec: SMSDecomposition, new_f0: np.ndarray) -> None:
        """Replace f0 in *dec* in-place, rescaling all harmonic tracks."""
        n = min(len(dec.hfreq), len(new_f0))
        for i in range(n):
            old_f0 = dec.hfreq[i, 0]
            f0_i   = new_f0[i]
            if old_f0 > 0 and f0_i > 0:
                ratio = f0_i / old_f0
                valid = dec.hfreq[i] > 0
                dec.hfreq[i, valid] *= ratio
            elif f0_i <= 0:
                # CREPE says unvoiced — zero out harmonics
                dec.hfreq[i, :] = 0
        dec.f0 = dec.hfreq[:, 0].copy()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_pair(self, target_y: np.ndarray, source_y: np.ndarray) -> None:
        if len(source_y) != len(target_y):
            raise ValueError(
                f"Source and target must have the same length "
                f"({len(target_y)} vs {len(source_y)}). "
                f"Use match_length() first."
            )

    # ------------------------------------------------------------------
    # Internal: spectral-envelope transfer via harmonic interpolation
    # ------------------------------------------------------------------

    @staticmethod
    def _transfer_magnitudes(
        src_dec: SMSDecomposition,
        tgt_dec: SMSDecomposition,
        alpha:   float,
    ) -> np.ndarray:
        """Build new harmonic magnitudes by blending source timbre onto target.

        For each frame the source spectral envelope (interpolated from its
        harmonic peaks) is evaluated at the target harmonic frequencies and
        blended with the target magnitudes according to *alpha*.
        """
        _SILENCE_DB = -200.0

        n_frames = min(len(tgt_dec.hfreq), len(src_dec.hfreq))
        new_hmag = tgt_dec.hmag.copy()

        for i in range(n_frames):
            tgt_valid = tgt_dec.hfreq[i] > 0
            if not np.any(tgt_valid):
                continue

            src_valid = src_dec.hfreq[i] > 0

            if not np.any(src_valid):
                # Source is silent in this frame — suppress target harmonics
                new_hmag[i, tgt_valid] = (
                    (1.0 - alpha) * tgt_dec.hmag[i, tgt_valid]
                    + alpha * _SILENCE_DB
                )
                continue

            src_freqs = src_dec.hfreq[i, src_valid]
            src_mags  = src_dec.hmag[i,  src_valid]
            tgt_freqs = tgt_dec.hfreq[i, tgt_valid]

            transferred = np.interp(tgt_freqs, src_freqs, src_mags)
            new_hmag[i, tgt_valid] = (
                alpha * transferred + (1.0 - alpha) * tgt_dec.hmag[i, tgt_valid]
            )
        return new_hmag

    # ------------------------------------------------------------------
    # Transfer methods  (mirrors Baseline)
    # ------------------------------------------------------------------

    def f0_transfer(
        self,
        target_y:    np.ndarray,
        source_y:    np.ndarray,
        output_path: Optional[str] = None,
        alpha:       float = 1.0,
        use_crepe:   bool  = False,
        target_f0:   Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Transfer: target F0 + source harmonic magnitudes + source stochastic.

        Keeps the *pitch contour* of the target while adopting the full
        *timbre* (spectral shape + noise character) of the source.

        If *target_f0* is provided it overrides the target pitch contour
        directly.  If *use_crepe* is True, CREPE is used instead of the
        built-in TWM f0 estimator.
        """
        self._validate_pair(target_y, source_y)

        tgt_dec = self.decompose(target_y)
        src_dec = self.decompose(source_y)

        if target_f0 is not None:
            self._replace_f0(tgt_dec, target_f0)
        elif use_crepe:
            self._replace_f0(tgt_dec, self._crepe_f0(target_y, len(tgt_dec.hfreq)))

        new_hmag = self._transfer_magnitudes(src_dec, tgt_dec, alpha)

        n = min(tgt_dec.stocEnv.shape[0], src_dec.stocEnv.shape[0])
        new_stocEnv = tgt_dec.stocEnv.copy()
        new_stocEnv[:n] = (
            alpha * src_dec.stocEnv[:n] + (1.0 - alpha) * tgt_dec.stocEnv[:n]
        )

        out = self.synthesize_hps(tgt_dec.hfreq, new_hmag,
                                  tgt_dec.hphase, new_stocEnv)

        if output_path is not None:
            sf.write(output_path, out, self.sample_rate)
        return out

    def f0_and_ap_transfer(
        self,
        target_y:    np.ndarray,
        source_y:    np.ndarray,
        output_path: Optional[str] = None,
        alpha:       float = 1.0,
        use_crepe:   bool  = False,
        target_f0:   Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Transfer: target F0 + source harmonic magnitudes + target stochastic.

        Keeps the *pitch contour* and *noise character* of the target
        while adopting only the *spectral envelope* of the source.

        If *target_f0* is provided it overrides the target pitch contour
        directly.  If *use_crepe* is True, CREPE is used instead of the
        built-in TWM f0 estimator.
        """
        self._validate_pair(target_y, source_y)

        tgt_dec = self.decompose(target_y)
        src_dec = self.decompose(source_y)

        if target_f0 is not None:
            self._replace_f0(tgt_dec, target_f0)
        elif use_crepe:
            self._replace_f0(tgt_dec, self._crepe_f0(target_y, len(tgt_dec.hfreq)))

        new_hmag = self._transfer_magnitudes(src_dec, tgt_dec, alpha)

        out = self.synthesize_hps(tgt_dec.hfreq, new_hmag,
                                  tgt_dec.hphase, tgt_dec.stocEnv)

        if output_path is not None:
            sf.write(output_path, out, self.sample_rate)
        return out
    

