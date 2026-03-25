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
from scipy.interpolate import interp1d
from scipy.signal import get_window

try:
    from sms_tools.core import (
        dft_analysis,
        estimate_f0_hps,
        harmonic_analysis,
        spectral_envelope,
        stochastic_residual,
        synthesize_harmonics,
        synthesize_stochastic,
    )
    from sms_tools.audio_io import read_wav, write_wav
    _HAS_SMS_TOOLS = True
except ImportError:
    _HAS_SMS_TOOLS = False


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
        f0, timeaxis = pw.harvest(y, fs)  # type: ignore[attr-defined]
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
        use_crepe: bool = False,
    ) -> np.ndarray:
        """Transfer: target F0 + source SP + source AP.

        Keeps the *pitch contour* of the target while adopting the
        *timbre* (spectral envelope + aperiodicity) of the source.
        """
        self._validate_pair(target_y, source_y)

        if use_crepe:
            target_f0 = self.crepe_f0(target_y)[0]
        else:
            target_f0, _, _ = self.decompose(target_y)
        _, source_sp, source_ap = self.decompose(source_y)

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
    ) -> np.ndarray:
        """Transfer: target F0 + source SP + target AP.

        Keeps the *pitch contour* and *breathiness / noise character* of
        the target while adopting the *spectral envelope* of the source.
        """
        self._validate_pair(target_y, source_y)


        target_f0, _, target_ap = self.decompose(target_y)
        _, source_sp, _ = self.decompose(source_y)
        if use_crepe:
            target_f0 = self.crepe_f0(target_y)[0]

        out = self.synthesize(target_f0, source_sp, target_ap, self.sample_rate)

        if output_path is not None:
            sf.write(output_path, out, self.sample_rate)
        return out





# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _env_at_freqs(env_dB: np.ndarray, freqs_hz: np.ndarray,
                  fs: int, N: int) -> np.ndarray:
    """Interpolate a spectral envelope (length hN, dB) at given frequencies (Hz)."""
    hN = len(env_dB)
    bin_freqs = np.linspace(0, fs / 2, hN)
    if len(freqs_hz) == 0:
        return np.array([])
    f = interp1d(bin_freqs, env_dB, kind='linear', fill_value='extrapolate')
    return f(np.clip(freqs_hz, 0, fs / 2))


def _resample_frames(arr: np.ndarray, n_out: int) -> np.ndarray:
    """Linearly resample a 2-D frame array (n_in × K) to (n_out × K)."""
    n_in, K = arr.shape
    if n_in == n_out:
        return arr
    t_in  = np.linspace(0, 1, n_in)
    t_out = np.linspace(0, 1, n_out)
    out   = np.zeros((n_out, K))
    for k in range(K):
        out[:, k] = interp1d(t_in, arr[:, k], kind='linear',
                             fill_value='extrapolate')(t_out)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# SMS decomposition result (named tuple-like dataclass)
# ─────────────────────────────────────────────────────────────────────────────

class SMSDecomposition:
    """Container for the three SMS/HPS components, mirroring WORLD's (f0, sp, ap).

    Attributes
    ----------
    f0        : ndarray, shape (n_frames,)
        Fundamental frequency per frame in Hz (0 = unvoiced).
    sp        : ndarray, shape (n_frames, hN)
        Spectral envelope per frame in dB  — analogue of WORLD's ``sp``.
    ap        : ndarray, shape (n_frames, stoch_coeff)
        Stochastic residual envelope per frame in dB — analogue of WORLD's ``ap``.
    frames_freqs  : list[ndarray]   Harmonic partial frequencies (Hz).
    frames_mags   : list[ndarray]   Harmonic partial magnitudes  (dB).
    frames_phases : list[ndarray]   Harmonic partial phases      (rad).
    H, N, M, fs   : int             Analysis parameters.
    n_samples     : int             Length of the padded analysis signal.
    """

    def __init__(self, raw: dict) -> None:
        self.f0             = raw['frames_f0']
        self.sp             = raw['frames_env']       # (F × hN)   dB
        self.ap             = raw['stoch_env']        # (F × K)    dB
        self.frames_freqs   = raw['frames_freqs']
        self.frames_mags    = raw['frames_mags']
        self.frames_phases  = raw['frames_phases']
        self.H              = raw['H']
        self.N              = raw['N']
        self.M              = raw['M']
        self.fs             = raw['fs']
        self.n_samples      = raw['n_samples']

    # ------------------------------------------------------------------
    # Convenience accessors matching WORLD tuple unpacking style:
    #   f0, sp, ap = baseline.decompose(y)
    # ------------------------------------------------------------------
    def __iter__(self):
        yield self.f0
        yield self.sp
        yield self.ap


# ─────────────────────────────────────────────────────────────────────────────
# BaselineSMS
# ─────────────────────────────────────────────────────────────────────────────

class BaselineSMS:
    """SMS/HPS-vocoder baseline for timbre transfer.

    Drop-in API replacement for the WORLD-based ``Baseline`` class.

    Parameters
    ----------
    sample_rate : int
        Working sample rate.  All audio loaded / synthesised uses this rate.
    window : str
        scipy window name used for STFT frames (default ``'hann'``).
    M : int
        Analysis window length in samples (default 1001).
    N : int
        FFT size (default 2048).
    H : int
        Hop size in samples (default 256).
    n_harm : int
        Maximum number of harmonics tracked per frame (default 20).
    min_f0, max_f0 : float
        Search range for the HPS f0 estimator in Hz.
    stoch_coeff : int
        Number of spectral coefficients used to represent the stochastic
        residual envelope (default 128).
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        *,
        window:      str   = 'hann',
        M:           int   = 1001,
        N:           int   = 2048,
        H:           int   = 256,
        n_harm:      int   = 20,
        min_f0:      float = 80.0,
        max_f0:      float = 1200.0,
        stoch_coeff: int   = 128,
    ) -> None:
        if not _HAS_SMS_TOOLS:
            raise ImportError(
                "BaselineSMS requires the 'sms_tools' package. "
                "Install it or use the WORLD-based Baseline class instead."
            )
        self.sample_rate  = sample_rate
        self._window      = window
        self._M           = M
        self._N           = N
        self._H           = H
        self._n_harm      = n_harm
        self._min_f0      = min_f0
        self._max_f0      = max_f0
        self._stoch_coeff = stoch_coeff

    # ------------------------------------------------------------------
    # I/O helpers  (mirrors Baseline)
    # ------------------------------------------------------------------

    @staticmethod
    def match_length(
        target_y: np.ndarray,
        source_y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(target_y, source_y)`` with *source_y* matched to target length.

        Identical behaviour to ``Baseline.match_length``: truncates if source
        is longer, wraps (pads) if shorter.
        """
        if len(source_y) >= len(target_y):
            return target_y, source_y[:len(target_y)]
        return target_y, np.pad(
            source_y,
            (0, len(target_y) - len(source_y)),
            mode='wrap',
        )

    # ------------------------------------------------------------------
    # Core SMS analysis
    # ------------------------------------------------------------------

    def _run_analysis(self, y: np.ndarray, fs: Optional[int] = None) -> dict:
        """Internal HPS analysis — returns the raw dict from ``hps_analysis``."""
        fs = fs or self.sample_rate
        M, N, H = self._M, self._N, self._H

        w = get_window(self._window, M)
        w = w / w.sum()

        # Zero-pad so every frame is fully covered
        x = np.concatenate([np.zeros(M // 2), y, np.zeros(M)])

        frames_freqs, frames_mags, frames_phases = [], [], []
        frames_env, frames_f0 = [], []

        n_frames = (len(x) - M) // H + 1
        for i in range(n_frames):
            start = i * H
            frame = x[start : start + M]
            if len(frame) < M:
                frame = np.pad(frame, (0, M - len(frame)))

            mX, pX = dft_analysis(frame, w, N)
            env     = spectral_envelope(mX)
            f0      = estimate_f0_hps(mX, fs, N,
                                      f0_min=self._min_f0, f0_max=self._max_f0)
            frames_f0.append(f0)

            if f0 > 0:
                hfreqs, hmags, hphases = harmonic_analysis(
                    mX, pX, f0, fs, N, self._n_harm)
            else:
                hfreqs = hmags = hphases = np.array([])

            frames_freqs.append(hfreqs)
            frames_mags.append(hmags)
            frames_phases.append(hphases)
            frames_env.append(env)

        frames_env = np.array(frames_env)   # (F × hN)
        frames_f0  = np.array(frames_f0)

        total_samples = len(x)
        harmonic_y = synthesize_harmonics(
            frames_freqs, frames_mags, frames_phases, H, fs, total_samples)

        stoch_env = stochastic_residual(x, harmonic_y, w, H, N, self._stoch_coeff)

        return {
            'frames_freqs' : frames_freqs,
            'frames_mags'  : frames_mags,
            'frames_phases': frames_phases,
            'frames_env'   : frames_env,
            'frames_f0'    : frames_f0,
            'stoch_env'    : stoch_env,
            'H'            : H,
            'N'            : N,
            'fs'           : fs,
            'M'            : M,
            'harmonic_y'   : harmonic_y,
            'n_samples'    : total_samples,
        }

    # ------------------------------------------------------------------
    # Public decompose / synthesize  (mirrors Baseline)
    # ------------------------------------------------------------------

    def decompose(
        self,
        y:  np.ndarray,
        fs: Optional[int] = None,
    ) -> SMSDecomposition:
        """SMS/HPS analysis: returns an :class:`SMSDecomposition`.

        The object is iterable as ``(f0, sp, ap)`` to match WORLD's tuple API::

            f0, sp, ap = baseline.decompose(y)

        Parameters
        ----------
        y  : ndarray   Mono waveform, float64, already at ``self.sample_rate``.
        fs : int, optional  Override sample rate.
        """
        raw = self._run_analysis(y, fs)
        return SMSDecomposition(raw)

    @staticmethod
    def synthesize(
        f0:  np.ndarray,
        sp:  np.ndarray,
        ap:  np.ndarray,
        fs:  int,
        *,
        H:   int = 256,
        N:   int = 2048,
        M:   int = 1001,
        frames_freqs:  Optional[list] = None,
        frames_phases: Optional[list] = None,
        stoch_coeff:   int = 128,
    ) -> np.ndarray:
        """SMS re-synthesis from ``(f0, sp, ap)``.

        Parameters
        ----------
        f0  : ndarray (F,)      Fundamental frequency per frame (Hz).
        sp  : ndarray (F × hN)  Spectral envelope per frame (dB).
        ap  : ndarray (F × K)   Stochastic residual envelope per frame (dB).
        fs  : int               Sample rate.

        Keyword-only (advanced)
        -----------------------
        frames_freqs, frames_phases : optional per-frame harmonic data.
            When provided, used for additive harmonic synthesis.
            When omitted, harmonics are regenerated from *f0* and *sp*.
        H, N, M, stoch_coeff : analysis parameters (must match decomposition).

        Notes
        -----
        If you obtained ``(f0, sp, ap)`` from :meth:`decompose` you can
        synthesize faithfully.  If you swapped components across two
        decompositions (timbre transfer), synthesis degrades gracefully:
        harmonic structure is rebuilt from the target f0 and the new sp.
        """
        n_frames   = len(f0)
        hN         = sp.shape[1]
        total_samp = n_frames * H + N

        # ── Rebuild harmonic parameters when not supplied ──────────────────
        if frames_freqs is None or frames_phases is None:
            new_freqs, new_mags, new_phases = [], [], []
            bin_freqs = np.linspace(0, fs / 2, hN)

            for i in range(n_frames):
                f0_i = f0[i]
                env_i = sp[i]
                if f0_i <= 0:
                    new_freqs.append(np.array([]))
                    new_mags.append(np.array([]))
                    new_phases.append(np.array([]))
                    continue

                h_freqs, h_mags, h_phases = [], [], []
                for h in range(1, 21):
                    fh = h * f0_i
                    if fh >= fs / 2:
                        break
                    # Read magnitude from spectral envelope at harmonic frequency
                    mag = float(interp1d(
                        bin_freqs, env_i, kind='linear',
                        fill_value='extrapolate')(np.clip(fh, 0, fs / 2)))
                    h_freqs.append(fh)
                    h_mags.append(mag)
                    h_phases.append(0.0)   # zero-phase for cross-decomposition synth

                new_freqs.append(np.array(h_freqs))
                new_mags.append(np.array(h_mags))
                new_phases.append(np.array(h_phases))
        else:
            # Re-shape magnitudes from provided sp (timbre already baked in)
            bin_freqs = np.linspace(0, fs / 2, hN)
            new_freqs  = frames_freqs
            new_phases = frames_phases
            new_mags   = []
            for i in range(n_frames):
                hf = frames_freqs[i]
                if len(hf) == 0:
                    new_mags.append(np.array([]))
                    continue
                mags = interp1d(
                    bin_freqs, sp[i], kind='linear',
                    fill_value='extrapolate')(np.clip(hf, 0, fs / 2))
                new_mags.append(mags)

        # ── Harmonic component ─────────────────────────────────────────────
        y_harm = synthesize_harmonics(new_freqs, new_mags, new_phases,
                                      H, fs, total_samp)

        # ── Stochastic component ───────────────────────────────────────────
        y_stoch = synthesize_stochastic(ap, H, N, fs, n_coeff=stoch_coeff)

        min_len = min(len(y_harm), len(y_stoch))
        y_out   = y_harm[:min_len] + y_stoch[:min_len]

        peak = np.max(np.abs(y_out))
        if peak > 0:
            y_out = y_out / peak * 0.9
        return y_out

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
    # Transfer methods  (mirrors Baseline)
    # ------------------------------------------------------------------

    def f0_transfer(
        self,
        target_y:    np.ndarray,
        source_y:    np.ndarray,
        output_path: Optional[str] = None,
        alpha:       float = 1.0,
    ) -> np.ndarray:
        """Transfer: target F0 + source SP (spectral envelope) + source AP (stochastic).

        Keeps the *pitch contour* of the target while adopting the full
        *timbre* (spectral shape + noise character) of the source.

        Analogue of ``Baseline.f0_transfer``.

        Parameters
        ----------
        target_y    : ndarray  Pitch / rhythm donor waveform.
        source_y    : ndarray  Timbre donor waveform (same length as target_y).
        output_path : str, optional  If given, write result to this WAV path.
        alpha       : float  Blend strength 0 (no transfer) → 1 (full transfer).

        Returns
        -------
        ndarray   Output waveform.
        """
        self._validate_pair(target_y, source_y)

        tgt_dec = self.decompose(target_y)
        src_dec = self.decompose(source_y)

        n_tgt = len(tgt_dec.frames_freqs)
        n_src = len(src_dec.frames_freqs)

        # ── Spectral envelope: blend source SP onto target frame count ─────
        src_sp_resampled = _resample_frames(src_dec.sp, n_tgt)          # (Ft × hN)
        new_sp = alpha * src_sp_resampled + (1.0 - alpha) * tgt_dec.sp  # (Ft × hN)

        # ── Stochastic residual: align to tgt AP frame count before blending
        n_ap_tgt = tgt_dec.ap.shape[0]
        src_ap_resampled = _resample_frames(src_dec.ap, n_ap_tgt)
        new_ap = alpha * src_ap_resampled + (1.0 - alpha) * tgt_dec.ap

        # ── Re-synthesise with TARGET f0 + blended SP + blended AP ────────
        out = self.synthesize(
            f0            = tgt_dec.f0,
            sp            = new_sp,
            ap            = new_ap,
            fs            = self.sample_rate,
            H             = self._H,
            N             = self._N,
            M             = self._M,
            frames_freqs  = tgt_dec.frames_freqs,
            frames_phases = tgt_dec.frames_phases,
            stoch_coeff   = self._stoch_coeff,
        )

        if output_path is not None:
            write_wav(output_path, out, self.sample_rate)
        return out

    def f0_and_ap_transfer(
        self,
        target_y:    np.ndarray,
        source_y:    np.ndarray,
        output_path: Optional[str] = None,
        alpha:       float = 1.0,
    ) -> np.ndarray:
        """Transfer: target F0 + source SP (spectral envelope) + target AP (stochastic).

        Keeps the *pitch contour* and *noise / breathiness character* of the
        target while adopting only the *spectral envelope* (vowel colour /
        instrument body resonance) of the source.

        Analogue of ``Baseline.f0_and_ap_transfer``.

        Parameters
        ----------
        target_y    : ndarray  Pitch / rhythm / noise-character donor waveform.
        source_y    : ndarray  Spectral-envelope (timbre colour) donor waveform.
        output_path : str, optional  If given, write result to this WAV path.
        alpha       : float  Blend strength 0 (no transfer) → 1 (full transfer).

        Returns
        -------
        ndarray   Output waveform.
        """
        self._validate_pair(target_y, source_y)

        tgt_dec = self.decompose(target_y)
        src_dec = self.decompose(source_y)

        n_tgt = len(tgt_dec.frames_freqs)

        # ── Spectral envelope only: resample source SP to target frame count
        src_sp_resampled = _resample_frames(src_dec.sp, n_tgt)
        new_sp = alpha * src_sp_resampled + (1.0 - alpha) * tgt_dec.sp

        # ── Stochastic residual: keep target AP entirely ───────────────────
        new_ap = tgt_dec.ap

        # ── Re-synthesise with TARGET f0 + blended SP + TARGET AP ─────────
        out = self.synthesize(
            f0            = tgt_dec.f0,
            sp            = new_sp,
            ap            = new_ap,
            fs            = self.sample_rate,
            H             = self._H,
            N             = self._N,
            M             = self._M,
            frames_freqs  = tgt_dec.frames_freqs,
            frames_phases = tgt_dec.frames_phases,
            stoch_coeff   = self._stoch_coeff,
        )

        if output_path is not None:
            write_wav(output_path, out, self.sample_rate)
        return out
    

