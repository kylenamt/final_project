"""Pitch-corrected timbre evaluation metrics.

Addresses the confound between pitch (F0) and spectral descriptors by
providing four complementary evaluation layers:

1. **Pitch stratification** — compare feature distributions within F0 bins.
2. **Spectral envelope descriptors** — cepstrally smooth the spectrum with
   an F0-adaptive order so that descriptors reflect the filter (timbre),
   not the excitation (pitch).
3. **Regression-based residuals** — model the expected pitch dependency on
   reference data, then evaluate the residual for synthesised signals.
4. **Decoupled accuracy** — F0 RMSE / Pearson-r and loudness RMSE /
   Pearson-r measured independently of timbre quality.

The class mirrors :class:`~evaluation.timbre_metrics.TimbreMetrics` in
constructor signature and output format so that both can be used
interchangeably in the evaluation pipeline.

Usage
-----
>>> pm = PitchMetrics(sample_rate=16000, frame_width_sec=0.25)

Extract spectral features *and* aligned F0:

>>> feats = pm.extract_features_with_f0(signal)

Pitch-stratified bins:

>>> bins = PitchMetrics.stratify_by_pitch(features_2d, f0)

Envelope-based descriptors:

>>> env = pm.extract_envelope_features(signal)

Regression residuals:

>>> models = PitchMetrics.fit_pitch_regression(ref_2d, ref_f0, keys)
>>> residuals = PitchMetrics.compute_residuals(synth_2d, synth_f0, models, keys)
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import crepe
import numpy as np
from scipy.fft import fft, ifft
from scipy.ndimage import median_filter
from scipy.signal import get_window
from scipy.stats import f as f_dist
from scipy.stats import pearsonr
from scipy.stats import t as t_dist

from .timbre_metrics import TimbreMetrics
from utils import load_audio

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

logger = logging.getLogger(__name__)

__all__ = ["PitchMetrics"]


# ----------------------------------------------------------------------
# Module-level worker for ProcessPoolExecutor (must be picklable)
# ----------------------------------------------------------------------

def _extract_with_f0_worker(
    file_path: Path,
    sr: int,
    frame_width_sec: float,
    overlap_pct: float,
    metrics: Optional[List[str]],
    feature_types: Optional[List[str]],
) -> Optional[Dict[str, np.ndarray]]:
    """Extract per-frame spectral features *and* F0 from a single WAV file.

    Returns ``None`` on error so the caller can skip failures.
    """
    try:
        audio, _ = load_audio(file_path, sr=sr, mono=True)
        pm = PitchMetrics(
            sample_rate=sr,
            frame_width_sec=frame_width_sec,
            overlap_pct=overlap_pct,
        )
        return pm.extract_features_with_f0(
            audio, metrics=metrics, feature_types=feature_types,
        )
    except Exception:
        logger.warning("Feature+F0 extraction failed: %s", file_path, exc_info=True)
        return None


class PitchMetrics:
    """Pitch-aware timbre feature extractor.

    Parameters
    ----------
    sample_rate : int
        Expected sample rate of the input audio (Hz).
    frame_width_sec : float
        Duration of each analysis frame in seconds.
    overlap_pct : float
        Fraction of overlap between consecutive frames (0.0 – 1.0).
    """

    DEFAULT_PITCH_BINS: Dict[str, Tuple[float, float]] = {
        "G3-D4": (196.0, 294.0),
        "D4-A4": (294.0, 440.0),
        "A4-E5": (440.0, 659.0),
        "E5+":   (659.0, 2000.0),
    }

    # Descriptor keys matching pytimbre SpectralMetrics output.
    _DESCRIPTOR_KEYS = (
        "spectral_centroid",
        "spectral_crest",
        "spectral_decrease",
        "spectral_flatness",
        "spectral_roll_off",
        "spectral_skewness",
        "spectral_spread",
    )

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_width_sec: float = 0.25,
        overlap_pct: float = 0.0,
    ):
        self.sample_rate = sample_rate
        self.frame_width_sec = frame_width_sec
        self.overlap_pct = overlap_pct
        self._tm = TimbreMetrics(
            sample_rate=sample_rate,
            frame_width_sec=frame_width_sec,
            overlap_pct=overlap_pct,
        )

    # ------------------------------------------------------------------
    # F0 extraction (CREPE, frame-aligned)
    # ------------------------------------------------------------------

    def extract_f0(
        self, signal: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract F0 and confidence aligned to the spectral analysis grid.

        Uses CREPE with ``step_size`` matched to :attr:`frame_width_sec` so
        that each F0 value corresponds to one spectral frame.

        Parameters
        ----------
        signal : np.ndarray
            1-D mono waveform at :attr:`sample_rate`.

        Returns
        -------
        f0_hz : np.ndarray, shape ``(n_frames,)``
        confidence : np.ndarray, shape ``(n_frames,)``
        """
        step_ms = int(self.frame_width_sec * 1000.0)
        _, f0, confidence, _ = crepe.predict(
            signal, self.sample_rate,
            model_capacity="full",
            step_size=step_ms,
            verbose=0,
        )
        f0 = np.asarray(f0, dtype=np.float64)
        confidence = np.asarray(confidence, dtype=np.float64)

        # Clip to the number of complete spectral frames.
        frame_samples = int(self.sample_rate * self.frame_width_sec) # samples per frame
        hop_samples = int(frame_samples * (1.0 - self.overlap_pct))
        n_spectral_frames = max(1, 1 + (len(signal) - frame_samples) // hop_samples)
        n = min(len(f0), n_spectral_frames)
        return f0[:n], confidence[:n]

    # ------------------------------------------------------------------
    # Combined feature + F0 extraction
    # ------------------------------------------------------------------

    def extract_features_with_f0(
        self,
        signal: np.ndarray,
        metrics: Optional[List[str]] = None,
        feature_types: Optional[List[str]] = None,
        confidence_threshold: float = 0.85,
    ) -> Dict[str, np.ndarray]:
        """Extract per-frame spectral features *and* F0 from a signal.

        Returns a dict whose keys are the spectral metric names (same as
        :meth:`TimbreMetrics.extract_series_from_array`) plus ``"f0_hz"``,
        ``"f0_confidence"``, and ``"spectral_flux"``.  All arrays have the
        same length.

        Parameters
        ----------
        confidence_threshold : float
            CREPE frames with confidence below this value have their F0
            set to NaN (treated as unvoiced by downstream methods).
        """
        series = self._tm.extract_series_from_array(
            signal, metrics=metrics, feature_types=feature_types,
        )
        if not series:
            return {}

        f0, conf = self.extract_f0(signal)

        # Mask low-confidence frames
        low_conf = conf < confidence_threshold
        f0[low_conf] = np.nan

        # Trim all arrays to the shortest common length.
        first_key = next(iter(series))
        n_spec = len(series[first_key])
        n = min(n_spec, len(f0))

        out: Dict[str, np.ndarray] = {}
        for k, v in series.items():
            out[k] = v[:n]
        out["f0_hz"] = f0[:n]
        out["f0_confidence"] = conf[:n]

        # Spectral flux (uses same framing as TimbreMetrics)
        # flux = self.compute_spectral_flux(signal)
        # out["spectral_flux"] = flux[:n]
        
        return out

    # ------------------------------------------------------------------
    # Batch extraction with F0
    # ------------------------------------------------------------------

    def extract_from_dir_with_f0(
        self,
        wav_dir: str | Path,
        metrics: Optional[List[str]] = None,
        feature_types: Optional[List[str]] = None,
        n_workers: int = 1,
    ) -> Dict[str, np.ndarray]:
        """Extract features + F0 from every WAV under *wav_dir* and concatenate.

        Same semantics as :meth:`TimbreMetrics.extract_from_dir` but the
        returned dict also contains ``"f0_hz"`` and ``"f0_confidence"``.
        """
        paths = sorted(Path(wav_dir).rglob("*.wav"))
        if not paths:
            raise FileNotFoundError(f"No .wav files found in {wav_dir}")

        logger.info("Extracting features+F0 from %d files in %s", len(paths), wav_dir)

        all_series: List[Dict[str, np.ndarray]] = []

        if n_workers <= 1:
            items = tqdm(paths, desc="Feature+F0 extraction") if tqdm else paths
            for p in items:
                result = _extract_with_f0_worker(
                    p, self.sample_rate, self.frame_width_sec,
                    self.overlap_pct, metrics, feature_types,
                )
                if result is not None:
                    all_series.append(result)
        else:
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = {
                    pool.submit(
                        _extract_with_f0_worker,
                        p, self.sample_rate, self.frame_width_sec,
                        self.overlap_pct, metrics, feature_types,
                    ): p
                    for p in paths
                }
                items = (
                    tqdm(as_completed(futures), desc="Feature+F0 extraction",
                         total=len(futures))
                    if tqdm else as_completed(futures)
                )
                for future in items:
                    result = future.result()
                    if result is not None:
                        all_series.append(result)

        if not all_series:
            raise ValueError(f"No features extracted from any file in {wav_dir}")

        common_keys = set(all_series[0].keys())
        for s in all_series[1:]:
            common_keys &= set(s.keys())
        feat_keys = sorted(common_keys)

        if not feat_keys:
            raise ValueError("No common feature keys across files")

        concatenated: Dict[str, np.ndarray] = {
            k: np.concatenate([s[k] for s in all_series]) for k in feat_keys
        }

        logger.info(
            "Concatenated %d files -> %d frames, %d features",
            len(all_series), len(concatenated[feat_keys[0]]), len(feat_keys),
        )
        return concatenated


    

    # ==================================================================
    # Layer 1: Pitch stratification
    # ==================================================================

    @staticmethod
    def stratify_by_pitch(
        features_2d: np.ndarray,
        f0: np.ndarray,
        bins: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> Dict[str, np.ndarray]:
        """Split *features_2d* into pitch bins based on *f0*.

        Parameters
        ----------
        features_2d : np.ndarray, shape ``(n_frames, n_features)``
        f0 : np.ndarray, shape ``(n_frames,)``
            Fundamental frequency in Hz.  Unvoiced frames (``<= 0`` or NaN)
            are excluded from all bins.
        bins : dict, optional
            ``{name: (lo_hz, hi_hz)}``.  Defaults to :attr:`DEFAULT_PITCH_BINS`.

        Returns
        -------
        dict
            ``{bin_name: np.ndarray}`` — subset of *features_2d* rows.
        """
        if bins is None:
            bins = PitchMetrics.DEFAULT_PITCH_BINS

        voiced = (f0 > 0) & np.isfinite(f0)
        result: Dict[str, np.ndarray] = {}
        for name, (lo, hi) in bins.items():
            mask = voiced & (f0 >= lo) & (f0 < hi)
            result[name] = features_2d[mask]
        return result
    # ==================================================================
    # Layer 3: Regression-based residual analysis
    # ==================================================================

    @staticmethod
    def f0_dynamics(
        f0: np.ndarray,
        dt: float = 1.0,
        median_window: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """First and second derivatives of F0 via finite differences.

        Parameters
        ----------
        f0 : np.ndarray, shape ``(n,)``
            F0 in Hz.
        dt : float
            Time step between frames in seconds
            (``frame_width_sec * (1 - overlap_pct)``).  Defaults to 1.0
            which gives derivatives in Hz/frame for backwards compatibility.
        median_window : int
            Size of the median filter applied to the F0 contour before
            computing derivatives to suppress estimation jitter.  Set to 1
            to disable.

        Returns
        -------
        f0_dot : np.ndarray
            F0 velocity (Hz/s when *dt* is in seconds).
        f0_ddot : np.ndarray
            F0 acceleration (Hz/s² when *dt* is in seconds).
        """
        f0_smooth = f0.copy()
        if median_window > 1:
            f0_smooth = median_filter(f0_smooth, size=median_window)

        f0_dot = np.zeros_like(f0_smooth)
        f0_ddot = np.zeros_like(f0_smooth)
        if len(f0_smooth) > 1:
            f0_dot[1:-1] = (f0_smooth[2:] - f0_smooth[:-2]) / (2.0 * dt)
            f0_dot[0] = (f0_smooth[1] - f0_smooth[0]) / dt
            f0_dot[-1] = (f0_smooth[-1] - f0_smooth[-2]) / dt
        if len(f0_smooth) > 2:
            f0_ddot[1:-1] = (
                f0_smooth[2:] - 2.0 * f0_smooth[1:-1] + f0_smooth[:-2]
            ) / (dt ** 2)
        return f0_dot, f0_ddot

    @staticmethod
    def _f0_to_midi(f0: np.ndarray) -> np.ndarray:
        """Convert Hz to MIDI pitch, guarding against zero / NaN."""
        safe = np.where((f0 > 0) & np.isfinite(f0), f0, np.nan)
        return 12.0 * np.log2(safe / 440.0) + 69.0

    @staticmethod
    def fit_pitch_regression(
        features_2d: np.ndarray,
        f0: np.ndarray,
        feat_keys: List[str],
        degree: int = 2,
        use_dynamics: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Fit per-descriptor regression models on MIDI pitch.

        For each descriptor column, fits:

            ``d(t) = β₀ + β₁p + β₂p² [+ β₃ṗ + β₄p̈]``

        where *p* = MIDI pitch.

        Parameters
        ----------
        features_2d : np.ndarray, shape ``(n_frames, n_features)``
        f0 : np.ndarray, shape ``(n_frames,)``
            F0 in Hz.
        feat_keys : list of str
            Column names matching *features_2d* columns.
        degree : int
            Polynomial degree for pitch (default 2).
        use_dynamics : bool
            If ``True``, append F0 velocity and acceleration regressors.

        Returns
        -------
        dict
            ``{feat_name: coefficients}`` — coefficient arrays from
            least-squares fit.
        """
        midi = PitchMetrics._f0_to_midi(f0)
        voiced = np.isfinite(midi)

        midi_v = midi[voiced]
        feats_v = features_2d[voiced]

        # Build design matrix: [1, p, p², ..., (ṗ, p̈)]
        cols = [np.ones(len(midi_v))]
        for d in range(1, degree + 1):
            cols.append(midi_v ** d)

        if use_dynamics:
            f0_dot, f0_ddot = PitchMetrics.f0_dynamics(f0)
            cols.append(f0_dot[voiced])
            cols.append(f0_ddot[voiced])

        X = np.column_stack(cols)

        models: Dict[str, np.ndarray] = {}
        for i, key in enumerate(feat_keys):
            y = feats_v[:, i]
            coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            models[key] = coeffs

        return models

    @staticmethod
    def compute_residuals(
        features_2d: np.ndarray,
        f0: np.ndarray,
        models: Dict[str, np.ndarray],
        feat_keys: List[str],
        use_dynamics: bool = False,
    ) -> np.ndarray:
        """Compute pitch-regression residuals for each descriptor.

        Parameters
        ----------
        features_2d : np.ndarray, shape ``(n_frames, n_features)``
        f0 : np.ndarray, shape ``(n_frames,)``
        models : dict
            Output of :meth:`fit_pitch_regression`.
        feat_keys : list of str
        use_dynamics : bool

        Returns
        -------
        np.ndarray, shape ``(n_frames, n_features)``
            Residual array.  Unvoiced frames are set to NaN.
        """
        midi = PitchMetrics._f0_to_midi(f0)

        # Build full design matrix (including unvoiced — will be NaN)
        degree = len(next(iter(models.values())))
        if use_dynamics:
            degree -= 2  # last two cols are dynamics
        else:
            degree -= 1  # subtract intercept col

        cols = [np.ones(len(midi))]
        for d in range(1, degree + 1):
            cols.append(midi ** d)

        if use_dynamics:
            f0_dot, f0_ddot = PitchMetrics.f0_dynamics(f0)
            cols.append(f0_dot)
            cols.append(f0_ddot)

        X = np.column_stack(cols)

        residuals = np.full_like(features_2d, np.nan)
        for i, key in enumerate(feat_keys):
            if key not in models:
                continue
            predicted = X @ models[key]
            residuals[:, i] = features_2d[:, i] - predicted

        return residuals

    @staticmethod
    def _ols_stats(
        X: np.ndarray, y: np.ndarray, coeffs: np.ndarray
    ) -> Dict[str, Any]:
        """Compute OLS regression statistics for a single descriptor.

        Returns R², residual sum of squares, coefficient standard errors,
        t-statistics, and p-values.
        """
        n, p = X.shape
        y_hat = X @ coeffs
        residuals = y - y_hat
        ss_res = float(residuals @ residuals)
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        df = n - p
        if df > 0:
            sigma2 = ss_res / df
            try:
                cov = sigma2 * np.linalg.inv(X.T @ X)
                se = np.sqrt(np.maximum(np.diag(cov), 0.0))
            except np.linalg.LinAlgError:
                se = np.full(p, np.nan)
            t_vals = np.where(se > 0, coeffs / se, 0.0)
            p_vals = 2.0 * t_dist.sf(np.abs(t_vals), df)
        else:
            se = np.full(p, np.nan)
            t_vals = np.full(p, np.nan)
            p_vals = np.full(p, np.nan)

        return {
            "r2": r2,
            "ss_res": ss_res,
            "coeffs": coeffs,
            "se": se,
            "t_values": t_vals,
            "p_values": p_vals,
            "df": df,
        }

    @staticmethod
    def fit_coupling_model(
        features_2d: np.ndarray,
        f0: np.ndarray,
        f0_dot: np.ndarray,
        f0_ddot: np.ndarray,
        feat_keys: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """Fit Model A (static) and Model B (with dynamics) per descriptor.

        **Model A**: ``d_i ~ β₀ + β₁·f0 + β₂·f0²``
        **Model B**: ``d_i ~ β₀ + β₁·f0 + β₂·f0² + β₃·ḟ0 + β₄·f̈0``

        Parameters
        ----------
        features_2d : np.ndarray, shape ``(n_frames, n_features)``
        f0 : np.ndarray, shape ``(n_frames,)``
            F0 in Hz.  Unvoiced / NaN frames are excluded.
        f0_dot, f0_ddot : np.ndarray
            F0 velocity and acceleration (from :meth:`f0_dynamics`).
        feat_keys : list of str

        Returns
        -------
        dict
            ``{feat_name: { "coeffs_a", "coeffs_b", "r2_a", "r2_b",
            "delta_r2", "f_stat", "f_pvalue", "coeff_pvalues_b", ... }}``
        """
        voiced = (f0 > 0) & np.isfinite(f0)
        f0_v = f0[voiced]
        feats_v = features_2d[voiced]
        dot_v = f0_dot[voiced]
        ddot_v = f0_ddot[voiced]
        n = len(f0_v)

        ones = np.ones(n)
        X_A = np.column_stack([ones, f0_v, f0_v ** 2])
        X_B = np.column_stack([ones, f0_v, f0_v ** 2, dot_v, ddot_v])

        p_a = X_A.shape[1]
        p_b = X_B.shape[1]

        results: Dict[str, Dict[str, Any]] = {}
        for i, key in enumerate(feat_keys):
            y = feats_v[:, i]

            coeffs_a, _, _, _ = np.linalg.lstsq(X_A, y, rcond=None)
            coeffs_b, _, _, _ = np.linalg.lstsq(X_B, y, rcond=None)

            stats_a = PitchMetrics._ols_stats(X_A, y, coeffs_a)
            stats_b = PitchMetrics._ols_stats(X_B, y, coeffs_b)

            delta_r2 = stats_b["r2"] - stats_a["r2"]

            # F-test: Model A (restricted) vs Model B (full)
            df_diff = p_b - p_a
            df_b = n - p_b
            if df_b > 0 and df_diff > 0 and stats_b["ss_res"] > 0:
                f_stat = (
                    (stats_a["ss_res"] - stats_b["ss_res"]) / df_diff
                ) / (stats_b["ss_res"] / df_b)
                f_pvalue = float(f_dist.sf(f_stat, df_diff, df_b))
            else:
                f_stat = float("nan")
                f_pvalue = float("nan")

            results[key] = {
                "coeffs_a": coeffs_a,
                "coeffs_b": coeffs_b,
                "r2_a": stats_a["r2"],
                "r2_b": stats_b["r2"],
                "delta_r2": delta_r2,
                "f_stat": float(f_stat),
                "f_pvalue": f_pvalue,
                "coeff_pvalues_a": stats_a["p_values"],
                "coeff_pvalues_b": stats_b["p_values"],
                "coeff_se_b": stats_b["se"],
            }

        return results

    @staticmethod
    def coupling_error(
        features_2d: np.ndarray,
        f0: np.ndarray,
        f0_dot: np.ndarray,
        f0_ddot: np.ndarray,
        ref_models: Dict[str, Dict[str, Any]],
        feat_keys: List[str],
    ) -> Dict[str, float]:
        """Coupling error: MSE between output descriptors and reference prediction.

        ``CE_i = mean((d_i_output - g_i_ref(f0, ḟ0, f̈0))²)``

        Parameters
        ----------
        features_2d : np.ndarray, shape ``(n_frames, n_features)``
            Descriptors from the output signal.
        f0, f0_dot, f0_ddot : np.ndarray
            F0 and its derivatives from the output signal.
        ref_models : dict
            Output of :meth:`fit_coupling_model` on the reference signal.
        feat_keys : list of str

        Returns
        -------
        dict
            ``{feat_name: coupling_error}``
        """
        voiced = (f0 > 0) & np.isfinite(f0)
        f0_v = f0[voiced]
        dot_v = f0_dot[voiced]
        ddot_v = f0_ddot[voiced]
        feats_v = features_2d[voiced]
        n = len(f0_v)

        ones = np.ones(n)
        X_B = np.column_stack([ones, f0_v, f0_v ** 2, dot_v, ddot_v])

        ce: Dict[str, float] = {}
        for i, key in enumerate(feat_keys):
            if key not in ref_models:
                continue
            predicted = X_B @ ref_models[key]["coeffs_b"]
            ce[key] = float(np.mean((feats_v[:, i] - predicted) ** 2))
        return ce

    @staticmethod
    def coefficient_distance(
        models_output: Dict[str, Dict[str, Any]],
        models_ref: Dict[str, Dict[str, Any]],
        feat_keys: List[str],
    ) -> Dict[str, float]:
        """L2 distance between output and reference Model B coefficients.

        ``delta_beta_i = ||β_output - β_ref||₂``

        Parameters
        ----------
        models_output, models_ref : dict
            Outputs of :meth:`fit_coupling_model`.
        feat_keys : list of str

        Returns
        -------
        dict
            ``{feat_name: distance}``
        """
        dist: Dict[str, float] = {}
        for key in feat_keys:
            if key in models_output and key in models_ref:
                beta_out = models_output[key]["coeffs_b"]
                beta_ref = models_ref[key]["coeffs_b"]
                dist[key] = float(np.linalg.norm(beta_out - beta_ref))
        return dist

    @staticmethod
    def correlation_summary(
        features_2d: np.ndarray,
        f0: np.ndarray,
        feat_keys: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """Pearson correlation between each descriptor and F0.

        Parameters
        ----------
        features_2d : np.ndarray, shape ``(n_frames, n_features)``
        f0 : np.ndarray, shape ``(n_frames,)``
        feat_keys : list of str

        Returns
        -------
        dict
            ``{feat_name: {"pearson_r": float, "pearson_p": float}}``
        """
        voiced = (f0 > 0) & np.isfinite(f0)
        f0_v = f0[voiced]
        feats_v = features_2d[voiced]

        result: Dict[str, Dict[str, float]] = {}
        for i, key in enumerate(feat_keys):
            col = feats_v[:, i]
            valid = np.isfinite(col)
            if valid.sum() < 3:
                result[key] = {"pearson_r": float("nan"), "pearson_p": float("nan")}
                continue
            r, p = pearsonr(f0_v[valid], col[valid])
            result[key] = {"pearson_r": float(r), "pearson_p": float(p)}
        return result

    