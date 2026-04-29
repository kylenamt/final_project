"""Pitch-regression fitting for pre-extracted spectral features.

Given per-frame spectral descriptors and an aligned F0 contour, fit each
descriptor as a linear model in user-chosen pitch-derived regressors.
OLS and L1 (LassoCV) share a single :meth:`PitchMetrics.fit` entry
point; the returned dicts share a common layout so :meth:`predict`,
:meth:`coupling_error`, and :meth:`coefficient_distance` work
identically regardless of fit method.

Typical usage
-------------
>>> ref_models = PitchMetrics.fit(
...     ref_features_2d, ref_f0, DESCRIPTORS,
...     regressors="B", method="ols", dt=0.25,
... )
>>> ce = PitchMetrics.coupling_error(
...     synth_features_2d, synth_f0, ref_models, DESCRIPTORS, dt=0.25,
... )
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.ndimage import median_filter
from scipy.stats import pearsonr, spearmanr
from scipy.stats import t as t_dist

__all__ = ["PitchMetrics"]

RegressorSpec = Union[str, Sequence[str]]


class PitchMetrics:
    """Stateless fitter for pitch-regression models on spectral features."""

    REGRESSOR_PRESETS: Dict[str, Tuple[str, ...]] = {
        "A":      ("p",),
        "B":      ("p", "p_dot", "p_ddot"),
        "poly2":  ("p", "p2"),
        "hybrid": ("f0", "log2_f0", "p_dot", "p_ddot"),
    }

    SUPPORTED_KEYS = ("p", "p2", "p_dot", "p_ddot", "f0", "f0_2", "log2_f0")

    # ------------------------------------------------------------------
    # F0 extraction (CREPE)
    # ------------------------------------------------------------------

    @staticmethod
    def extract_f0(
        audio: np.ndarray,
        sample_rate: int = 16_000,
        frame_width_sec: float = 0.25,
        overlap_pct: float = 0.0,
        viterbi: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract CREPE F0 aligned to a fixed frame grid.

        DDSP ``compute_f0`` is called with ``frame_rate=250.0``
        (target 4 ms spacing between successive F0 estimates).

        Parameters
        ----------
        audio : np.ndarray
            1-D mono waveform (expected at 16 kHz, per CREPE).
        sample_rate : int
            Audio sample rate; CREPE expects 16 kHz.
        frame_width_sec : float
            Analysis frame width in seconds.
        overlap_pct : float
            Fractional overlap between consecutive frames in ``[0, 1)``.
        viterbi : bool
            Whether to smooth the CREPE output with Viterbi decoding.

        Returns
        -------
        f0_hz : (n_frames,) float64 array
        f0_confidence : (n_frames,) float64 array, in ``[0, 1]``.
        """
        if sample_rate != 16_000:
            raise ValueError(
                f"CREPE expects 16 kHz audio, got sample_rate={sample_rate}"
            )
        if not (0.0 <= overlap_pct < 1.0):
            raise ValueError("overlap_pct must be in [0, 1)")
        if frame_width_sec <= 0:
            raise ValueError("frame_width_sec must be > 0")

        from ddsp.spectral_ops import compute_f0  # optional dep; keeps import cheap

        # Use DDSP compute_f0 frame_rate=250.0 (target 4 ms estimate spacing).
        frame_rate = 250.0

        audio = np.asarray(audio, dtype=np.float32)
        f0, conf = compute_f0(audio, frame_rate=frame_rate, viterbi=viterbi)
        return (
            np.asarray(f0, dtype=np.float64),
            np.asarray(conf, dtype=np.float64),
        )

    # ------------------------------------------------------------------
    # F0 transforms and dynamics
    # ------------------------------------------------------------------

    @staticmethod
    def _transform_f0(f0: np.ndarray, mode: str) -> np.ndarray:
        """Map F0 (Hz) to the pitch regressor. ``hz`` passes through;
        ``log2`` returns ``log2(f0)`` with non-positive frames as NaN."""
        if mode == "hz":
            return np.asarray(f0, dtype=np.float64)
        if mode == "log2":
            f0 = np.asarray(f0, dtype=np.float64)
            safe = np.where((f0 > 0) & np.isfinite(f0), f0, np.nan)
            return np.log2(safe)
        raise ValueError(f"Unknown pitch_transform {mode!r}; expected 'hz' or 'log2'")

    @staticmethod
    def f0_dynamics(
        f0: np.ndarray,
        dt: float = 1.0,
        median_window: int = 5,
        pitch_transform: str = "log2",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """First and second time-derivatives of the pitch contour.

        Returns ``(p_dot, p_ddot)`` with the same length as *f0*, computed
        on the transformed contour (``log2(f0)`` by default so derivatives
        are in octaves/s, octaves/s²).
        """
        base = PitchMetrics._transform_f0(f0, pitch_transform)
        smooth = base.copy()
        if median_window > 1:
            smooth = median_filter(smooth, size=median_window)

        p_dot = np.zeros_like(smooth)
        p_ddot = np.zeros_like(smooth)
        if len(smooth) > 1:
            p_dot[1:-1] = (smooth[2:] - smooth[:-2]) / (2.0 * dt)
            p_dot[0] = (smooth[1] - smooth[0]) / dt
            p_dot[-1] = (smooth[-1] - smooth[-2]) / dt
        if len(smooth) > 2:
            p_ddot[1:-1] = (
                smooth[2:] - 2.0 * smooth[1:-1] + smooth[:-2]
            ) / (dt ** 2)
        return p_dot, p_ddot

    # ------------------------------------------------------------------
    # Design matrix
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_regressors(regressors: RegressorSpec) -> Tuple[str, ...]:
        if isinstance(regressors, str):
            if regressors not in PitchMetrics.REGRESSOR_PRESETS:
                raise ValueError(
                    f"Unknown regressor preset {regressors!r}. "
                    f"Known: {sorted(PitchMetrics.REGRESSOR_PRESETS)}"
                )
            return PitchMetrics.REGRESSOR_PRESETS[regressors]
        keys = tuple(regressors)
        unknown = [k for k in keys if k not in PitchMetrics.SUPPORTED_KEYS]
        if unknown:
            raise ValueError(
                f"Unknown regressor keys: {unknown}. "
                f"Supported: {PitchMetrics.SUPPORTED_KEYS}"
            )
        if not keys:
            raise ValueError("regressors must contain at least one key")
        return keys

    @staticmethod
    def build_design(
        f0: np.ndarray,
        regressors: RegressorSpec,
        *,
        dt: float,
        pitch_transform: str = "log2",
        median_window: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[str, ...]]:
        """Build the (n_frames, 1 + k) design matrix and voiced mask.

        Column 0 is the intercept; the remaining columns follow the order
        of *regressors*. Frames where any required column is NaN/inf are
        marked unvoiced in the returned mask.
        """
        keys = PitchMetrics._resolve_regressors(regressors)
        f0 = np.asarray(f0, dtype=np.float64)
        n = f0.shape[0]

        need_dynamics = any(k in ("p_dot", "p_ddot") for k in keys)
        if need_dynamics:
            p_dot, p_ddot = PitchMetrics.f0_dynamics(
                f0, dt=dt, median_window=median_window,
                pitch_transform=pitch_transform,
            )
        else:
            p_dot = p_ddot = None

        p_trans = PitchMetrics._transform_f0(f0, pitch_transform)
        log2_f0 = PitchMetrics._transform_f0(f0, "log2")

        col_map = {
            "p":       p_trans,
            "p2":      p_trans ** 2,
            "p_dot":   p_dot,
            "p_ddot":  p_ddot,
            "f0":      f0,
            "f0_2":    f0 ** 2,
            "log2_f0": log2_f0,
        }

        cols = [np.ones(n, dtype=np.float64)]
        for k in keys:
            cols.append(col_map[k])
        X = np.column_stack(cols)

        voiced = np.all(np.isfinite(X), axis=1)
        return X, voiced, keys

    # ------------------------------------------------------------------
    # Fit dispatch
    # ------------------------------------------------------------------

    @staticmethod
    def fit(
        features_2d: np.ndarray,
        f0: np.ndarray,
        feat_keys: Sequence[str],
        regressors: RegressorSpec = "B",
        method: str = "ols",
        *,
        dt: float,
        pitch_transform: str = "log2",
        median_window: int = 5,
        alphas: Optional[np.ndarray] = None,
        cv: int = 5,
        random_state: int = 0,
    ) -> Dict[str, Dict[str, Any]]:
        """Fit one linear model per descriptor column of *features_2d*.

        Parameters
        ----------
        features_2d : (n_frames, n_features) array
        f0 : (n_frames,) array, Hz
        feat_keys : column names for *features_2d*
        regressors : preset name or list of regressor keys (see
            :attr:`REGRESSOR_PRESETS` and :attr:`SUPPORTED_KEYS`).
        method : ``"ols"`` or ``"l1"``.
        dt : frame step in seconds, used for F0 derivatives.

        Returns
        -------
        dict
            ``{feat_name: model_dict}``. Every model dict contains
            ``coeffs``, ``r2``, ``adj_r2``, ``ss_res``, ``n``, ``k``,
            ``regressors``, ``method``, ``pitch_transform``,
            ``dt``, ``median_window``. OLS adds ``se``, ``t_values``,
            ``p_values``, ``aic``, ``bic``. L1 adds ``alpha``,
            ``n_nonzero``, ``nonzero_mask``.
        """
        X, voiced, keys = PitchMetrics.build_design(
            f0, regressors, dt=dt, pitch_transform=pitch_transform,
            median_window=median_window,
        )
        features_2d = np.asarray(features_2d, dtype=np.float64)
        if features_2d.shape[0] != X.shape[0]:
            raise ValueError(
                f"features_2d has {features_2d.shape[0]} frames but f0 has "
                f"{X.shape[0]}"
            )
        if features_2d.shape[1] != len(feat_keys):
            raise ValueError(
                f"features_2d has {features_2d.shape[1]} columns but "
                f"feat_keys has {len(feat_keys)}"
            )

        base = {
            "regressors": keys,
            "method": method,
            "pitch_transform": pitch_transform,
            "dt": dt,
            "median_window": median_window,
        }

        results: Dict[str, Dict[str, Any]] = {}
        for i, key in enumerate(feat_keys):
            y_full = features_2d[:, i]
            col_mask = voiced & np.isfinite(y_full)
            X_v = X[col_mask]
            y_v = y_full[col_mask]

            if method == "ols":
                fit_out = PitchMetrics._fit_ols(X_v, y_v)
            elif method == "l1":
                fit_out = PitchMetrics._fit_l1(
                    X_v, y_v, alphas=alphas, cv=cv, random_state=random_state,
                )
            else:
                raise ValueError(f"Unknown method {method!r}; expected 'ols' or 'l1'")

            # Scale constants captured on the fit sample so downstream
            # metrics can be reported in scale-invariant form.
            y_mean = float(y_v.mean()) if y_v.size else float("nan")
            y_std = float(y_v.std(ddof=0)) if y_v.size else float("nan")
            x_slopes = X_v[:, 1:]
            x_std = (
                x_slopes.std(axis=0, ddof=0)
                if x_slopes.size
                else np.zeros(len(keys))
            )

            results[key] = {
                **base,
                **fit_out,
                "y_mean": y_mean,
                "y_std": y_std,
                "x_std": np.asarray(x_std, dtype=np.float64),
            }

        return results

    # ------------------------------------------------------------------
    # OLS
    # ------------------------------------------------------------------

    @staticmethod
    def _fit_ols(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        n, k = X.shape
        coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coeffs
        ss_res = float(resid @ resid)
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        df = n - k
        adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / df if df > 0 else float("nan")

        if n > 0 and ss_res > 0:
            sigma2_mle = ss_res / n
            log_lik = -0.5 * n * (np.log(2.0 * np.pi) + np.log(sigma2_mle) + 1.0)
            aic = 2.0 * k - 2.0 * log_lik
            bic = k * np.log(n) - 2.0 * log_lik
        else:
            aic = bic = float("nan")

        if df > 0 and ss_res > 0:
            sigma2 = ss_res / df
            try:
                cov = sigma2 * np.linalg.inv(X.T @ X)
                se = np.sqrt(np.maximum(np.diag(cov), 0.0))
                t_vals = np.where(se > 0, coeffs / se, 0.0)
                p_vals = 2.0 * t_dist.sf(np.abs(t_vals), df)
            except np.linalg.LinAlgError:
                se = np.full(k, np.nan)
                t_vals = np.full(k, np.nan)
                p_vals = np.full(k, np.nan)
        else:
            se = np.full(k, np.nan)
            t_vals = np.full(k, np.nan)
            p_vals = np.full(k, np.nan)

        return {
            "coeffs": coeffs,
            "r2": r2,
            "adj_r2": adj_r2,
            "ss_res": ss_res,
            "n": int(n),
            "k": int(k),
            "se": se,
            "t_values": t_vals,
            "p_values": p_vals,
            "aic": float(aic),
            "bic": float(bic),
        }

    # ------------------------------------------------------------------
    # L1 (LassoCV on standardized slopes)
    # ------------------------------------------------------------------

    @staticmethod
    def _fit_l1(
        X: np.ndarray,
        y: np.ndarray,
        *,
        alphas: Optional[np.ndarray],
        cv: int,
        random_state: int,
    ) -> Dict[str, Any]:
        from sklearn.linear_model import LassoCV  # optional dep

        n, k = X.shape
        slopes = X[:, 1:]  # drop the intercept column
        mu = slopes.mean(axis=0)
        sigma = slopes.std(axis=0)
        sigma_safe = np.where(sigma > 0, sigma, 1.0)
        X_std = (slopes - mu) / sigma_safe

        lasso = LassoCV(
            alphas=100 if alphas is None else alphas,
            cv=cv,
            random_state=random_state,
            max_iter=10_000,
        )
        lasso.fit(X_std, y)
        b_std = lasso.coef_
        c_std = float(lasso.intercept_)

        beta = np.zeros(k, dtype=np.float64)
        beta[0] = c_std - float(np.sum(mu * (b_std / sigma_safe)))
        beta[1:] = b_std / sigma_safe

        y_hat = X @ beta
        resid = y - y_hat
        ss_res = float(resid @ resid)
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        n_slopes = k - 1
        df = n - k
        adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / df if df > 0 else float("nan")

        zero_tol = 1e-10
        nonzero_mask = np.abs(beta) > zero_tol
        nonzero_mask[0] = True

        return {
            "coeffs": beta,
            "r2": r2,
            "adj_r2": adj_r2,
            "ss_res": ss_res,
            "n": int(n),
            "k": int(k),
            "alpha": float(lasso.alpha_),
            "n_nonzero": int(np.sum(np.abs(beta[1:]) > zero_tol)),
            "nonzero_mask": nonzero_mask,
            "aic": float("nan"),
            "bic": float("nan"),
            "se": np.full(k, np.nan),
            "t_values": np.full(k, np.nan),
            "p_values": np.full(k, np.nan),
        }

    # ------------------------------------------------------------------
    # Prediction and evaluation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _design_from_model(f0: np.ndarray, model: Dict[str, Any]):
        return PitchMetrics.build_design(
            f0,
            model["regressors"],
            dt=model["dt"],
            pitch_transform=model["pitch_transform"],
            median_window=model["median_window"],
        )

    @staticmethod
    def predict(
        f0: np.ndarray,
        models: Dict[str, Dict[str, Any]],
        feat_keys: Sequence[str],
    ) -> np.ndarray:
        """Predict each descriptor from *f0* using the stored models.

        The design matrix is rebuilt from the ``regressors`` /
        ``pitch_transform`` / ``dt`` / ``median_window`` recorded on each
        model, so callers do not need to repass F0 derivatives. Unvoiced
        frames are returned as NaN.
        """
        f0 = np.asarray(f0, dtype=np.float64)
        out = np.full((f0.shape[0], len(feat_keys)), np.nan)
        for i, key in enumerate(feat_keys):
            model = models.get(key)
            if model is None:
                continue
            X, voiced, _ = PitchMetrics._design_from_model(f0, model)
            out[voiced, i] = X[voiced] @ model["coeffs"]
        return out

    @staticmethod
    def coupling_error(
        features_2d: np.ndarray,
        f0: np.ndarray,
        ref_models: Dict[str, Dict[str, Any]],
        feat_keys: Sequence[str],
        normalize: str = "none",
    ) -> Dict[str, float]:
        """Per-descriptor prediction error vs the reference model.

        Parameters
        ----------
        normalize : {"none", "nmse", "nrmse"}
            ``"none"``  — raw mean-squared error in the descriptor's own
            units (not comparable across descriptors with different scales).
            ``"nmse"``  — ``mse / var_ref``, where ``var_ref`` is the
            reference fit-sample variance of the descriptor. 0 is perfect,
            1 is as bad as predicting the reference mean. Dimensionless,
            so values *are* comparable across descriptors.
            ``"nrmse"`` — ``sqrt(mse) / std_ref``, same idea but in
            std-deviation units.
        """
        if normalize not in ("none", "nmse", "nrmse"):
            raise ValueError(
                f"Unknown normalize {normalize!r}; "
                f"expected 'none', 'nmse', or 'nrmse'"
            )

        features_2d = np.asarray(features_2d, dtype=np.float64)
        predicted = PitchMetrics.predict(f0, ref_models, feat_keys)

        ce: Dict[str, float] = {}
        for i, key in enumerate(feat_keys):
            if key not in ref_models:
                continue
            diff = features_2d[:, i] - predicted[:, i]
            mask = np.isfinite(diff)
            if not mask.any():
                ce[key] = float("nan")
                continue
            mse = float(np.mean(diff[mask] ** 2))

            if normalize == "none":
                ce[key] = mse
                continue

            y_std = ref_models[key].get("y_std", float("nan"))
            if not (np.isfinite(y_std) and y_std > 0):
                ce[key] = float("nan")
                continue
            if normalize == "nmse":
                ce[key] = mse / (y_std ** 2)
            else:  # nrmse
                ce[key] = float(np.sqrt(mse)) / y_std
        return ce

    @staticmethod
    def coefficient_distance(
        models_output: Dict[str, Dict[str, Any]],
        models_ref: Dict[str, Dict[str, Any]],
        feat_keys: Sequence[str],
        normalize: str = "none",
    ) -> Dict[str, float]:
        """L2 distance between output and reference coefficient vectors.

        Raises ``ValueError`` if a descriptor was fit with mismatched
        regressor layouts on the two sides — comparing those coefficients
        would be meaningless.

        Parameters
        ----------
        normalize : {"none", "standardized"}
            ``"none"`` — raw L2 on coefficients in their original units
            (not comparable across descriptors with different scales).
            ``"standardized"`` — rescale each slope coefficient by
            ``x_std_ref[j] / y_std_ref`` before computing the distance,
            so the comparison is in dimensionless standardized-beta units
            (an "effect-size" distance). The intercept column is dropped
            because it has no regressor scale.
        """
        if normalize not in ("none", "standardized"):
            raise ValueError(
                f"Unknown normalize {normalize!r}; "
                f"expected 'none' or 'standardized'"
            )

        dist: Dict[str, float] = {}
        for key in feat_keys:
            if key not in models_output or key not in models_ref:
                continue
            m_out = models_output[key]
            m_ref = models_ref[key]
            if tuple(m_out["regressors"]) != tuple(m_ref["regressors"]):
                raise ValueError(
                    f"regressor mismatch for {key!r}: "
                    f"{m_out['regressors']} vs {m_ref['regressors']}"
                )

            diff = m_out["coeffs"] - m_ref["coeffs"]
            if normalize == "none":
                dist[key] = float(np.linalg.norm(diff))
                continue

            y_std = m_ref.get("y_std", float("nan"))
            x_std = m_ref.get("x_std")
            if (
                x_std is None
                or not (np.isfinite(y_std) and y_std > 0)
                or not np.all(np.isfinite(x_std))
            ):
                dist[key] = float("nan")
                continue
            # Drop intercept (diff[0]) — standardized betas are slopes only.
            scale = np.asarray(x_std, dtype=np.float64) / y_std
            dist[key] = float(np.linalg.norm(diff[1:] * scale))
        return dist

    @staticmethod
    def correlation_summary(
        features_2d: np.ndarray,
        f0: np.ndarray,
        feat_keys: Sequence[str],
        pitch_transform: str = "log2",
        method: str = "pearson",
    ) -> Dict[str, Dict[str, float]]:
        """Correlation between each descriptor and the (transformed) pitch.

        Parameters
        ----------
        method : {"pearson", "spearman", "both"}
            ``"pearson"`` (default) — linear correlation; result keys are
            ``pearson_r`` and ``pearson_p``. Backward compatible.
            ``"spearman"`` — rank correlation; result keys are
            ``spearman_r`` and ``spearman_p``.
            ``"both"`` — returns all four keys per descriptor.
        """
        if method not in ("pearson", "spearman", "both"):
            raise ValueError(
                f"Unknown method {method!r}; "
                f"expected 'pearson', 'spearman', or 'both'"
            )

        features_2d = np.asarray(features_2d, dtype=np.float64)
        p = PitchMetrics._transform_f0(f0, pitch_transform)
        voiced = np.isfinite(p)
        p_v = p[voiced]
        feats_v = features_2d[voiced]

        want_pearson = method in ("pearson", "both")
        want_spearman = method in ("spearman", "both")

        nan_entry: Dict[str, float] = {}
        if want_pearson:
            nan_entry["pearson_r"] = float("nan")
            nan_entry["pearson_p"] = float("nan")
        if want_spearman:
            nan_entry["spearman_r"] = float("nan")
            nan_entry["spearman_p"] = float("nan")

        result: Dict[str, Dict[str, float]] = {}
        for i, key in enumerate(feat_keys):
            col = feats_v[:, i]
            valid = np.isfinite(col)
            if valid.sum() < 3:
                result[key] = dict(nan_entry)
                continue
            entry: Dict[str, float] = {}
            if want_pearson:
                r, p_val = pearsonr(p_v[valid], col[valid])
                entry["pearson_r"] = float(r)
                entry["pearson_p"] = float(p_val)
            if want_spearman:
                r, p_val = spearmanr(p_v[valid], col[valid])
                entry["spearman_r"] = float(r)
                entry["spearman_p"] = float(p_val)
            result[key] = entry
        return result
