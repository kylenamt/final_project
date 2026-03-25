"""Distributional distance metrics for timbre evaluation.

This module provides the ``Loss`` class with two statistical distance
measures used to compare per-frame feature distributions extracted by
``TimbreMetrics``:

- **MMD** (Maximum Mean Discrepancy) — kernel-based two-sample test
  statistic using the unbiased quadratic-time estimator with an RBF kernel.
- **Wasserstein** distance — earth-mover distance; exact 1-D solution via
  ``scipy.stats`` and sliced approximation for higher dimensions.

Quick usage::

    from evaluation.loss import Loss

    loss = Loss()
    distance = loss.mmd(features_a, features_b)
    distance = loss.wasserstein(features_a, features_b)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance as _wasserstein_1d


class Loss:
    """Statistical distance metrics for comparing feature distributions.

    Parameters
    ----------
    kernel : str
        Kernel used by MMD.  Only ``"rbf"`` is supported.
    bandwidth : float or None
        RBF bandwidth (sigma).  When *None* the median heuristic is used.
    n_projections : int
        Number of random projections for sliced Wasserstein (multi-dim).
    """

    def __init__(
        self,
        kernel: str = "rbf",
        bandwidth: Optional[float] = None,
        n_projections: int = 50,
    ):
        if kernel != "rbf":
            raise ValueError(f"Unsupported kernel: {kernel!r}. Only 'rbf' is supported.")
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.n_projections = n_projections

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_inputs(
        X: np.ndarray, Y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Validate and reshape inputs to 2-D float64 arrays.

        1-D arrays ``(n,)`` are promoted to column vectors ``(n, 1)``.
        """
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)

        if X.ndim == 1:
            X = X[:, np.newaxis]
        if Y.ndim == 1:
            Y = Y[:, np.newaxis]

        if X.ndim != 2 or Y.ndim != 2:
            raise ValueError("X and Y must be 1-D or 2-D arrays.")
        if X.shape[1] != Y.shape[1]:
            raise ValueError(
                f"Feature dimensions must match: X has {X.shape[1]}, "
                f"Y has {Y.shape[1]}."
            )
        if X.shape[0] < 2 or Y.shape[0] < 2:
            raise ValueError("X and Y must each contain at least 2 samples.")

        return X, Y

    # ------------------------------------------------------------------
    # Kernel helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _median_heuristic(X: np.ndarray, Y: np.ndarray) -> float:
        """Compute bandwidth via the median heuristic.

        Returns the median of all pairwise Euclidean distances between the
        concatenated samples.  A small epsilon is added to avoid zero bandwidth.
        """
        Z = np.concatenate([X, Y], axis=0)
        dists = cdist(Z, Z, metric="euclidean")
        # Take upper triangle (excluding diagonal zeros)
        triu_idx = np.triu_indices_from(dists, k=1)
        median_dist = float(np.median(dists[triu_idx]))
        return max(median_dist, 1e-10)

    @staticmethod
    def _rbf_kernel(
        X: np.ndarray, Y: np.ndarray, bandwidth: float
    ) -> np.ndarray:
        """Compute the RBF (Gaussian) kernel matrix.

        k(x, y) = exp(-||x - y||^2 / (2 * sigma^2))
        """
        sq_dists = cdist(X, Y, metric="sqeuclidean")
        return np.exp(-sq_dists / (2.0 * bandwidth ** 2))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def mmd(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Maximum Mean Discrepancy (unbiased estimator).

        Parameters
        ----------
        X : np.ndarray, shape ``(m,)`` or ``(m, d)``
            Samples from the first distribution.
        Y : np.ndarray, shape ``(n,)`` or ``(n, d)``
            Samples from the second distribution.

        Returns
        -------
        float
            MMD value (square root of the unbiased MMD^2 estimate, clamped
            to zero to avoid negative values from estimation variance).
        """
        X, Y = self._validate_inputs(X, Y)
        m, n = X.shape[0], Y.shape[0]

        sigma = self.bandwidth if self.bandwidth is not None else self._median_heuristic(X, Y)

        K_XX = self._rbf_kernel(X, X, sigma)
        K_YY = self._rbf_kernel(Y, Y, sigma)
        K_XY = self._rbf_kernel(X, Y, sigma)

        # Unbiased estimator: exclude diagonal from K_XX and K_YY
        np.fill_diagonal(K_XX, 0.0)
        np.fill_diagonal(K_YY, 0.0)

        mmd_sq = (
            K_XX.sum() / (m * (m - 1))
            + K_YY.sum() / (n * (n - 1))
            - 2.0 * K_XY.sum() / (m * n)
        )

        return float(np.sqrt(max(mmd_sq, 0.0)))

    def wasserstein(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Wasserstein-1 (earth mover's) distance.

        Parameters
        ----------
        X : np.ndarray, shape ``(m,)`` or ``(m, d)``
            Samples from the first distribution.
        Y : np.ndarray, shape ``(n,)`` or ``(n, d)``
            Samples from the second distribution.

        Returns
        -------
        float
            For 1-D inputs, the exact Wasserstein-1 distance.
            For multi-dimensional inputs, the sliced Wasserstein distance
            (mean of 1-D Wasserstein distances over random projections).
        """
        X_raw = np.asarray(X, dtype=np.float64)
        Y_raw = np.asarray(Y, dtype=np.float64)

        # 1-D fast path
        if X_raw.ndim == 1 and Y_raw.ndim == 1:
            return float(_wasserstein_1d(X_raw, Y_raw))

        # Multi-dimensional: sliced Wasserstein
        X_2d, Y_2d = self._validate_inputs(X_raw, Y_raw)
        d = X_2d.shape[1]

        if d == 1:
            return float(_wasserstein_1d(X_2d[:, 0], Y_2d[:, 0]))

        rng = np.random.default_rng()
        directions = rng.standard_normal((self.n_projections, d))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)

        total = 0.0
        for theta in directions:
            proj_x = X_2d @ theta
            proj_y = Y_2d @ theta
            total += _wasserstein_1d(proj_x, proj_y)

        return float(total / self.n_projections)
