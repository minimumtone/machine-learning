"""
Out-of-Distribution (OOD) Detection Module
外挿（OOD）検知モジュール

Detects samples that lie outside the training distribution using:
  - Mahalanobis distance
  - k-Nearest Neighbour distance (k=10)
  - Normalised composite OOD score
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class OODResult:
    """Container for OOD detection results."""

    # Per-sample scores (length = n_query)
    mahalanobis_scores: np.ndarray
    knn_scores: np.ndarray
    composite_scores: np.ndarray  # normalised [0, 1]

    # Binary classification at given threshold
    is_ood: np.ndarray  # bool array
    ood_threshold: float

    # Summary statistics
    ood_ratio: float  # fraction of OOD samples
    n_total: int
    n_ood: int


class OODDetector:
    """OOD detector based on Mahalanobis + kNN distance in feature space.

    Usage::

        detector = OODDetector(k=10, threshold_quantile=0.95)
        detector.fit(X_train)
        result = detector.score(X_query)

    Parameters
    ----------
    k : int
        Number of neighbours for kNN distance (default 10).
    threshold_quantile : float
        Quantile on training distances to set OOD threshold (default 0.95).
    """

    def __init__(
        self,
        k: int = 10,
        threshold_quantile: float = 0.95,
    ) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if not 0 < threshold_quantile < 1:
            raise ValueError(
                f"threshold_quantile must be in (0,1), got {threshold_quantile}"
            )
        self._k = k
        self._threshold_q = threshold_quantile
        self._scaler: Optional[StandardScaler] = None
        self._cov_inv: Optional[np.ndarray] = None
        self._mean: Optional[np.ndarray] = None
        self._nn: Optional[NearestNeighbors] = None
        self._fitted = False
        self._train_composite: Optional[np.ndarray] = None
        self._ood_threshold: float = 0.0
        # Normalization params saved at fit-time for consistent scoring
        self._maha_min: float = 0.0
        self._maha_max: float = 1.0
        self._knn_min: float = 0.0
        self._knn_max: float = 1.0

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X_train: pd.DataFrame) -> "OODDetector":
        """Fit the OOD detector on training features.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training feature matrix (n_train, n_features).
        """
        logger.info("Fitting OOD detector on %d samples, %d features",
                     X_train.shape[0], X_train.shape[1])

        self._scaler = StandardScaler()
        # CRITICAL: Force C-contiguous layout at the DataFrame → numpy boundary.
        # pandas 3.0 DataFrame.values returns F-contiguous (column-major) arrays
        # when the BlockManager is fragmented.  StandardScaler preserves layout,
        # so X_scaled inherits F-contiguous.  Row slices of an F-contiguous 2-D
        # array have non-unit stride, which causes scipy.spatial.distance.
        # mahalanobis (and BLAS/LAPACK in general) to SIGSEGV.
        X_train_arr = np.ascontiguousarray(
            X_train.to_numpy(dtype="float64", na_value=np.nan)
        )
        X_scaled = np.ascontiguousarray(self._scaler.fit_transform(X_train_arr))

        # Mahalanobis setup
        self._mean = X_scaled.mean(axis=0)
        # np.cov returns a 0-d array when n_features==1; atleast_2d normalises
        # that so downstream np.eye() and np.linalg.inv() always see a 2-D matrix.
        cov = np.atleast_2d(np.cov(X_scaled, rowvar=False))
        # Regularise for numerical stability (also handles rank-deficient case
        # when n_train < n_features, e.g. FS_MAGPIE with a small fold).
        cov += np.eye(cov.shape[0]) * 1e-6
        try:
            self._cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            logger.warning("Covariance matrix singular – using pseudo-inverse")
            self._cov_inv = np.linalg.pinv(cov)

        # kNN setup — use k+1 neighbours so that when querying training
        # data we can drop the self-reference (distance-0 hit).
        actual_k = min(self._k, X_scaled.shape[0] - 1)
        if actual_k < 1:
            raise ValueError(
                f"OODDetector requires at least 2 training samples, "
                f"got {X_scaled.shape[0]}. Cannot fit kNN with k={self._k}."
            )
        self._actual_k = actual_k  # save for consistent use in score()
        self._nn = NearestNeighbors(n_neighbors=actual_k + 1, metric="euclidean")
        self._nn.fit(X_scaled)

        # Compute training set scores for threshold calibration.
        # _knn_batch_train excludes self-reference (first column).
        maha_train = self._mahalanobis_batch(X_scaled)
        knn_train = self._knn_batch_train(X_scaled)

        # Save training min/max for consistent normalization at score-time
        self._maha_min = float(maha_train.min())
        self._maha_max = float(maha_train.max())
        self._knn_min = float(knn_train.min())
        self._knn_max = float(knn_train.max())

        composite_train = self._combine(
            maha_train, knn_train,
            maha_min=self._maha_min, maha_max=self._maha_max,
            knn_min=self._knn_min, knn_max=self._knn_max,
        )
        self._train_composite = composite_train
        self._ood_threshold = float(np.quantile(composite_train, self._threshold_q))

        self._fitted = True
        logger.info("OOD detector fitted. Threshold (q=%.2f) = %.4f",
                     self._threshold_q, self._ood_threshold)
        return self

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------

    def score(self, X_query: pd.DataFrame) -> OODResult:
        """Score query samples for OOD-ness.

        Parameters
        ----------
        X_query : pd.DataFrame
            Query feature matrix (n_query, n_features).

        Returns
        -------
        OODResult
        """
        if not self._fitted:
            raise RuntimeError("OODDetector.fit() must be called before score()")

        # Same C-contiguous guarantee as in fit().
        X_query_arr = np.ascontiguousarray(
            X_query.to_numpy(dtype="float64", na_value=np.nan)
        )
        X_scaled = np.ascontiguousarray(self._scaler.transform(X_query_arr))

        maha = self._mahalanobis_batch(X_scaled)
        knn = self._knn_batch(X_scaled)
        composite = self._combine(
            maha, knn,
            maha_min=self._maha_min, maha_max=self._maha_max,
            knn_min=self._knn_min, knn_max=self._knn_max,
        )

        is_ood = composite > self._ood_threshold
        n_ood = int(is_ood.sum())
        n_total = len(composite)

        logger.info("OOD scoring: %d / %d samples flagged (%.1f%%)",
                     n_ood, n_total, 100 * n_ood / max(n_total, 1))

        return OODResult(
            mahalanobis_scores=maha,
            knn_scores=knn,
            composite_scores=composite,
            is_ood=is_ood,
            ood_threshold=self._ood_threshold,
            ood_ratio=n_ood / max(n_total, 1),
            n_total=n_total,
            n_ood=n_ood,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _mahalanobis_batch(self, X_scaled: np.ndarray) -> np.ndarray:
        """Compute Mahalanobis distance for each row.

        Clamps NaN/Inf results (caused by floating-point noise in the
        covariance inverse) to 0.0 so downstream normalisation doesn't
        propagate NaN into the composite score.
        """
        dists = np.array([
            mahalanobis(
                np.ascontiguousarray(x), self._mean, self._cov_inv
            )
            for x in X_scaled
        ])
        return np.nan_to_num(dists, nan=0.0, posinf=0.0, neginf=0.0)

    def _knn_batch_train(self, X_scaled: np.ndarray) -> np.ndarray:
        """Compute mean kNN distance for training data (excludes self).

        The fitted NearestNeighbors uses k+1 neighbours.  The first
        column is always the query point itself (distance ≈ 0), so we
        drop it.
        """
        dists, _ = self._nn.kneighbors(X_scaled)
        return dists[:, 1:].mean(axis=1)

    def _knn_batch(self, X_scaled: np.ndarray) -> np.ndarray:
        """Compute mean kNN distance for query data (no self-reference).

        Query points are not part of the fitted index, so there is no
        self-reference.  However, the fitted model has k+1 neighbours;
        we take only the first ``_actual_k`` columns to keep the same
        semantics as training (``_actual_k`` may be < ``_k`` when the
        training set is small).
        """
        dists, _ = self._nn.kneighbors(X_scaled)
        return dists[:, :self._actual_k].mean(axis=1)

    @staticmethod
    def _combine(
        maha: np.ndarray,
        knn: np.ndarray,
        maha_min: float,
        maha_max: float,
        knn_min: float,
        knn_max: float,
        w_maha: float = 0.5,
        w_knn: float = 0.5,
    ) -> np.ndarray:
        """Normalise and combine Mahalanobis + kNN into composite score.

        Normalization uses training-set min/max so that fit() and
        score() operate on the same scale.  Query scores may exceed
        [0, 1] when they are further from the training distribution
        than any training sample – this is intentional.
        """
        def _norm(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
            if hi - lo < 1e-12:
                return np.zeros_like(arr)
            return (arr - lo) / (hi - lo)

        return (
            w_maha * _norm(maha, maha_min, maha_max)
            + w_knn * _norm(knn, knn_min, knn_max)
        )
