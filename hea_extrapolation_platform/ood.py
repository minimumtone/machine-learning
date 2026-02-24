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
        X_scaled = self._scaler.fit_transform(X_train.values)

        # Mahalanobis setup
        self._mean = X_scaled.mean(axis=0)
        cov = np.cov(X_scaled, rowvar=False)
        # Regularise for numerical stability
        cov += np.eye(cov.shape[0]) * 1e-6
        try:
            self._cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            logger.warning("Covariance matrix singular – using pseudo-inverse")
            self._cov_inv = np.linalg.pinv(cov)

        # kNN setup
        actual_k = min(self._k, X_scaled.shape[0] - 1)
        if actual_k < 1:
            actual_k = 1
        self._nn = NearestNeighbors(n_neighbors=actual_k, metric="euclidean")
        self._nn.fit(X_scaled)

        # Compute training set scores for threshold calibration
        maha_train = self._mahalanobis_batch(X_scaled)
        knn_train = self._knn_batch(X_scaled)
        composite_train = self._combine(maha_train, knn_train)
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

        X_scaled = self._scaler.transform(X_query.values)

        maha = self._mahalanobis_batch(X_scaled)
        knn = self._knn_batch(X_scaled)
        composite = self._combine(maha, knn)

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
        """Compute Mahalanobis distance for each row."""
        dists = np.array([
            mahalanobis(x, self._mean, self._cov_inv) for x in X_scaled
        ])
        return dists

    def _knn_batch(self, X_scaled: np.ndarray) -> np.ndarray:
        """Compute mean kNN distance for each row."""
        dists, _ = self._nn.kneighbors(X_scaled)
        return dists.mean(axis=1)

    @staticmethod
    def _combine(
        maha: np.ndarray,
        knn: np.ndarray,
        w_maha: float = 0.5,
        w_knn: float = 0.5,
    ) -> np.ndarray:
        """Normalise and combine Mahalanobis + kNN into [0, 1] composite."""
        def _norm(arr: np.ndarray) -> np.ndarray:
            lo, hi = arr.min(), arr.max()
            if hi - lo < 1e-12:
                return np.zeros_like(arr)
            return (arr - lo) / (hi - lo)

        return w_maha * _norm(maha) + w_knn * _norm(knn)
