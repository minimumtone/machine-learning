"""
Data Splitting Strategies for HEA Extrapolation Platform
データ分割戦略モジュール

Three splitting policies:
  RandomCVSplitter         - Standard K-fold cross-validation
  CompositionBlockSplitter - Cluster-based splits on composition space
  ElementExclusionSplitter - Hold-out alloys containing a specific element
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Type alias for a train/test index split
SplitIndices = Tuple[np.ndarray, np.ndarray]


class BaseSplitter(ABC):
    """Abstract base class for all splitting strategies."""

    name: str = "base"

    @abstractmethod
    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        compositions: Optional[pd.DataFrame] = None,
    ) -> Generator[SplitIndices, None, None]:
        """Yield (train_indices, test_indices) tuples."""
        ...

    @abstractmethod
    def n_splits(self) -> int:
        """Return number of folds."""
        ...


class RandomCVSplitter(BaseSplitter):
    """Standard K-fold cross-validation with optional shuffling.

    Parameters
    ----------
    n_folds : int
        Number of folds (default 5).
    seed : int
        Random seed for shuffling.
    """

    name = "RandomCV"

    def __init__(self, n_folds: int = 5, seed: int = 42) -> None:
        if n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {n_folds}")
        self._n_folds = n_folds
        self._seed = seed

    def n_splits(self) -> int:
        return self._n_folds

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        compositions: Optional[pd.DataFrame] = None,
    ) -> Generator[SplitIndices, None, None]:
        kf = KFold(
            n_splits=self._n_folds, shuffle=True, random_state=self._seed,
        )
        logger.debug("RandomCV split: n_folds=%d, seed=%d", self._n_folds, self._seed)
        for train_idx, test_idx in kf.split(X):
            yield train_idx, test_idx


class CompositionBlockSplitter(BaseSplitter):
    """Cluster-based splitting on composition vectors.

    1. Standardise composition vectors.
    2. k-means clustering (k = n_folds).
    3. Each cluster forms one test fold.

    This prevents same-family alloys leaking between train/test.

    Parameters
    ----------
    n_folds : int
        Number of clusters / folds (default 5).
    seed : int
        Random seed for k-means.
    """

    name = "CompositionBlock"

    def __init__(self, n_folds: int = 5, seed: int = 42) -> None:
        if n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {n_folds}")
        self._n_folds = n_folds
        self._seed = seed

    def n_splits(self) -> int:
        return self._n_folds

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        compositions: Optional[pd.DataFrame] = None,
    ) -> Generator[SplitIndices, None, None]:
        if compositions is None:
            raise ValueError(
                "CompositionBlockSplitter requires the `compositions` DataFrame "
                "(element columns with atomic fractions)."
            )

        scaler = StandardScaler()
        comp_scaled = scaler.fit_transform(compositions.values)

        actual_k = min(self._n_folds, len(compositions))
        km = KMeans(
            n_clusters=actual_k,
            random_state=self._seed,
            n_init=10,
            max_iter=300,
        )
        labels = km.fit_predict(comp_scaled)
        all_idx = np.arange(len(compositions))

        logger.debug(
            "CompositionBlock split: k=%d, cluster sizes=%s",
            actual_k,
            [int((labels == k).sum()) for k in range(actual_k)],
        )

        for fold_label in range(actual_k):
            test_mask = labels == fold_label
            if test_mask.sum() == 0:
                logger.warning("Empty cluster %d – skipping fold", fold_label)
                continue
            yield all_idx[~test_mask], all_idx[test_mask]


class ElementExclusionSplitter(BaseSplitter):
    """Hold out all alloys containing a specific element.

    For each target element, alloys *containing* that element go to the test
    set and the remaining alloys form the training set.

    Parameters
    ----------
    target_elements : list of str
        Elements to test exclusion on.  Each produces one fold.
    min_test_size : int
        Minimum number of test samples; elements with fewer matching alloys
        are skipped with a warning.
    """

    name = "ElementExclusion"

    def __init__(
        self,
        target_elements: Optional[List[str]] = None,
        min_test_size: int = 5,
    ) -> None:
        self._target_elements = target_elements or ["Co", "Ni", "Ti"]
        self._min_test_size = min_test_size
        self._actual_n_splits: Optional[int] = None  # set after split()

    def n_splits(self) -> int:
        """Return the number of valid folds.

        Before ``split()`` has been called, returns the *maximum possible*
        number of folds (i.e. ``len(target_elements)``).  After ``split()``
        has completed, returns the *actual* number of folds that were
        yielded (may be fewer if elements were skipped).
        """
        if self._actual_n_splits is not None:
            return self._actual_n_splits
        return len(self._target_elements)

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        compositions: Optional[pd.DataFrame] = None,
    ) -> Generator[SplitIndices, None, None]:
        if compositions is None:
            raise ValueError(
                "ElementExclusionSplitter requires the `compositions` DataFrame."
            )

        all_idx = np.arange(len(compositions))
        actual_folds = 0

        for elem in self._target_elements:
            if elem not in compositions.columns:
                logger.warning(
                    "Element '%s' not found in composition columns – skipping", elem
                )
                continue
            test_mask = compositions[elem].values > 0
            n_test = int(test_mask.sum())
            n_train = int((~test_mask).sum())
            if n_test < self._min_test_size:
                logger.warning(
                    "Element '%s' has only %d test samples (< %d) – skipping",
                    elem, n_test, self._min_test_size,
                )
                continue
            if n_train < self._min_test_size:
                logger.warning(
                    "Element '%s' exclusion leaves only %d train samples – skipping",
                    elem, n_train,
                )
                continue
            logger.debug(
                "ElementExclusion fold: elem=%s, train=%d, test=%d",
                elem, n_train, n_test,
            )
            actual_folds += 1
            yield all_idx[~test_mask], all_idx[test_mask]

        self._actual_n_splits = actual_folds


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

SPLITTER_REGISTRY: Dict[str, type] = {
    "RandomCV": RandomCVSplitter,
    "CompositionBlock": CompositionBlockSplitter,
    "ElementExclusion": ElementExclusionSplitter,
}


def get_splitter(name: str, **kwargs) -> BaseSplitter:
    """Instantiate a splitter by name.

    Parameters
    ----------
    name : str
        One of 'RandomCV', 'CompositionBlock', 'ElementExclusion'.
    **kwargs
        Forwarded to the splitter constructor.
    """
    if name not in SPLITTER_REGISTRY:
        raise ValueError(
            f"Unknown splitter '{name}'. Available: {list(SPLITTER_REGISTRY.keys())}"
        )
    return SPLITTER_REGISTRY[name](**kwargs)
