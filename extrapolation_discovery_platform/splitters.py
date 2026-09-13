"""
Data Splitting Strategies for the HEA Feature Design Framework
データ分割戦略モジュール

Three splitting policies:
  RandomCVSplitter         - Standard K-fold cross-validation
  CompositionBlockSplitter - Cluster-based splits on composition space
  ElementExclusionSplitter - Hold-out alloys containing a specific element
  CompositionGroupCVSplitter - Exact-composition group cross-validation
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
    fold_labels: List[str]

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
        self.fold_labels = []

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
        self.fold_labels = []
        for i, (train_idx, test_idx) in enumerate(kf.split(X)):
            self.fold_labels.append(f"fold{i}")
            yield train_idx, test_idx


class CompositionBlockSplitter(BaseSplitter):
    """Cluster-based splitting on composition vectors.

    1. Standardise composition vectors.
    2. k-means clustering (k = 3 × n_folds).
    3. Greedily group clusters into n_folds size-balanced folds;
       each group of whole clusters forms one test fold.

    This prevents same-family alloys leaking between train/test while
    keeping fold sizes reasonably balanced.

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
        self._actual_n_splits: Optional[int] = None
        self.fold_labels = []

    def n_splits(self) -> int:
        """Return number of folds.

        Before ``split()`` has been fully consumed, returns the
        *requested* number of folds (upper bound).  After ``split()``
        completes, returns the *actual* number of non-empty clusters
        yielded (may be fewer if some clusters were empty).

        .. warning::
            Callers that use ``n_splits()`` to pre-allocate job lists
            should treat the pre-split value as an **upper bound** and
            handle fewer actual folds gracefully.  Alternatively, call
            ``list(splitter.split(...))`` first, then use ``len()``.
        """
        if self._actual_n_splits is not None:
            return self._actual_n_splits
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
        # CRITICAL: Force C-contiguous layout before BLAS calls in
        # StandardScaler / KMeans.  pandas 3.0 .values on fragmented
        # DataFrames returns F-contiguous arrays → SIGSEGV in BLAS.
        comp_scaled = scaler.fit_transform(
            np.ascontiguousarray(
                compositions.to_numpy(dtype="float64", na_value=np.nan)
            )
        )

        n = len(compositions)
        n_folds = min(self._n_folds, n)
        # クラスタをそのまま fold にするとサイズが極端に不均衡になり得る
        # （例: 5 fold で 114:38:5:34:96）。細かめにクラスタリングした上で
        # クラスタ単位のまま貪欲ビンパッキングで n_folds 群に束ねることで、
        # 組成ファミリの分離を保ちながら fold サイズを平準化する。
        actual_k = min(n_folds * 3, n)
        km = KMeans(
            n_clusters=actual_k,
            random_state=self._seed,
            n_init=10,
            max_iter=300,
        )
        labels = km.fit_predict(comp_scaled)
        all_idx = np.arange(n)

        cluster_sizes = [(k, int((labels == k).sum())) for k in range(actual_k)]
        cluster_sizes = [(k, s) for k, s in cluster_sizes if s > 0]
        logger.debug(
            "CompositionBlock split: k=%d, cluster sizes=%s",
            actual_k, [s for _, s in cluster_sizes],
        )

        # 大きいクラスタから順に、現在最小の fold に割り当てる（貪欲法）
        fold_members: List[List[int]] = [[] for _ in range(n_folds)]
        fold_totals = np.zeros(n_folds, dtype=int)
        for k, size in sorted(cluster_sizes, key=lambda t: -t[1]):
            dest = int(np.argmin(fold_totals))
            fold_members[dest].append(k)
            fold_totals[dest] += size

        logger.debug("CompositionBlock fold sizes (balanced)=%s", fold_totals.tolist())

        actual_folds = 0
        self.fold_labels = []
        for members in fold_members:
            if not members:
                logger.warning("Empty fold – skipping")
                continue
            test_mask = np.isin(labels, members)
            if test_mask.sum() == 0:
                continue
            self.fold_labels.append(f"fold{actual_folds}")
            actual_folds += 1
            yield all_idx[~test_mask], all_idx[test_mask]

        self._actual_n_splits = actual_folds


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
        presence_threshold: float = 0.001,
        min_train_size: int = 50,
        max_test_fraction: float = 0.4,
    ) -> None:
        self._target_elements = target_elements or ["Co", "Ni", "Ti"]
        self._min_test_size = min_test_size
        self._presence_threshold = presence_threshold
        self._min_train_size = min_train_size
        self._max_test_fraction = max_test_fraction
        self._actual_n_splits: Optional[int] = None  # set after split()
        self.fold_labels = []

    def n_splits(self) -> int:
        """Return the number of valid folds.

        Before ``split()`` has been called, returns the *maximum possible*
        number of folds (i.e. ``len(target_elements)``).  After ``split()``
        has completed, returns the *actual* number of folds that were
        yielded (may be fewer if elements were skipped).

        .. warning::
            ``_actual_n_splits`` is only updated when the ``split()``
            generator has been **fully consumed**.  Calling ``n_splits()``
            while iterating through ``split()`` will still return the
            pre-split estimate.
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
        self.fold_labels = []

        for elem in self._target_elements:
            if elem not in compositions.columns:
                logger.warning(
                    "Element '%s' not found in composition columns – skipping", elem
                )
                continue
            test_mask = np.ascontiguousarray(
                compositions[elem].to_numpy(dtype="float64")
            ) >= self._presence_threshold
            n_test = int(test_mask.sum())
            n_train = int((~test_mask).sum())
            if n_test < self._min_test_size:
                logger.warning(
                    "Element '%s' has only %d test samples (< %d) – skipping",
                    elem, n_test, self._min_test_size,
                )
                continue
            if n_train < self._min_train_size:
                logger.warning(
                    "Element '%s' exclusion leaves only %d train samples (< %d) – skipping",
                    elem, n_train, self._min_train_size,
                )
                continue
            if len(compositions) > 0 and n_test / len(compositions) > self._max_test_fraction:
                logger.warning(
                    "Element '%s' has %d/%d test samples (%.1f%% > %.1f%%) – skipping",
                    elem, n_test, len(compositions),
                    100.0 * n_test / len(compositions),
                    100.0 * self._max_test_fraction,
                )
                continue
            logger.debug(
                "ElementExclusion fold: elem=%s, train=%d, test=%d",
                elem, n_train, n_test,
            )
            self.fold_labels.append(elem)
            actual_folds += 1
            yield all_idx[~test_mask], all_idx[test_mask]

        self._actual_n_splits = actual_folds


class CompositionGroupCVSplitter(BaseSplitter):
    """exact-composition GroupKFold: identical compositions never straddle train/test"""

    name = "CompositionGroupCV"

    def __init__(self, n_folds: int = 5, seed: int = 42) -> None:
        if n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {n_folds}")
        self._n_folds = n_folds
        self._seed = seed
        self._actual_n_splits: Optional[int] = None
        self.fold_labels = []

    def n_splits(self) -> int:
        if self._actual_n_splits is not None:
            return self._actual_n_splits
        return self._n_folds

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        compositions: Optional[pd.DataFrame] = None,
    ) -> Generator[SplitIndices, None, None]:
        if compositions is None:
            raise ValueError(
                "CompositionGroupCVSplitter requires the `compositions` DataFrame."
            )
        if len(compositions) != len(X):
            raise ValueError("X and compositions must have the same number of rows.")

        comp_keys = [
            tuple(np.round(row.astype(float), 4))
            for row in compositions.to_numpy(dtype="float64", na_value=np.nan)
        ]
        groups: Dict[Tuple[float, ...], List[int]] = {}
        for idx, key in enumerate(comp_keys):
            groups.setdefault(key, []).append(idx)
        n_groups = len(groups)
        self.fold_labels = []
        self._actual_n_splits = min(self._n_folds, n_groups)
        if n_groups < 2:
            logger.warning(
                "CompositionGroupCV requires at least 2 unique composition groups; "
                "found %d",
                n_groups,
            )
            self._actual_n_splits = 0
            return

        rng = np.random.RandomState(self._seed)
        group_items = list(groups.items())
        rng.shuffle(group_items)
        fold_groups: List[List[Tuple[float, ...]]] = [
            [] for _ in range(self._actual_n_splits)
        ]
        fold_sizes = np.zeros(self._actual_n_splits, dtype=int)
        for key, indices in group_items:
            dest = int(np.argmin(fold_sizes))
            fold_groups[dest].append(key)
            fold_sizes[dest] += len(indices)

        all_idx = np.arange(len(compositions))
        for fold_idx, keys in enumerate(fold_groups):
            mask = np.array([key in keys for key in comp_keys], dtype=bool)
            if not mask.any():
                continue
            self.fold_labels.append(f"group{fold_idx}")
            yield all_idx[~mask], all_idx[mask]


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

SPLITTER_REGISTRY: Dict[str, type] = {
    "RandomCV": RandomCVSplitter,
    "CompositionBlock": CompositionBlockSplitter,
    "CompositionGroupCV": CompositionGroupCVSplitter,
    "ElementExclusion": ElementExclusionSplitter,
}


def get_splitter(name: str, **kwargs) -> BaseSplitter:
    """Instantiate a splitter by name.

    Parameters
    ----------
    name : str
        One of 'RandomCV', 'CompositionBlock', 'CompositionGroupCV',
        'ElementExclusion'.
    **kwargs
        Forwarded to the splitter constructor.
    """
    if name not in SPLITTER_REGISTRY:
        raise ValueError(
            f"Unknown splitter '{name}'. Available: {list(SPLITTER_REGISTRY.keys())}"
        )
    return SPLITTER_REGISTRY[name](**kwargs)
