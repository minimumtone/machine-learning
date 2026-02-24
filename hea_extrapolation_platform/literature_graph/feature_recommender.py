"""
Feature Recommender from Literature Evidence (MVP)
文献由来 FeatureSet 生成支援

Collects key_features from top literature workflow matches,
cross-references with the platform's FeatureCatalog, and
proposes augmented FeatureSet variants.

Rules (MVP)
-----------
- Collect key_features from top-5 literature WFs by frequency.
- Registered features → "FeatureSet expansion candidates".
- Unregistered features → flagged but excluded from auto-generation.
- Max additions m <= 5 (prevent feature explosion).
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from hea_extrapolation_platform.features import FeatureCatalog, FeatureSetName
from hea_extrapolation_platform.literature_graph.schemas import Workflow
from hea_extrapolation_platform.literature_graph.search import (
    LiteratureSearchEngine,
    SearchResult,
    StructuredFilter,
)

logger = logging.getLogger(__name__)

# Maximum number of additional features from literature
MAX_ADDITIONAL_FEATURES = 5


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class FeatureRecommendation:
    """A recommended FeatureSet derived from literature evidence.

    Attributes
    ----------
    name : str
        Human-readable name for the proposed set (e.g. "FS_BASE+LIT_TOP3").
    base_features : List[str]
        Columns from the base FeatureSet.
    added_features : List[str]
        Columns added from literature evidence.
    all_features : List[str]
        base_features + added_features.
    evidence_workflows : List[SearchResult]
        Literature WFs that informed the recommendation.
    unregistered_features : List[str]
        Features found in literature but not in FeatureCatalog.
    feature_frequency : Dict[str, int]
        Feature → occurrence count in top literature WFs.
    """
    name: str
    base_features: List[str] = field(default_factory=list)
    added_features: List[str] = field(default_factory=list)
    all_features: List[str] = field(default_factory=list)
    evidence_workflows: List[SearchResult] = field(default_factory=list)
    unregistered_features: List[str] = field(default_factory=list)
    feature_frequency: Dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Feature Recommender
# ---------------------------------------------------------------------------

class LiteratureFeatureRecommender:
    """Generate FeatureSet candidates from literature search results.

    Parameters
    ----------
    search_engine : LiteratureSearchEngine
        Configured search engine with loaded index.
    max_additions : int
        Maximum number of features to add on top of the base set.
    """

    def __init__(
        self,
        search_engine: LiteratureSearchEngine,
        max_additions: int = MAX_ADDITIONAL_FEATURES,
    ) -> None:
        self._engine = search_engine
        self._max_add = max_additions
        self._all_registered = self._collect_registered_features()

    @staticmethod
    def _collect_registered_features() -> Set[str]:
        """Collect all feature names registered in FeatureCatalog."""
        all_cols: Set[str] = set()
        for fs_name in FeatureCatalog.list_sets():
            all_cols.update(FeatureCatalog.columns(fs_name))
        return all_cols

    def recommend(
        self,
        query: str,
        base_set: FeatureSetName = FeatureSetName.FS_BASE,
        structured_filter: Optional[StructuredFilter] = None,
        top_n_wf: int = 5,
    ) -> FeatureRecommendation:
        """Generate a FeatureSet recommendation from literature.

        Parameters
        ----------
        query : str
            Natural language or workflow_text query.
        base_set : FeatureSetName
            Base feature set to augment.
        structured_filter : StructuredFilter, optional
        top_n_wf : int
            Number of top workflows to consider.

        Returns
        -------
        FeatureRecommendation
        """
        results, feat_counts = self._engine.search_for_features(
            query, structured_filter=structured_filter, top_n=top_n_wf,
        )

        base_cols = set(FeatureCatalog.columns(base_set))
        registered = self._all_registered
        freq_dict = dict(feat_counts)

        # Separate registered vs unregistered
        candidates_registered: List[Tuple[str, int]] = []
        unregistered: List[str] = []

        for feat, count in feat_counts:
            if feat in registered and feat not in base_cols:
                candidates_registered.append((feat, count))
            elif feat not in registered:
                unregistered.append(feat)

        # Select top-m registered features not already in base
        added = [f for f, _ in candidates_registered[: self._max_add]]

        base_list = sorted(base_cols)
        all_features = base_list + added
        name = f"{base_set.value}+LIT_TOP{len(added)}"

        rec = FeatureRecommendation(
            name=name,
            base_features=base_list,
            added_features=added,
            all_features=all_features,
            evidence_workflows=results,
            unregistered_features=unregistered,
            feature_frequency=freq_dict,
        )

        logger.info(
            "Recommendation '%s': base=%d + added=%d features (unregistered=%d)",
            name, len(base_list), len(added), len(unregistered),
        )
        return rec

    def recommend_thermo_only(
        self,
        query: str,
        structured_filter: Optional[StructuredFilter] = None,
        top_n_wf: int = 5,
    ) -> FeatureRecommendation:
        """Generate a recommendation restricted to thermodynamic features.

        Filters added features to only include those in FS_THERMO.
        """
        full_rec = self.recommend(
            query, base_set=FeatureSetName.FS_BASE,
            structured_filter=structured_filter, top_n_wf=top_n_wf,
        )

        thermo_cols = set(FeatureCatalog.columns(FeatureSetName.FS_THERMO))
        base_cols = set(FeatureCatalog.columns(FeatureSetName.FS_BASE))
        thermo_only = thermo_cols - base_cols

        added_thermo = [f for f in full_rec.added_features if f in thermo_only]
        base_list = full_rec.base_features
        all_features = base_list + added_thermo

        return FeatureRecommendation(
            name=f"FS_BASE+LIT_THERMO{len(added_thermo)}",
            base_features=base_list,
            added_features=added_thermo,
            all_features=all_features,
            evidence_workflows=full_rec.evidence_workflows,
            unregistered_features=full_rec.unregistered_features,
            feature_frequency=full_rec.feature_frequency,
        )
