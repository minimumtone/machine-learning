"""
Two-Stage Search Engine for Literature Graph (MVP)
2段構え検索エンジン

Stage 1: Embedding-based retrieval (top-K=30 candidates)
Stage 2: Structured filtering on graph attributes + re-ranking

This ensures results are both semantically relevant AND structurally
consistent with the user's experimental context.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from hea_extrapolation_platform.literature_graph.schemas import (
    Paper,
    Workflow,
    load_jsonl,
)
from hea_extrapolation_platform.literature_graph.vector_index import (
    VectorIndex,
    embed_workflow_texts,
)
from hea_extrapolation_platform.literature_graph.workflow_text import (
    generate_workflow_text,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Search result container
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """A single search result with score and metadata."""
    workflow: Workflow
    paper: Optional[Paper]
    embedding_distance: float
    final_score: float
    matched_filters: Dict[str, bool] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Structured filters
# ---------------------------------------------------------------------------

@dataclass
class StructuredFilter:
    """Structured filter criteria for stage-2 filtering.

    All fields are optional; only non-None fields are applied.
    """
    materials_domain: Optional[str] = None
    task: Optional[str] = None
    inputs: Optional[str] = None
    split_policy: Optional[str] = None
    max_data_size: Optional[int] = None
    min_data_size: Optional[int] = None
    model_family: Optional[str] = None
    year_min: Optional[int] = None
    year_max: Optional[int] = None

    def matches(self, workflow: Workflow, paper: Optional[Paper] = None) -> Tuple[bool, Dict[str, bool]]:
        """Check if a workflow matches the filter criteria.

        Returns (passes, detail_dict) where detail_dict shows per-field match.
        """
        details: Dict[str, bool] = {}
        passes = True

        if self.materials_domain is not None and paper is not None:
            match = paper.materials_domain.lower() == self.materials_domain.lower()
            details["materials_domain"] = match
            if not match:
                passes = False

        if self.task is not None and paper is not None:
            match = paper.task.lower() == self.task.lower()
            details["task"] = match
            if not match:
                passes = False

        if self.inputs is not None:
            match = workflow.inputs.lower() == self.inputs.lower()
            details["inputs"] = match
            if not match:
                passes = False

        if self.split_policy is not None:
            match = workflow.split_policy.lower() == self.split_policy.lower()
            details["split_policy"] = match
            if not match:
                passes = False

        if self.max_data_size is not None:
            match = workflow.data_size_n <= self.max_data_size
            details["max_data_size"] = match
            if not match:
                passes = False

        if self.min_data_size is not None:
            match = workflow.data_size_n >= self.min_data_size
            details["min_data_size"] = match
            if not match:
                passes = False

        if self.model_family is not None:
            match = workflow.model_family.lower() == self.model_family.lower()
            details["model_family"] = match
            if not match:
                passes = False

        if self.year_min is not None and paper is not None:
            match = paper.year >= self.year_min
            details["year_min"] = match
            if not match:
                passes = False

        if self.year_max is not None and paper is not None:
            match = paper.year <= self.year_max
            details["year_max"] = match
            if not match:
                passes = False

        return passes, details


# ---------------------------------------------------------------------------
# Search Engine
# ---------------------------------------------------------------------------

class LiteratureSearchEngine:
    """Two-stage search: embedding retrieval + structured filtering.

    Parameters
    ----------
    index : VectorIndex
        Pre-built vector index of workflow texts.
    workflows : list of Workflow
        All known workflows.
    papers : list of Paper
        All known papers (for structured filtering).
    embedding_top_k : int
        Number of candidates from stage 1 (default 30).
    final_top_n : int
        Number of results to return after stage 2 (default 10).
    """

    def __init__(
        self,
        index: VectorIndex,
        workflows: List[Workflow],
        papers: List[Paper],
        embedding_top_k: int = 30,
        final_top_n: int = 10,
    ) -> None:
        self._index = index
        self._wf_map: Dict[str, Workflow] = {w.workflow_id: w for w in workflows}
        self._paper_map: Dict[str, Paper] = {p.paper_id: p for p in papers}
        self._embedding_top_k = embedding_top_k
        self._final_top_n = final_top_n

    def search(
        self,
        query: str,
        structured_filter: Optional[StructuredFilter] = None,
        top_n: Optional[int] = None,
    ) -> List[SearchResult]:
        """Execute two-stage search.

        Parameters
        ----------
        query : str
            Natural-language query or workflow_text-like string.
        structured_filter : StructuredFilter, optional
            Stage-2 filter criteria.
        top_n : int, optional
            Override final_top_n.

        Returns
        -------
        list of SearchResult, sorted by final_score (ascending = best).
        """
        top_n = top_n or self._final_top_n

        # Stage 1: Embedding retrieval
        query_vec = embed_workflow_texts([query])
        candidates = self._index.search(query_vec[0], top_k=self._embedding_top_k)

        if not candidates:
            logger.warning("No candidates found for query: %s", query[:80])
            return []

        logger.info("Stage 1: %d embedding candidates", len(candidates))

        # Stage 2: Structured filtering + scoring
        results: List[SearchResult] = []
        for wf_id, dist in candidates:
            wf = self._wf_map.get(wf_id)
            if wf is None:
                continue
            paper = self._paper_map.get(wf.paper_id)

            if structured_filter is not None:
                passes, details = structured_filter.matches(wf, paper)
                if not passes:
                    continue
            else:
                details = {}

            results.append(SearchResult(
                workflow=wf,
                paper=paper,
                embedding_distance=dist,
                final_score=dist,  # MVP: use raw distance as score
                matched_filters=details,
            ))

        # Sort by final score (lower distance = better)
        results.sort(key=lambda r: r.final_score)
        results = results[:top_n]

        logger.info(
            "Stage 2: %d results after filtering (returning top %d)",
            len(results), top_n,
        )
        return results

    def search_for_features(
        self,
        query: str,
        structured_filter: Optional[StructuredFilter] = None,
        top_n: int = 5,
    ) -> Tuple[List[SearchResult], List[Tuple[str, int]]]:
        """Search and aggregate key_features from top results.

        Returns
        -------
        results : list of SearchResult
        feature_counts : list of (feature_name, count), sorted by frequency desc.
        """
        results = self.search(query, structured_filter=structured_filter, top_n=top_n)

        # Aggregate key_features
        feature_freq: Dict[str, int] = {}
        for r in results:
            for feat in r.workflow.key_features:
                feat_lower = feat.strip()
                feature_freq[feat_lower] = feature_freq.get(feat_lower, 0) + 1

        feature_counts = sorted(feature_freq.items(), key=lambda x: -x[1])
        return results, feature_counts
