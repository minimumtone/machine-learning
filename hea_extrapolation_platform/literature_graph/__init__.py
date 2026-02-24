"""
Literature Graph Module (MVP)
文献メタデータグラフモジュール

Provides a lightweight, JSON-based knowledge graph of published ML workflows
for materials science. This MVP uses JSONL files + FAISS for vector search,
with a clear abstraction layer for future migration to Neo4j / Milvus.

Sub-modules
-----------
schemas          Data models (Paper, Workflow, Edge) as frozen dataclasses
workflow_text    Canonical text generation for embedding
seed_data        Built-in seed dataset (20-50 HEA papers)
vector_index     VectorIndex abstraction + FAISS backend
search           Two-stage search (embedding + structured filter)
feature_recommender  FeatureSet generation from literature evidence
"""

from hea_extrapolation_platform.literature_graph.schemas import (  # noqa: F401
    Paper,
    Workflow,
    Edge,
    load_jsonl,
)
from hea_extrapolation_platform.literature_graph.search import (  # noqa: F401
    LiteratureSearchEngine,
    SearchResult,
    StructuredFilter,
)
from hea_extrapolation_platform.literature_graph.vector_index import (  # noqa: F401
    VectorIndex,
)
from hea_extrapolation_platform.literature_graph.feature_recommender import (  # noqa: F401
    LiteratureFeatureRecommender,
    FeatureRecommendation,
)
from hea_extrapolation_platform.literature_graph.workflow_text import (  # noqa: F401
    generate_workflow_text,
)
