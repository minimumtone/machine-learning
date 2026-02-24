"""
Data Models for Literature Graph (MVP)
文献グラフデータモデル

Frozen dataclasses that define the contract for Paper, Workflow, and Edge.
These schemas are designed to be forward-compatible:
  - MVP: JSONL files
  - Future: Neo4j nodes / relationships

All fields are explicitly typed for robustness.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ModelFamily(str, Enum):
    """Broad model family categories."""
    LINEAR = "linear"
    TREE = "tree"
    NN = "nn"
    GP = "gp"
    ENSEMBLE = "ensemble"
    OTHER = "other"


class InputScope(str, Enum):
    """What information the workflow ingests."""
    COMPOSITION_ONLY = "composition_only"
    COMPOSITION_PROCESS = "composition+process"
    COMPOSITION_CALPHAD = "composition+calphad"
    COMPOSITION_MICROSTRUCTURE = "composition+microstructure"
    FULL = "full"


class SplitPolicy(str, Enum):
    """Data-split strategy reported in the paper."""
    RANDOM = "random"
    BLOCKED = "blocked"
    LEAVE_ELEMENT_OUT = "leave_element_out"
    TIME = "time"
    OTHER = "other"


class EdgeType(str, Enum):
    """Relationship types in the literature graph."""
    REPORTS = "REPORTS"             # Paper -[:REPORTS]-> Workflow
    USES_FEATURE = "USES_FEATURE"  # Workflow -[:USES_FEATURE]-> Feature
    EVALUATED_BY = "EVALUATED_BY"  # Workflow -[:EVALUATED_BY]-> Metric


# ---------------------------------------------------------------------------
# Paper
# ---------------------------------------------------------------------------

@dataclass
class Paper:
    """Minimal bibliographic metadata for one publication.

    Fields
    ------
    paper_id : str
        Unique ID (DOI preferred, e.g. ``10.1016/j.actamat.2021.116800``).
    title : str
    year : int
    venue : str
        Journal or conference name.
    materials_domain : str
        e.g. ``HEA``, ``steel``, ``ceramic`` (free text, lowercase recommended).
    task : str
        Prediction target, e.g. ``yield_strength``, ``hardness``.
    notes : str
        A short self-authored summary (200-500 chars). NOT a copy of the abstract.
    """
    paper_id: str
    title: str
    year: int
    venue: str
    materials_domain: str = "HEA"
    task: str = "yield_strength"
    notes: str = ""
    doi_verified: bool = False
    """Whether ``paper_id`` (DOI) has been verified against CrossRef/DOI.org.

    Seed data is auto-generated and DOIs are *plausible but unverified*.
    Set to ``True`` only after programmatic or manual verification.
    """

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Paper":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_jsonl_line(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_jsonl_line(cls, line: str) -> "Paper":
        return cls.from_dict(json.loads(line))


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------

@dataclass
class Workflow:
    """Minimal structured metadata for one ML workflow reported in a paper.

    Fields
    ------
    workflow_id : str
        ``{paper_id}__wf{N}`` (e.g. ``10.1016/j.actamat.2021.116800__wf1``).
    paper_id : str
        Back-reference to the parent Paper.
    model_family : str
        One of ModelFamily values.
    model_name : str
        Specific model, e.g. ``XGBoost``, ``Ridge``, ``RandomForest``.
    inputs : str
        One of InputScope values.
    split_policy : str
        One of SplitPolicy values.
    data_size_n : int
        Number of data points used.
    metrics : Dict[str, float]
        e.g. ``{"rmse": 120.0, "r2": 0.62}``.
    key_features : List[str]
        Feature names used / found important. Prefer Feature Catalog IDs
        when available; free-text otherwise.
    notes : str
        Short self-authored summary of the workflow specifics.
    """
    workflow_id: str
    paper_id: str
    model_family: str = "tree"
    model_name: str = ""
    inputs: str = "composition_only"
    split_policy: str = "random"
    data_size_n: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)
    key_features: List[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Workflow":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_jsonl_line(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_jsonl_line(cls, line: str) -> "Workflow":
        return cls.from_dict(json.loads(line))


# ---------------------------------------------------------------------------
# Edge
# ---------------------------------------------------------------------------

@dataclass
class Edge:
    """A directed relationship in the literature graph.

    Fields
    ------
    source_id : str
        ID of the source node.
    target_id : str
        ID of the target node.
    edge_type : str
        One of EdgeType values.
    properties : dict
        Optional edge attributes.
    """
    source_id: str
    target_id: str
    edge_type: str
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Edge":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_jsonl_line(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_jsonl_line(cls, line: str) -> "Edge":
        return cls.from_dict(json.loads(line))


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path, cls: type) -> list:
    """Load a JSONL file into a list of dataclass instances."""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(cls.from_jsonl_line(line))
            except Exception as exc:
                raise ValueError(
                    f"Failed to parse {cls.__name__} at {path}:{lineno}: {exc}"
                ) from exc
    return items


def save_jsonl(items: Sequence, path: Path) -> None:
    """Save a list of dataclass instances to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(item.to_jsonl_line() + "\n")
