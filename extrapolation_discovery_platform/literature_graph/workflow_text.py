"""
Workflow Text Generator for Literature Graph
ワークフローテキスト生成モジュール

Generates a canonical text representation of a Workflow for embedding.
The template is fixed to ensure forward-compatibility when migrating
from manual entry to automatic extraction.

Template
--------
::

    [Task] {task} regression
    [Domain] {domain}
    [Inputs] {inputs}
    [Model] {model_name}
    [Split] {split_policy}
    [Data] N={data_size_n}
    [Metrics] {metrics_str}
    [KeyFeatures] {features_csv}
    [Notes] {notes}
"""

from __future__ import annotations

from typing import Optional

from extrapolation_discovery_platform.literature_graph.schemas import Paper, Workflow


def generate_workflow_text(
    workflow: Workflow,
    paper: Optional[Paper] = None,
) -> str:
    """Generate canonical text for embedding from a Workflow (+ optional Paper).

    Parameters
    ----------
    workflow : Workflow
        Structured workflow metadata.
    paper : Paper, optional
        Parent paper for domain / task fallback.

    Returns
    -------
    str
        Multi-line canonical text.
    """
    domain = "HEA"
    task = "yield_strength"
    if paper is not None:
        domain = paper.materials_domain or domain
        task = paper.task or task

    # Metrics string
    if workflow.metrics:
        metrics_parts = []
        for k, v in sorted(workflow.metrics.items()):
            if isinstance(v, float):
                metrics_parts.append(f"{k.upper()}={v:.4g}")
            else:
                metrics_parts.append(f"{k.upper()}={v}")
        metrics_str = ", ".join(metrics_parts)
    else:
        metrics_str = "N/A"

    # Key features CSV
    features_csv = ", ".join(workflow.key_features) if workflow.key_features else "N/A"

    lines = [
        f"[Task] {task} regression",
        f"[Domain] {domain}",
        f"[Inputs] {workflow.inputs}",
        f"[Model] {workflow.model_name or workflow.model_family}",
        f"[Split] {workflow.split_policy}",
        f"[Data] N={workflow.data_size_n}",
        f"[Metrics] {metrics_str}",
        f"[KeyFeatures] {features_csv}",
        f"[Notes] {workflow.notes}" if workflow.notes else "[Notes] N/A",
    ]
    return "\n".join(lines)
