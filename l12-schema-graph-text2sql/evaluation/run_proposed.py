"""Run the proposed schema-graph-assisted Text-to-SQL method."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from llm.sql_generator import pipeline


def run_proposed(
    dataset_path: Path | None = None,
    output_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Run the proposed method over the evaluation dataset."""
    if dataset_path is None:
        dataset_path = Path(__file__).parent / "evaluation_dataset.jsonl"
    if output_path is None:
        output_path = dataset_path.parent / "proposed_result.json"

    results: list[dict[str, Any]] = []
    with dataset_path.open() as f:
        queries = [json.loads(line) for line in f if line.strip()]

    for q in queries:
        t0 = time.time()
        result = pipeline(q["question"])
        total_ms = int((time.time() - t0) * 1000)

        results.append({
            "id": q["id"],
            "method": "proposed",
            "question": q["question"],
            "difficulty": q["difficulty"],
            "sql": result["sql"],
            "conditions": result["conditions"],
            "linked_tables": result["linked_schema"]["required_tables"],
            "linked_columns": result["linked_schema"]["required_columns"],
            "model": result.get("model", ""),
            "tokens": result.get("tokens", 0),
            "latency_ms": total_ms,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Proposed method: {len(results)} queries processed -> {output_path}")
    return results


if __name__ == "__main__":
    run_proposed()
