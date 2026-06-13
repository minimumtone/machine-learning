"""Run the proposed schema-graph-assisted Text-to-SQL method."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from llm.sql_generator import build_schema_context_from_db, pipeline


def run_proposed(
    dataset_path: Path | None = None,
    output_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Run the proposed method over the evaluation dataset."""
    if dataset_path is None:
        dataset_path = Path(__file__).parent / "evaluation_dataset.jsonl"
    if output_path is None:
        output_path = dataset_path.parent / "proposed_result.json"

    # Try to use full 30-table schema context from live DB
    join_list = None
    all_columns = None
    try:
        import psycopg
        import os
        conn = psycopg.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            dbname=os.getenv("POSTGRES_DB", "l12_materials"),
            user=os.getenv("POSTGRES_USER", "l12_user"),
            password=os.getenv("POSTGRES_PASSWORD", "l12_password"),
        )
        ctx = build_schema_context_from_db(conn)
        join_list = ctx["join_list"]
        all_columns = ctx["all_columns"]
        conn.close()
        print(f"Using full schema: {ctx['n_tables']} tables, {ctx['n_columns']} columns")
    except Exception:
        print("DB unavailable; using 5-table fallback")

    results: list[dict[str, Any]] = []
    with dataset_path.open() as f:
        queries = [json.loads(line) for line in f if line.strip()]

    for q in queries:
        t0 = time.time()
        result = pipeline(
            q["question"],
            join_list=join_list,
            all_columns=all_columns,
        )
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
