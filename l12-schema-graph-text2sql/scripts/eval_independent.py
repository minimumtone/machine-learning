#!/usr/bin/env python3
"""Evaluate the full pipeline on the independent (expert-designed) query set.

Runs the current full pipeline over evaluation/expert_evaluation_dataset.jsonl
and saves per-query accuracy plus difficulty-level summaries. This provides an
external-validity check against queries not authored by the pipeline designers.

Usage:
    python scripts/eval_independent.py [--dataset PATH] [--output PATH]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import psycopg  # noqa: E402

from evaluation.metrics import normalize_limit  # noqa: E402
from graph.graph_builder import build_table_graph  # noqa: E402
from graph.join_path_generator import get_allowed_join_list  # noqa: E402
from graph.schema_parser import get_columns, get_foreign_keys, get_tables  # noqa: E402
from llm.sql_generator import pipeline as sql_pipeline  # noqa: E402
from scripts.eval_ablation import CONNINFO, compute_metrics, execute_sql  # noqa: E402
from scripts.provenance import build_provenance  # noqa: E402

DEFAULT_DATASET = PROJECT / "evaluation" / "expert_evaluation_dataset.jsonl"
DEFAULT_OUTPUT = PROJECT / "evaluation" / "independent_eval_results.json"
DIFFICULTY_ORDER = ["easy", "medium", "hard", "very_hard"]


def load_dataset(path: Path) -> list[dict]:
    """Load a JSONL evaluation dataset into a list of query dicts."""
    queries = []
    with open(path) as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    return queries


def summarize(results: list[dict]) -> dict:
    """Summarize row recall plus stricter result-set metrics when present."""
    if not results:
        return {"overall": 0.0, "by_difficulty": {}, "avg_latency": 0.0}
    overall = sum(r.get("recall", r.get("accuracy", 0.0)) for r in results) / len(results)
    by_diff: dict[str, list[float]] = {}
    for r in results:
        by_diff.setdefault(r["difficulty"], []).append(r.get("recall", r.get("accuracy", 0.0)))
    out = {
        "overall": overall,  # historical name; row-level recall
        "overall_recall": overall,
        "by_difficulty": {d: sum(a) / len(a) for d, a in by_diff.items()},
        "avg_latency": sum(r["latency_s"] for r in results) / len(results),
    }
    for key in ("precision", "f1", "exact_match"):
        if all(key in r for r in results):
            out[f"overall_{key}"] = sum(r[key] for r in results) / len(results)
    return out


def main() -> None:
    """Run the full pipeline over the independent query set and save results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    model = os.getenv("LLM_MODEL", "gpt-5.5")
    print(f"Model: {model}")
    print("Connecting to PostgreSQL...")
    conn = psycopg.connect(CONNINFO)

    print("Loading schema...")
    tables = get_tables(conn)
    columns = {t: get_columns(conn, t) for t in tables}
    fks = get_foreign_keys(conn)
    table_graph = build_table_graph(fks)
    if not table_graph.has_edge("composition", "element"):
        table_graph.add_edge(
            "composition", "element",
            source_column="element", target_column="symbol",
        )
    allowed_columns = [
        f"{t}.{c.column_name}" for t, cols in columns.items() for c in cols
    ]
    allowed_joins = get_allowed_join_list(table_graph)

    queries = load_dataset(args.dataset)
    print(f"Independent queries: {len(queries)}")

    def exec_fn(sql: str) -> dict:
        return execute_sql(conn, sql)

    results: list[dict] = []
    existing_ids: set[str] = set()
    if args.output.exists():
        with open(args.output) as f:
            prev = json.load(f)
        results = prev.get("results", [])
        existing_ids = {r["qid"] for r in results}
        print(f"Resuming: {len(existing_ids)} queries already done")

    for i, q in enumerate(queries):
        qid = q["id"]
        if qid in existing_ids:
            continue
        print(f"[{i + 1}/{len(queries)}] {qid} ({q['difficulty']})...",
              end=" ", flush=True)
        t0 = time.time()
        try:
            pipe_result = sql_pipeline(
                user_query=q["question"],
                join_list=allowed_joins,
                all_columns=allowed_columns,
                skip_intent_check=True,
                n_best=3,
                execute_fn=exec_fn,
                table_graph=table_graph,
            )
            sql = pipe_result.get("sql", "")
            if sql:
                sql = normalize_limit(sql)
            metrics = compute_metrics(conn, sql, qid)
        except Exception as e:
            print(f"ERROR: {type(e).__name__}: {e!s:.80s}")
            metrics, sql = {"recall": 0.0, "precision": 0.0, "f1": 0.0, "exact_match": 0.0}, ""
        elapsed = time.time() - t0
        print(f"recall={metrics['recall']:.1%} exact={metrics['exact_match']:.0%}  {elapsed:.1f}s")
        results.append({
            "qid": qid,
            "difficulty": q["difficulty"],
            "accuracy": metrics["recall"],  # historical field
            **metrics,
            "latency_s": round(elapsed, 1),
            "sql": sql,
        })
        with open(args.output, "w") as f:
            json.dump({
                "model": model,
                "provenance": build_provenance(args.dataset),
                "dataset": str(args.dataset.name),
                "n_queries": len(queries),
                "summary": summarize(results),
                "results": results,
            }, f, ensure_ascii=False, indent=2)

    summary = summarize(results)
    print(f"\nOverall: {summary['overall']:.1%}")
    for d in DIFFICULTY_ORDER:
        if d in summary["by_difficulty"]:
            print(f"  {d:12s}: {summary['by_difficulty'][d]:.1%}")
    print(f"Saved to {args.output}")
    conn.close()


if __name__ == "__main__":
    main()
