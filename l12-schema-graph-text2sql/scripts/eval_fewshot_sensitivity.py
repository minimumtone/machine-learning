#!/usr/bin/env python3
"""Few-shot sensitivity analysis: measure the effect of varying k (number of
few-shot examples retrieved) on pipeline accuracy.

Runs the full pipeline with k = 1, 3, 5, 10, 15 over all 100 evaluation
queries. The default pipeline uses k=3.

Output: evaluation/fewshot_sensitivity_results.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))


from evaluation.metrics import execution_accuracy_full, normalize_limit  # noqa: E402
from scripts.provenance import (  # noqa: E402
    assert_resumable,
    build_provenance,
)
from graph.graph_builder import build_table_graph  # noqa: E402
from graph.join_path_generator import get_allowed_join_list  # noqa: E402
from graph.schema_parser import get_foreign_keys, get_tables, get_columns  # noqa: E402
from llm.sql_generator import pipeline as sql_pipeline  # noqa: E402
from scripts.eval_db import open_eval_connection, run_model_sql  # noqa: E402

EVAL_DIR = PROJECT / "evaluation"
RESULTS_DIR = EVAL_DIR / "expected_results"

CONNINFO = (
    f"host={os.getenv('POSTGRES_HOST', 'localhost')} "
    f"port={os.getenv('POSTGRES_PORT', '5432')} "
    f"dbname={os.getenv('POSTGRES_DB', 'l12_materials')} "
    f"user={os.getenv('POSTGRES_USER', 'l12_user')} "
    f"password={os.getenv('POSTGRES_PASSWORD', 'l12_password')}"
)

K_VALUES = [1, 3, 5, 10, 15]


def load_queries():
    queries = []
    with open(EVAL_DIR / "evaluation_dataset.jsonl") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    return queries


def load_expected(qid):
    path = RESULTS_DIR / f"{qid}.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        return data.get("rows", []), data.get("columns", [])
    return [], []


def execute_sql(conn, sql):
    return run_model_sql(conn, sql)



def compute_accuracy(conn, sql, qid):
    expected_rows, expected_columns = load_expected(qid)
    if not sql:
        return 0.0
    exec_result = execute_sql(conn, sql)
    if not exec_result.get("success"):
        return 0.0
    metrics = execution_accuracy_full(
        exec_result["rows"], expected_rows,
        exec_result["columns"], expected_columns,
    )
    return metrics.get("recall", 0.0)


def run_k_condition(
    conn, queries, k_value,
    allowed_joins, allowed_columns, table_graph, exec_fn,
):
    """Run the full pipeline with a specific few-shot k value."""
    import llm.few_shot_store as fs_mod
    import llm.sql_generator as sg_mod

    orig_retrieve_fs = fs_mod.retrieve_similar
    orig_retrieve_sg = sg_mod.retrieve_similar

    def patched_retrieve(query: str, top_k: int = 3) -> list[dict[str, Any]]:
        return orig_retrieve_fs(query, top_k=k_value)

    fs_mod.retrieve_similar = patched_retrieve
    sg_mod.retrieve_similar = patched_retrieve

    results = []
    try:
        for i, q in enumerate(queries):
            qid = q["id"]
            question = q["question"]
            difficulty = q["difficulty"]

            print(
                f"  [{i+1}/{len(queries)}] {qid} ({difficulty})...",
                end=" ", flush=True,
            )

            t0 = time.time()
            try:
                pipe_result = sql_pipeline(
                    user_query=question,
                    join_list=allowed_joins,
                    all_columns=allowed_columns,
                    skip_intent_check=True,
                    n_best=3,
                    execute_fn=exec_fn,
                    table_graph=table_graph,
                )
                elapsed = time.time() - t0
                sql = pipe_result.get("sql", "")
                if sql:
                    sql = normalize_limit(sql)
                acc = compute_accuracy(conn, sql, qid)
                print(f"acc={acc:.1%}  {elapsed:.1f}s")

                results.append({
                    "qid": qid,
                    "difficulty": difficulty,
                    "accuracy": acc,
                    "latency_s": round(elapsed, 1),
                    "sql": sql,
                })
            except Exception as e:
                elapsed = time.time() - t0
                print(f"ERROR: {type(e).__name__}: {e!s:.80s}  {elapsed:.1f}s")
                results.append({
                    "qid": qid,
                    "difficulty": difficulty,
                    "accuracy": 0.0,
                    "latency_s": round(elapsed, 1),
                    "sql": "",
                })
    finally:
        fs_mod.retrieve_similar = orig_retrieve_fs
        sg_mod.retrieve_similar = orig_retrieve_sg

    return results


def main():
    model = os.getenv("LLM_MODEL", "gpt-5.5")
    out_path = PROJECT / "evaluation" / "fewshot_sensitivity_results.json"

    # Allow specifying which k values to run
    k_start = os.getenv("FEWSHOT_K_START", "")
    k_values = K_VALUES
    if k_start:
        k_start_val = int(k_start)
        k_values = [k for k in K_VALUES if k >= k_start_val]

    print(f"Model: {model}")
    print(f"K values: {k_values}")
    print("Connecting to PostgreSQL...")
    conn = open_eval_connection(CONNINFO, suite="main")

    print("Loading schema...")
    tables = get_tables(conn)
    columns = {}
    for t in tables:
        columns[t] = get_columns(conn, t)
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

    print("Loading queries...")
    all_queries = load_queries()
    print(f"Total queries: {len(all_queries)}")

    def exec_fn(sql):
        return execute_sql(conn, sql)

    # Load existing results if resuming
    all_results: dict[str, Any] = {}
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
        assert_resumable(
            {**existing.get("provenance", {}),
             "model": existing.get("model", "")},
            {**build_provenance(EVAL_DIR / "evaluation_dataset.jsonl"),
             "model": model},
            force="--force-stale-resume" in sys.argv,
            what=out_path.name)
        all_results = existing.get("conditions", {})
        print(f"Loaded existing results: {list(all_results.keys())}")

    for k in k_values:
        cond_name = f"k={k}"
        if cond_name in all_results:
            print(f"\nSkipping {cond_name} (already exists)")
            continue

        print(f"\n{'='*70}")
        print(f"CONDITION: k={k} (few-shot examples)")
        print(f"{'='*70}")

        results = run_k_condition(
            conn, all_queries, k,
            allowed_joins, allowed_columns, table_graph, exec_fn,
        )

        # Compute summary
        total_acc = sum(r["accuracy"] for r in results) / len(results)
        by_diff: dict[str, list[float]] = {}
        for r in results:
            by_diff.setdefault(r["difficulty"], []).append(r["accuracy"])
        diff_summary = {
            d: sum(accs) / len(accs) for d, accs in by_diff.items()
        }
        avg_latency = sum(r["latency_s"] for r in results) / len(results)

        print(f"\n  Overall: {total_acc:.1%}")
        for d in ["easy", "medium", "hard", "very_hard"]:
            if d in diff_summary:
                print(f"  {d:12s}: {diff_summary[d]:.1%}")
        print(f"  Avg latency: {avg_latency:.1f}s")

        all_results[cond_name] = {
            "k": k,
            "overall": total_acc,
            "by_difficulty": diff_summary,
            "avg_latency": avg_latency,
            "results": results,
        }

        # Save after each k (incremental)
        with open(out_path, "w") as f:
            json.dump({
                "model": model,
                "provenance": build_provenance(EVAL_DIR / "evaluation_dataset.jsonl"),
                "n_queries": len(all_queries),
                "k_values": K_VALUES,
                "conditions": all_results,
            }, f, ensure_ascii=False, indent=2)
        print(f"  Saved to {out_path}")

    # Final summary
    print(f"\n{'='*70}")
    print("FEW-SHOT SENSITIVITY SUMMARY")
    print(f"{'='*70}")
    print(
        f"{'k':>5s} {'Overall':>8s} {'Easy':>8s} {'Medium':>8s} "
        f"{'Hard':>8s} {'VHard':>8s} {'Latency':>8s}"
    )
    print("-" * 60)

    for k in K_VALUES:
        cond_name = f"k={k}"
        if cond_name not in all_results:
            continue
        r = all_results[cond_name]
        diff = r["by_difficulty"]
        print(
            f"{k:>5d} {r['overall']:7.1%} {diff.get('easy',0):7.1%} "
            f"{diff.get('medium',0):7.1%} {diff.get('hard',0):7.1%} "
            f"{diff.get('very_hard',0):7.1%} {r['avg_latency']:6.1f}s"
        )

    conn.close()
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
