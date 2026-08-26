#!/usr/bin/env python3
"""A/B test: measure hybrid reranker impact on SQL generation accuracy.

Runs queries through the full pipeline with n_best=3 (reranker active)
and compares against the existing proposed_result.csv baseline.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import psycopg  # noqa: E402

from evaluation.metrics import execution_accuracy_full, normalize_limit  # noqa: E402
from graph.graph_builder import build_table_graph  # noqa: E402
from graph.join_path_generator import get_allowed_join_list  # noqa: E402
from graph.schema_parser import get_foreign_keys, get_tables, get_columns  # noqa: E402
from llm.sql_generator import pipeline as sql_pipeline  # noqa: E402

EVAL_DIR = PROJECT / "evaluation"
RESULTS_DIR = EVAL_DIR / "expected_results"

CONNINFO = (
    f"host={os.getenv('POSTGRES_HOST', 'localhost')} "
    f"port={os.getenv('POSTGRES_PORT', '5432')} "
    f"dbname={os.getenv('POSTGRES_DB', 'l12_materials')} "
    f"user={os.getenv('POSTGRES_USER', 'l12_user')} "
    f"password={os.getenv('POSTGRES_PASSWORD', 'l12_password')}"
)


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
    try:
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout = '10s'")
            cur.execute(sql)
            columns = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchall()
        return {"success": True, "columns": columns, "rows": [list(r) for r in rows], "row_count": len(rows)}
    except Exception as e:
        conn.rollback()
        return {"success": False, "error": str(e), "rows": [], "row_count": 0, "columns": []}


def run_pipeline_no_reranker(question, allowed_joins, allowed_columns,
                             table_graph, exec_fn):
    """Run the pipeline with the hybrid reranker disabled (baseline arm)."""
    import llm.reranker as reranker_mod
    orig_rerank_sql = reranker_mod.rerank_sql_candidates
    orig_rerank_schema = reranker_mod.rerank_schema_tables
    reranker_mod.rerank_sql_candidates = lambda q, c, **kw: c
    reranker_mod.rerank_schema_tables = lambda q, t, **kw: t
    try:
        return sql_pipeline(
            user_query=question,
            join_list=allowed_joins,
            all_columns=allowed_columns,
            skip_intent_check=True,
            n_best=3,
            execute_fn=exec_fn,
            table_graph=table_graph,
        )
    finally:
        reranker_mod.rerank_sql_candidates = orig_rerank_sql
        reranker_mod.rerank_schema_tables = orig_rerank_schema


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


def main():
    model = os.getenv("LLM_MODEL", "gpt-5.5")
    sample_size = int(os.getenv("EVAL_SAMPLE", "20"))

    print(f"Model: {model}")
    print(f"Sample size: {sample_size}")
    print("Connecting to PostgreSQL...")
    conn = psycopg.connect(CONNINFO)

    print("Loading schema...")
    tables = get_tables(conn)
    columns = {}
    for t in tables:
        columns[t] = get_columns(conn, t)
    fks = get_foreign_keys(conn)
    table_graph = build_table_graph(fks)
    if not table_graph.has_edge("composition", "element"):
        table_graph.add_edge("composition", "element",
                             source_column="element", target_column="symbol")
    allowed_columns = [f"{t}.{c.column_name}" for t, cols in columns.items() for c in cols]
    allowed_joins = get_allowed_join_list(table_graph)

    print("Loading queries...")
    all_queries = load_queries()

    # Sample: balanced across difficulty
    by_diff = {}
    for q in all_queries:
        by_diff.setdefault(q["difficulty"], []).append(q)

    sample = []
    per_diff = max(1, sample_size // len(by_diff))
    for diff in ["easy", "medium", "hard", "very_hard"]:
        qs = by_diff.get(diff, [])
        sample.extend(qs[:per_diff])
    sample = sample[:sample_size]

    print(f"Selected {len(sample)} queries")
    print()

    def exec_fn(sql):
        return execute_sql(conn, sql)

    results = []
    for i, q in enumerate(sample):
        qid = q["id"]
        question = q["question"]
        difficulty = q["difficulty"]

        print(f"[{i+1}/{len(sample)}] {qid} ({difficulty})...", end=" ", flush=True)

        t0 = time.time()
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
        n_best_info = pipe_result.get("n_best_info", {})
        reranked = n_best_info.get("reranked", False)

        tb0 = time.time()
        base_result = run_pipeline_no_reranker(
            question, allowed_joins, allowed_columns, table_graph, exec_fn)
        base_elapsed = time.time() - tb0
        base_sql = base_result.get("sql", "")
        if base_sql:
            base_sql = normalize_limit(base_sql)
        base_acc = compute_accuracy(conn, base_sql, qid)

        delta_str = ""
        if base_acc is not None:
            delta = acc - base_acc
            delta_str = f"  delta={delta:+.1%}"

        base_str = f"{base_acc:.1%}" if base_acc is not None else "N/A"
        print(f"acc={acc:.1%}  base={base_str}{delta_str}  reranked={reranked}  {elapsed:.1f}s")

        results.append({
            "qid": qid,
            "difficulty": difficulty,
            "accuracy_reranker": acc,
            "accuracy_baseline": base_acc,
            "reranked": reranked,
            "latency_s": round(elapsed, 1),
            "latency_baseline_s": round(base_elapsed, 1),
            "sql": sql,
            "sql_baseline": base_sql,
        })

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY — Hybrid Reranker (n_best=3)")
    print("=" * 70)

    total_reranker = sum(r["accuracy_reranker"] for r in results) / len(results)
    has_base = [r for r in results if r["accuracy_baseline"] is not None]
    total_baseline = sum(r["accuracy_baseline"] for r in has_base) / len(has_base) if has_base else 0

    print(f"\nOverall ({len(results)} queries):")
    print(f"  Reranker:  {total_reranker:.1%}")
    if has_base:
        print(f"  Baseline:  {total_baseline:.1%}")
        print(f"  Delta:     {total_reranker - total_baseline:+.1%}")

    for diff in ["easy", "medium", "hard", "very_hard"]:
        dr = [r for r in results if r["difficulty"] == diff]
        if dr:
            avg_r = sum(r["accuracy_reranker"] for r in dr) / len(dr)
            db = [r for r in dr if r["accuracy_baseline"] is not None]
            avg_b = sum(r["accuracy_baseline"] for r in db) / len(db) if db else 0
            delta = avg_r - avg_b if db else 0
            print(f"  {diff:12s}: reranker={avg_r:.1%}  baseline={avg_b:.1%}  delta={delta:+.1%}")

    reranked_count = sum(1 for r in results if r["reranked"])
    print(f"\nReranked: {reranked_count}/{len(results)}")
    avg_latency = sum(r["latency_s"] for r in results) / len(results)
    print(f"Avg latency: {avg_latency:.1f}s")

    # Improvement/regression breakdown
    improved = [r for r in results if r["accuracy_baseline"] is not None and r["accuracy_reranker"] > r["accuracy_baseline"]]
    regressed = [r for r in results if r["accuracy_baseline"] is not None and r["accuracy_reranker"] < r["accuracy_baseline"]]
    same = [r for r in results if r["accuracy_baseline"] is not None and r["accuracy_reranker"] == r["accuracy_baseline"]]
    print(f"\nImproved: {len(improved)}  Same: {len(same)}  Regressed: {len(regressed)}")
    for r in improved:
        print(f"  +{r['qid']}: {r['accuracy_baseline']:.1%} → {r['accuracy_reranker']:.1%}")
    for r in regressed:
        print(f"  -{r['qid']}: {r['accuracy_baseline']:.1%} → {r['accuracy_reranker']:.1%}")

    # Save
    out_path = PROJECT / "evaluation" / "reranker_eval_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": model,
            "sample_size": len(results),
            "overall_reranker": total_reranker,
            "overall_baseline": total_baseline,
            "delta": total_reranker - total_baseline if has_base else None,
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_path}")
    conn.close()


if __name__ == "__main__":
    main()
