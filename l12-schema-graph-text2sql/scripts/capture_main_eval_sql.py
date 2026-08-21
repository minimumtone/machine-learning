#!/usr/bin/env python3
"""Capture per-query generated SQL for the main 100-query evaluation set.

This script runs the full pipeline (n_best=3, hybrid reranker, SQLGuard) over
`evaluation/evaluation_dataset.jsonl` and writes:
  - evaluation/generated_sql/main/<qid>.sql
  - evaluation/main_eval_with_sql.json

It does NOT overwrite `ablation_results.json` or `ablation_run_*.json`.  It is
distributed as a re-run option for reviewers who want to audit the generated SQL.
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

from evaluation.metrics import execution_accuracy_full, normalize_limit  # noqa: E402
from graph.graph_builder import build_table_graph  # noqa: E402
from graph.join_path_generator import get_allowed_join_list  # noqa: E402
from graph.schema_parser import get_foreign_keys, get_tables, get_columns  # noqa: E402
from llm.sql_generator import pipeline as sql_pipeline  # noqa: E402

EVAL_DIR = PROJECT / "evaluation"
RESULTS_DIR = EVAL_DIR / "expected_results"
OUT_DIR = EVAL_DIR / "generated_sql" / "main"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONNINFO = (
    f"host={os.getenv('POSTGRES_HOST', 'localhost')} "
    f"port={os.getenv('POSTGRES_PORT', '5432')} "
    f"dbname={os.getenv('POSTGRES_DB', 'l12_materials')} "
    f"user={os.getenv('POSTGRES_USER', 'l12_user')} "
    f"password={os.getenv('POSTGRES_PASSWORD', 'l12_password')}"
)


def load_queries(limit: int | None = None):
    queries = []
    with open(EVAL_DIR / "evaluation_dataset.jsonl") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    if limit is not None:
        queries = queries[:limit]
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Run only first N queries for testing")
    parser.add_argument("--out", type=Path, default=EVAL_DIR / "main_eval_with_sql.json", help="Aggregate JSON output")
    args = parser.parse_args()

    model = os.getenv("LLM_MODEL", "gpt-5.5")
    print(f"Model: {model}")
    print("Connecting to PostgreSQL...")

    conn = psycopg.connect(CONNINFO)
    tables = get_tables(conn)
    columns = {t: get_columns(conn, t) for t in tables}
    fks = get_foreign_keys(conn)
    table_graph = build_table_graph(fks)
    if not table_graph.has_edge("composition", "element"):
        table_graph.add_edge("composition", "element", source_column="element", target_column="symbol")
    allowed_columns = [f"{t}.{c.column_name}" for t, cols in columns.items() for c in cols]
    allowed_joins = get_allowed_join_list(table_graph)

    print("Loading queries...")
    all_queries = load_queries(args.limit)
    print(f"Total queries: {len(all_queries)}")

    def exec_fn(sql):
        return execute_sql(conn, sql)

    results = []
    for i, q in enumerate(all_queries):
        qid = q["id"]
        question = q["question"]
        difficulty = q["difficulty"]
        print(f"[{i+1}/{len(all_queries)}] {qid} ({difficulty})...", end=" ", flush=True)

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

            sql_path = OUT_DIR / f"{qid}.sql"
            sql_path.write_text(sql.rstrip() + "\n", encoding="utf-8")

            results.append({
                "qid": qid,
                "difficulty": difficulty,
                "question": question,
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
                "question": question,
                "accuracy": 0.0,
                "latency_s": round(elapsed, 1),
                "sql": "",
                "error": f"{type(e).__name__}: {e!s}",
            })

    conn.close()

    # Summary
    total_acc = sum(r["accuracy"] for r in results) / len(results)
    by_diff = {}
    for r in results:
        by_diff.setdefault(r["difficulty"], []).append(r["accuracy"])
    diff_summary = {d: sum(accs) / len(accs) for d, accs in by_diff.items()}
    avg_latency = sum(r["latency_s"] for r in results) / len(results)

    out_data = {
        "model": model,
        "n_queries": len(results),
        "overall": total_acc,
        "by_difficulty": diff_summary,
        "avg_latency": avg_latency,
        "results": results,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
    print(f"\nOverall: {total_acc:.1%}")
    for d in ["easy", "medium", "hard", "very_hard"]:
        if d in diff_summary:
            print(f"  {d:12s}: {diff_summary[d]:.1%}")
    print(f"Results written to {args.out}")
    print(f"SQL logs written to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
