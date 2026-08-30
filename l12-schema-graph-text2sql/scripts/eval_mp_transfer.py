#!/usr/bin/env python3
"""Evaluate the pipeline on a real Materials Project-flavored transfer schema."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))


from evaluation.metrics import common_column_exact_overlap, execution_accuracy_full, normalize_limit  # noqa: E402
from evaluation.metrics_strict import exact_result_set_match  # noqa: E402
from graph.graph_builder import build_table_graph  # noqa: E402
from graph.join_path_generator import get_allowed_join_list  # noqa: E402
from graph.schema_parser import get_columns, get_foreign_keys, get_tables  # noqa: E402
from llm.sql_generator import pipeline as sql_pipeline  # noqa: E402
from scripts.eval_independent import load_dataset, summarize  # noqa: E402
from scripts.provenance import build_provenance  # noqa: E402
from scripts.eval_db import open_eval_connection, run_model_sql  # noqa: E402

DB_NAME = "mp_transfer"
DATASET = PROJECT / "evaluation" / "mp_transfer_evaluation_dataset.jsonl"
RESULTS_DIR = PROJECT / "evaluation" / "expected_results_mp_transfer"
DEFAULT_OUTPUT = PROJECT / "evaluation" / "mp_transfer_eval_results.json"
DIFFICULTY_ORDER = ["easy", "medium", "hard", "very_hard"]


def _mp_conninfo() -> str:
    from scripts.eval_ablation import CONNINFO
    base = os.getenv("CONNINFO", CONNINFO)
    return base.replace(f"dbname={os.getenv('POSTGRES_DB', 'l12_materials')}", f"dbname={DB_NAME}")


def execute_sql(conn, sql: str) -> dict:
    return run_model_sql(conn, sql)



def compute_metrics(conn, sql: str, qid: str) -> dict[str, float]:
    """Compute row recall, precision, F1, and exact result-set match."""
    path = RESULTS_DIR / f"{qid}.json"
    with open(path) as f:
        data = json.load(f)
    expected_rows, expected_columns = data.get("rows", []), data.get("columns", [])
    if not sql:
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0, "exact_match": 0.0}
    exec_result = execute_sql(conn, sql)
    if not exec_result.get("success"):
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0, "exact_match": 0.0}
    metrics = execution_accuracy_full(
        exec_result["rows"], expected_rows,
        exec_result["columns"], expected_columns,
    )
    metrics["common_column_exact_overlap"] = common_column_exact_overlap(
        exec_result["rows"], expected_rows,
        exec_result["columns"], expected_columns,
    )
    metrics["exact_match"] = 1.0 if exact_result_set_match(
        exec_result["rows"], expected_rows,
        exec_result["columns"], expected_columns,
        ordered=bool(data.get("ordered")),
    ) else 0.0
    return metrics


def compute_accuracy(conn, sql: str, qid: str) -> float:
    """Backward-compatible alias: historical accuracy equals row recall."""
    return compute_metrics(conn, sql, qid)["recall"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    os.environ.setdefault(
        "SQL_PROMPT_TEMPLATE",
        str(PROJECT / "llm" / "prompt_templates" / "sql_generation_prompt_mp.md"),
    )
    os.environ.setdefault(
        "FEW_SHOT_STORE",
        str(PROJECT / "llm" / "few_shot_examples_mp.json"),
    )
    # Use the same LLM reranker setting as other transfer tests.
    os.environ.setdefault("USE_LLM_RERANKER", "1")

    model = os.getenv("LLM_MODEL", "gpt-5.5")
    print(f"Model: {model}")

    conn = open_eval_connection(_mp_conninfo(), suite="mp_transfer")
    print("Loading MP transfer schema...")
    tables = get_tables(conn)
    columns = {t: get_columns(conn, t) for t in tables}
    fks = get_foreign_keys(conn)
    table_graph = build_table_graph(fks)
    allowed_columns = [
        f"{t}.{c.column_name}" for t, cols in columns.items() for c in cols
    ]
    allowed_joins = get_allowed_join_list(table_graph)
    print(f"Tables: {len(tables)}, columns: {len(allowed_columns)}")

    queries = load_dataset(DATASET)
    print(f"MP transfer queries: {len(queries)}")

    def exec_fn(sql: str) -> dict:
        return execute_sql(conn, sql)

    results: list[dict] = []
    for i, q in enumerate(queries):
        qid = q["id"]
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
                "provenance": build_provenance(
                    DATASET,
                    gold_dir=PROJECT / "evaluation" / "gold_sql_mp",
                    prompt_path=PROJECT / "llm" / "prompt_templates"
                    / "sql_generation_prompt_mp.md"),
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
