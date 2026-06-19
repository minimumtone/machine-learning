#!/usr/bin/env python3
"""Evaluate Japanese Cross-Encoder reranker on VH 20 queries.

Compares: ms-marco-MiniLM (current) vs hotchpotch/japanese-reranker-cross-encoder-xsmall-v1
Only affects Few-shot example reranking (the Cross-Encoder component).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import psycopg

from llm.sql_generator import pipeline as sql_pipeline
from graph.graph_builder import build_table_graph
from scripts.eval_ablation import (
    load_queries, get_tables, get_columns, get_foreign_keys,
    get_allowed_join_list, normalize_limit, compute_accuracy,
)

CONNINFO = (
    f"host={os.getenv('POSTGRES_HOST', 'localhost')} "
    f"port={os.getenv('POSTGRES_PORT', '5433')} "
    f"dbname={os.getenv('POSTGRES_DB', 'l12_materials')} "
    f"user={os.getenv('POSTGRES_USER', 'l12_user')} "
    f"password={os.getenv('POSTGRES_PASSWORD', 'l12_password')}"
)


def run_vh_with_model(conn, vh_queries, allowed_joins, allowed_columns, table_graph, exec_fn, ce_model_name):
    """Run VH queries with a specific Cross-Encoder model."""
    import llm.reranker as reranker_mod

    # Swap Cross-Encoder model
    orig_model = reranker_mod._CROSS_ENCODER_MODEL
    orig_ce = reranker_mod._cross_encoder
    reranker_mod._CROSS_ENCODER_MODEL = ce_model_name
    reranker_mod._cross_encoder = None  # Force reload

    results = []
    for i, q in enumerate(vh_queries):
        qid = q["id"]
        question = q["question"]
        print(f"  [{i+1}/{len(vh_queries)}] {qid}...", end=" ", flush=True)

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
            results.append({"qid": qid, "accuracy": acc, "latency_s": round(elapsed, 1)})
        except Exception as e:
            elapsed = time.time() - t0
            print(f"ERROR: {e!s:.60s}  {elapsed:.1f}s")
            results.append({"qid": qid, "accuracy": 0.0, "latency_s": round(elapsed, 1)})

    # Restore
    reranker_mod._CROSS_ENCODER_MODEL = orig_model
    reranker_mod._cross_encoder = orig_ce

    return results


def main():
    conn = psycopg.connect(CONNINFO)
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

    all_queries = load_queries()
    vh_queries = [q for q in all_queries if q["difficulty"] == "very_hard"]
    print(f"VH queries: {len(vh_queries)}")

    exec_fn = lambda sql: __import__('llm.sql_generator', fromlist=['execute_sql']).execute_sql(conn, sql)

    models = [
        ("ms-marco (current)", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        ("japanese-xsmall", "hotchpotch/japanese-reranker-cross-encoder-xsmall-v1"),
    ]

    all_results = {}
    for label, model_name in models:
        print(f"\n{'='*60}")
        print(f"Cross-Encoder: {label} ({model_name})")
        print(f"{'='*60}")
        results = run_vh_with_model(conn, vh_queries, allowed_joins, allowed_columns, table_graph, exec_fn, model_name)
        avg_acc = sum(r["accuracy"] for r in results) / len(results)
        avg_lat = sum(r["latency_s"] for r in results) / len(results)
        print(f"\n  Overall VH accuracy: {avg_acc:.1%}")
        print(f"  Avg latency: {avg_lat:.1f}s")
        all_results[label] = {
            "model": model_name,
            "overall_accuracy": avg_acc,
            "avg_latency": avg_lat,
            "results": results,
        }

    # Save
    out_path = PROJECT / "evaluation" / "jp_reranker_vh_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out_path}")

    # Comparison
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    for label, data in all_results.items():
        print(f"  {label}: {data['overall_accuracy']:.1%}  (lat: {data['avg_latency']:.1f}s)")

    # Per-query diff
    if len(all_results) == 2:
        labels = list(all_results.keys())
        r1 = {r["qid"]: r["accuracy"] for r in all_results[labels[0]]["results"]}
        r2 = {r["qid"]: r["accuracy"] for r in all_results[labels[1]]["results"]}
        print(f"\nPer-query diff ({labels[1]} - {labels[0]}):")
        for qid in sorted(r1.keys()):
            diff = r2.get(qid, 0) - r1.get(qid, 0)
            if abs(diff) > 0.001:
                print(f"  {qid}: {r1[qid]:.1%} -> {r2[qid]:.1%}  ({diff:+.1%})")

    conn.close()


if __name__ == "__main__":
    main()
