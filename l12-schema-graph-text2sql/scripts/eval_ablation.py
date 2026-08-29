#!/usr/bin/env python3
"""Ablation study: measure contribution of each pipeline component.

Runs 7 conditions (including the full model) over all evaluation queries.
Each condition disables one component to measure its contribution.

Conditions:
  1. full         — Full pipeline (n_best=3 + hybrid reranker)
  2. no_graph     — No Steiner tree (5-table fallback joins)
  3. no_reranker  — n_best=3 but reranker disabled
  4. no_nbest     — n_best=1 (single candidate, no reranker)
  5. no_dict      — No materials domain dictionary
  6. no_fewshot   — No few-shot examples
  7. no_guard     — No SQLGuard validation in scoring
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))


from evaluation.metrics import common_column_exact_overlap, execution_accuracy_full, normalize_limit  # noqa: E402
from evaluation.metrics_strict import exact_result_set_match  # noqa: E402
from scripts.provenance import (  # noqa: E402
    assert_resumable,
    build_provenance,
)
from graph.graph_builder import build_table_graph  # noqa: E402
from graph.join_path_generator import get_allowed_join_list  # noqa: E402
from graph.schema_parser import get_foreign_keys, get_tables, get_columns  # noqa: E402
from llm.sql_generator import pipeline as sql_pipeline  # noqa: E402

EVAL_DIR = PROJECT / "evaluation"
RESULTS_DIR = EVAL_DIR / "expected_results"

from scripts.db_conninfo import CONNINFO  # noqa: E402
from scripts.eval_db import open_eval_connection, run_model_sql  # noqa: E402


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



def compute_metrics(conn, sql, qid):
    """Return recall, precision, F1 and exact result-set match."""
    expected_rows, expected_columns = load_expected(qid)
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
    ) else 0.0
    return metrics


def compute_accuracy(conn, sql, qid):
    """Backward-compatible alias for the historical primary metric (recall)."""
    return compute_metrics(conn, sql, qid)["recall"]


def run_condition(conn, queries, condition, allowed_joins, allowed_columns, table_graph, exec_fn):
    """Run a single ablation condition over all queries."""
    results = []
    
    for i, q in enumerate(queries):
        qid = q["id"]
        question = q["question"]
        difficulty = q["difficulty"]
        
        print(f"  [{i+1}/{len(queries)}] {qid} ({difficulty})...", end=" ", flush=True)
        
        t0 = time.time()
        
        try:
            pipe_result: dict[str, Any] = {}
            if condition == "full":
                pipe_result = sql_pipeline(
                    user_query=question,
                    join_list=allowed_joins,
                    all_columns=allowed_columns,
                    skip_intent_check=True,
                    n_best=3,
                    execute_fn=exec_fn,
                    table_graph=table_graph,
                )
            
            elif condition == "no_graph":
                pipe_result = sql_pipeline(
                    user_query=question,
                    join_list=None,
                    all_columns=None,
                    skip_intent_check=True,
                    n_best=3,
                    execute_fn=exec_fn,
                    table_graph=None,
                )
            
            elif condition == "no_reranker":
                with patch.dict(os.environ, {"OPENAI_API_KEY_RERANK": "", "RERANK_MODEL": "__disabled__"}):
                    import llm.reranker as reranker_mod
                    orig_rerank_sql = reranker_mod.rerank_sql_candidates
                    orig_rerank_schema = reranker_mod.rerank_schema_tables
                    reranker_mod.rerank_sql_candidates = lambda q, c, **kw: c
                    reranker_mod.rerank_schema_tables = lambda q, t, **kw: t
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
                    finally:
                        reranker_mod.rerank_sql_candidates = orig_rerank_sql
                        reranker_mod.rerank_schema_tables = orig_rerank_schema
            
            elif condition == "no_nbest":
                pipe_result = sql_pipeline(
                    user_query=question,
                    join_list=allowed_joins,
                    all_columns=allowed_columns,
                    skip_intent_check=True,
                    n_best=1,
                    execute_fn=exec_fn,
                    table_graph=table_graph,
                )
            
            elif condition == "no_dict":
                import llm.schema_linker as sl_mod
                orig_ctm = sl_mod.CONDITION_TABLE_MAP
                orig_ccm = sl_mod.CONDITION_COLUMN_MAP
                orig_map = sl_mod.map_conditions
                sl_mod.CONDITION_TABLE_MAP = {}
                sl_mod.CONDITION_COLUMN_MAP = {}
                sl_mod.map_conditions = lambda c: {}
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
                finally:
                    sl_mod.CONDITION_TABLE_MAP = orig_ctm
                    sl_mod.CONDITION_COLUMN_MAP = orig_ccm
                    sl_mod.map_conditions = orig_map
            
            elif condition == "no_fewshot":
                import llm.few_shot_store as fs_mod
                orig_retrieve = fs_mod.retrieve_similar
                fs_mod.retrieve_similar = lambda *a, **kw: []
                import llm.sql_generator as sg_mod
                orig_sg_retrieve = sg_mod.retrieve_similar
                sg_mod.retrieve_similar = lambda *a, **kw: []
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
                finally:
                    fs_mod.retrieve_similar = orig_retrieve
                    sg_mod.retrieve_similar = orig_sg_retrieve
            
            elif condition == "no_guard":
                import safety.sql_validator as sv_mod
                orig_validate = sv_mod.validate_sql
                sv_mod.validate_sql = lambda sql, **kw: {"valid": True, "errors": []}
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
                finally:
                    sv_mod.validate_sql = orig_validate
            
            elapsed = time.time() - t0
            sql = pipe_result.get("sql", "")
            if sql:
                sql = normalize_limit(sql)
            
            metrics = compute_metrics(conn, sql, qid)
            acc = metrics["recall"]
            print(f"recall={acc:.1%} exact={metrics['exact_match']:.0%}  {elapsed:.1f}s")
            
            results.append({
                "qid": qid,
                "difficulty": difficulty,
                "accuracy": acc,  # historical field; equals row-level recall
                "recall": metrics["recall"],
                "precision": metrics["precision"],
                "f1": metrics["f1"],
                "exact_match": metrics["exact_match"],
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
                "recall": 0.0,
                "precision": 0.0,
                "f1": 0.0,
                "exact_match": 0.0,
                "latency_s": round(elapsed, 1),
                "sql": "",
            })
    
    return results


def main():
    model = os.getenv("LLM_MODEL", "gpt-5.5")
    
    conditions = ["full", "no_graph", "no_reranker", "no_nbest", "no_dict", "no_fewshot", "no_guard"]
    
    # Allow resuming from a specific condition
    start_from = os.getenv("ABLATION_START", "full")
    if start_from in conditions:
        start_idx = conditions.index(start_from)
        conditions = conditions[start_idx:]
    
    print(f"Model: {model}")
    print(f"Conditions: {conditions}")
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
        table_graph.add_edge("composition", "element",
                             source_column="element", target_column="symbol")
    allowed_columns = [f"{t}.{c.column_name}" for t, cols in columns.items() for c in cols]
    allowed_joins = get_allowed_join_list(table_graph)
    
    print("Loading queries...")
    all_queries = load_queries()
    print(f"Total queries: {len(all_queries)}")
    
    def exec_fn(sql):
        return execute_sql(conn, sql)
    
    # Load existing results if resuming
    out_path = Path(os.getenv("ABLATION_RESULTS",
                              str(PROJECT / "evaluation" / "ablation_results.json")))
    all_results = {}
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
    
    for cond in conditions:
        if cond in all_results:
            print(f"CONDITION: {cond} already completed, skipping")
            continue
        print(f"\n{'='*70}")
        print(f"CONDITION: {cond}")
        print(f"{'='*70}")
        
        results = run_condition(conn, all_queries, cond, allowed_joins, allowed_columns, table_graph, exec_fn)
        
        # Compute summary
        total_acc = sum(r["accuracy"] for r in results) / len(results)
        by_diff = {}
        for r in results:
            by_diff.setdefault(r["difficulty"], []).append(r["accuracy"])
        diff_summary = {d: sum(accs)/len(accs) for d, accs in by_diff.items()}
        avg_latency = sum(r["latency_s"] for r in results) / len(results)
        
        print(f"\n  Overall: {total_acc:.1%}")
        for d in ["easy", "medium", "hard", "very_hard"]:
            if d in diff_summary:
                print(f"  {d:12s}: {diff_summary[d]:.1%}")
        print(f"  Avg latency: {avg_latency:.1f}s")
        
        all_results[cond] = {
            "overall": total_acc,
            "by_difficulty": diff_summary,
            "avg_latency": avg_latency,
            "results": results,
        }
        
        # Save after each condition (incremental save)
        with open(out_path, "w") as f:
            json.dump({
                "model": model,
                "provenance": build_provenance(EVAL_DIR / "evaluation_dataset.jsonl"),
                "n_queries": len(all_queries),
                "conditions": all_results,
            }, f, ensure_ascii=False, indent=2)
        print(f"  Saved to {out_path}")
    
    # Final summary table
    print(f"\n{'='*70}")
    print("ABLATION SUMMARY")
    print(f"{'='*70}")
    print(f"{'Condition':15s} {'Overall':>8s} {'Easy':>8s} {'Medium':>8s} {'Hard':>8s} {'VHard':>8s} {'Latency':>8s}")
    print("-" * 70)
    
    full_acc = all_results.get("full", {}).get("overall", 0)
    for cond in ["full", "no_graph", "no_reranker", "no_nbest", "no_dict", "no_fewshot", "no_guard"]:
        if cond not in all_results:
            continue
        r = all_results[cond]
        overall = r["overall"]
        delta = overall - full_acc
        diff = r["by_difficulty"]
        print(f"{cond:15s} {overall:7.1%} {diff.get('easy',0):7.1%} {diff.get('medium',0):7.1%} "
              f"{diff.get('hard',0):7.1%} {diff.get('very_hard',0):7.1%} {r['avg_latency']:6.1f}s "
              f"({'baseline' if cond == 'full' else f'{delta:+.1%}'})")
    
    conn.close()
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
