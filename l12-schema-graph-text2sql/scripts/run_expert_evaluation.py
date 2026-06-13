#!/usr/bin/env python3
"""Run evaluation pipeline on expert-designed 100 queries.
Produces comparison: 著者設計100件 vs 独立設計100件.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import psycopg

from evaluation.metrics import (
    execution_accuracy_full,
    normalize_limit as _normalize_limit,
    syntax_validity,
)
from graph.graph_builder import build_table_graph
from graph.join_path_generator import generate_joins_for_tables, get_allowed_join_list
from graph.schema_parser import get_foreign_keys, get_tables, get_columns
from llm.entity_extractor import extract_conditions
from llm.schema_linker import link_schema
from llm.sql_generator import generate_sql_via_llm

EVAL_DIR = PROJECT / "evaluation"
CONNINFO = (
    f"host={os.getenv('POSTGRES_HOST', 'localhost')} "
    f"port={os.getenv('POSTGRES_PORT', '5432')} "
    f"dbname={os.getenv('POSTGRES_DB', 'l12_materials')} "
    f"user={os.getenv('POSTGRES_USER', 'l12_user')} "
    f"password={os.getenv('POSTGRES_PASSWORD', 'l12_password')}"
)


def load_expert_dataset() -> list[dict]:
    queries = []
    with open(EVAL_DIR / "expert_evaluation_dataset.jsonl") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    return queries


def load_expected(qid: str):
    path = EVAL_DIR / "expected_results" / f"{qid}.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        return data.get("rows", []), data.get("columns", [])
    return [], []


def execute_sql(conn, sql: str) -> dict:
    try:
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout = '10s'")
            cur.execute(sql)
            columns = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchall()
        return {
            "success": True,
            "columns": columns,
            "rows": [list(r) for r in rows],
            "row_count": len(rows),
        }
    except Exception as e:
        conn.rollback()
        return {"success": False, "error": str(e), "rows": [], "row_count": 0}


def proposed_pipeline(query, table_graph, allowed_columns, allowed_joins, api_key, model):
    t0 = time.time()
    conditions = extract_conditions(query)
    linked = link_schema(conditions)
    required_tables = linked["required_tables"]

    # Provide ALL columns from required tables (not just linker subset)
    required_columns = [
        c for c in allowed_columns
        if c.split(".")[0] in required_tables
    ]

    join_clause = generate_joins_for_tables(table_graph, required_tables)
    join_list = []
    for line in join_clause.split("\n"):
        m = re.search(r"ON\s+(.+)", line, re.IGNORECASE)
        if m:
            join_list.append(m.group(1).strip())
    # Also include all allowed joins relevant to required tables
    for j in allowed_joins:
        # Fix B10: word-boundary match
        if any(re.search(r'\b' + re.escape(t) + r'\b', j) for t in required_tables) and j not in join_list:
            join_list.append(j)

    result = generate_sql_via_llm(
        user_query=query,
        allowed_tables=required_tables,
        allowed_columns=required_columns,
        allowed_joins=join_list if join_list else allowed_joins,
        model=model,
        api_key=api_key,
    )
    latency_ms = int((time.time() - t0) * 1000)
    result["latency_ms"] = latency_ms
    result["linked_tables"] = required_tables
    return result


def normalize_limit(sql: str) -> str:
    """Fix B2: Only add LIMIT if absent — delegate to metrics.normalize_limit."""
    return _normalize_limit(sql)


def main():
    api_key = os.getenv("OPENAI_API_KEY", "")
    model = os.getenv("LLM_MODEL", "gpt-5.5")
    has_llm = bool(api_key and api_key != "your_api_key_here")

    print(f"LLM: {has_llm}, model: {model}")
    conn = psycopg.connect(CONNINFO)

    # Get schema info (all 30 tables)
    tables = get_tables(conn)
    all_allowed_tables = list(tables)
    columns_map = {}
    for t in tables:
        columns_map[t] = get_columns(conn, t)
    fks = get_foreign_keys(conn)
    table_graph = build_table_graph(fks)

    # Add logical join: element.symbol = composition.element (no FK but valid)
    if not table_graph.has_edge("composition", "element"):
        table_graph.add_edge(
            "composition", "element",
            source_column="element",
            target_column="symbol",
        )

    allowed_columns = []
    for t, cols in columns_map.items():
        for c in cols:
            allowed_columns.append(f"{t}.{c.column_name}")
    allowed_joins = get_allowed_join_list(table_graph)

    print(f"Schema: {len(all_allowed_tables)} tables, {len(allowed_columns)} columns, {len(allowed_joins)} joins")

    queries = load_expert_dataset()
    print(f"Expert queries: {len(queries)}")

    results = []
    correct = 0
    exec_success = 0
    syntax_ok = 0
    total = len(queries)

    for i, q in enumerate(queries):
        qid = q["id"]
        question = q["question"]
        difficulty = q.get("difficulty", "unknown")
        expected_rows, expected_columns = load_expected(qid)

        print(f"\r  [{i+1}/{total}] {qid} ({difficulty})...", end="", flush=True)

        try:
            gen = proposed_pipeline(question, table_graph, allowed_columns, allowed_joins, api_key, model)
            sql = gen.get("sql", "")
            tokens = gen.get("tokens", 0)
            latency_ms = gen.get("latency_ms", 0)
        except Exception as e:
            sql = ""
            tokens = 0
            latency_ms = 0
            gen = {"error": str(e)}

        sql = normalize_limit(sql)

        # Execute
        if sql:
            exec_result = execute_sql(conn, sql)
        else:
            exec_result = {"success": False, "error": "empty SQL", "rows": [], "row_count": 0}

        # Fix B4: Type normalization is now handled inside execution_accuracy_full
        # via _normalize_value — no need for manual str() conversion
        actual_rows = exec_result.get("rows", [])

        # Metrics — Fix B1: use full metrics (precision/recall/F1)
        is_syntax = syntax_validity(sql) if sql else False
        is_exec = exec_result.get("success", False)
        result_columns = exec_result.get("columns", None)
        acc_full = execution_accuracy_full(
            actual_rows, expected_rows,
            result_columns=result_columns,
            expected_columns=expected_columns,
        )
        exec_acc = acc_full["recall"]
        exec_prec = acc_full["precision"]
        exec_f1 = acc_full["f1"]
        is_correct = exec_f1 >= 0.8  # Use F1, not recall-only

        if is_syntax:
            syntax_ok += 1
        if is_exec:
            exec_success += 1
        if is_correct:
            correct += 1

        result_entry = {
            "id": qid,
            "question": question,
            "difficulty": difficulty,
            "generated_sql": sql,
            "execution_success": is_exec,
            "execution_accuracy": exec_acc,
            "execution_precision": exec_prec,
            "execution_f1": exec_f1,
            "is_correct": is_correct,
            "expected_row_count": len(expected_rows),
            "actual_row_count": exec_result.get("row_count", 0),
            "tokens": tokens,
            "latency_ms": latency_ms,
            "error": gen.get("error") or exec_result.get("error"),
        }
        results.append(result_entry)

    print()

    # Summary by difficulty
    diff_stats = {}
    for r in results:
        d = r["difficulty"]
        if d not in diff_stats:
            diff_stats[d] = {"total": 0, "correct": 0}
        diff_stats[d]["total"] += 1
        if r["is_correct"]:
            diff_stats[d]["correct"] += 1

    # Mean execution accuracy (continuous, distinct from binary correct rate)
    mean_exec_acc = sum(r["execution_accuracy"] for r in results) / len(results) * 100

    print("\n" + "="*60)
    print("EXPERT EVALUATION RESULTS (Proposed Method)")
    print("="*60)
    print(f"Total queries:       {total}")
    print(f"Syntax valid:        {syntax_ok}/{total} ({100*syntax_ok/total:.1f}%)")
    print(f"Execution success:   {exec_success}/{total} ({100*exec_success/total:.1f}%)")
    print(f"Correct (acc>=0.8):  {correct}/{total} ({100*correct/total:.1f}%)")
    print()
    print("By difficulty:")
    for d in ["easy", "medium", "hard", "very_hard"]:
        if d in diff_stats:
            s = diff_stats[d]
            print(f"  {d:12s}: {s['correct']}/{s['total']} ({100*s['correct']/s['total']:.1f}%)")

    print()
    print("COMPARISON:")
    print("  著者設計100件:   (re-evaluate with same pipeline)")
    print(f"  独立設計100件:   {100*correct/total:.1f}% (F1-based)")

    # Save detailed results
    output_path = EVAL_DIR / "expert_evaluation_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "paper_ref": {
                "summary": "Table (tab:independent_eval) -- author vs independent evaluation",
                "by_difficulty": "Table (tab:independent_eval) difficulty rows",
                "results": "Supplementary (tab:sup_expert_detail), (tab:sup_expert_category)",
            },
            "summary": {
                "total": total,
                "syntax_valid": syntax_ok,
                "execution_success": exec_success,
                "correct": correct,
                "accuracy": round(100 * correct / total, 1),
                "by_difficulty": {d: {"correct": s["correct"], "total": s["total"],
                                      "accuracy": round(100*s["correct"]/s["total"], 1)}
                                  for d, s in diff_stats.items()},
            },
            "comparison": {
                "note": ("author_designed and expert_designed were evaluated with different "
                         "pipeline versions. Current author-designed representative run "
                         "execution_accuracy is 70.6% (3-run mean 70.9%). "
                         "Expert-designed mean execution_accuracy is "
                         f"{round(mean_exec_acc, 1)}%, and binary correct rate is "
                         f"{round(100*correct/total, 1)}%."),
                "expert_designed": {
                    "queries": total,
                    "binary_correct_rate": round(100 * correct / total, 1),
                    "mean_execution_accuracy": round(mean_exec_acc, 1),
                },
            },
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")
    conn.close()


if __name__ == "__main__":
    main()
