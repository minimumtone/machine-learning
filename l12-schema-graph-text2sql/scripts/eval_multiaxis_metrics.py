#!/usr/bin/env python3
"""Multi-axis evaluation: compute EM, SELECT-column precision, JOIN accuracy,
syntax validity, and execution validity for the full pipeline.

Runs the pipeline once on all 100 queries, captures the generated SQL,
and computes multiple metrics in a single pass.

Output: evaluation/multiaxis_results.json
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import psycopg  # noqa: E402

from evaluation.metrics import (  # noqa: E402
    execution_accuracy_full,
    hallucinated_column_rate,
    hallucinated_join_rate,
    hallucinated_table_rate,
    normalize_limit,
    syntax_validity,
)
from graph.graph_builder import build_table_graph  # noqa: E402
from graph.join_path_generator import get_allowed_join_list  # noqa: E402
from graph.schema_parser import get_foreign_keys, get_tables, get_columns  # noqa: E402
from llm.sql_generator import pipeline as sql_pipeline  # noqa: E402

EVAL_DIR = PROJECT / "evaluation"
RESULTS_DIR = EVAL_DIR / "expected_results"
GOLD_SQL_DIR = EVAL_DIR / "gold_sql"

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
    data: dict[str, Any] = {}
    if path.exists():
        with open(path) as f:
            data = json.load(f)
    # Load gold SQL from separate file
    gold_path = GOLD_SQL_DIR / f"{qid}.sql"
    if gold_path.exists():
        data["gold_sql"] = gold_path.read_text(encoding="utf-8").strip()
    return data


def execute_sql(conn, sql):
    try:
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout = '10s'")
            cur.execute(sql)
            columns = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchall()
        return {
            "success": True, "columns": columns,
            "rows": [list(r) for r in rows], "row_count": len(rows),
        }
    except Exception as e:
        conn.rollback()
        return {
            "success": False, "error": str(e),
            "rows": [], "row_count": 0, "columns": [],
        }


def _extract_select_columns(sql: str) -> list[str]:
    """Extract column names from SELECT clause."""
    if not sql:
        return []
    sql_upper = sql.strip().upper()
    # Handle CTE
    if sql_upper.startswith("WITH"):
        # Find the main SELECT after CTEs
        # Simple approach: find last SELECT
        parts = re.split(r'\bSELECT\b', sql, flags=re.IGNORECASE)
        if len(parts) > 1:
            select_part = parts[-1]
        else:
            return []
    else:
        select_part = re.sub(r'^SELECT\s+', '', sql.strip(), flags=re.IGNORECASE)

    # Find FROM position
    from_match = re.search(r'\bFROM\b', select_part, re.IGNORECASE)
    if from_match:
        select_part = select_part[:from_match.start()]

    # Split by comma, handling nested parentheses
    cols = []
    depth = 0
    current = ""
    for ch in select_part:
        if ch == '(':
            depth += 1
            current += ch
        elif ch == ')':
            depth -= 1
            current += ch
        elif ch == ',' and depth == 0:
            cols.append(current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        cols.append(current.strip())

    # Extract the alias or column name
    result = []
    for col in cols:
        col = col.strip()
        if not col:
            continue
        # Check for AS alias
        as_match = re.search(r'\bAS\s+(\w+)\s*$', col, re.IGNORECASE)
        if as_match:
            result.append(as_match.group(1).lower())
        else:
            # Use last word/identifier
            parts = col.split('.')
            last = parts[-1].strip().lower()
            # Remove any trailing parentheses/functions
            result.append(re.sub(r'[^a-z0-9_]', '', last))
    return result


def _extract_join_tables(sql: str) -> list[str]:
    """Extract tables mentioned in FROM/JOIN clauses."""
    if not sql:
        return []
    tables = []
    for m in re.finditer(r'(?:FROM|JOIN)\s+(\w+)', sql, re.IGNORECASE):
        t = m.group(1).lower()
        if t.upper() not in ("SELECT", "ON", "WHERE", "AND", "OR",
                              "LEFT", "RIGHT", "INNER", "OUTER",
                              "CROSS", "FULL"):
            tables.append(t)
    return tables


def _exact_match(gen_sql: str, gold_sql: str, conn) -> bool:
    """Check if generated and gold SQL produce identical result sets."""
    if not gen_sql or not gold_sql:
        return False
    gen_result = execute_sql(conn, gen_sql)
    gold_result = execute_sql(conn, gold_sql)
    if not gen_result["success"] or not gold_result["success"]:
        return False
    # Sort and compare
    gen_rows = sorted(
        [tuple(str(v) for v in r) for r in gen_result["rows"]],
    )
    gold_rows = sorted(
        [tuple(str(v) for v in r) for r in gold_result["rows"]],
    )
    return gen_rows == gold_rows


def _select_column_precision(gen_sql: str, gold_sql: str) -> float:
    """Precision of SELECT columns: what fraction of generated columns
    match gold columns."""
    gen_cols = _extract_select_columns(gen_sql)
    gold_cols = _extract_select_columns(gold_sql)
    if not gen_cols:
        return 0.0
    if not gold_cols:
        return 1.0  # No gold cols to compare
    gold_set = set(gold_cols)
    matched = sum(1 for c in gen_cols if c in gold_set)
    return matched / len(gen_cols)


def _join_match_rate(gen_sql: str, gold_sql: str) -> float:
    """What fraction of gold JOIN tables appear in generated SQL."""
    gen_tables = set(_extract_join_tables(gen_sql))
    gold_tables = set(_extract_join_tables(gold_sql))
    if not gold_tables:
        return 1.0
    matched = len(gen_tables & gold_tables)
    return matched / len(gold_tables)


def main():
    out_path = PROJECT / "evaluation" / "multiaxis_results.json"

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
        table_graph.add_edge(
            "composition", "element",
            source_column="element", target_column="symbol",
        )
    allowed_columns = [
        f"{t}.{c.column_name}" for t, cols in columns.items() for c in cols
    ]
    allowed_joins = get_allowed_join_list(table_graph)
    allowed_tables = list(columns.keys())

    print("Loading queries...")
    all_queries = load_queries()
    print(f"Total queries: {len(all_queries)}")

    def exec_fn(sql):
        return execute_sql(conn, sql)

    results = []
    for i, q in enumerate(all_queries):
        qid = q["id"]
        question = q["question"]
        difficulty = q["difficulty"]

        print(
            f"[{i+1}/{len(all_queries)}] {qid} ({difficulty})...",
            end=" ", flush=True,
        )

        expected = load_expected(qid)
        gold_sql = expected.get("gold_sql", "")
        expected_rows = expected.get("rows", [])
        expected_columns = expected.get("columns", [])

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
            gen_sql = pipe_result.get("sql", "")
            if gen_sql:
                gen_sql = normalize_limit(gen_sql)

            # Execute generated SQL
            gen_result = execute_sql(conn, gen_sql) if gen_sql else {
                "success": False, "rows": [], "columns": [],
            }

            # Metric 1: Recall (standard ablation metric)
            acc_metrics = execution_accuracy_full(
                gen_result.get("rows", []), expected_rows,
                gen_result.get("columns", []), expected_columns,
            )

            # Metric 2: Exact Match
            em = _exact_match(gen_sql, gold_sql, conn) if gold_sql else False

            # Metric 3: SELECT column precision
            sel_prec = _select_column_precision(gen_sql, gold_sql) if gold_sql else 0.0

            # Metric 4: JOIN match rate
            join_rate = _join_match_rate(gen_sql, gold_sql) if gold_sql else 0.0

            # Metric 5: Syntax validity
            syn_valid = syntax_validity(gen_sql)

            # Metric 6: Execution validity
            exec_valid = gen_result.get("success", False)

            # Metric 7: Hallucinated table/column rates
            gen_tables = _extract_join_tables(gen_sql)
            gen_cols_list = [
                f"{t}.{c}" for t, c in
                re.findall(r'(\w+)\.(\w+)', gen_sql)
            ] if gen_sql else []
            hall_table = hallucinated_table_rate(gen_tables, allowed_tables)
            hall_col = hallucinated_column_rate(gen_cols_list, allowed_columns, gen_sql)
            hall_join = hallucinated_join_rate(gen_sql, allowed_joins)

            print(
                f"recall={acc_metrics['recall']:.1%} "
                f"EM={'Y' if em else 'N'} "
                f"sel={sel_prec:.0%} "
                f"join={join_rate:.0%} "
                f"{elapsed:.1f}s"
            )

            results.append({
                "qid": qid,
                "difficulty": difficulty,
                "gen_sql": gen_sql,
                "recall": acc_metrics["recall"],
                "precision": acc_metrics["precision"],
                "f1": acc_metrics["f1"],
                "exact_match": em,
                "select_column_precision": sel_prec,
                "join_match_rate": join_rate,
                "syntax_valid": syn_valid,
                "execution_valid": exec_valid,
                "hallucinated_table_rate": hall_table,
                "hallucinated_column_rate": hall_col,
                "hallucinated_join_rate": hall_join,
                "latency_s": round(elapsed, 1),
            })

        except Exception as e:
            elapsed = time.time() - t0
            print(f"ERROR: {type(e).__name__}: {e!s:.80s}  {elapsed:.1f}s")
            results.append({
                "qid": qid,
                "difficulty": difficulty,
                "gen_sql": "",
                "recall": 0.0,
                "precision": 0.0,
                "f1": 0.0,
                "exact_match": False,
                "select_column_precision": 0.0,
                "join_match_rate": 0.0,
                "syntax_valid": False,
                "execution_valid": False,
                "hallucinated_table_rate": 0.0,
                "hallucinated_column_rate": 0.0,
                "hallucinated_join_rate": 0.0,
                "latency_s": round(elapsed, 1),
            })

    # Aggregate
    n = len(results)
    agg: dict[str, Any] = {
        "n_queries": n,
        "recall_mean": sum(r["recall"] for r in results) / n,
        "precision_mean": sum(r["precision"] for r in results) / n,
        "f1_mean": sum(r["f1"] for r in results) / n,
        "exact_match_rate": sum(1 for r in results if r["exact_match"]) / n,
        "select_column_precision_mean": sum(r["select_column_precision"] for r in results) / n,
        "join_match_rate_mean": sum(r["join_match_rate"] for r in results) / n,
        "syntax_validity_rate": sum(1 for r in results if r["syntax_valid"]) / n,
        "execution_validity_rate": sum(1 for r in results if r["execution_valid"]) / n,
        "hallucinated_table_rate_mean": sum(r["hallucinated_table_rate"] for r in results) / n,
        "hallucinated_column_rate_mean": sum(r["hallucinated_column_rate"] for r in results) / n,
        "hallucinated_join_rate_mean": sum(r["hallucinated_join_rate"] for r in results) / n,
    }

    # By difficulty
    by_diff: dict[str, dict[str, Any]] = {}
    for diff in ["easy", "medium", "hard", "very_hard"]:
        diff_results = [r for r in results if r["difficulty"] == diff]
        if not diff_results:
            continue
        nd = len(diff_results)
        by_diff[diff] = {
            "n": nd,
            "recall": sum(r["recall"] for r in diff_results) / nd,
            "precision": sum(r["precision"] for r in diff_results) / nd,
            "f1": sum(r["f1"] for r in diff_results) / nd,
            "exact_match": sum(1 for r in diff_results if r["exact_match"]) / nd,
            "select_col_prec": sum(r["select_column_precision"] for r in diff_results) / nd,
            "join_match": sum(r["join_match_rate"] for r in diff_results) / nd,
        }

    output = {
        "model": os.getenv("LLM_MODEL", "gpt-5.5"),
        "aggregate": agg,
        "by_difficulty": by_diff,
        "results": results,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # Print summary
    print(f"\n{'='*70}")
    print("MULTI-AXIS EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"Recall (EX):         {agg['recall_mean']:.1%}")
    print(f"Precision:           {agg['precision_mean']:.1%}")
    print(f"F1:                  {agg['f1_mean']:.1%}")
    print(f"Exact Match:         {agg['exact_match_rate']:.1%}")
    print(f"SELECT col prec:     {agg['select_column_precision_mean']:.1%}")
    print(f"JOIN match rate:     {agg['join_match_rate_mean']:.1%}")
    print(f"Syntax validity:     {agg['syntax_validity_rate']:.1%}")
    print(f"Execution validity:  {agg['execution_validity_rate']:.1%}")
    print(f"Halluc. table rate:  {agg['hallucinated_table_rate_mean']:.1%}")
    print(f"Halluc. column rate: {agg['hallucinated_column_rate_mean']:.1%}")
    print(f"Halluc. JOIN rate:   {agg['hallucinated_join_rate_mean']:.1%}")

    print(f"\n{'Difficulty':>12s} {'Recall':>8s} {'Prec':>8s} {'F1':>8s} {'EM':>8s} {'SelCol':>8s} {'JoinM':>8s}")
    print("-" * 62)
    for diff in ["easy", "medium", "hard", "very_hard"]:
        if diff in by_diff:
            d = by_diff[diff]
            print(
                f"{diff:>12s} {d['recall']:7.1%} {d['precision']:7.1%} "
                f"{d['f1']:7.1%} {d['exact_match']:7.1%} "
                f"{d['select_col_prec']:7.1%} {d['join_match']:7.1%}"
            )

    conn.close()
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
