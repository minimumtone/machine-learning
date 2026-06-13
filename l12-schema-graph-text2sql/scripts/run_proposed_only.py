#!/usr/bin/env python3
"""Re-run ONLY the proposed method, keeping baseline results from existing CSV.

Outputs: evaluation/proposed_result.csv (overwritten with new results).
"""
from __future__ import annotations

import csv
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
    hallucinated_column_rate,
    hallucinated_join_rate,
    hallucinated_table_rate,
    normalize_limit,
    syntax_validity,
)
from graph.graph_builder import build_table_graph
from graph.join_path_generator import generate_joins_for_tables, get_allowed_join_list
from graph.schema_parser import get_foreign_keys, get_tables, get_columns
from llm.entity_extractor import extract_conditions
from llm.repair_loop import execution_repair_loop, detect_superset
from llm.schema_linker import link_schema
from llm.sql_generator import generate_sql_via_llm
from llm.intent_classifier import classify_query_type
from llm.sql_sanity_checker import check_sql_sanity
from safety.sql_validator import (
    extract_columns_from_sql,
    extract_tables_from_sql,
)

EVAL_DIR = PROJECT / "evaluation"
RESULTS_DIR = EVAL_DIR / "expected_results"
GOLD_DIR = EVAL_DIR / "gold_sql"

CONNINFO = (
    f"host={os.getenv('POSTGRES_HOST', 'localhost')} "
    f"port={os.getenv('POSTGRES_PORT', '5432')} "
    f"dbname={os.getenv('POSTGRES_DB', 'l12_materials')} "
    f"user={os.getenv('POSTGRES_USER', 'l12_user')} "
    f"password={os.getenv('POSTGRES_PASSWORD', 'l12_password')}"
)

ALLOWED_TABLES = [
    "material_entry", "composition", "structure",
    "calculation", "calculated_property", "phase_stability",
    "prototype_definition",
    "alloy_system", "application_domain", "band_structure",
    "defect_type", "density_of_states", "elastic_tensor",
    "element", "element_property", "experimental_measurement",
    "grain_boundary", "literature_reference", "magnetic_property",
    "material_alloy_system", "material_application", "material_defect",
    "material_reference", "material_synthesis", "measured_property",
    "phase_diagram_entry", "space_group", "surface_energy",
    "synthesis_method", "thermal_property",
]


def load_evaluation_dataset():
    queries = []
    with open(EVAL_DIR / "evaluation_dataset.jsonl") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    return queries


def load_expected_results(qid):
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
        return {
            "success": True,
            "columns": columns,
            "rows": [list(r) for r in rows],
            "row_count": len(rows),
        }
    except Exception as e:
        conn.rollback()
        return {"success": False, "error": str(e), "rows": [], "row_count": 0}


def get_schema_info(conn):
    tables = get_tables(conn)
    columns = {}
    for t in tables:
        columns[t] = get_columns(conn, t)
    fks = get_foreign_keys(conn)
    table_graph = build_table_graph(fks)

    if not table_graph.has_edge("composition", "element"):
        table_graph.add_edge(
            "composition", "element",
            source_column="element",
            target_column="symbol",
        )

    allowed_columns = []
    for t, cols in columns.items():
        for c in cols:
            allowed_columns.append(f"{t}.{c.column_name}")

    allowed_joins = get_allowed_join_list(table_graph)
    return table_graph, allowed_columns, allowed_joins


def proposed_schema_graph(query, table_graph, allowed_columns, allowed_joins, api_key, model):
    t0 = time.time()
    conditions = extract_conditions(query)
    coverage = conditions.get("_coverage", {})
    linked = link_schema(conditions)
    required_tables = linked["required_tables"]

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
    for j in allowed_joins:
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
    result["linked_columns"] = required_columns
    result["coverage"] = coverage
    result["effective_joins"] = join_list if join_list else allowed_joins
    return result


def main():
    api_key = os.getenv("OPENAI_API_KEY", "")
    model = os.getenv("LLM_MODEL", "gpt-5.5")

    print(f"Model: {model}")
    print("Connecting to PostgreSQL...")
    conn = psycopg.connect(CONNINFO)

    print("Loading schema info...")
    table_graph, allowed_columns, allowed_joins = get_schema_info(conn)

    print("Loading evaluation dataset...")
    queries = load_evaluation_dataset()
    print(f"  {len(queries)} queries loaded")

    results = []
    CSV_FIELDS = [
        "query_id", "question", "difficulty", "hop_count", "method", "sql",
        "syntax_valid", "execution_valid",
        "execution_accuracy", "execution_precision", "execution_recall", "execution_f1",
        "raw_execution_accuracy", "raw_execution_precision", "raw_execution_f1",
        "hallucinated_table_rate", "hallucinated_column_rate", "hallucinated_join_rate",
        "token_usage", "latency_ms", "sanity_regen", "repair_attempts", "repair_tokens",
    ]

    output_path = EVAL_DIR / "proposed_result.csv"

    for i, q in enumerate(queries):
        qid = q["id"]
        question = q["question"]
        difficulty = q["difficulty"]
        hop_count = q.get("hop_count", 1)
        expected_rows, expected_columns = load_expected_results(qid)

        print(f"\r  [{i+1}/{len(queries)}] {qid} ({difficulty})...", end="", flush=True)

        try:
            gen = proposed_schema_graph(
                question, table_graph, allowed_columns, allowed_joins, api_key, model
            )
            sql = gen.get("sql", "")
            tokens = gen.get("tokens", 0)
            latency_ms = gen.get("latency_ms", 0)
            sanity_regen = 0
            repair_attempts = 0
            repair_tokens = 0

            if sql:
                sql = normalize_limit(sql)

            # Rule-based sanity check: detect structural mismatches
            if sql:
                qt_info = classify_query_type(question)
                sanity = check_sql_sanity(question, sql, qt_info["query_type"])
                if not sanity["sane"]:
                    # Regenerate with corrective hints
                    corrective = sanity["correction_hints"]
                    gen2 = proposed_schema_graph(
                        question + "\n\n[修正指示] " + corrective,
                        table_graph, allowed_columns, allowed_joins, api_key, model
                    )
                    if gen2.get("sql", ""):
                        sql = normalize_limit(gen2["sql"])
                        tokens += gen2.get("tokens", 0)
                        sanity_regen += 1

            exec_result = execute_sql(conn, sql) if sql else {"success": False, "rows": [], "row_count": 0}

            # Repair loop
            if sql:
                query_conditions = extract_conditions(question)
                needs_repair = (
                    not exec_result.get("success", False)
                    or exec_result.get("row_count", 0) == 0
                )
                if not needs_repair and exec_result.get("success"):
                    superset_check = detect_superset(
                        sql, exec_result.get("row_count", 0), query_conditions,
                    )
                    if superset_check["is_superset"]:
                        needs_repair = True

                if needs_repair:
                    coverage = gen.get("coverage", {})
                    effective_joins = gen.get("effective_joins", allowed_joins)
                    linked_tables = gen.get("linked_tables", ALLOWED_TABLES)
                    linked_cols = gen.get("linked_columns", allowed_columns)

                    def _exec_fn(s):
                        return execute_sql(conn, normalize_limit(s))

                    repair_result = execution_repair_loop(
                        original_sql=sql,
                        question=question,
                        execute_fn=_exec_fn,
                        allowed_tables=linked_tables,
                        allowed_columns=linked_cols,
                        allowed_joins=effective_joins,
                        coverage=coverage,
                        conditions=query_conditions,
                        required_tables=linked_tables,
                        max_retries=3,
                        model=model,
                        api_key=api_key,
                    )
                    if repair_result.get("repaired", False):
                        sql = normalize_limit(repair_result["sql"])
                        exec_result = repair_result["exec_result"]
                    elif repair_result["exec_result"].get("success", False):
                        sql = repair_result["sql"]
                        exec_result = repair_result["exec_result"]
                    repair_attempts += len(repair_result.get("attempts", []))
                    repair_tokens = repair_result.get("repair_tokens", 0)
                    tokens += repair_tokens
                    latency_ms += repair_result.get("repair_latency_ms", 0)

            # Metrics
            gen_tables = extract_tables_from_sql(sql)
            gen_columns = extract_columns_from_sql(sql)
            is_syntax_valid = syntax_validity(sql)
            is_exec_valid = exec_result.get("success", False)
            result_columns = exec_result.get("columns", None)

            acc_full = execution_accuracy_full(
                exec_result.get("rows", []), expected_rows,
                result_columns=result_columns,
                expected_columns=expected_columns,
            )
            raw_acc_full = execution_accuracy_full(
                exec_result.get("rows", []), expected_rows,
            )

            h_table = hallucinated_table_rate(gen_tables, ALLOWED_TABLES)
            h_column = hallucinated_column_rate(gen_columns, allowed_columns, sql=sql)
            h_join = hallucinated_join_rate(sql, allowed_joins)

            row = {
                "query_id": qid,
                "question": question,
                "difficulty": difficulty,
                "hop_count": hop_count,
                "method": "proposed",
                "sql": sql,
                "syntax_valid": is_syntax_valid,
                "execution_valid": is_exec_valid,
                "execution_accuracy": acc_full["recall"],
                "execution_precision": acc_full["precision"],
                "execution_recall": acc_full["recall"],
                "execution_f1": acc_full["f1"],
                "raw_execution_accuracy": raw_acc_full["recall"],
                "raw_execution_precision": raw_acc_full["precision"],
                "raw_execution_f1": raw_acc_full["f1"],
                "hallucinated_table_rate": h_table,
                "hallucinated_column_rate": h_column,
                "hallucinated_join_rate": h_join,
                "token_usage": tokens,
                "latency_ms": latency_ms,
                "sanity_regen": sanity_regen,
                "repair_attempts": repair_attempts,
                "repair_tokens": repair_tokens,
            }
            results.append(row)

        except Exception as e:
            print(f" ERROR: {e}")
            results.append({
                "query_id": qid, "question": question, "difficulty": difficulty,
                "hop_count": hop_count, "method": "proposed", "sql": "",
                "syntax_valid": False, "execution_valid": False,
                "execution_accuracy": 0.0, "execution_precision": 0.0,
                "execution_recall": 0.0, "execution_f1": 0.0,
                "raw_execution_accuracy": 0.0, "raw_execution_precision": 0.0,
                "raw_execution_f1": 0.0,
                "hallucinated_table_rate": 0.0, "hallucinated_column_rate": 0.0,
                "hallucinated_join_rate": 0.0,
                "token_usage": 0, "latency_ms": 0,
                "sanity_regen": 0, "repair_attempts": 0, "repair_tokens": 0,
            })

    print(f"\n\nWriting results to {output_path}...")
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    # Summary
    total = len(results)
    avg_acc = sum(float(r["execution_accuracy"]) for r in results) / total if total else 0
    syntax_ok = sum(1 for r in results if r["syntax_valid"])
    exec_ok = sum(1 for r in results if r["execution_valid"])
    print(f"\nProposed: {total} queries")
    print(f"  Avg accuracy: {avg_acc:.1%}")
    print(f"  Syntax valid: {syntax_ok}/{total}")
    print(f"  Execution valid: {exec_ok}/{total}")

    # By difficulty
    for diff in ["easy", "medium", "hard", "very_hard"]:
        subset = [r for r in results if r["difficulty"] == diff]
        if subset:
            acc = sum(float(r["execution_accuracy"]) for r in subset) / len(subset)
            print(f"  {diff}: {acc:.1%} ({len(subset)} queries)")

    conn.close()


if __name__ == "__main__":
    main()
