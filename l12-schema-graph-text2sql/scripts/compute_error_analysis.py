#!/usr/bin/env python3
"""Per-condition error counts for the ablation error-analysis table.

For each of the 7 ablation conditions, counts over the 100 stored
run-1 generated SQLs:

- syntax errors (sqlglot parse failure)
- execution errors (statement fails against the live database)
- table hallucinations (queries referencing a non-existent table)
- JOIN hallucinations (queries with at least one JOIN outside the FK list)

Outputs evaluation/error_analysis_counts.json.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import psycopg  # noqa: E402

from evaluation.metrics import (  # noqa: E402
    hallucinated_join_rate,
    hallucinated_table_rate,
    syntax_validity,
)
from graph.graph_builder import build_table_graph  # noqa: E402
from graph.join_path_generator import get_allowed_join_list  # noqa: E402
from graph.schema_parser import get_foreign_keys, get_tables  # noqa: E402

DSN = (
    f"host={os.environ.get('DB_HOST', 'localhost')} "
    f"port={os.environ.get('DB_PORT', '5432')} "
    f"dbname={os.environ.get('DB_NAME', 'l12_materials')} "
    f"user={os.environ.get('DB_USER', 'l12_user')} "
    f"password={os.environ.get('DB_PASSWORD', 'l12_password')}"
)

CONDITIONS = [
    "full", "no_fewshot", "no_dict", "no_reranker",
    "no_guard", "no_nbest", "no_graph",
]

_NON_TABLE = {"SELECT", "ON", "WHERE", "AND", "OR", "LEFT", "RIGHT",
              "INNER", "OUTER", "CROSS", "FULL"}


def _extract_tables(sql: str) -> list[str]:
    return [
        m.group(1).lower()
        for m in re.finditer(r"(?:FROM|JOIN)\s+(\w+)", sql, re.IGNORECASE)
        if m.group(1).upper() not in _NON_TABLE
    ]


def main() -> None:
    run = json.loads((ROOT / "evaluation" / "ablation_run_1.json").read_text())
    out: dict[str, dict[str, int]] = {}
    with psycopg.connect(DSN) as conn:
        valid_tables = get_tables(conn)
        table_graph = build_table_graph(get_foreign_keys(conn))
        if not table_graph.has_edge("composition", "element"):
            table_graph.add_edge(
                "composition", "element",
                source_column="element", target_column="symbol",
            )
        allowed_joins = get_allowed_join_list(table_graph)
        for cond in CONDITIONS:
            results = run["conditions"][cond]["results"]
            syn = exe = tab = joi = 0
            for r in results:
                sql = r.get("sql") or ""
                if not sql or not syntax_validity(sql):
                    syn += 1
                    exe += 1
                    continue
                cte_names = [
                    m.group(1).lower() for m in re.finditer(
                        r"(?:WITH|,)\s*(\w+)\s+AS\s*\(", sql, re.IGNORECASE)
                ]
                gen_tables = [
                    t for t in _extract_tables(sql) if t not in cte_names
                ]
                if hallucinated_table_rate(gen_tables, valid_tables) > 0:
                    tab += 1
                if hallucinated_join_rate(sql, allowed_joins) > 0:
                    joi += 1
                try:
                    with conn.cursor() as cur:
                        cur.execute(sql)
                        cur.fetchall()
                except Exception:
                    exe += 1
                    conn.rollback()
            out[cond] = {
                "syntax_errors": syn,
                "execution_errors": exe,
                "table_hallucinations": tab,
                "join_hallucinations": joi,
                "n_queries": len(results),
            }
            print(cond, out[cond])

    dst = ROOT / "evaluation" / "error_analysis_counts.json"
    dst.write_text(json.dumps(
        {"_note": "Counts over run-1 stored SQLs per ablation condition",
         "source_file": "evaluation/ablation_run_1.json",
         "counts": out}, indent=2, ensure_ascii=False))
    print(f"Wrote {dst}")


if __name__ == "__main__":
    main()
