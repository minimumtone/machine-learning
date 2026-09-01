#!/usr/bin/env python3
"""Independent English validation: 25 English-authored queries, full pipeline.

The questions in evaluation/independent_en_dataset.jsonl were authored in
English directly against the schema (not translations of the Japanese
evaluation sets); gold SQL and expected results were created afterwards
and verified against the fixture DB (scripts/gen_independent_en_expected.py).

Usage:
    python scripts/eval_independent_en.py [--run N]

Saves to evaluation/independent_en_run{N}.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from graph.graph_builder import build_table_graph  # noqa: E402
from graph.join_path_generator import get_allowed_join_list  # noqa: E402
from graph.schema_parser import get_columns, get_foreign_keys, get_tables  # noqa: E402
from scripts.db_conninfo import CONNINFO  # noqa: E402
import scripts.eval_ablation as ablation  # noqa: E402
from scripts.eval_db import open_eval_connection  # noqa: E402
from scripts.provenance import build_provenance  # noqa: E402

EVAL_DIR = PROJECT / "evaluation"
DATASET = EVAL_DIR / "independent_en_dataset.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=int, default=1)
    args = ap.parse_args()

    out_path = EVAL_DIR / f"independent_en_run{args.run}.json"
    if out_path.exists():
        print(f"{out_path} already exists, skipping")
        return

    # Score against the independent EN expected results.
    ablation.RESULTS_DIR = EVAL_DIR / "expected_results_independent_en"

    conn = open_eval_connection(CONNINFO, suite="main")
    tables = get_tables(conn)
    columns = {t: get_columns(conn, t) for t in tables}
    fks = get_foreign_keys(conn)
    table_graph = build_table_graph(fks)
    if not table_graph.has_edge("composition", "element"):
        table_graph.add_edge("composition", "element",
                             source_column="element", target_column="symbol")
    allowed_columns = [f"{t}.{c.column_name}" for t, cols in columns.items() for c in cols]
    allowed_joins = get_allowed_join_list(table_graph)

    with open(DATASET) as f:
        queries = [json.loads(line) for line in f if line.strip()]
    print(f"Independent EN queries: {len(queries)}  run: {args.run}")

    def exec_fn(sql):
        return ablation.execute_sql(conn, sql)

    results = ablation.run_condition(conn, queries, "full", allowed_joins,
                                     allowed_columns, table_graph, exec_fn)

    total = sum(r["recall"] for r in results) / len(results)
    by_diff: dict[str, list[float]] = {}
    for r in results:
        by_diff.setdefault(r["difficulty"], []).append(r["recall"])
    summary = {
        "run": args.run,
        "condition": "full",
        "language": "en_independent",
        "overall_recall": total,
        "by_difficulty": {d: sum(v) / len(v) for d, v in by_diff.items()},
        "results": results,
        "provenance": build_provenance(DATASET),
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nOverall recall: {total:.1%}")
    for d, v in summary["by_difficulty"].items():
        print(f"  {d}: {v:.1%}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
