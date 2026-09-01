#!/usr/bin/env python3
"""Paired language robustness evaluation (Japanese vs English).

Runs the full pipeline condition over the 100-query ablation dataset in
both languages. The English dataset (evaluation_dataset_en.jsonl) shares
identical gold SQL, expected results, database, and pipeline
configuration with the Japanese dataset; only the natural-language
question differs.

Usage:
    python scripts/eval_language_paired.py --language en --run 1
    python scripts/eval_language_paired.py --language ja --run 1

Each run saves to evaluation/language_paired_{language}_run{N}.json
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
from scripts.eval_ablation import execute_sql, run_condition  # noqa: E402
from scripts.eval_db import open_eval_connection  # noqa: E402
from scripts.provenance import build_provenance  # noqa: E402

EVAL_DIR = PROJECT / "evaluation"

DATASETS = {
    "ja": EVAL_DIR / "evaluation_dataset.jsonl",
    "en": EVAL_DIR / "evaluation_dataset_en.jsonl",
}


def load_queries(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", choices=["ja", "en"], required=True)
    ap.add_argument("--run", type=int, required=True)
    args = ap.parse_args()

    dataset = DATASETS[args.language]
    out_path = EVAL_DIR / f"language_paired_{args.language}_run{args.run}.json"
    if out_path.exists():
        print(f"{out_path} already exists, skipping")
        return

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

    queries = load_queries(dataset)
    print(f"Language: {args.language}  run: {args.run}  queries: {len(queries)}")

    def exec_fn(sql):
        return execute_sql(conn, sql)

    results = run_condition(conn, queries, "full", allowed_joins,
                            allowed_columns, table_graph, exec_fn)

    total = sum(r["recall"] for r in results) / len(results)
    by_diff: dict[str, list[float]] = {}
    for r in results:
        by_diff.setdefault(r["difficulty"], []).append(r["recall"])
    summary = {
        "language": args.language,
        "run": args.run,
        "condition": "full",
        "overall_recall": total,
        "by_difficulty": {d: sum(v) / len(v) for d, v in by_diff.items()},
        "results": results,
        "provenance": build_provenance(dataset),
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nOverall recall: {total:.1%}")
    for d, v in summary["by_difficulty"].items():
        print(f"  {d}: {v:.1%}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
