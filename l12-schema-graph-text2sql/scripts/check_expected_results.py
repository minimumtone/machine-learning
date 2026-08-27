#!/usr/bin/env python3
"""Check expected_results consistency against the current database.

Executes every gold SQL file and compares the result to the stored
expected_results JSON: column names, row order for ORDER BY queries, row
multiset otherwise, and typed value tolerance (see scripts/gold_compare.py
for the full comparison policy). Also reports orphan expected-results files
that no longer have a gold SQL.

Usage:
    python scripts/check_expected_results.py [--update]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import psycopg  # noqa: E402

from scripts.build_transfer_db import transfer_conninfo  # noqa: E402
from scripts.db_conninfo import CONNINFO  # noqa: E402
from scripts.gold_compare import (  # noqa: E402
    normalize_rows,
    rows_match,
    sql_is_ordered,
    validate_expected_schema,
)

GOLD_DIR = PROJECT / "evaluation" / "gold_sql"
RESULTS_DIR = PROJECT / "evaluation" / "expected_results"


def _write_expected(path: Path, columns: list[str], rows: list,
                    ordered: bool) -> None:
    payload: dict = {}
    if path.exists():
        try:
            with open(path) as f:
                prior = json.load(f)
            if isinstance(prior, dict):
                # Preserve annotation keys (e.g. expected_empty, purpose).
                payload = {k: v for k, v in prior.items()
                           if k not in ("columns", "ordered", "rows")}
        except (json.JSONDecodeError, OSError):
            pass
    payload.update({"columns": columns, "ordered": ordered, "rows": rows})
    with open(path, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    """Compare gold SQL execution results with stored expected results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true",
                        help="Rewrite stale/missing expected_results files")
    args = parser.parse_args()

    conn = psycopg.connect(CONNINFO)
    try:
        transfer_conn = psycopg.connect(transfer_conninfo())
    except psycopg.OperationalError:
        transfer_conn = None
    stale, missing, ok, skipped = [], [], [], []
    column_mismatch, order_mismatch, malformed = [], [], []

    gold_ids = {p.stem for p in GOLD_DIR.glob("*.sql")}
    expected_ids = {p.stem for p in RESULTS_DIR.glob("*.json")}
    orphans = sorted(expected_ids - gold_ids)

    for sql_path in sorted(GOLD_DIR.glob("*.sql")):
        qid = sql_path.stem
        if qid.startswith("q_transfer"):
            if transfer_conn is None:
                skipped.append(qid)
                continue
            active_conn = transfer_conn
        else:
            active_conn = conn
        expected_path = RESULTS_DIR / f"{qid}.json"
        sql = sql_path.read_text()
        ordered = sql_is_ordered(sql)
        try:
            with active_conn.cursor() as cur:
                cur.execute("SET statement_timeout = '30s'")
                cur.execute(sql)  # type: ignore[arg-type]
                columns = ([d.name for d in cur.description]
                           if cur.description else [])
                rows = [list(r) for r in cur.fetchall()]
        except Exception as e:
            active_conn.rollback()
            print(f"{qid}: GOLD SQL ERROR: {e!s:.100s}")
            continue
        rows_norm = normalize_rows(rows)
        if not expected_path.exists():
            missing.append(qid)
            if args.update:
                _write_expected(expected_path, columns, rows_norm, ordered)
            continue
        with open(expected_path) as f:
            expected = json.load(f)
        schema_err = validate_expected_schema(expected)
        expected_ordered = expected.get("ordered", ordered) \
            if isinstance(expected, dict) else ordered
        expected_columns = expected.get("columns") \
            if isinstance(expected, dict) else None
        expected_rows = normalize_rows(expected.get("rows", [])) \
            if isinstance(expected, dict) else []
        if schema_err:
            malformed.append(qid)
            print(f"{qid}: MALFORMED expected JSON: {schema_err}")
        elif columns != expected_columns:
            column_mismatch.append(qid)
            print(f"{qid}: COLUMN MISMATCH (stored {expected_columns}, "
                  f"current {columns})")
        elif rows_match(rows_norm, expected_rows, ordered=expected_ordered):
            ok.append(qid)
        elif expected_ordered and rows_match(rows_norm, expected_rows,
                                             ordered=False):
            order_mismatch.append(qid)
            print(f"{qid}: ORDER MISMATCH (same multiset, different order)")
        else:
            stale.append(qid)
            print(f"{qid}: STALE (stored {len(expected_rows)} rows, "
                  f"current {len(rows_norm)} rows)")
        if args.update and qid not in ok:
            _write_expected(expected_path, columns, rows_norm, ordered)
    conn.close()
    if transfer_conn is not None:
        transfer_conn.close()
    if skipped:
        print(f"skipped {len(skipped)} transfer queries: transfer DB not built "
              "(run `python scripts/build_transfer_db.py` first)")
    print(f"\nok={len(ok)} stale={len(stale)} "
          f"order_mismatch={len(order_mismatch)} "
          f"column_mismatch={len(column_mismatch)} "
          f"missing={len(missing)} malformed={len(malformed)} "
          f"orphan={len(orphans)} skipped={len(skipped)}")
    if missing:
        print("missing:", missing)
    if orphans:
        print("orphan expected_results (no matching gold SQL):", orphans)
        sys.exit(1)
    if stale or order_mismatch or column_mismatch or malformed:
        sys.exit(1)


if __name__ == "__main__":
    main()
