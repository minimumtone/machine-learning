#!/usr/bin/env python3
"""Check expected_results consistency against the current database.

Executes every gold SQL file and compares the result rows to the stored
expected_results JSON. Reports queries whose stored expectation is stale
(e.g. after new data such as pure elements was added to the database).

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
from scripts.eval_ablation import CONNINFO  # noqa: E402

GOLD_DIR = PROJECT / "evaluation" / "gold_sql"
RESULTS_DIR = PROJECT / "evaluation" / "expected_results"


def _normalize_cell(v: object) -> object:
    """Normalize a cell for comparison (numeric strings/Decimals to float)."""
    if isinstance(v, bool) or v is None:
        return v
    try:
        return round(float(v), 6)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(v)


def _normalize_rows(rows: list) -> list:
    """Normalize rows to a sorted, comparison-stable representation."""
    return sorted(str([_normalize_cell(c) for c in row]) for row in rows)


def main() -> None:
    """Compare gold SQL execution results with stored expected results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true",
                        help="Rewrite stale expected_results files")
    args = parser.parse_args()

    conn = psycopg.connect(CONNINFO)
    try:
        transfer_conn = psycopg.connect(transfer_conninfo())
    except psycopg.OperationalError:
        transfer_conn = None
    stale, missing, ok, skipped = [], [], [], []
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
        try:
            with active_conn.cursor() as cur:
                cur.execute("SET statement_timeout = '30s'")
                cur.execute(sql)  # type: ignore[arg-type]
                columns = [d[0] for d in cur.description] if cur.description else []
                rows = [list(r) for r in cur.fetchall()]
        except Exception as e:
            active_conn.rollback()
            print(f"{qid}: GOLD SQL ERROR: {e!s:.100s}")
            continue
        rows_json = json.loads(json.dumps(rows, default=str))
        if not expected_path.exists():
            missing.append(qid)
            if args.update:
                with open(expected_path, "w") as f:
                    json.dump({"columns": columns, "rows": rows_json},
                              f, ensure_ascii=False, indent=2, default=str)
            continue
        with open(expected_path) as f:
            expected = json.load(f)
        if _normalize_rows(rows_json) == _normalize_rows(expected.get("rows", [])):
            ok.append(qid)
        else:
            stale.append(qid)
            print(f"{qid}: STALE (stored {len(expected.get('rows', []))} rows, "
                  f"current {len(rows_json)} rows)")
            if args.update:
                with open(expected_path, "w") as f:
                    json.dump({"columns": columns, "rows": rows_json},
                              f, ensure_ascii=False, indent=2, default=str)
    conn.close()
    if transfer_conn is not None:
        transfer_conn.close()
    if skipped:
        print(f"skipped {len(skipped)} transfer queries: transfer DB not built "
              "(run `python scripts/build_transfer_db.py` first)")
    print(f"\nok={len(ok)} stale={len(stale)} missing={len(missing)} "
          f"skipped={len(skipped)}")
    if missing:
        print("missing:", missing)


if __name__ == "__main__":
    main()
