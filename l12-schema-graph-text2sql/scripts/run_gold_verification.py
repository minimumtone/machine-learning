#!/usr/bin/env python3
"""Standalone gold-SQL re-verification for the SQL distribution package.

Executes every gold SQL file against the fixture databases and compares the
rows to the stored expected_results JSON, so a third party can re-verify the
"all gold queries pass, expected results not stale" claim from the package
alone (no LLM, no API key, only psycopg + a loaded database).

Main-database queries run against L12_DSN; q_transfer_* queries run against
TRANSFER_DSN when it is set (otherwise they are skipped and reported).

Usage:
    L12_DSN="host=127.0.0.1 port=5432 dbname=l12_materials user=l12_user \
password=..." \
    TRANSFER_DSN="host=127.0.0.1 port=5432 dbname=oqmd_transfer user=l12_user \
password=..." \
    python scripts/run_gold_verification.py

Exit status: 0 when every executed query matches its expected results and no
expectation file is missing; 1 otherwise.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import psycopg

PROJECT = Path(__file__).resolve().parent.parent
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


def main() -> int:
    """Run all gold SQL and compare to expected_results; return exit code."""
    main_dsn = os.environ.get("L12_DSN")
    if not main_dsn:
        print("ERROR: set L12_DSN to the main fixture database DSN",
              file=sys.stderr)
        return 1
    transfer_dsn = os.environ.get("TRANSFER_DSN")

    main_conn = psycopg.connect(main_dsn)
    transfer_conn = psycopg.connect(transfer_dsn) if transfer_dsn else None

    ok, stale, missing, errors, skipped = [], [], [], [], []
    for sql_path in sorted(GOLD_DIR.glob("*.sql")):
        qid = sql_path.stem
        conn = transfer_conn if qid.startswith("q_transfer") else main_conn
        if conn is None:
            skipped.append(qid)
            continue
        expected_path = RESULTS_DIR / f"{qid}.json"
        try:
            with conn.cursor() as cur:
                cur.execute(sql_path.read_text())
                rows = cur.fetchall()
        except psycopg.Error as exc:
            conn.rollback()
            errors.append((qid, str(exc).splitlines()[0]))
            continue
        if not expected_path.exists():
            missing.append(qid)
            continue
        expected = json.loads(expected_path.read_text())
        if _normalize_rows(rows) != _normalize_rows(expected["rows"]):
            stale.append(qid)
        else:
            ok.append(qid)

    print(f"ok={len(ok)} stale={len(stale)} missing={len(missing)} "
          f"errors={len(errors)} skipped={len(skipped)}")
    for qid in stale:
        print(f"STALE   {qid}")
    for qid in missing:
        print(f"MISSING {qid}")
    for qid, msg in errors:
        print(f"ERROR   {qid}: {msg}")
    if skipped:
        print(f"skipped (TRANSFER_DSN not set): {', '.join(skipped)}")
    return 0 if not (stale or missing or errors) else 1


if __name__ == "__main__":
    sys.exit(main())
