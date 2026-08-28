#!/usr/bin/env python3
"""Check expected_results consistency against the current databases.

Executes every gold SQL file (main, transfer, and the obfuscated transfer
suite) and compares the result to the stored expected_results JSON: column
names, row order for ORDER BY queries, row multiset otherwise, and typed
value tolerance (see scripts/gold_compare.py for the full comparison
policy). The stored "ordered" flag is also validated against the SQL's
actual top-level ORDER BY (an order-contract mismatch fails the run).
Also reports orphan expected-results files that no longer have a gold SQL.

All database connections are forced READ ONLY with a 30 s
statement_timeout (--update only rewrites JSON files, never the DB).
Skipped transfer/obfuscated queries fail the run unless
--allow-missing-transfer is given, so a partial run can never be mistaken
for a full check.

Usage:
    python scripts/check_expected_results.py [--update]
        [--allow-missing-transfer]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import psycopg  # noqa: E402

from scripts.build_obfuscated_transfer_db import (  # noqa: E402
    obfuscated_conninfo,
)
from scripts.build_transfer_db import transfer_conninfo  # noqa: E402
from scripts.db_conninfo import CONNINFO  # noqa: E402
from scripts.fixture_guard import assert_valid_fixture  # noqa: E402
from scripts.transfer_guard import assert_valid_transfer  # noqa: E402
from scripts.gold_compare import (  # noqa: E402
    normalize_rows,
    rows_match,
    sql_is_ordered,
    validate_expected_schema,
)

GOLD_DIR = PROJECT / "evaluation" / "gold_sql"
RESULTS_DIR = PROJECT / "evaluation" / "expected_results"
OBF_GOLD_DIR = PROJECT / "evaluation" / "gold_sql_obfuscated"
OBF_RESULTS_DIR = PROJECT / "evaluation" / "expected_results_obfuscated"


def _connect_readonly(conninfo: str) -> psycopg.Connection:
    conn = psycopg.connect(conninfo)
    with conn.cursor() as cur:
        cur.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY")
        cur.execute("SET statement_timeout = '30s'")
    conn.commit()
    return conn


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


class Buckets:
    """Per-run result buckets shared by all suites."""

    def __init__(self) -> None:
        self.ok: list[str] = []
        self.stale: list[str] = []
        self.missing: list[str] = []
        self.skipped: list[str] = []
        self.column_mismatch: list[str] = []
        self.order_mismatch: list[str] = []
        self.order_contract_mismatch: list[str] = []
        self.malformed: list[str] = []


def check_suite(gold_dir: Path, results_dir: Path,
                conn: psycopg.Connection | None, b: Buckets,
                args: argparse.Namespace, suite: str,
                qid_filter=lambda qid: True) -> None:
    """Execute one gold-SQL suite and compare against expected results."""
    for sql_path in sorted(gold_dir.glob("*.sql")):
        qid = sql_path.stem
        if not qid_filter(qid):
            continue
        if conn is None:
            b.skipped.append(f"{suite}:{qid}")
            continue
        expected_path = results_dir / f"{qid}.json"
        sql = sql_path.read_text()
        ordered = sql_is_ordered(sql)
        try:
            with conn.cursor() as cur:
                cur.execute(sql)  # type: ignore[arg-type]
                columns = ([d.name for d in cur.description]
                           if cur.description else [])
                rows = [list(r) for r in cur.fetchall()]
        except Exception as e:
            conn.rollback()
            print(f"{qid}: GOLD SQL ERROR: {e!s:.100s}")
            continue
        rows_norm = normalize_rows(rows)
        if not expected_path.exists():
            b.missing.append(qid)
            if args.update:
                _write_expected(expected_path, columns, rows_norm, ordered)
            continue
        with open(expected_path) as f:
            expected = json.load(f)
        schema_err = validate_expected_schema(expected)
        expected_ordered = expected["ordered"] \
            if isinstance(expected, dict) and schema_err is None else ordered
        expected_columns = expected.get("columns") \
            if isinstance(expected, dict) else None
        expected_rows = normalize_rows(expected.get("rows", [])) \
            if isinstance(expected, dict) else []
        if schema_err:
            b.malformed.append(qid)
            print(f"{qid}: MALFORMED expected JSON: {schema_err}")
        elif expected_ordered != ordered:
            # The stored flag is a contract: it must agree with the SQL's
            # actual top-level ORDER BY, so editing a gold SQL cannot
            # silently downgrade (or overstate) the comparison mode.
            b.order_contract_mismatch.append(qid)
            print(f"{qid}: ORDER CONTRACT MISMATCH (stored ordered="
                  f"{expected_ordered}, SQL has ORDER BY={ordered})")
        elif columns != expected_columns:
            b.column_mismatch.append(qid)
            print(f"{qid}: COLUMN MISMATCH (stored {expected_columns}, "
                  f"current {columns})")
        elif rows_match(rows_norm, expected_rows, ordered=expected_ordered):
            b.ok.append(qid)
        elif expected_ordered and rows_match(rows_norm, expected_rows,
                                             ordered=False):
            b.order_mismatch.append(qid)
            print(f"{qid}: ORDER MISMATCH (same multiset, different order)")
        else:
            b.stale.append(qid)
            print(f"{qid}: STALE (stored {len(expected_rows)} rows, "
                  f"current {len(rows_norm)} rows)")
        if args.update and qid not in b.ok:
            _write_expected(expected_path, columns, rows_norm, ordered)


def main() -> None:
    """Compare gold SQL execution results with stored expected results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true",
                        help="Rewrite stale/missing expected_results files")
    parser.add_argument("--allow-missing-transfer", action="store_true",
                        help="Treat skipped transfer/obfuscated queries as "
                             "non-fatal (default: skipping fails the run)")
    args = parser.parse_args()

    # Guard every connection before any query runs: 007 marker +
    # fingerprint + validate_fixture_integrity() for the main DB,
    # transfer marker + validate_transfer_integrity() for the others.
    conn = _connect_readonly(CONNINFO)
    assert_valid_fixture(conn)
    try:
        transfer_conn = _connect_readonly(transfer_conninfo())
        assert_valid_transfer(transfer_conn)
    except psycopg.OperationalError:
        transfer_conn = None
    try:
        obf_conn = _connect_readonly(obfuscated_conninfo())
        assert_valid_transfer(obf_conn)
    except psycopg.OperationalError:
        obf_conn = None

    b = Buckets()

    orphans = sorted(
        f"{results_dir.name}/{qid}"
        for gold_dir, results_dir in ((GOLD_DIR, RESULTS_DIR),
                                      (OBF_GOLD_DIR, OBF_RESULTS_DIR))
        for qid in ({p.stem for p in results_dir.glob("*.json")}
                    - {p.stem for p in gold_dir.glob("*.sql")})
    )

    check_suite(GOLD_DIR, RESULTS_DIR, conn, b, args, "main",
                lambda qid: not qid.startswith("q_transfer"))
    check_suite(GOLD_DIR, RESULTS_DIR, transfer_conn, b, args, "transfer",
                lambda qid: qid.startswith("q_transfer"))
    check_suite(OBF_GOLD_DIR, OBF_RESULTS_DIR, obf_conn, b, args,
                "obfuscated")

    conn.close()
    for c in (transfer_conn, obf_conn):
        if c is not None:
            c.close()
    if b.skipped:
        print(f"skipped {len(b.skipped)} queries (DB not built): "
              + ", ".join(b.skipped[:5])
              + (" ..." if len(b.skipped) > 5 else ""))
    print(f"\nok={len(b.ok)} stale={len(b.stale)} "
          f"order_mismatch={len(b.order_mismatch)} "
          f"order_contract_mismatch={len(b.order_contract_mismatch)} "
          f"column_mismatch={len(b.column_mismatch)} "
          f"missing={len(b.missing)} malformed={len(b.malformed)} "
          f"orphan={len(orphans)} skipped={len(b.skipped)}")
    if b.missing:
        print("missing:", b.missing)
    if orphans:
        print("orphan expected_results (no matching gold SQL):", orphans)
        sys.exit(1)
    if (b.stale or b.order_mismatch or b.order_contract_mismatch
            or b.column_mismatch or b.malformed):
        sys.exit(1)
    if b.missing and not args.update:
        sys.exit(1)
    if b.skipped and not args.allow_missing_transfer:
        print("FAIL: queries were skipped and "
              "--allow-missing-transfer was not given")
        sys.exit(1)


if __name__ == "__main__":
    main()
