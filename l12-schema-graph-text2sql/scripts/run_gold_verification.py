#!/usr/bin/env python3
"""Standalone gold-SQL re-verification for the SQL distribution package.

Executes every gold SQL file against the fixture databases and compares the
result to the stored expected_results JSON (column names, row order for
ORDER BY queries, row multiset otherwise, typed value tolerance — see
scripts/gold_compare.py for the full comparison policy), so a third party
can re-verify the package claims alone (no LLM, no API key; requires
psycopg + sqlglot — see requirements-repro.txt — and the loaded databases).

Main-database queries run against L12_DSN; q_transfer_* queries run against
TRANSFER_DSN; the obfuscated transfer suite (evaluation/gold_sql_obfuscated)
runs against OBF_TRANSFER_DSN. Suites whose DSN is not set are skipped and
reported — and skipping is a FAILURE unless --allow-skip is passed, so a
partial run can never be mistaken for full verification by its exit code.
All connections are forced READ ONLY with a 30 s statement_timeout, and
each connection is pinned to a single REPEATABLE READ snapshot: the guard
and every gold query of a suite see one and the same database state, so a
concurrent writer cannot make part of the run verify a different state
(each gold query runs inside a SAVEPOINT, so a SQL error rolls back only
that query — the outer snapshot transaction, and hence the single-state
guarantee for the remaining queries, is preserved; the run still fails).

Usage:
    L12_DSN="postgresql://l12_user:...@127.0.0.1:5432/l12_materials" \
    TRANSFER_DSN="postgresql://l12_user:...@127.0.0.1:5432/oqmd_transfer" \
    OBF_TRANSFER_DSN="postgresql://l12_user:...@127.0.0.1:5432/oqmd_transfer_obfuscated" \
    python scripts/run_gold_verification.py

Exit status: 0 when every query (none skipped, unless --allow-skip) matches
its expected results and there are no missing expectations, orphan expected
files, malformed expected JSON, column mismatches, or order mismatches;
1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import psycopg

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from scripts.fixture_guard import assert_valid_fixture  # noqa: E402
from scripts.mp_guard import assert_valid_mp_transfer  # noqa: E402
from scripts.transfer_guard import assert_valid_transfer  # noqa: E402
from scripts.gold_compare import (  # noqa: E402
    normalize_rows,
    rows_match,
    sql_is_ordered,
    validate_expected_schema,
)

SUITES = [
    # (name, gold dir, expected dir, DSN env var, qid filter)
    ("main", "gold_sql", "expected_results", "L12_DSN",
     lambda qid: not qid.startswith("q_transfer")),
    ("transfer", "gold_sql", "expected_results", "TRANSFER_DSN",
     lambda qid: qid.startswith("q_transfer")),
    ("obfuscated", "gold_sql_obfuscated", "expected_results_obfuscated",
     "OBF_TRANSFER_DSN", lambda qid: True),
    ("mp", "gold_sql_mp", "expected_results_mp_transfer",
     "MP_DSN", lambda qid: True),
]

# ok=N must mean "this distribution's schema answered N queries", not
# "some database answered N queries": every connection is guarded before
# any query runs. The main DB must carry the 007 marker with an
# unchanged schema fingerprint and pass validate_fixture_integrity()
# now; the transfer DBs must carry their initialization marker and pass
# validate_transfer_integrity() now.
GUARDS = {
    "L12_DSN": assert_valid_fixture,
    "TRANSFER_DSN": assert_valid_transfer,
    "OBF_TRANSFER_DSN": assert_valid_transfer,
    # The MP transfer DB carries no marker table; its guard compares the
    # live table contents' SHA-256 against the pinned snapshot digest.
    "MP_DSN": assert_valid_mp_transfer,
}


def _connect_readonly(dsn: str) -> psycopg.Connection:
    """READ ONLY + REPEATABLE READ: after the initial commit below, the
    first statement (the suite guard) opens one snapshot that every
    subsequent gold query shares until the run ends."""
    conn = psycopg.connect(dsn)
    conn.read_only = True
    conn.isolation_level = psycopg.IsolationLevel.REPEATABLE_READ
    with conn.cursor() as cur:
        cur.execute("SET statement_timeout = '30s'")
    conn.commit()
    return conn


def main() -> int:
    """Run all gold SQL suites and compare to expected results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-skip", action="store_true",
        help="Treat suites skipped for a missing DSN as non-fatal "
             "(default: any skipped query fails the run)")
    args = parser.parse_args()
    if not os.environ.get("L12_DSN"):
        print("ERROR: set L12_DSN to the main fixture database DSN",
              file=sys.stderr)
        return 1

    conns: dict[str, psycopg.Connection | None] = {}
    for _, _, _, env, _ in SUITES:
        dsn = os.environ.get(env)
        if env not in conns:
            conn = _connect_readonly(dsn) if dsn else None
            if conn is not None:
                GUARDS[env](conn)
            conns[env] = conn

    ok: list[str] = []
    stale: list[str] = []
    order_mismatch: list[str] = []
    order_contract_mismatch: list[str] = []
    column_mismatch: list[str] = []
    missing: list[str] = []
    malformed: list[tuple[str, str]] = []
    errors: list[tuple[str, str]] = []
    skipped: list[str] = []
    orphans: list[str] = []

    for name, gold_dirname, expected_dirname, env, qid_filter in SUITES:
        gold_dir = PROJECT / "evaluation" / gold_dirname
        results_dir = PROJECT / "evaluation" / expected_dirname
        conn = conns[env]

        gold_ids = {p.stem for p in gold_dir.glob("*.sql") if qid_filter(p.stem)}
        if name != "transfer":
            # orphan check is per expected-results directory; the main and
            # transfer suites share one directory, checked once under "main"
            expected_ids = {p.stem for p in results_dir.glob("*.json")}
            all_gold = {p.stem for p in gold_dir.glob("*.sql")}
            orphans.extend(
                f"{expected_dirname}/{qid}"
                for qid in sorted(expected_ids - all_gold)
            )

        for sql_path in sorted(gold_dir.glob("*.sql")):
            qid = sql_path.stem
            if qid not in gold_ids:
                continue
            if conn is None:
                skipped.append(f"{name}:{qid}")
                continue
            sql = sql_path.read_text()
            expected_path = results_dir / f"{qid}.json"
            try:
                # SAVEPOINT: a failing gold query must roll back only
                # itself, not the outer REPEATABLE READ transaction, so
                # every later query still sees the same snapshot.
                with conn.cursor() as cur:
                    cur.execute("SAVEPOINT gold_query")
                    try:
                        cur.execute(sql)
                        actual_columns = (
                            [d.name for d in cur.description]
                            if cur.description else []
                        )
                        rows = cur.fetchall()
                        cur.execute("RELEASE SAVEPOINT gold_query")
                    except psycopg.Error:
                        cur.execute("ROLLBACK TO SAVEPOINT gold_query")
                        cur.execute("RELEASE SAVEPOINT gold_query")
                        raise
            except psycopg.Error as exc:
                errors.append((qid, str(exc).splitlines()[0]))
                continue
            if not expected_path.exists():
                missing.append(qid)
                continue
            expected = json.loads(expected_path.read_text())
            schema_err = validate_expected_schema(expected)
            if schema_err:
                malformed.append((qid, schema_err))
                continue
            if actual_columns != expected["columns"]:
                column_mismatch.append(qid)
                continue
            # The stored "ordered" flag is a contract, not a trusted input:
            # it must agree with the SQL's actual top-level ORDER BY, so a
            # gold SQL edit cannot silently downgrade an ordered comparison
            # (or claim an order the SQL does not enforce).
            ordered = expected["ordered"]
            if sql_is_ordered(sql) != ordered:
                order_contract_mismatch.append(qid)
                continue
            actual_norm = normalize_rows(rows)
            expected_norm = normalize_rows(expected["rows"])
            if rows_match(actual_norm, expected_norm, ordered=ordered):
                ok.append(qid)
            elif ordered and rows_match(actual_norm, expected_norm,
                                        ordered=False):
                order_mismatch.append(qid)
            else:
                stale.append(qid)

    ordered_metadata_missing = sum(
        1 for _qid, msg in malformed if "'ordered'" in msg)
    print(f"ok={len(ok)} stale={len(stale)} "
          f"order_mismatch={len(order_mismatch)} "
          f"order_contract_mismatch={len(order_contract_mismatch)} "
          f"column_mismatch={len(column_mismatch)} "
          f"missing={len(missing)} malformed={len(malformed)} "
          f"ordered_metadata_missing={ordered_metadata_missing} "
          f"orphan={len(orphans)} errors={len(errors)} "
          f"skipped={len(skipped)}")
    for qid in stale:
        print(f"STALE           {qid}")
    for qid in order_mismatch:
        print(f"ORDER_MISMATCH  {qid}")
    for qid in order_contract_mismatch:
        print(f"ORDER_CONTRACT_MISMATCH {qid}: expected JSON 'ordered' flag "
              "disagrees with the SQL's top-level ORDER BY")
    for qid in column_mismatch:
        print(f"COLUMN_MISMATCH {qid}")
    for qid in missing:
        print(f"MISSING         {qid}")
    for qid, msg in malformed:
        print(f"MALFORMED       {qid}: {msg}")
    for qid in orphans:
        print(f"ORPHAN          {qid}")
    for qid, msg in errors:
        print(f"ERROR           {qid}: {msg}")
    if skipped:
        print(f"skipped (DSN not set): {', '.join(skipped)}")
    failed = (stale or order_mismatch or order_contract_mismatch
              or column_mismatch or missing or malformed or orphans
              or errors)
    if skipped and not args.allow_skip:
        print("FAIL: suites were skipped and --allow-skip was not given",
              file=sys.stderr)
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
