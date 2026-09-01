#!/usr/bin/env python3
"""Generate/verify expected results for the independent English validation set.

Executes each gold SQL in evaluation/gold_sql_independent_en/ against the
main fixture DB (READ ONLY, REPEATABLE READ) and writes/verifies
evaluation/expected_results_independent_en/{qid}.json with the same schema
as the main suite (columns / ordered / semantic_ordered / rows).

Usage:
    python scripts/gen_independent_en_expected.py [--update]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import psycopg  # noqa: E402

from scripts.db_conninfo import CONNINFO  # noqa: E402
from scripts.fixture_guard import assert_valid_fixture  # noqa: E402
from scripts.gold_compare import normalize_rows, rows_match, sql_is_ordered  # noqa: E402

GOLD_DIR = PROJECT / "evaluation" / "gold_sql_independent_en"
RESULTS_DIR = PROJECT / "evaluation" / "expected_results_independent_en"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true")
    args = ap.parse_args()

    conn = psycopg.connect(CONNINFO)
    conn.read_only = True
    conn.isolation_level = psycopg.IsolationLevel.REPEATABLE_READ
    with conn.cursor() as cur:
        cur.execute("SET statement_timeout = '30s'")
    conn.commit()
    assert_valid_fixture(conn)

    failures = []
    for sql_path in sorted(GOLD_DIR.glob("*.sql")):
        qid = sql_path.stem
        sql = sql_path.read_text().strip()
        ordered = sql_is_ordered(sql)
        with conn.cursor() as cur:
            cur.execute("SAVEPOINT q")
            try:
                cur.execute(sql)
                columns = [d.name for d in cur.description]
                rows = normalize_rows(cur.fetchall())
            except Exception as e:
                cur.execute("ROLLBACK TO SAVEPOINT q")
                failures.append(f"{qid}: EXEC ERROR {e}")
                continue
            cur.execute("RELEASE SAVEPOINT q")
        if not rows:
            failures.append(f"{qid}: EMPTY RESULT")
            continue
        out = RESULTS_DIR / f"{qid}.json"
        payload = {"columns": columns, "ordered": ordered,
                   "semantic_ordered": ordered, "rows": rows}
        if args.update or not out.exists():
            with open(out, "w") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
            print(f"{qid}: wrote {len(rows)} rows, {len(columns)} cols, ordered={ordered}")
        else:
            with open(out) as f:
                prior = json.load(f)
            if prior["columns"] != columns or not rows_match(
                    prior["rows"], rows, ordered=ordered):
                failures.append(f"{qid}: STALE expected result")
            else:
                print(f"{qid}: OK ({len(rows)} rows)")

    if failures:
        print("\nFAILURES:")
        for f_ in failures:
            print(" ", f_)
        sys.exit(1)
    print("\nAll independent EN gold queries verified.")


if __name__ == "__main__":
    main()
