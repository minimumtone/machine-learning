"""Shared guarded, read-only snapshot connection for model-SQL evaluation.

Every eval_*.py script executes model-generated SQL.  That SQL is
untrusted (the LLM-only baseline and the no_guard ablation run it
without SQLGuard), so the evaluation connection must not be able to
mutate the fixture:

- the connection is READ ONLY and pinned to a single REPEATABLE READ
  snapshot for the whole run;
- the suite's fixture/transfer/MP guard is asserted inside that
  snapshot before any model SQL runs;
- each model query executes inside a SAVEPOINT so a failing query
  cannot abort the snapshot transaction.
"""
from __future__ import annotations

from typing import Any

import psycopg

from scripts.fixture_guard import assert_valid_fixture
from scripts.mp_guard import assert_valid_mp_transfer
from scripts.transfer_guard import assert_valid_transfer

_GUARDS = {
    "main": assert_valid_fixture,
    "transfer": assert_valid_transfer,
    "mp_transfer": assert_valid_mp_transfer,
}


def open_eval_connection(conninfo: str, suite: str = "main",
                         statement_timeout: str = "10s") -> psycopg.Connection:
    """Guarded READ ONLY + REPEATABLE READ connection for one eval run."""
    conn = psycopg.connect(conninfo)
    conn.read_only = True
    conn.isolation_level = psycopg.IsolationLevel.REPEATABLE_READ
    conn.execute(f"SET statement_timeout = '{statement_timeout}'")
    _GUARDS[suite](conn)
    return conn


def run_model_sql(conn: psycopg.Connection, sql: str) -> dict[str, Any]:
    """Execute one model-generated query inside a SAVEPOINT."""
    try:
        conn.execute("SAVEPOINT eval_q")
    except Exception as e:  # noqa: BLE001 — snapshot itself is broken
        return {"success": False, "error": str(e),
                "rows": [], "row_count": 0, "columns": []}
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            columns = [d.name for d in cur.description] if cur.description else []
            rows = [list(r) for r in cur.fetchall()]
        conn.execute("RELEASE SAVEPOINT eval_q")
        return {"success": True, "columns": columns,
                "rows": rows, "row_count": len(rows)}
    except Exception as e:  # noqa: BLE001 — scored as execution failure
        conn.execute("ROLLBACK TO SAVEPOINT eval_q")
        return {"success": False, "error": str(e),
                "rows": [], "row_count": 0, "columns": []}
