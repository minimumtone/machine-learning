"""Guard: only audit/verify a fully initialized, drift-free fixture DB.

Audit scripts connect directly to the main fixture DB. Run against a
partially loaded DB (say 001-003 only) they would audit whatever tables
exist and could report success for an incomplete fixture. This guard
requires:

1. the version='007' row in schema_initialization_status (written only
   after all 006 integrity assertions passed), and
2. the recorded schema fingerprint to equal a freshly computed
   compute_schema_fingerprint() (no schema drift since initialization).
"""
from __future__ import annotations

import psycopg


def assert_initialized_fixture(conn: psycopg.Connection) -> None:
    """Raise RuntimeError unless the DB is an initialized 007 fixture
    with an unchanged schema fingerprint."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_name = 'schema_initialization_status')
        """)
        row = cur.fetchone()
        assert row is not None
        if not row[0]:
            raise RuntimeError(
                "fixture DB has no schema_initialization_status table; "
                "load db/001...007 before auditing")
        cur.execute("""
            SELECT schema_fingerprint FROM schema_initialization_status
            WHERE version = '007'
        """)
        marker = cur.fetchone()
        if marker is None:
            raise RuntimeError(
                "fixture DB has no version='007' initialization marker; "
                "db/007_initialization_marker.sql did not complete")
        cur.execute("SELECT compute_schema_fingerprint()")
        row = cur.fetchone()
        assert row is not None
        if row[0] != marker[0]:
            raise RuntimeError(
                "schema fingerprint mismatch: recorded "
                f"{marker[0][:12]}..., current {row[0][:12]}... — schema "
                "drifted since initialization")
