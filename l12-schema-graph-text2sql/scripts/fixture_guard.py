"""Guards: only audit/verify a fully initialized fixture DB.

Audit scripts connect directly to the main fixture DB. Run against a
partially loaded DB (say 001-003 only) they would audit whatever tables
exist and could report success for an incomplete fixture.

What each layer guarantees (they are deliberately separate):

- assert_initialized_fixture: SCHEMA IDENTITY ONLY — the version='007'
  marker exists and the recorded schema fingerprint equals a freshly
  computed compute_schema_fingerprint() (no schema drift since
  initialization). It says nothing about the data.
- validate_fixture_integrity() (db/006): cross-row DATA invariants,
  re-checked against the CURRENT data, so post-initialization data
  edits that violate an invariant are caught even though they leave the
  schema fingerprint unchanged.
- scripts/audit_semantics.py: chemistry/prototype semantics.
- scripts/run_gold_verification.py: benchmark output regression.

assert_valid_fixture combines the first two and is what verifiers,
audits and the transfer builder should call: data drift outside the 006
invariants remains undetectable by construction (the fixture is
immutable by contract; see README).
"""
from __future__ import annotations

import psycopg


def assert_initialized_fixture(conn: psycopg.Connection) -> None:
    """Raise RuntimeError unless the DB is an initialized 007 fixture
    with an unchanged schema fingerprint (schema identity only; use
    assert_valid_fixture to also re-check the 006 data invariants)."""
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


def assert_valid_fixture(conn: psycopg.Connection) -> None:
    """Schema identity (007 marker + fingerprint) AND a fresh run of the
    006 cross-row data invariants against the current data."""
    assert_initialized_fixture(conn)
    with conn.cursor() as cur:
        cur.execute("SELECT validate_fixture_integrity()")
