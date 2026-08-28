"""Guard for the transfer / obfuscated-transfer databases.

Mirrors scripts/fixture_guard.py for the main fixture: verifiers and
audits must not run against a partially built transfer DB (schema
applied but integrity checks never passed). The guard requires:

1. the transfer_initialization_status marker table with its version row
   (written by db/transfer_integrity_checks.sql only after
   validate_transfer_integrity() passed at build time), and
2. a FRESH run of validate_transfer_integrity() against the current
   data, so post-build data edits that violate an invariant are caught
   ("passes the integrity checks NOW", not "passed them once").

Both databases expose the same unrenamed marker table and validator
function name (the obfuscation deliberately leaves them untouched and
rewrites only the validator body's identifiers).
"""
from __future__ import annotations

import psycopg


def assert_valid_transfer(conn: psycopg.Connection) -> None:
    """Raise RuntimeError unless the transfer DB carries the
    initialization marker and currently passes its integrity checks."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_name = 'transfer_initialization_status')
        """)
        row = cur.fetchone()
        assert row is not None
        if not row[0]:
            raise RuntimeError(
                "transfer DB has no transfer_initialization_status table; "
                "run db/transfer_integrity_checks.sql (via "
                "scripts/build_transfer_db.py) before auditing")
        cur.execute(
            "SELECT 1 FROM transfer_initialization_status "
            "WHERE version = '001'")
        if cur.fetchone() is None:
            raise RuntimeError(
                "transfer DB has no version='001' initialization marker; "
                "db/transfer_integrity_checks.sql did not complete")
        cur.execute("SELECT validate_transfer_integrity()")
