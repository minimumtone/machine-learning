"""Guard for the transfer / obfuscated-transfer databases.

Mirrors scripts/fixture_guard.py for the main fixture: verifiers and
audits must not run against a partially built transfer DB (schema
applied but integrity checks never passed). The guard requires:

1. the transfer_initialization_status marker table with its version row
   (written by db/transfer_integrity_checks.sql only after
   validate_transfer_integrity() passed at build time),
2. that the recorded schema fingerprint equals a freshly recomputed
   compute_transfer_schema_fingerprint() — post-build schema drift
   (ALTER TABLE / DROP CONSTRAINT / ...) is rejected even when the
   change is invisible to the data validator (mirrors the main
   fixture_guard's schema-identity check), and
3. a FRESH run of validate_transfer_integrity() against the current
   data, so post-build data edits that violate an invariant are caught
   ("passes the integrity checks NOW", not "passed them once").

Scope: schema identity is guaranteed by the fingerprint; data drift is
guaranteed only to the extent the validator's cross-row invariants
express it (same caveat as the main fixture guard).

Both databases expose the same unrenamed marker table, validator and
fingerprint function names (the obfuscation deliberately leaves them
untouched and rewrites only the validator body's identifiers; the
obfuscated DB is sealed with its own post-rename fingerprint).
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
        try:
            cur.execute(
                "SELECT schema_fingerprint "
                "FROM transfer_initialization_status "
                "WHERE version = '001'")
        except psycopg.errors.UndefinedColumn as exc:
            raise RuntimeError(
                "transfer DB marker has no schema_fingerprint column (built "
                "by a pre-fingerprint package revision); rebuild via "
                "scripts/build_transfer_db.py") from exc
        row = cur.fetchone()
        if row is None:
            raise RuntimeError(
                "transfer DB has no version='001' initialization marker; "
                "db/transfer_integrity_checks.sql did not complete")
        stored_fp = row[0]
        cur.execute("SELECT compute_transfer_schema_fingerprint()")
        fp_row = cur.fetchone()
        assert fp_row is not None
        current_fp = fp_row[0]
        if stored_fp != current_fp:
            raise RuntimeError(
                "transfer DB schema fingerprint mismatch: marker records "
                f"{stored_fp[:12]}… but the current schema computes "
                f"{current_fp[:12]}… — the schema was changed after "
                "initialization; rebuild the transfer DB via "
                "scripts/build_transfer_db.py")
        cur.execute("SELECT validate_transfer_integrity()")
