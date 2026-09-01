#!/usr/bin/env python3
"""Guard for the Materials Project transfer database.

The MP transfer DB has no marker table; its fixture identity is defined
by the pinned snapshot ``db/mp_transfer_snapshot.json.gz``. This guard
verifies two things before a single gold query runs:

1. Data identity: the live tables are re-exported in the snapshot's
   canonical record form and the SHA-256 digest must equal the
   snapshot's recorded digest, so any added, removed, or edited row in
   the three audited tables fails verification.
2. Schema identity: a catalog-derived fingerprint (columns with exact
   types / nullability / defaults, and all table constraints in the
   public schema) must equal the pinned ``MP_SCHEMA_FINGERPRINT``, so
   ALTER TABLE / DROP CONSTRAINT / extra objects are detected even
   though they would not change the exported records.

Out of scope (as for the other suites' fingerprints): indexes, roles,
GRANTs, and server GUC settings.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import psycopg

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from scripts.build_mp_transfer_db import (  # noqa: E402
    SNAPSHOT_PATH,
    _records_sha256,
    load_snapshot,
)

_ENTRY_COLS = [
    "entry_id", "formula", "chemsys", "nelements", "crystal_system",
    "spacegroup_symbol", "energy_per_atom", "energy_above_hull",
    "band_gap", "volume", "lattice_a", "lattice_b", "lattice_c",
    "is_stable",
]

# Pinned catalog fingerprint of the schema created by
# scripts/build_mp_transfer_db.py (_SCHEMA_SQL). Recompute with
# compute_mp_schema_fingerprint() after an intentional schema change.
MP_SCHEMA_FINGERPRINT = (
    "1aa87ff7da44e7aa5df66188b09d5cdd1454400c4307ab84bb25115fa59d066a"
)

_SCHEMA_PARTS_SQL = """
    WITH parts AS (
        SELECT 'col:' || c.relname || '.' || a.attname
               || ':' || format_type(a.atttypid, a.atttypmod)
               || ':' || a.attnotnull
               || ':' || COALESCE(pg_get_expr(d.adbin, d.adrelid), '')
               AS part,
               1 AS grp, c.relname AS k1, a.attnum::TEXT AS k2
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum > 0
             AND NOT a.attisdropped
        LEFT JOIN pg_attrdef d ON d.adrelid = c.oid AND d.adnum = a.attnum
        WHERE n.nspname = 'public' AND c.relkind IN ('r', 'v', 'm')
        UNION ALL
        SELECT 'con:' || c.relname || ':' || con.conname
               || ':' || pg_get_constraintdef(con.oid),
               2, c.relname, con.conname
        FROM pg_constraint con
        JOIN pg_class c ON c.oid = con.conrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public'
    )
    SELECT part FROM parts ORDER BY grp, k1, k2
"""


def compute_mp_schema_fingerprint(conn: psycopg.Connection) -> str:
    """SHA-256 over the public-schema column/constraint catalog."""
    with conn.cursor() as cur:
        cur.execute(_SCHEMA_PARTS_SQL)
        parts = [r[0] for r in cur.fetchall()]
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def assert_valid_mp_transfer(conn: psycopg.Connection) -> None:
    """Raise RuntimeError unless the DB matches the pinned snapshot."""
    schema_fp = compute_mp_schema_fingerprint(conn)
    if schema_fp != MP_SCHEMA_FINGERPRINT:
        raise RuntimeError(
            f"MP transfer DB schema fingerprint {schema_fp} != pinned "
            f"{MP_SCHEMA_FINGERPRINT} — schema drift detected; rebuild "
            "with scripts/build_mp_transfer_db.py")
    snap = load_snapshot()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT " + ", ".join(_ENTRY_COLS)
            + " FROM mp_entries ORDER BY entry_id")
        entries = [dict(zip(_ENTRY_COLS, r)) for r in cur.fetchall()]
        cur.execute("SELECT entry_id, element, atomic_fraction "
                    "FROM mp_element_ratios ORDER BY entry_id, element")
        ratios = [dict(zip(["entry_id", "element", "atomic_fraction"], r))
                  for r in cur.fetchall()]
        cur.execute("SELECT symbol, atomic_number, name "
                    "FROM mp_elements ORDER BY symbol")
        elements = [dict(zip(["symbol", "atomic_number", "name"], r))
                    for r in cur.fetchall()]
    digest = _records_sha256(entries, ratios, elements)
    recorded = snap["_meta"]["records_sha256"]
    if digest != recorded:
        raise RuntimeError(
            f"MP transfer DB does not match the pinned snapshot "
            f"{SNAPSHOT_PATH.name}: data sha256 {digest} != recorded "
            f"{recorded} — rebuild with scripts/build_mp_transfer_db.py")
