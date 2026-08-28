#!/usr/bin/env python3
"""Guard for the Materials Project transfer database.

The MP transfer DB has no marker table; its fixture identity is defined
by the pinned snapshot ``db/mp_transfer_snapshot.json.gz``. This guard
re-exports the live tables in the snapshot's canonical record form and
compares the SHA-256 digest against the snapshot's recorded digest, so
any schema or data drift (added, removed, or edited rows/columns) fails
verification before a single gold query runs.
"""
from __future__ import annotations

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


def assert_valid_mp_transfer(conn: psycopg.Connection) -> None:
    """Raise RuntimeError unless the DB matches the pinned snapshot."""
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
