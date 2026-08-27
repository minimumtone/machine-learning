#!/usr/bin/env python3
"""Build the transfer-test database from the L1_2 source database.

Creates a separate database with an OQMD-flavored schema (different table and
column names) and populates it from the existing L1_2 data. Used for the
schema-transfer experiment: evaluating the pipeline on an unseen schema.

Usage:
    python scripts/build_transfer_db.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import psycopg  # noqa: E402
from psycopg import sql as pgsql  # noqa: E402

from scripts.db_conninfo import CONNINFO  # noqa: E402

TRANSFER_DB = os.getenv("TRANSFER_DB", "oqmd_transfer")

# DROP DATABASE guard: this builder may only ever drop a transfer DB it
# owns the naming convention for, never system DBs or the main fixture DB.
FORBIDDEN_DB_NAMES = {
    "postgres",
    "template0",
    "template1",
    os.getenv("POSTGRES_DB", "l12_materials"),
}


def assert_safe_transfer_db(name: str) -> None:
    """Refuse to drop/recreate anything outside the transfer-DB namespace."""
    if name in FORBIDDEN_DB_NAMES or not name.startswith("oqmd_transfer"):
        raise RuntimeError(
            f"Refusing to drop database {name!r}: TRANSFER_DB must start "
            "with 'oqmd_transfer' and must not name a system or main DB")


SCHEMA_SQL = PROJECT / "db" / "transfer_schema.sql"
INTEGRITY_SQL = PROJECT / "db" / "transfer_integrity_checks.sql"


def transfer_conninfo() -> str:
    """Connection string for the transfer database."""
    return (
        f"host={os.getenv('POSTGRES_HOST', 'localhost')} "
        f"port={os.getenv('POSTGRES_PORT', '5432')} "
        f"dbname={TRANSFER_DB} "
        f"user={os.getenv('POSTGRES_USER', 'l12_user')} "
        f"password={os.getenv('POSTGRES_PASSWORD', 'l12_password')}"
    )


def recreate_database() -> None:
    """Drop and recreate the transfer database."""
    assert_safe_transfer_db(TRANSFER_DB)
    admin = psycopg.connect(CONNINFO, autocommit=True)
    with admin.cursor() as cur:
        cur.execute(
            pgsql.SQL("DROP DATABASE IF EXISTS {}").format(pgsql.Identifier(TRANSFER_DB))
        )
        cur.execute(
            pgsql.SQL("CREATE DATABASE {}").format(pgsql.Identifier(TRANSFER_DB))
        )
    admin.close()


def main() -> None:
    """Recreate the transfer DB, apply its schema, and copy data across."""
    recreate_database()
    src = psycopg.connect(CONNINFO)
    dst = psycopg.connect(transfer_conninfo())
    with dst.cursor() as cur:
        cur.execute(SCHEMA_SQL.read_text())  # type: ignore[arg-type]

    def copy(select_sql: str, insert_sql: str) -> int:
        with src.cursor() as sc, dst.cursor() as dc:
            sc.execute(select_sql)  # type: ignore[arg-type]
            rows = sc.fetchall()
            dc.executemany(insert_sql, rows)  # type: ignore[arg-type]
        return len(rows)

    n = copy(
        "SELECT symbol, name, atomic_number, atomic_mass FROM element",
        "INSERT INTO oqmd_elements VALUES (%s, %s, %s, %s)",
    )
    print(f"oqmd_elements: {n}")

    n = copy(
        """
        SELECT m.entry_id, m.formula,
               COALESCE(s.prototype, s.strukturbericht),
               s.space_group_number, s.crystal_system,
               s.lattice_a, s.volume_per_atom
        FROM material_entry m
        LEFT JOIN structure s ON s.entry_id = m.entry_id
        """,
        "INSERT INTO oqmd_entries VALUES (%s, %s, %s, %s, %s, %s, %s) "
        "ON CONFLICT (entry_key) DO NOTHING",
    )
    print(f"oqmd_entries: {n}")

    n = copy(
        """
        SELECT stability_id, entry_id, formation_energy_per_atom,
               energy_above_hull, band_gap
        FROM phase_stability
        """,
        "INSERT INTO oqmd_formation_energies "
        "(fe_key, entry_key, delta_e, hull_distance, gap_ev) "
        "VALUES (%s, %s, %s, %s, %s)",
    )
    print(f"oqmd_formation_energies: {n}")

    n = copy(
        """
        SELECT composition_id, entry_id, element, atomic_fraction, site_label
        FROM composition
        """,
        "INSERT INTO oqmd_element_ratios VALUES (%s, %s, %s, %s, %s)",
    )
    print(f"oqmd_element_ratios: {n}")

    n = copy(
        """
        SELECT 'ref_' || pure_ref_id, element_symbol, ground_state_spacegroup,
               delta_e, volume_per_atom, n_polymorphs
        FROM pure_element_reference
        -- The transfer DB carries a single energy convention; copying the
        -- divergence-test set as well would violate the UNIQUE(symbol) key
        -- and mix conventions in the weighted reference sums.
        WHERE reference_set = 'L12-FIXTURE-PBE-v1'
        """,
        "INSERT INTO oqmd_reference_states VALUES (%s, %s, %s, %s, %s, %s)",
    )
    print(f"oqmd_reference_states: {n}")

    dst.commit()

    # Post-load cross-row assertions (ratio sums, reference coverage).
    with dst.cursor() as cur:
        cur.execute(INTEGRITY_SQL.read_text())  # type: ignore[arg-type]
    dst.commit()
    print("Transfer integrity checks passed.")
    src.close()
    dst.close()
    print("Transfer DB built.")


if __name__ == "__main__":
    main()
