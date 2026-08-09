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

from scripts.eval_ablation import CONNINFO  # noqa: E402

TRANSFER_DB = os.getenv("TRANSFER_DB", "oqmd_transfer")
SCHEMA_SQL = PROJECT / "db" / "transfer_schema.sql"


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
               energy_above_hull, is_stable, band_gap
        FROM phase_stability
        """,
        "INSERT INTO oqmd_formation_energies VALUES (%s, %s, %s, %s, %s, %s)",
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
               energy_per_atom, volume_per_atom, n_polymorphs
        FROM pure_element_reference
        """,
        "INSERT INTO oqmd_reference_states VALUES (%s, %s, %s, %s, %s, %s)",
    )
    print(f"oqmd_reference_states: {n}")

    dst.commit()
    src.close()
    dst.close()
    print("Transfer DB built.")


if __name__ == "__main__":
    main()
