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
from psycopg.conninfo import make_conninfo  # noqa: E402

from scripts.db_conninfo import CONNINFO  # noqa: E402
from scripts.fixture_guard import assert_valid_fixture  # noqa: E402
from scripts.transfer_guard import assert_valid_transfer  # noqa: E402

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


def git_commit() -> str:
    """Build provenance for the transfer marker (same convention as the
    main 007 marker): the GIT_COMMIT env var, the package's GIT_COMMIT
    file, or 'unknown'."""
    env = os.getenv("GIT_COMMIT", "").strip()
    if env:
        return env
    f = PROJECT / "GIT_COMMIT"
    if f.is_file():
        content = f.read_text().strip()
        if content:
            return content
    return "unknown"


def transfer_conninfo() -> str:
    """Connection string for the transfer database."""
    return make_conninfo(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=TRANSFER_DB,
        user=os.getenv("POSTGRES_USER", "l12_user"),
        password=os.getenv("POSTGRES_PASSWORD", "l12_password"),
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
    """Validate the source fixture, then recreate the transfer DB, apply
    its schema, and copy data across.

    Fail-safe ordering: ALL source validation (fixture guard and
    reference-set contract) runs BEFORE the existing transfer DB is
    dropped, so a broken source never destroys a healthy transfer DB.

    Snapshot consistency: the source connection is pinned to a single
    REPEATABLE READ READ ONLY snapshot that covers the guard, the
    contract checks and every copy SELECT, so the transfer DB is an
    exact copy of one source state (never a mix of states seen by
    successive READ COMMITTED statements).
    """
    src = psycopg.connect(CONNINFO)
    src.read_only = True
    src.isolation_level = psycopg.IsolationLevel.REPEATABLE_READ

    # The transfer DB carries a single energy convention. If the main DB
    # ever mixed reference sets in phase_stability, the copied formation
    # energies would silently combine conventions with the single-set
    # oqmd_reference_states below, so require exactly one set up front.
    with src.cursor() as sc:
        sc.execute("SELECT DISTINCT reference_set FROM phase_stability")
        ref_sets = {row[0] for row in sc.fetchall()}
    if ref_sets != {"L12-FIXTURE-PBE-v1"}:
        raise RuntimeError(
            "transfer build requires phase_stability to use exactly "
            f"'L12-FIXTURE-PBE-v1', got {sorted(ref_sets)}"
        )
    # The set NAME alone does not guarantee the reference master still
    # carries the fixture semantics, so pin the full master tuple too.
    expected_master = (
        "L12-FIXTURE-PBE-v1", "DFT", "PBE",
        "synthetic fixture (elemental references adopted from OQMD)",
        "OQMD standard reference-energy fit "
        "(adopted for elemental references)",
    )
    with src.cursor() as sc:
        sc.execute(
            "SELECT reference_set, method, functional, source, fit_name "
            "FROM reference_energy_set "
            "WHERE reference_set = 'L12-FIXTURE-PBE-v1'")
        master = sc.fetchone()
    if master is None or tuple(master) != expected_master:
        raise RuntimeError(
            "reference_energy_set master row for 'L12-FIXTURE-PBE-v1' "
            f"does not match the fixture contract: got {master!r}, "
            f"expected {expected_master!r}"
        )
    # Only build from a fully initialized main fixture whose schema is
    # unchanged since initialization AND whose data currently passes the
    # 006 invariants (schema-only guarding would let schema-preserving
    # data edits flow into the transfer copy).
    assert_valid_fixture(src)

    # Source validated: only now may the existing transfer DB be touched.
    recreate_database()
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

    # Post-load cross-row assertions (ratio sums, reference coverage):
    # installs validate_transfer_integrity() and
    # compute_transfer_schema_fingerprint(), runs the validator, and
    # writes the fingerprint-sealed transfer_initialization_status marker
    # on success.
    with dst.cursor() as cur:
        cur.execute("SELECT set_config('l12.git_commit', %s, false)",
                    (git_commit(),))
        cur.execute(INTEGRITY_SQL.read_text())  # type: ignore[arg-type]
    dst.commit()
    assert_valid_transfer(dst)
    print("Transfer integrity checks passed (marker written).")
    src.close()
    dst.close()
    print("Transfer DB built.")


if __name__ == "__main__":
    main()
