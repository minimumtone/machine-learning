#!/usr/bin/env python3
"""Build a Materials Project-flavored transfer database.

Fetches real MP summary data via the public REST API and ingests it into a
small PostgreSQL schema (mp_entries, mp_element_ratios, mp_elements).
Used for transfer/generalization experiment D.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import psycopg
from psycopg import sql as pgsql
import requests

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from scripts.eval_ablation import CONNINFO  # noqa: E402

DB_NAME = "mp_transfer"
MP_API_URL = "https://api.materialsproject.org/materials/summary/"
CHEMSYS_LIST = [
    "Ni-Al", "Co-Al", "Fe-Al", "Ti-Al", "Co-Ti", "Ni-Ti",
    "Zr-Al", "Sc-Al", "Pt-Al", "Ni-Co", "Al-Mg", "Al-Cu",
    "Al-Ag", "Al-Au", "Fe-Ni", "Cr-Al", "Mn-Al", "V-Al",
    "Nb-Al", "Mo-Al", "W-Al", "Rh-Al", "Pd-Al", "Hf-Al",
    "Ta-Al", "Al-Zn", "Al-Si",
]
MP_FIELDS = (
    "material_id,formula_pretty,elements,composition,energy_per_atom,"
    "energy_above_hull,band_gap,volume,nsites,symmetry,"
    "structure.lattice.a,structure.lattice.b,structure.lattice.c"
)
PER_PAGE = 100
REQUEST_TIMEOUT = 60
SLEEP_BETWEEN = 0.3

_ELEMENT_PROPERTIES: dict[str, dict[str, Any]] = {
    "H": {"atomic_number": 1, "name": "Hydrogen"},
    "He": {"atomic_number": 2, "name": "Helium"},
    "Li": {"atomic_number": 3, "name": "Lithium"},
    "Be": {"atomic_number": 4, "name": "Beryllium"},
    "B": {"atomic_number": 5, "name": "Boron"},
    "C": {"atomic_number": 6, "name": "Carbon"},
    "N": {"atomic_number": 7, "name": "Nitrogen"},
    "O": {"atomic_number": 8, "name": "Oxygen"},
    "F": {"atomic_number": 9, "name": "Fluorine"},
    "Ne": {"atomic_number": 10, "name": "Neon"},
    "Na": {"atomic_number": 11, "name": "Sodium"},
    "Mg": {"atomic_number": 12, "name": "Magnesium"},
    "Al": {"atomic_number": 13, "name": "Aluminum"},
    "Si": {"atomic_number": 14, "name": "Silicon"},
    "P": {"atomic_number": 15, "name": "Phosphorus"},
    "S": {"atomic_number": 16, "name": "Sulfur"},
    "Cl": {"atomic_number": 17, "name": "Chlorine"},
    "Ar": {"atomic_number": 18, "name": "Argon"},
    "K": {"atomic_number": 19, "name": "Potassium"},
    "Ca": {"atomic_number": 20, "name": "Calcium"},
    "Sc": {"atomic_number": 21, "name": "Scandium"},
    "Ti": {"atomic_number": 22, "name": "Titanium"},
    "V": {"atomic_number": 23, "name": "Vanadium"},
    "Cr": {"atomic_number": 24, "name": "Chromium"},
    "Mn": {"atomic_number": 25, "name": "Manganese"},
    "Fe": {"atomic_number": 26, "name": "Iron"},
    "Co": {"atomic_number": 27, "name": "Cobalt"},
    "Ni": {"atomic_number": 28, "name": "Nickel"},
    "Cu": {"atomic_number": 29, "name": "Copper"},
    "Zn": {"atomic_number": 30, "name": "Zinc"},
    "Ga": {"atomic_number": 31, "name": "Gallium"},
    "Ge": {"atomic_number": 32, "name": "Germanium"},
    "As": {"atomic_number": 33, "name": "Arsenic"},
    "Se": {"atomic_number": 34, "name": "Selenium"},
    "Br": {"atomic_number": 35, "name": "Bromine"},
    "Kr": {"atomic_number": 36, "name": "Krypton"},
    "Rb": {"atomic_number": 37, "name": "Rubidium"},
    "Sr": {"atomic_number": 38, "name": "Strontium"},
    "Y": {"atomic_number": 39, "name": "Yttrium"},
    "Zr": {"atomic_number": 40, "name": "Zirconium"},
    "Nb": {"atomic_number": 41, "name": "Niobium"},
    "Mo": {"atomic_number": 42, "name": "Molybdenum"},
    "Tc": {"atomic_number": 43, "name": "Technetium"},
    "Ru": {"atomic_number": 44, "name": "Ruthenium"},
    "Rh": {"atomic_number": 45, "name": "Rhodium"},
    "Pd": {"atomic_number": 46, "name": "Palladium"},
    "Ag": {"atomic_number": 47, "name": "Silver"},
    "Cd": {"atomic_number": 48, "name": "Cadmium"},
    "In": {"atomic_number": 49, "name": "Indium"},
    "Sn": {"atomic_number": 50, "name": "Tin"},
    "Sb": {"atomic_number": 51, "name": "Antimony"},
    "Te": {"atomic_number": 52, "name": "Tellurium"},
    "I": {"atomic_number": 53, "name": "Iodine"},
    "Xe": {"atomic_number": 54, "name": "Xenon"},
    "Cs": {"atomic_number": 55, "name": "Cesium"},
    "Ba": {"atomic_number": 56, "name": "Barium"},
    "La": {"atomic_number": 57, "name": "Lanthanum"},
    "Ce": {"atomic_number": 58, "name": "Cerium"},
    "Pr": {"atomic_number": 59, "name": "Praseodymium"},
    "Nd": {"atomic_number": 60, "name": "Neodymium"},
    "Pm": {"atomic_number": 61, "name": "Promethium"},
    "Sm": {"atomic_number": 62, "name": "Samarium"},
    "Eu": {"atomic_number": 63, "name": "Europium"},
    "Gd": {"atomic_number": 64, "name": "Gadolinium"},
    "Tb": {"atomic_number": 65, "name": "Terbium"},
    "Dy": {"atomic_number": 66, "name": "Dysprosium"},
    "Ho": {"atomic_number": 67, "name": "Holmium"},
    "Er": {"atomic_number": 68, "name": "Erbium"},
    "Tm": {"atomic_number": 69, "name": "Thulium"},
    "Yb": {"atomic_number": 70, "name": "Ytterbium"},
    "Lu": {"atomic_number": 71, "name": "Lutetium"},
    "Hf": {"atomic_number": 72, "name": "Hafnium"},
    "Ta": {"atomic_number": 73, "name": "Tantalum"},
    "W": {"atomic_number": 74, "name": "Tungsten"},
    "Re": {"atomic_number": 75, "name": "Rhenium"},
    "Os": {"atomic_number": 76, "name": "Osmium"},
    "Ir": {"atomic_number": 77, "name": "Iridium"},
    "Pt": {"atomic_number": 78, "name": "Platinum"},
    "Au": {"atomic_number": 79, "name": "Gold"},
    "Hg": {"atomic_number": 80, "name": "Mercury"},
    "Tl": {"atomic_number": 81, "name": "Thallium"},
    "Pb": {"atomic_number": 82, "name": "Lead"},
    "Bi": {"atomic_number": 83, "name": "Bismuth"},
    "Po": {"atomic_number": 84, "name": "Polonium"},
    "At": {"atomic_number": 85, "name": "Astatine"},
    "Rn": {"atomic_number": 86, "name": "Radon"},
    "Fr": {"atomic_number": 87, "name": "Francium"},
    "Ra": {"atomic_number": 88, "name": "Radium"},
    "Ac": {"atomic_number": 89, "name": "Actinium"},
    "Th": {"atomic_number": 90, "name": "Thorium"},
    "Pa": {"atomic_number": 91, "name": "Protactinium"},
    "U": {"atomic_number": 92, "name": "Uranium"},
    "Np": {"atomic_number": 93, "name": "Neptunium"},
    "Pu": {"atomic_number": 94, "name": "Plutonium"},
    "Am": {"atomic_number": 95, "name": "Americium"},
    "Cm": {"atomic_number": 96, "name": "Curium"},
}


def mp_conninfo(db: str = DB_NAME) -> str:
    """Build a psycopg connection string for the MP transfer DB."""
    base = os.getenv("CONNINFO", CONNINFO)
    return base.replace(f"dbname={os.getenv('POSTGRES_DB', 'l12_materials')}", f"dbname={db}")


def fetch_chemsys(chemsys: str, api_key: str) -> list[dict[str, Any]]:
    """Fetch one page of MP summary data for a given chemical system."""
    headers = {"X-API-KEY": api_key}
    params: dict[str, Any] = {
        "chemsys": chemsys,
        "_per_page": PER_PAGE,
        "_page": 0,
        "_fields": MP_FIELDS,
    }
    response = requests.get(MP_API_URL, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json().get("data", [])


def _create_db() -> None:
    """Create the transfer database if it does not yet exist."""
    admin_conninfo = mp_conninfo("postgres")
    admin = psycopg.connect(admin_conninfo, autocommit=True)
    with admin.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_NAME,))
        if not cur.fetchone():
            cur.execute(pgsql.SQL("CREATE DATABASE {}").format(pgsql.Identifier(DB_NAME)))
    admin.close()


def _create_schema(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            DROP TABLE IF EXISTS mp_element_ratios CASCADE;
            DROP TABLE IF EXISTS mp_entries CASCADE;
            DROP TABLE IF EXISTS mp_elements CASCADE;

            CREATE TABLE mp_elements (
                symbol TEXT PRIMARY KEY,
                atomic_number INTEGER NOT NULL,
                name TEXT
            );

            CREATE TABLE mp_entries (
                entry_id TEXT PRIMARY KEY,
                formula TEXT NOT NULL,
                chemsys TEXT NOT NULL,
                nelements INTEGER NOT NULL,
                crystal_system TEXT,
                spacegroup_symbol TEXT,
                energy_per_atom DOUBLE PRECISION,
                energy_above_hull DOUBLE PRECISION,
                band_gap DOUBLE PRECISION,
                volume DOUBLE PRECISION,
                lattice_a DOUBLE PRECISION,
                lattice_b DOUBLE PRECISION,
                lattice_c DOUBLE PRECISION,
                is_stable BOOLEAN
            );

            CREATE TABLE mp_element_ratios (
                entry_id TEXT REFERENCES mp_entries(entry_id),
                element TEXT REFERENCES mp_elements(symbol),
                atomic_fraction DOUBLE PRECISION NOT NULL,
                PRIMARY KEY (entry_id, element)
            );
        """)
    conn.commit()


def _ingest(entries: list[dict[str, Any]], conn: psycopg.Connection) -> None:
    seen_elements: set[str] = set()
    entries_rows: list[tuple[Any, ...]] = []
    ratios_rows: list[tuple[str, str, float]] = []

    for doc in entries:
        entry_id = doc["material_id"]
        formula = doc["formula_pretty"]
        elements = sorted(doc.get("elements", []))
        composition = doc.get("composition", {})
        chemsys = "-".join(elements)
        nelements = doc.get("nelements", len(elements))
        symmetry = doc.get("symmetry", {}) or {}
        crystal_system = symmetry.get("crystal_system")
        spacegroup_symbol = symmetry.get("symbol")
        energy_per_atom = doc.get("energy_per_atom")
        energy_above_hull = doc.get("energy_above_hull")
        band_gap = doc.get("band_gap")
        volume = doc.get("volume")
        lattice = (doc.get("structure") or {}).get("lattice", {}) or {}
        lattice_a = lattice.get("a")
        lattice_b = lattice.get("b")
        lattice_c = lattice.get("c")
        is_stable = energy_above_hull == 0.0

        entries_rows.append((
            entry_id, formula, chemsys, nelements, crystal_system,
            spacegroup_symbol, energy_per_atom, energy_above_hull, band_gap,
            volume, lattice_a, lattice_b, lattice_c, is_stable,
        ))

        for elem in elements:
            seen_elements.add(elem)
            frac = doc.get("composition", {}).get(elem, 0.0) / sum(composition.values()) if composition else 0.0
            ratios_rows.append((entry_id, elem, frac))

    element_rows = [
        (sym, _ELEMENT_PROPERTIES.get(sym, {}).get("atomic_number"), _ELEMENT_PROPERTIES.get(sym, {}).get("name"))
        for sym in sorted(seen_elements)
    ]

    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO mp_elements (symbol, atomic_number, name) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
            element_rows,
        )
        cur.executemany(
            """
            INSERT INTO mp_entries (
                entry_id, formula, chemsys, nelements, crystal_system, spacegroup_symbol,
                energy_per_atom, energy_above_hull, band_gap, volume,
                lattice_a, lattice_b, lattice_c, is_stable
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (entry_id) DO NOTHING
            """,
            entries_rows,
        )
        cur.executemany(
            "INSERT INTO mp_element_ratios (entry_id, element, atomic_fraction) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
            ratios_rows,
        )
    conn.commit()


def main() -> None:
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        raise RuntimeError("MP_API_KEY environment variable is required")

    _create_db()
    conn = psycopg.connect(mp_conninfo())
    _create_schema(conn)

    all_entries: list[dict[str, Any]] = []
    for chemsys in CHEMSYS_LIST:
        print(f"Fetching {chemsys}...")
        try:
            chunk = fetch_chemsys(chemsys, api_key)
            print(f"  -> {len(chunk)} entries")
            all_entries.extend(chunk)
        except Exception as exc:
            print(f"  FAILED {chemsys}: {exc}")
        time.sleep(SLEEP_BETWEEN)

    print(f"Total entries fetched: {len(all_entries)}")
    _ingest(all_entries, conn)

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM mp_entries")
        row = cur.fetchone()
        n_entries = row[0] if row else 0
        cur.execute("SELECT COUNT(*) FROM mp_element_ratios")
        row = cur.fetchone()
        n_ratios = row[0] if row else 0
    print(f"Ingested {n_entries} entries / {n_ratios} element ratios")

    conn.close()


if __name__ == "__main__":
    main()
