#!/usr/bin/env python3
"""Build the Materials Project-flavored transfer database.

Default (reproducible) path: rebuild the DB from the pinned snapshot
``db/mp_transfer_snapshot.json.gz`` — no network access or API key
needed. Every expected result under
``evaluation/expected_results_mp_transfer/`` was generated from this
snapshot.

Optional live path: ``--refresh-from-api`` re-fetches all chemical
systems from the MP API (requires ``MP_API_KEY`` and ``requests``),
following pagination to completion. The fetch is all-or-nothing: any
chemsys failure, incomplete pagination, or duplicate material_id aborts
before anything is written. On success a new snapshot (with fetch
metadata and a SHA-256 over the records) is saved and the DB is built
from it.

In both paths the database is first built into a temporary database
(``mp_transfer_build_tmp``) and only swapped into place after row
counts match the snapshot, so a mid-build failure never destroys a
previously valid ``mp_transfer``.
"""
from __future__ import annotations

import argparse
import datetime
import gzip
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import psycopg
from psycopg import sql as pgsql

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from scripts.db_conninfo import mp_conninfo  # noqa: E402, F401

DB_NAME = "mp_transfer"
TMP_DB_NAME = "mp_transfer_build_tmp"
SNAPSHOT_PATH = PROJECT / "db" / "mp_transfer_snapshot.json.gz"
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


# ---------------------------------------------------------------------------
# Snapshot handling
# ---------------------------------------------------------------------------

def _records_sha256(entries: list[dict[str, Any]],
                    ratios: list[dict[str, Any]],
                    elements: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        {"entries": entries, "ratios": ratios, "elements": elements},
        sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_snapshot(path: Path = SNAPSHOT_PATH) -> dict[str, Any]:
    """Load and integrity-check the pinned snapshot."""
    if not path.exists():
        raise FileNotFoundError(
            f"snapshot not found: {path} — either restore the packaged "
            "snapshot or run with --refresh-from-api")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        snap = json.load(f)
    meta = snap.get("_meta", {})
    digest = _records_sha256(snap["entries"], snap["ratios"], snap["elements"])
    if digest != meta.get("records_sha256"):
        raise ValueError(
            f"snapshot integrity check failed: records sha256 {digest} "
            f"!= recorded {meta.get('records_sha256')}")
    return snap


def save_snapshot(entries: list[dict[str, Any]],
                  ratios: list[dict[str, Any]],
                  elements: list[dict[str, Any]],
                  fetch_meta: dict[str, Any],
                  path: Path = SNAPSHOT_PATH) -> None:
    # Canonical record order: must match the ORDER BY used by
    # mp_guard.assert_valid_mp_transfer so digests are comparable.
    entries = sorted(entries, key=lambda e: e["entry_id"])
    ratios = sorted(ratios, key=lambda r: (r["entry_id"], r["element"]))
    elements = sorted(elements, key=lambda el: el["symbol"])
    meta = dict(fetch_meta)
    meta["records_sha256"] = _records_sha256(entries, ratios, elements)
    meta["n_entries"] = len(entries)
    meta["n_ratios"] = len(ratios)
    meta["n_elements"] = len(elements)
    snap = {"_meta": meta, "entries": entries, "ratios": ratios,
            "elements": elements}
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(snap, f, ensure_ascii=False, sort_keys=True)
    print(f"Snapshot saved: {path} (sha256={meta['records_sha256']})")


# ---------------------------------------------------------------------------
# Live API fetch (--refresh-from-api)
# ---------------------------------------------------------------------------

def fetch_chemsys_all_pages(chemsys: str, api_key: str) -> list[dict[str, Any]]:
    """Fetch every page of MP summary data for one chemical system."""
    import requests  # optional dependency; only needed for live refresh

    headers = {"X-API-KEY": api_key}
    docs: list[dict[str, Any]] = []
    page = 0
    while True:
        params: dict[str, Any] = {
            "chemsys": chemsys,
            "_per_page": PER_PAGE,
            "_page": page,
            "_fields": MP_FIELDS,
        }
        response = requests.get(MP_API_URL, headers=headers, params=params,
                                timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        chunk = response.json().get("data", [])
        docs.extend(chunk)
        if len(chunk) < PER_PAGE:
            # Pagination provably complete: the API returned a short page.
            return docs
        page += 1
        time.sleep(SLEEP_BETWEEN)


def _normalize_doc(doc: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    elements = sorted(doc.get("elements", []))
    composition = doc.get("composition", {})
    symmetry = doc.get("symmetry", {}) or {}
    lattice = (doc.get("structure") or {}).get("lattice", {}) or {}
    energy_above_hull = doc.get("energy_above_hull")
    entry = {
        "entry_id": doc["material_id"],
        "formula": doc["formula_pretty"],
        "chemsys": "-".join(elements),
        "nelements": doc.get("nelements", len(elements)),
        "crystal_system": symmetry.get("crystal_system"),
        "spacegroup_symbol": symmetry.get("symbol"),
        "energy_per_atom": doc.get("energy_per_atom"),
        "energy_above_hull": energy_above_hull,
        "band_gap": doc.get("band_gap"),
        "volume": doc.get("volume"),
        "lattice_a": lattice.get("a"),
        "lattice_b": lattice.get("b"),
        "lattice_c": lattice.get("c"),
        "is_stable": energy_above_hull == 0.0,
    }
    total = sum(composition.values()) if composition else 0.0
    ratios = [
        {"entry_id": doc["material_id"], "element": elem,
         "atomic_fraction": (composition.get(elem, 0.0) / total) if total else 0.0}
        for elem in elements
    ]
    return entry, ratios


def refresh_from_api() -> dict[str, Any]:
    """All-or-nothing live fetch: returns a validated snapshot dict.

    Fetches everything into memory first, verifies that every chemsys
    succeeded, pagination completed, and material IDs are unique — only
    then writes the snapshot. The DB is not touched here at all.
    """
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        raise RuntimeError("MP_API_KEY environment variable is required "
                           "for --refresh-from-api")

    per_chemsys: dict[str, list[dict[str, Any]]] = {}
    for chemsys in CHEMSYS_LIST:
        print(f"Fetching {chemsys}...")
        # No try/except: any chemsys failure aborts the whole refresh
        # before anything is persisted.
        docs = fetch_chemsys_all_pages(chemsys, api_key)
        print(f"  -> {len(docs)} entries")
        per_chemsys[chemsys] = docs
        time.sleep(SLEEP_BETWEEN)

    all_docs = [d for docs in per_chemsys.values() for d in docs]
    ids = [d["material_id"] for d in all_docs]
    if len(ids) != len(set(ids)):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate material IDs in fetch: {dupes[:10]}")

    entries: list[dict[str, Any]] = []
    ratios: list[dict[str, Any]] = []
    seen_elements: set[str] = set()
    for doc in all_docs:
        entry, entry_ratios = _normalize_doc(doc)
        entries.append(entry)
        ratios.extend(entry_ratios)
        seen_elements.update(r["element"] for r in entry_ratios)
    elements = [
        {"symbol": sym,
         "atomic_number": _ELEMENT_PROPERTIES.get(sym, {}).get("atomic_number"),
         "name": _ELEMENT_PROPERTIES.get(sym, {}).get("name")}
        for sym in sorted(seen_elements)
    ]

    fetch_meta = {
        "fetched_at_utc": datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="seconds"),
        "api_endpoint": MP_API_URL,
        "fields": MP_FIELDS,
        "chemsys_list": CHEMSYS_LIST,
        "per_page": PER_PAGE,
        "per_chemsys_counts": {c: len(v) for c, v in per_chemsys.items()},
    }
    save_snapshot(entries, ratios, elements, fetch_meta)
    return load_snapshot()


# ---------------------------------------------------------------------------
# DB build (tmp DB + swap)
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
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
"""


def _admin_conn() -> psycopg.Connection:
    return psycopg.connect(mp_conninfo("postgres"), autocommit=True)


def _drop_db(admin: psycopg.Connection, name: str) -> None:
    with admin.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,))
        if cur.fetchone():
            cur.execute(pgsql.SQL("DROP DATABASE {} WITH (FORCE)")
                        .format(pgsql.Identifier(name)))


def _ingest_snapshot(conn: psycopg.Connection, snap: dict[str, Any]) -> None:
    with conn.cursor() as cur:
        cur.execute(_SCHEMA_SQL)
        cur.executemany(
            "INSERT INTO mp_elements (symbol, atomic_number, name) "
            "VALUES (%(symbol)s, %(atomic_number)s, %(name)s)",
            snap["elements"],
        )
        cur.executemany(
            """
            INSERT INTO mp_entries (
                entry_id, formula, chemsys, nelements, crystal_system,
                spacegroup_symbol, energy_per_atom, energy_above_hull,
                band_gap, volume, lattice_a, lattice_b, lattice_c, is_stable
            ) VALUES (
                %(entry_id)s, %(formula)s, %(chemsys)s, %(nelements)s,
                %(crystal_system)s, %(spacegroup_symbol)s,
                %(energy_per_atom)s, %(energy_above_hull)s, %(band_gap)s,
                %(volume)s, %(lattice_a)s, %(lattice_b)s, %(lattice_c)s,
                %(is_stable)s
            )
            """,
            snap["entries"],
        )
        cur.executemany(
            "INSERT INTO mp_element_ratios (entry_id, element, atomic_fraction) "
            "VALUES (%(entry_id)s, %(element)s, %(atomic_fraction)s)",
            snap["ratios"],
        )
    conn.commit()


def build_db_from_snapshot(snap: dict[str, Any]) -> None:
    """Build into a tmp DB, verify counts, then swap into place."""
    admin = _admin_conn()
    try:
        _drop_db(admin, TMP_DB_NAME)
        with admin.cursor() as cur:
            cur.execute(pgsql.SQL("CREATE DATABASE {}")
                        .format(pgsql.Identifier(TMP_DB_NAME)))
        try:
            conn = psycopg.connect(mp_conninfo(TMP_DB_NAME))
            try:
                _ingest_snapshot(conn, snap)
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM mp_entries")
                    n_entries = cur.fetchone()[0]
                    cur.execute("SELECT COUNT(*) FROM mp_element_ratios")
                    n_ratios = cur.fetchone()[0]
                    cur.execute("SELECT COUNT(*) FROM mp_elements")
                    n_elements = cur.fetchone()[0]
            finally:
                conn.close()
            meta = snap["_meta"]
            if (n_entries, n_ratios, n_elements) != (
                    meta["n_entries"], meta["n_ratios"], meta["n_elements"]):
                raise ValueError(
                    f"tmp DB counts {(n_entries, n_ratios, n_elements)} != "
                    f"snapshot counts {(meta['n_entries'], meta['n_ratios'], meta['n_elements'])}")
            # Integrity verified: swap tmp into place. Only now is the
            # previous mp_transfer destroyed.
            _drop_db(admin, DB_NAME)
            with admin.cursor() as cur:
                cur.execute(pgsql.SQL("ALTER DATABASE {} RENAME TO {}")
                            .format(pgsql.Identifier(TMP_DB_NAME),
                                    pgsql.Identifier(DB_NAME)))
        except Exception:
            _drop_db(admin, TMP_DB_NAME)
            raise
    finally:
        admin.close()
    print(f"Built {DB_NAME}: {snap['_meta']['n_entries']} entries / "
          f"{snap['_meta']['n_ratios']} element ratios / "
          f"{snap['_meta']['n_elements']} elements "
          f"(snapshot sha256={snap['_meta']['records_sha256']})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-from-api", action="store_true",
        help="Re-fetch all chemical systems from the MP API (requires "
             "MP_API_KEY + requests), save a new snapshot, then build. "
             "Default: build from the pinned snapshot.")
    args = parser.parse_args()

    if args.refresh_from_api:
        snap = refresh_from_api()
    else:
        snap = load_snapshot()
        print(f"Loaded snapshot {SNAPSHOT_PATH.name}: "
              f"{snap['_meta']['n_entries']} entries "
              f"(fetched_at={snap['_meta'].get('fetched_at_utc') or snap['_meta'].get('exported_at_utc', 'unknown')})")
    build_db_from_snapshot(snap)


if __name__ == "__main__":
    main()
