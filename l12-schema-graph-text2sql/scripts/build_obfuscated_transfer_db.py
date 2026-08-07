#!/usr/bin/env python3
"""Build an obfuscated copy of the transfer database.

Renames table names and all column names to random English identifiers.
Columns that share the same old name across tables keep the same new name,
so existing SQL queries can be translated by a simple identifier substitution.
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterator

import psycopg

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from scripts.eval_ablation import CONNINFO  # noqa: E402

SRC_DB = os.getenv("TRANSFER_DB", "oqmd_transfer")
OBF_DB = f"{SRC_DB}_obfuscated"
MAPPING_PATH = PROJECT / "db" / "obfuscated_transfer_mapping.json"

WORDS = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
    "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
    "quebec", "romeo", "sierra", "tango", "uniform", "victor", "whiskey",
    "xray", "yankee", "zulu", "apollo", "baker", "calypso", "draco", "eagle",
    "falcon", "gemini", "halo", "iris", "juno", "kronos", "luna", "mars",
    "nova", "orion", "pegasus", "quasar", "rhea", "solar", "terra", "umbra",
    "vega", "wolf", "xenon", "yeti", "zenith",
]


def _name_stream(prefix: str, rng: random.Random) -> Iterator[str]:
    """Yield unique prefixed words in random order, repeating with suffixes."""
    words = WORDS.copy()
    rng.shuffle(words)
    suffix = 0
    while True:
        for w in words:
            suf = f"_{suffix}" if suffix else ""
            yield f"{prefix}_{w}{suf}"
        suffix += 1


def main() -> None:
    seed = int(os.getenv("OBFUSCATE_SEED", "42"))
    rng = random.Random(seed)

    print("Terminating existing transfer connections...")
    admin = psycopg.connect(CONNINFO, autocommit=True)
    with admin.cursor() as cur:
        cur.execute("""
            SELECT pg_terminate_backend(pid)
            FROM pg_stat_activity
            WHERE datname IN (%s, %s, 'l12_materials')
              AND pid <> pg_backend_pid()
        """, (SRC_DB, OBF_DB))
        print("Dropped existing obfuscated DB (if any)...")
        cur.execute(f'DROP DATABASE IF EXISTS "{OBF_DB}"')  # type: ignore[arg-type]
        print(f"Creating obfuscated DB from template {SRC_DB}...")
        cur.execute(f'CREATE DATABASE "{OBF_DB}" WITH TEMPLATE "{SRC_DB}"')  # type: ignore[arg-type]
    admin.close()

    obf_conninfo = (
        f"host={os.getenv('POSTGRES_HOST', 'localhost')} "
        f"port={os.getenv('POSTGRES_PORT', '5432')} "
        f"dbname={OBF_DB} "
        f"user={os.getenv('POSTGRES_USER', 'l12_user')} "
        f"password={os.getenv('POSTGRES_PASSWORD', 'l12_password')}"
    )
    print(f"Connecting to {OBF_DB}...")
    conn = psycopg.connect(obf_conninfo)
    with conn.cursor() as cur:
        print("Fetching table list...")
        cur.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_type = 'BASE TABLE'
            ORDER BY table_name
        """)
        tables = [r[0] for r in cur.fetchall()]

    tbl_stream = _name_stream("tbl", rng)
    table_map = {old_t: next(tbl_stream) for old_t in tables}
    col_stream = _name_stream("col", rng)

    mapping: dict[str, Any] = {"seed": seed, "tables": table_map, "columns": {}}
    global_col_map: dict[str, str] = {}

    for i, old_t in enumerate(tables):
        new_t = table_map[old_t]
        print(f"Obfuscating table {i+1}/{len(tables)}: {old_t} -> {new_t}")
        with conn.cursor() as cur:
            cur.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = %s
                ORDER BY ordinal_position
            """, (old_t,))
            cols = [r[0] for r in cur.fetchall()]
        col_map: dict[str, str] = {}
        for old_c in cols:
            if old_c not in global_col_map:
                global_col_map[old_c] = next(col_stream)
            new_c = global_col_map[old_c]
            col_map[old_c] = new_c
            with conn.cursor() as cur:
                cur.execute(
                    f'ALTER TABLE "{old_t}" RENAME COLUMN "{old_c}" TO "{new_c}"'  # type: ignore[arg-type]
                )
        mapping["columns"][old_t] = {"new_table": new_t, "columns": col_map}
        with conn.cursor() as cur:
            cur.execute(f'ALTER TABLE "{old_t}" RENAME TO "{new_t}"')  # type: ignore[arg-type]
    print("Committing schema renames...")
    conn.commit()
    conn.close()

    MAPPING_PATH.write_text(json.dumps(mapping, ensure_ascii=False, indent=2))
    print(f"Obfuscated DB {OBF_DB} built.")
    print(f"Mapping saved to {MAPPING_PATH}")


if __name__ == "__main__":
    main()
