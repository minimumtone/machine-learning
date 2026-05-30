"""Load seed CSV files into PostgreSQL."""
from __future__ import annotations

import csv
import os
from pathlib import Path

import psycopg


def get_connection_string() -> str:
    user = os.getenv("POSTGRES_USER", "l12_user")
    password = os.getenv("POSTGRES_PASSWORD", "l12_password")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "l12_materials")
    return f"host={host} port={port} dbname={db} user={user} password={password}"


LOAD_ORDER = [
    ("material_entry", "seed_l12_entries.csv"),
    ("composition", "seed_composition.csv"),
    ("structure", "seed_structure.csv"),
    ("calculation", "seed_calculation.csv"),
    ("calculated_property", "seed_properties.csv"),
    ("phase_stability", "seed_phase_stability.csv"),
    ("prototype_definition", "seed_prototype_definition.csv"),
]


def load_csv_into_table(
    conn: psycopg.Connection,
    table: str,
    csv_path: Path,
) -> int:
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    if not rows:
        return 0
    columns = list(rows[0].keys())
    placeholders = ", ".join(["%s"] * len(columns))
    col_names = ", ".join(columns)
    sql = f"INSERT INTO {table} ({col_names}) VALUES ({placeholders}) ON CONFLICT DO NOTHING"
    count = 0
    with conn.cursor() as cur:
        for row in rows:
            values = [row[c] for c in columns]
            cur.execute(sql, values)
            count += cur.rowcount
    conn.commit()
    return count


def load_all(seed_dir: Path | None = None) -> dict[str, int]:
    if seed_dir is None:
        seed_dir = Path(__file__).resolve().parent.parent / "db" / "seed"
    results: dict[str, int] = {}
    conn = psycopg.connect(get_connection_string())
    try:
        for table, filename in LOAD_ORDER:
            csv_path = seed_dir / filename
            if not csv_path.exists():
                print(f"  SKIP {filename} (not found)")
                continue
            n = load_csv_into_table(conn, table, csv_path)
            results[table] = n
            print(f"  {table}: {n} rows inserted")
    finally:
        conn.close()
    return results


if __name__ == "__main__":
    print("Loading seed data ...")
    load_all()
    print("Done.")
