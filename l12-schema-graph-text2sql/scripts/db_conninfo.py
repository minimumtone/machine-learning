"""Environment-driven connection string for the main fixture database.

Uses psycopg's make_conninfo so credentials containing spaces or quotes
are escaped correctly; no LLM-pipeline dependencies.
"""
from __future__ import annotations

import os

from psycopg.conninfo import make_conninfo

CONNINFO = make_conninfo(
    host=os.getenv("POSTGRES_HOST", "localhost"),
    port=os.getenv("POSTGRES_PORT", "5432"),
    dbname=os.getenv("POSTGRES_DB", "l12_materials"),
    user=os.getenv("POSTGRES_USER", "l12_user"),
    password=os.getenv("POSTGRES_PASSWORD", "l12_password"),
)


def main_conninfo() -> str:
    """Connection string for the main fixture DB."""
    return os.getenv("CONNINFO", CONNINFO)


def mp_conninfo(db: str = "mp_transfer") -> str:
    """Connection string for the Materials Project transfer DB."""
    base = os.getenv("CONNINFO", CONNINFO)
    return base.replace(
        f"dbname={os.getenv('POSTGRES_DB', 'l12_materials')}", f"dbname={db}")
