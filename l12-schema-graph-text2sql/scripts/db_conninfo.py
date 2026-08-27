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
