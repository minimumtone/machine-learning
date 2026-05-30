"""Execute validated SQL safely against PostgreSQL."""
from __future__ import annotations

import os
import time
from typing import Any

import psycopg

from safety.sql_validator import validate_sql


def get_readonly_connection_string() -> str:
    user = os.getenv("POSTGRES_USER", "l12_user")
    password = os.getenv("POSTGRES_PASSWORD", "l12_password")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "l12_materials")
    return f"host={host} port={port} dbname={db} user={user} password={password}"


def execute_sql(
    sql: str,
    timeout_seconds: int | None = None,
    validate: bool = True,
) -> dict[str, Any]:
    """Validate and execute SQL, returning results or errors."""
    if timeout_seconds is None:
        timeout_seconds = int(os.getenv("SQL_TIMEOUT_SECONDS", "10"))

    if validate:
        validation = validate_sql(sql)
        if not validation["valid"]:
            return {
                "success": False,
                "errors": validation["errors"],
                "rows": [],
                "columns": [],
            }
        sql = validation["sql"]

    conninfo = get_readonly_connection_string()
    t0 = time.time()
    try:
        conn = psycopg.connect(conninfo)
        try:
            with conn.cursor() as cur:
                cur.execute(f"SET statement_timeout = '{timeout_seconds * 1000}'")
                cur.execute(sql)
                columns = [desc[0] for desc in cur.description] if cur.description else []
                rows = cur.fetchall()
            latency_ms = int((time.time() - t0) * 1000)
            return {
                "success": True,
                "rows": [list(r) for r in rows],
                "columns": columns,
                "row_count": len(rows),
                "latency_ms": latency_ms,
            }
        finally:
            conn.close()
    except Exception as e:
        return {
            "success": False,
            "errors": [str(e)],
            "rows": [],
            "columns": [],
        }
