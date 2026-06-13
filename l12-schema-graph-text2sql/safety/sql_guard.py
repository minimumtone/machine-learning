"""Execute validated SQL safely against PostgreSQL.

Includes empty-result diagnosis to distinguish:
  - Case A: query ran correctly but no rows matched the filter conditions
  - Case B: referenced entities (elements, prototypes, etc.) do not exist in the DB
"""
from __future__ import annotations

import os
import re
import time
import typing
from typing import Any

from safety.sql_validator import validate_sql

if typing.TYPE_CHECKING:
    import psycopg


def get_readonly_connection_string() -> str:
    user = os.getenv("POSTGRES_USER", "l12_user")
    password = os.getenv("POSTGRES_PASSWORD", "l12_password")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "l12_materials")
    return f"host={host} port={port} dbname={db} user={user} password={password}"


# ── Empty-result diagnosis ────────────────────────────────────────────


# Patterns to extract literal values from WHERE clauses
_ELEMENT_PAT = re.compile(
    r"(?:c\w*\.element\s*=\s*'([^']+)')", re.IGNORECASE,
)
_PROTOTYPE_PAT = re.compile(
    r"(?:s\w*\.(?:prototype|strukturbericht)\s*=\s*'([^']+)')", re.IGNORECASE,
)
_FORMULA_PAT = re.compile(
    r"(?:m\w*\.(?:formula|reduced_formula)\s*=\s*'([^']+)')", re.IGNORECASE,
)


def _extract_referenced_entities(sql: str) -> dict[str, list[str]]:
    """Extract element/prototype/formula literal values from a SQL string."""
    entities: dict[str, list[str]] = {
        "elements": [],
        "prototypes": [],
        "formulas": [],
    }
    for m in _ELEMENT_PAT.finditer(sql):
        val = m.group(1)
        if val not in entities["elements"]:
            entities["elements"].append(val)
    for m in _PROTOTYPE_PAT.finditer(sql):
        val = m.group(1)
        if val not in entities["prototypes"]:
            entities["prototypes"].append(val)
    for m in _FORMULA_PAT.finditer(sql):
        val = m.group(1)
        if val not in entities["formulas"]:
            entities["formulas"].append(val)
    return entities


def diagnose_empty_result(
    sql: str,
    conn: "psycopg.Connection",
) -> dict[str, Any]:
    """When a query returns 0 rows, check whether referenced entities exist in the DB.

    Returns a dict with:
      - diagnosis: "no_matching_data" | "entity_not_found"
      - message: human-readable explanation (Japanese)
      - missing_entities: list of {type, value} for entities not found in DB
      - existing_entities: list of {type, value} for entities confirmed in DB
    """
    entities = _extract_referenced_entities(sql)
    missing: list[dict[str, str]] = []
    existing: list[dict[str, str]] = []

    checks: list[tuple[str, str, str, str]] = []
    # (entity_type, value, check_sql, label)
    for elem in entities["elements"]:
        checks.append((
            "element", elem,
            "SELECT 1 FROM composition WHERE element = %s LIMIT 1",
            f"元素 '{elem}'",
        ))
    for proto in entities["prototypes"]:
        checks.append((
            "prototype", proto,
            "SELECT 1 FROM structure WHERE prototype = %s OR strukturbericht = %s LIMIT 1",
            f"プロトタイプ '{proto}'",
        ))
    for formula in entities["formulas"]:
        checks.append((
            "formula", formula,
            "SELECT 1 FROM material_entry WHERE formula = %s OR reduced_formula = %s LIMIT 1",
            f"化学式 '{formula}'",
        ))

    try:
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout = '5s'")
            for etype, value, check_sql, label in checks:
                params = (value,) if check_sql.count("%s") == 1 else (value, value)
                cur.execute(check_sql, params)
                row = cur.fetchone()
                entry = {"type": etype, "value": value, "label": label}
                if row:
                    existing.append(entry)
                else:
                    missing.append(entry)
    except Exception as e:
        return {"diagnosis": "db_error", "message": f"診断中にDBエラー: {e}", "suggestion": ""}

    if missing:
        names = "、".join(e["label"] for e in missing)
        return {
            "diagnosis": "entity_not_found",
            "message": (
                f"指定された{names}はデータベースに登録されていません。"
                "そのため該当するエントリは存在しません。"
            ),
            "missing_entities": missing,
            "existing_entities": existing,
        }

    return {
        "diagnosis": "no_matching_data",
        "message": (
            "検索条件に合致するエントリはありませんでした。"
            "指定された条件（元素・プロトタイプ等）はデータベースに存在しますが、"
            "すべての条件を同時に満たすエントリが見つかりませんでした。"
        ),
        "missing_entities": [],
        "existing_entities": existing,
    }


# ── SQL execution ─────────────────────────────────────────────────────


def execute_sql(
    sql: str,
    timeout_seconds: int | None = None,
    validate: bool = True,
    diagnose_if_empty: bool = True,
) -> dict[str, Any]:
    """Validate and execute SQL, returning results or errors.

    When *diagnose_if_empty* is True (default) and the result set is empty,
    an additional diagnosis is performed to distinguish:
      - "entity_not_found": a referenced element/prototype/formula does not
        exist in the database at all.
      - "no_matching_data": all referenced entities exist but the combination
        of filter conditions yielded no rows.
    The diagnosis is returned under the ``empty_diagnosis`` key.
    """
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
        import psycopg
        conn = psycopg.connect(conninfo)
        try:
            with conn.cursor() as cur:
                cur.execute("SET TRANSACTION READ ONLY")
                cur.execute(f"SET statement_timeout = '{timeout_seconds * 1000}'")
                cur.execute(sql)
                columns = [desc[0] for desc in cur.description] if cur.description else []
                rows = cur.fetchall()
            latency_ms = int((time.time() - t0) * 1000)

            result: dict[str, Any] = {
                "success": True,
                "rows": [list(r) for r in rows],
                "columns": columns,
                "row_count": len(rows),
                "latency_ms": latency_ms,
            }

            if len(rows) == 0 and diagnose_if_empty:
                result["empty_diagnosis"] = diagnose_empty_result(sql, conn)

            return result
        finally:
            conn.close()
    except Exception as e:
        return {
            "success": False,
            "errors": [str(e)],
            "rows": [],
            "columns": [],
        }
