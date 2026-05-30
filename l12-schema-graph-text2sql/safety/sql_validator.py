"""Validate generated SQL for safety and schema compliance."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

try:
    import sqlglot
    HAS_SQLGLOT = True
except ImportError:
    HAS_SQLGLOT = False


FORBIDDEN_KEYWORDS = [
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE",
    "CREATE", "GRANT", "REVOKE", "COPY",
]

DEFAULT_LIMIT = 100


def _load_allowed_schema(path: Path | None = None) -> dict[str, Any]:
    if path is None:
        path = Path(__file__).parent / "allowed_schema.yaml"
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_forbidden_keywords(sql: str) -> list[str]:
    """Return list of forbidden keywords found in the SQL."""
    upper = sql.upper()
    return [kw for kw in FORBIDDEN_KEYWORDS if re.search(rf"\b{kw}\b", upper)]


def check_multiple_statements(sql: str) -> bool:
    """Return True if SQL contains multiple statements."""
    stripped = sql.strip().rstrip(";")
    return ";" in stripped


def check_select_only(sql: str) -> bool:
    """Return True if the SQL is a SELECT statement."""
    stripped = sql.strip().upper()
    return stripped.startswith("SELECT") or stripped.startswith("WITH")


def check_limit(sql: str) -> tuple[bool, str]:
    """Check if SQL has a LIMIT clause; if not, append one."""
    if re.search(r"\bLIMIT\b", sql, re.IGNORECASE):
        return True, sql
    sql_with_limit = sql.rstrip().rstrip(";") + f"\nLIMIT {DEFAULT_LIMIT};"
    return False, sql_with_limit


def extract_tables_from_sql(sql: str) -> list[str]:
    """Extract table names referenced in the SQL (basic regex approach)."""
    tables: set[str] = set()
    for m in re.finditer(r"\bFROM\s+(\w+)", sql, re.IGNORECASE):
        tables.add(m.group(1).lower())
    for m in re.finditer(r"\bJOIN\s+(\w+)", sql, re.IGNORECASE):
        tables.add(m.group(1).lower())
    return sorted(tables)


def extract_columns_from_sql(sql: str) -> list[str]:
    """Extract column references (table.column patterns)."""
    cols: set[str] = set()
    for m in re.finditer(r"(\w+)\.(\w+)", sql):
        cols.add(f"{m.group(1)}.{m.group(2)}")
    return sorted(cols)


def check_allowed_tables(
    sql: str,
    allowed_tables: list[str],
) -> list[str]:
    """Return list of disallowed tables found in SQL."""
    used = extract_tables_from_sql(sql)
    allowed_lower = {t.lower() for t in allowed_tables}
    return [t for t in used if t not in allowed_lower]


def validate_sql(
    sql: str,
    schema_path: Path | None = None,
) -> dict[str, Any]:
    """Full validation pipeline. Returns dict with 'valid' and 'errors'."""
    errors: list[str] = []
    warnings: list[str] = []
    schema = _load_allowed_schema(schema_path)
    allowed_tables = schema.get("allowed_tables", [])

    if check_multiple_statements(sql):
        errors.append("Multiple SQL statements detected")

    forbidden = check_forbidden_keywords(sql)
    if forbidden:
        errors.append(f"Forbidden keywords: {', '.join(forbidden)}")

    if not check_select_only(sql):
        errors.append("Only SELECT statements are allowed")

    has_limit, sql = check_limit(sql)
    if not has_limit:
        warnings.append(f"LIMIT clause added (default {DEFAULT_LIMIT})")

    bad_tables = check_allowed_tables(sql, allowed_tables)
    if bad_tables:
        errors.append(f"Disallowed tables: {', '.join(bad_tables)}")

    if HAS_SQLGLOT:
        try:
            sqlglot.parse(sql, dialect="postgres")
        except Exception as e:
            errors.append(f"SQL parse error: {e}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "sql": sql,
    }
