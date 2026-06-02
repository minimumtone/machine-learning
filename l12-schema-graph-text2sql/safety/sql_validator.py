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

DEFAULT_LIMIT = 10000
MAX_SUBQUERY_DEPTH = 3
DISALLOWED_FUNCTIONS = [
    "pg_sleep", "dblink", "lo_import", "lo_export",
    "pg_read_file", "pg_ls_dir", "pg_stat_file",
]


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


def check_allowed_columns(
    sql: str,
    allowed_columns: list[str] | None = None,
    schema_path: Path | None = None,
) -> list[str]:
    """Return list of disallowed column references (alias.column) found in SQL.

    If allowed_columns is None, loads from allowed_schema.yaml.
    """
    schema = _load_allowed_schema(schema_path)
    if allowed_columns is None:
        allowed_columns = schema.get("allowed_columns", [])
    if not allowed_columns:
        return []
    used = extract_columns_from_sql(sql)
    allowed_lower = {c.lower() for c in allowed_columns}
    alias_to_table = {
        "m": "material_entry", "c": "composition", "s": "structure",
        "ps": "phase_stability", "calc": "calculation", "cp": "calculated_property",
    }
    disallowed: list[str] = []
    for col_ref in used:
        parts = col_ref.split(".")
        if len(parts) != 2:
            continue
        alias, col = parts
        table = alias_to_table.get(alias.lower(), alias.lower())
        canonical = f"{table}.{col}".lower()
        if canonical not in allowed_lower:
            disallowed.append(col_ref)
    return disallowed


def check_join_validity(
    sql: str,
    schema_path: Path | None = None,
) -> list[str]:
    """Check that JOINs use valid FK relationships."""
    schema = _load_allowed_schema(schema_path)
    allowed_joins = schema.get("allowed_joins", [])
    valid_pairs: set[tuple[str, str, str, str]] = set()
    for j in allowed_joins:
        valid_pairs.add((
            j["source_table"].lower(),
            j["source_column"].lower(),
            j["target_table"].lower(),
            j["target_column"].lower(),
        ))
        valid_pairs.add((
            j["target_table"].lower(),
            j["target_column"].lower(),
            j["source_table"].lower(),
            j["source_column"].lower(),
        ))
    alias_to_table = {
        "m": "material_entry", "c": "composition", "s": "structure",
        "ps": "phase_stability", "calc": "calculation", "cp": "calculated_property",
    }
    warnings: list[str] = []
    for m in re.finditer(
        r"JOIN\s+(\w+)\s+(\w+)\s+ON\s+(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)",
        sql, re.IGNORECASE,
    ):
        table = m.group(1).lower()
        alias1 = m.group(3).lower()
        col1 = m.group(4).lower()
        alias2 = m.group(5).lower()
        col2 = m.group(6).lower()
        t1 = alias_to_table.get(alias1, alias1)
        t2 = alias_to_table.get(alias2, alias2)
        if (t1, col1, t2, col2) not in valid_pairs:
            warnings.append(f"Non-FK JOIN: {t1}.{col1} = {t2}.{col2}")
    return warnings


def check_disallowed_functions(sql: str) -> list[str]:
    """Check for dangerous SQL functions."""
    upper = sql.upper()
    return [fn for fn in DISALLOWED_FUNCTIONS if fn.upper() in upper]


def check_subquery_depth(sql: str) -> int:
    """Estimate subquery nesting depth by counting nested SELECT keywords."""
    depth = 0
    max_depth = 0
    for char in sql:
        if char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
        max_depth = max(max_depth, depth)
    return max_depth


def validate_sql(
    sql: str,
    schema_path: Path | None = None,
) -> dict[str, Any]:
    """Full validation pipeline. Returns dict with 'valid', 'errors', 'warnings', and 'classification'.

    Classification values:
      accepted, modified, rejected_syntax, rejected_security,
      rejected_schema, rejected_join, rejected_complexity,
      clarification_required
    """
    errors: list[str] = []
    warnings: list[str] = []
    classification = "accepted"
    schema = _load_allowed_schema(schema_path)
    allowed_tables = schema.get("allowed_tables", [])

    if check_multiple_statements(sql):
        errors.append("Multiple SQL statements detected")
        classification = "rejected_security"

    forbidden = check_forbidden_keywords(sql)
    if forbidden:
        errors.append(f"Forbidden keywords: {', '.join(forbidden)}")
        classification = "rejected_security"

    if not check_select_only(sql):
        errors.append("Only SELECT statements are allowed")
        classification = "rejected_security"

    bad_funcs = check_disallowed_functions(sql)
    if bad_funcs:
        errors.append(f"Disallowed functions: {', '.join(bad_funcs)}")
        classification = "rejected_security"

    has_limit, sql = check_limit(sql)
    if not has_limit:
        warnings.append(f"LIMIT clause added (default {DEFAULT_LIMIT})")
        if classification == "accepted":
            classification = "modified"

    bad_tables = check_allowed_tables(sql, allowed_tables)
    if bad_tables:
        errors.append(f"Disallowed tables: {', '.join(bad_tables)}")
        classification = "rejected_schema"

    # Column whitelist check (if defined in schema)
    allowed_columns = schema.get("allowed_columns", [])
    if allowed_columns:
        bad_cols = check_allowed_columns(sql, allowed_columns, schema_path)
        if bad_cols:
            errors.append(f"Disallowed columns: {', '.join(bad_cols)}")
            classification = "rejected_schema"

    # JOIN validity check
    join_warnings = check_join_validity(sql, schema_path)
    if join_warnings:
        for jw in join_warnings:
            warnings.append(jw)

    # Subquery depth check
    depth = check_subquery_depth(sql)
    if depth > MAX_SUBQUERY_DEPTH:
        errors.append(f"Subquery depth {depth} exceeds max {MAX_SUBQUERY_DEPTH}")
        classification = "rejected_complexity"

    if HAS_SQLGLOT:
        try:
            sqlglot.parse(sql, dialect="postgres")
        except Exception as e:
            errors.append(f"SQL parse error: {e}")
            if classification == "accepted":
                classification = "rejected_syntax"

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "classification": classification,
        "sql": sql,
    }
