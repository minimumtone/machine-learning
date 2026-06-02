"""Validate generated SQL for safety and schema compliance."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

try:
    import sqlglot
    from sqlglot import exp as sqlglot_exp
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


# ── String-literal-aware helpers ──


def _strip_literals(sql: str) -> str:
    """Replace string literals and comments with placeholders to avoid false positives."""
    # Remove block comments
    out = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    # Remove line comments
    out = re.sub(r"--[^\n]*", " ", out)
    # Replace single-quoted strings with placeholder
    out = re.sub(r"'[^']*'", "'__LIT__'", out)
    return out


# ── AST-based extraction (preferred when sqlglot is available) ──


def _ast_extract_tables(sql: str) -> list[str]:
    """Extract real table names using sqlglot AST, excluding CTE aliases."""
    try:
        parsed = sqlglot.parse(sql, dialect="postgres")
    except Exception:
        return _regex_extract_tables(sql)

    tables: set[str] = set()
    cte_names: set[str] = set()

    for stmt in parsed:
        if stmt is None:
            continue
        for cte in stmt.find_all(sqlglot_exp.CTE):
            alias_node = cte.args.get("alias")
            if alias_node:
                cte_names.add(alias_node.name.lower())
        for tbl in stmt.find_all(sqlglot_exp.Table):
            name = tbl.name.lower()
            if name and name not in cte_names:
                tables.add(name)

    return sorted(tables)


def _ast_extract_columns(sql: str) -> list[str]:
    """Extract table.column references using sqlglot AST."""
    try:
        parsed = sqlglot.parse(sql, dialect="postgres")
    except Exception:
        return _regex_extract_columns(sql)

    cols: set[str] = set()
    for stmt in parsed:
        if stmt is None:
            continue
        for col in stmt.find_all(sqlglot_exp.Column):
            table_node = col.args.get("table")
            col_name = col.name
            if table_node and col_name:
                cols.add(f"{table_node.name}.{col_name}")

    return sorted(cols)


def _ast_check_multiple_statements(sql: str) -> bool:
    """Check for multiple statements via sqlglot parse."""
    try:
        stmts = sqlglot.parse(sql, dialect="postgres")
        real = [s for s in stmts if s is not None]
        return len(real) > 1
    except Exception:
        return _regex_check_multiple_statements(sql)


def _ast_check_forbidden_keywords(sql: str) -> list[str]:
    """Check forbidden keywords via AST node types."""
    try:
        parsed = sqlglot.parse(sql, dialect="postgres")
    except Exception:
        return _regex_check_forbidden_keywords(sql)

    found: list[str] = []
    node_keyword_map = {
        "Insert": "INSERT",
        "Update": "UPDATE",
        "Delete": "DELETE",
        "Drop": "DROP",
        "AlterTable": "ALTER",
        "Create": "CREATE",
    }
    for stmt in parsed:
        if stmt is None:
            continue
        stype = type(stmt).__name__
        if stype in node_keyword_map:
            kw = node_keyword_map[stype]
            if kw not in found:
                found.append(kw)
        # Also check for TRUNCATE, GRANT, REVOKE, COPY in the raw SQL
        # as sqlglot may not have AST nodes for all of these
    # Supplement with literal-aware regex for keywords not covered by AST
    clean = _strip_literals(sql).upper()
    for kw in FORBIDDEN_KEYWORDS:
        if kw not in found and re.search(rf"\b{kw}\b", clean):
            found.append(kw)
    return found


def _ast_check_disallowed_functions(sql: str) -> list[str]:
    """Check for disallowed functions via AST."""
    try:
        parsed = sqlglot.parse(sql, dialect="postgres")
    except Exception:
        return _regex_check_disallowed_functions(sql)

    found: list[str] = []
    for stmt in parsed:
        if stmt is None:
            continue
        for func in stmt.find_all(sqlglot_exp.Anonymous, sqlglot_exp.Func):
            name = getattr(func, "name", "") or type(func).__name__
            name_lower = name.lower()
            for dis in DISALLOWED_FUNCTIONS:
                if dis.lower() == name_lower and dis not in found:
                    found.append(dis)
    return found


def _ast_subquery_depth(sql: str) -> int:
    """Compute max subquery nesting via AST."""
    try:
        parsed = sqlglot.parse(sql, dialect="postgres")
    except Exception:
        return _regex_subquery_depth(sql)

    def _depth(node: sqlglot_exp.Expression, current: int) -> int:
        max_d = current
        for child in node.iter_expressions():
            d = current
            if isinstance(child, sqlglot_exp.Subquery):
                d = current + 1
            max_d = max(max_d, _depth(child, d))
        return max_d

    overall = 0
    for stmt in parsed:
        if stmt is not None:
            overall = max(overall, _depth(stmt, 0))
    return overall


def _ast_check_limit(sql: str) -> bool:
    """Check if outermost statement has a LIMIT clause via AST."""
    try:
        parsed = sqlglot.parse(sql, dialect="postgres")
    except Exception:
        return bool(re.search(r"\bLIMIT\b", sql, re.IGNORECASE))

    for stmt in parsed:
        if stmt is None:
            continue
        limit_node = stmt.find(sqlglot_exp.Limit)
        if limit_node is not None:
            # Verify it's at the top level, not inside a subquery
            parent = limit_node.parent
            while parent is not None:
                if isinstance(parent, sqlglot_exp.Subquery):
                    return False  # LIMIT is inside subquery, not outer
                parent = parent.parent if hasattr(parent, 'parent') else None
            return True
    return False


# ── Regex-based fallbacks ──


def _regex_extract_tables(sql: str) -> list[str]:
    """Regex fallback for table extraction."""
    clean = _strip_literals(sql)
    tables: set[str] = set()
    for m in re.finditer(r"\bFROM\s+(\w+)", clean, re.IGNORECASE):
        tables.add(m.group(1).lower())
    for m in re.finditer(r"\bJOIN\s+(\w+)", clean, re.IGNORECASE):
        tables.add(m.group(1).lower())
    return sorted(tables)


def _regex_extract_columns(sql: str) -> list[str]:
    """Regex fallback for column extraction."""
    clean = _strip_literals(sql)
    cols: set[str] = set()
    for m in re.finditer(r"(\w+)\.(\w+)", clean):
        cols.add(f"{m.group(1)}.{m.group(2)}")
    return sorted(cols)


def _regex_check_multiple_statements(sql: str) -> bool:
    """Regex fallback: check for multiple statements, ignoring string literals."""
    clean = _strip_literals(sql)
    stripped = clean.strip().rstrip(";")
    return ";" in stripped


def _regex_check_forbidden_keywords(sql: str) -> list[str]:
    """Regex fallback for forbidden keywords, ignoring comments/literals."""
    clean = _strip_literals(sql).upper()
    return [kw for kw in FORBIDDEN_KEYWORDS if re.search(rf"\b{kw}\b", clean)]


def _regex_check_disallowed_functions(sql: str) -> list[str]:
    """Regex fallback for disallowed functions."""
    clean = _strip_literals(sql).upper()
    return [fn for fn in DISALLOWED_FUNCTIONS if fn.upper() in clean]


def _regex_subquery_depth(sql: str) -> int:
    """Regex fallback for subquery depth, ignoring string literals."""
    clean = _strip_literals(sql)
    depth = 0
    max_depth = 0
    for char in clean:
        if char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
        max_depth = max(max_depth, depth)
    return max_depth


# ── Public API (dispatch to AST or regex) ──


def check_forbidden_keywords(sql: str) -> list[str]:
    """Return list of forbidden keywords found in the SQL."""
    if HAS_SQLGLOT:
        return _ast_check_forbidden_keywords(sql)
    return _regex_check_forbidden_keywords(sql)


def check_multiple_statements(sql: str) -> bool:
    """Return True if SQL contains multiple statements."""
    if HAS_SQLGLOT:
        return _ast_check_multiple_statements(sql)
    return _regex_check_multiple_statements(sql)


def check_select_only(sql: str) -> bool:
    """Return True if the SQL is a SELECT statement."""
    clean = _strip_literals(sql)
    stripped = clean.strip().upper()
    return stripped.startswith("SELECT") or stripped.startswith("WITH")


def check_limit(sql: str) -> tuple[bool, str]:
    """Check if outermost SQL has a LIMIT clause; if not, append one."""
    if HAS_SQLGLOT:
        has = _ast_check_limit(sql)
    else:
        # Regex: only match LIMIT not inside parentheses (rough heuristic)
        clean = _strip_literals(sql)
        # Remove content inside parentheses to skip subquery LIMITs
        no_parens = re.sub(r"\([^)]*\)", "", clean)
        has = bool(re.search(r"\bLIMIT\b", no_parens, re.IGNORECASE))

    if has:
        return True, sql
    sql_with_limit = sql.rstrip().rstrip(";") + f"\nLIMIT {DEFAULT_LIMIT};"
    return False, sql_with_limit


def extract_tables_from_sql(sql: str) -> list[str]:
    """Extract table names referenced in the SQL."""
    if HAS_SQLGLOT:
        return _ast_extract_tables(sql)
    return _regex_extract_tables(sql)


def extract_columns_from_sql(sql: str) -> list[str]:
    """Extract column references (table.column patterns)."""
    if HAS_SQLGLOT:
        return _ast_extract_columns(sql)
    return _regex_extract_columns(sql)


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
    """Return list of disallowed column references found in SQL."""
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
    # Match JOINs with or without explicit alias
    for m in re.finditer(
        r"JOIN\s+(\w+)(?:\s+(\w+))?\s+ON\s+(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)",
        _strip_literals(sql), re.IGNORECASE,
    ):
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
    if HAS_SQLGLOT:
        return _ast_check_disallowed_functions(sql)
    return _regex_check_disallowed_functions(sql)


def check_tautology(sql: str) -> list[str]:
    """Detect tautological conditions (e.g., OR 1=1, OR true, WHERE 1=1)."""
    warnings: list[str] = []
    clean = _strip_literals(sql).upper()
    patterns = [
        r"\bOR\s+1\s*=\s*1\b",
        r"\bOR\s+TRUE\b",
        r"\bWHERE\s+1\s*=\s*1\b",
        r"\bOR\s+(\d+)\s*=\s*\1\b",
    ]
    for pat in patterns:
        if re.search(pat, clean):
            warnings.append(f"Tautological condition detected")
    return warnings


def check_subquery_depth(sql: str) -> int:
    """Estimate subquery nesting depth."""
    if HAS_SQLGLOT:
        return _ast_subquery_depth(sql)
    return _regex_subquery_depth(sql)


# ── Classification priority ──

_CLASSIFICATION_PRIORITY = {
    "rejected_security": 0,
    "rejected_syntax": 1,
    "rejected_schema": 2,
    "rejected_complexity": 3,
    "modified": 4,
    "accepted": 5,
}


def validate_sql(
    sql: str,
    schema_path: Path | None = None,
) -> dict[str, Any]:
    """Full validation pipeline.

    Classification priority: rejected_security > rejected_syntax >
    rejected_schema > rejected_complexity > modified > accepted.
    """
    errors: list[str] = []
    warnings: list[str] = []
    classification = "accepted"
    schema = _load_allowed_schema(schema_path)
    allowed_tables = schema.get("allowed_tables", [])

    def _escalate(new_cls: str) -> None:
        nonlocal classification
        if _CLASSIFICATION_PRIORITY.get(new_cls, 99) < _CLASSIFICATION_PRIORITY.get(classification, 99):
            classification = new_cls

    if check_multiple_statements(sql):
        errors.append("Multiple SQL statements detected")
        _escalate("rejected_security")

    forbidden = check_forbidden_keywords(sql)
    if forbidden:
        errors.append(f"Forbidden keywords: {', '.join(forbidden)}")
        _escalate("rejected_security")

    if not check_select_only(sql):
        errors.append("Only SELECT statements are allowed")
        _escalate("rejected_security")

    bad_funcs = check_disallowed_functions(sql)
    if bad_funcs:
        errors.append(f"Disallowed functions: {', '.join(bad_funcs)}")
        _escalate("rejected_security")

    tautologies = check_tautology(sql)
    if tautologies:
        errors.append("Tautological condition detected (possible injection)")
        _escalate("rejected_security")

    has_limit, sql = check_limit(sql)
    if not has_limit:
        warnings.append(f"LIMIT clause added (default {DEFAULT_LIMIT})")
        _escalate("modified")

    bad_tables = check_allowed_tables(sql, allowed_tables)
    if bad_tables:
        errors.append(f"Disallowed tables: {', '.join(bad_tables)}")
        _escalate("rejected_schema")

    allowed_columns = schema.get("allowed_columns", [])
    if allowed_columns:
        bad_cols = check_allowed_columns(sql, allowed_columns, schema_path)
        if bad_cols:
            errors.append(f"Disallowed columns: {', '.join(bad_cols)}")
            _escalate("rejected_schema")

    join_warnings = check_join_validity(sql, schema_path)
    for jw in join_warnings:
        warnings.append(jw)

    depth = check_subquery_depth(sql)
    if depth > MAX_SUBQUERY_DEPTH:
        errors.append(f"Subquery depth {depth} exceeds max {MAX_SUBQUERY_DEPTH}")
        _escalate("rejected_complexity")

    if HAS_SQLGLOT:
        try:
            sqlglot.parse(sql, dialect="postgres")
        except Exception as e:
            errors.append(f"SQL parse error: {e}")
            _escalate("rejected_syntax")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "classification": classification,
        "sql": sql,
    }
