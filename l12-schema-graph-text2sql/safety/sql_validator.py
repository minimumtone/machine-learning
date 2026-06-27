"""Validate generated SQL for safety and schema compliance."""
from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    import sqlglot
    from sqlglot import expressions as sqlglot_exp
    HAS_SQLGLOT = True
else:
    try:
        import sqlglot
        from sqlglot import exp as sqlglot_exp
        HAS_SQLGLOT = True
    except ImportError:
        sqlglot = None
        sqlglot_exp = None
        HAS_SQLGLOT = False

logger = logging.getLogger(__name__)

FORBIDDEN_KEYWORDS = [
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE",
    "CREATE", "GRANT", "REVOKE", "COPY",
]

DEFAULT_LIMIT = 10000
MAX_SUBQUERY_DEPTH = 3
DISALLOWED_FUNCTIONS = [
    "pg_sleep", "dblink", "lo_import", "lo_export",
    "pg_read_file", "pg_ls_dir", "pg_stat_file",
    "pg_read_binary_file", "pg_stat_statements",
    "query_to_xml", "query_to_json",
]

# System/catalog tables that should never be queried by generated SQL
SYSTEM_TABLE_PREFIXES = [
    "pg_",       # pg_shadow, pg_authid, pg_roles, pg_stat_*, etc.
    "information_schema.",
]
SYSTEM_TABLES = [
    "pg_shadow", "pg_authid", "pg_roles", "pg_user",
    "pg_stat_activity", "pg_stat_statements",
    "pg_catalog", "pg_class", "pg_namespace",
    "pg_proc", "pg_settings",
]

# Column type registry for type-safety checks
# Maps table.column -> expected type category
COLUMN_TYPE_REGISTRY: dict[str, str] = {
    # numeric columns
    "structure.lattice_a": "numeric",
    "structure.lattice_b": "numeric",
    "structure.lattice_c": "numeric",
    "phase_stability.energy_above_hull": "numeric",
    "phase_stability.formation_energy_per_atom": "numeric",
    "calculated_property.value": "numeric",
    "composition.atomic_fraction": "numeric",
    "composition.weight_fraction": "numeric",
    # text columns
    "material_entry.formula": "text",
    "material_entry.reduced_formula": "text",
    "composition.element": "text",
    "structure.prototype": "text",
    "structure.strukturbericht": "text",
    "structure.space_group": "text",
    "calculation.method": "text",
    "calculated_property.property_name": "text",
    "calculated_property.unit": "text",
}


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
    assert sqlglot is not None and sqlglot_exp is not None
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
    assert sqlglot is not None and sqlglot_exp is not None
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
    assert sqlglot is not None
    try:
        stmts = sqlglot.parse(sql, dialect="postgres")
        real = [s for s in stmts if s is not None]
        return len(real) > 1
    except Exception:
        return _regex_check_multiple_statements(sql)


def _ast_check_forbidden_keywords(sql: str) -> list[str]:
    """Check forbidden keywords via AST node types."""
    assert sqlglot is not None
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
    assert sqlglot is not None and sqlglot_exp is not None
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
    assert sqlglot is not None and sqlglot_exp is not None
    try:
        parsed = sqlglot.parse(sql, dialect="postgres")
    except Exception:
        return _regex_subquery_depth(sql)

    def _depth(node: Any, current: int) -> int:
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
    assert sqlglot is not None and sqlglot_exp is not None
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
                parent = getattr(parent, "parent", None)
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
        lhs, rhs = m.group(1), m.group(2)
        # Skip numeric literals (e.g. 180.0, 3.57)
        if lhs.isdigit() or rhs.isdigit():
            continue
        cols.add(f"{lhs}.{rhs}")
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


def check_system_tables(sql: str) -> list[str]:
    """Detect references to system/catalog tables (pg_shadow, pg_authid, etc.).

    These tables may expose sensitive metadata even with read-only access.
    """
    used = extract_tables_from_sql(sql)
    violations: list[str] = []
    for table in used:
        tl = table.lower()
        if tl in SYSTEM_TABLES:
            violations.append(table)
            continue
        for prefix in SYSTEM_TABLE_PREFIXES:
            if tl.startswith(prefix):
                violations.append(table)
                break
    return violations


def check_cte_bodies_select_only(sql: str) -> list[str]:
    """Verify that all CTE bodies are pure SELECT statements.

    PostgreSQL supports writable CTEs (INSERT/UPDATE/DELETE inside WITH).
    This check ensures no CTE body contains data-modification statements.
    """
    if not HAS_SQLGLOT or sqlglot is None or sqlglot_exp is None:
        return _regex_check_cte_bodies(sql)

    try:
        parsed = sqlglot.parse(sql, dialect="postgres")
    except Exception:
        return _regex_check_cte_bodies(sql)

    violations: list[str] = []
    dml_types = {"Insert", "Update", "Delete"}

    for stmt in parsed:
        if stmt is None:
            continue
        for cte in stmt.find_all(sqlglot_exp.CTE):
            alias_node = cte.args.get("alias")
            cte_name = alias_node.name if alias_node else "<unnamed>"
            # The CTE body is the 'this' expression
            body = cte.args.get("this")
            if body is None:
                continue
            body_type = type(body).__name__
            if body_type in dml_types:
                violations.append(
                    f"CTE '{cte_name}' contains {body_type.upper()} (writable CTE)"
                )
            # Also check for DML nested inside the CTE body
            for node in body.walk():
                actual_node = node[0] if isinstance(node, tuple) else node
                if type(actual_node).__name__ in dml_types:
                    if f"CTE '{cte_name}'" not in " ".join(violations):
                        violations.append(
                            f"CTE '{cte_name}' contains nested {type(actual_node).__name__.upper()}"
                        )
                    break
    return violations


def _regex_check_cte_bodies(sql: str) -> list[str]:
    """Regex fallback for CTE body DML detection."""
    violations: list[str] = []
    clean = _strip_literals(sql)
    # Find WITH ... AS (...) patterns
    cte_pattern = re.compile(
        r"\b(\w+)\s+AS\s*\(", re.IGNORECASE
    )
    for m in cte_pattern.finditer(clean):
        cte_name = m.group(1)
        # Skip if it's not after WITH or a comma
        start = m.start()
        prefix = clean[:start].rstrip()
        if not (prefix.upper().endswith("WITH") or prefix.endswith(",")):
            continue
        # Extract the CTE body (find matching parenthesis)
        paren_start = m.end() - 1
        depth = 0
        body_end = paren_start
        for i in range(paren_start, len(clean)):
            if clean[i] == '(':
                depth += 1
            elif clean[i] == ')':
                depth -= 1
                if depth == 0:
                    body_end = i
                    break
        body = clean[paren_start + 1:body_end].upper()
        for kw in ["INSERT", "UPDATE", "DELETE"]:
            if re.search(rf"\b{kw}\b", body):
                violations.append(
                    f"CTE '{cte_name}' contains {kw} (writable CTE)"
                )
                break
    return violations


def check_column_type_safety(
    sql: str,
    type_registry: dict[str, str] | None = None,
) -> list[str]:
    """Detect type mismatches in WHERE conditions.

    Catches cases like:
      WHERE energy_above_hull = 'stable'  (string compared to numeric column)
      WHERE element > 100                  (numeric compared to text column)
    """
    if type_registry is None:
        type_registry = COLUMN_TYPE_REGISTRY

    warnings: list[str] = []
    alias_to_table = {
        "m": "material_entry", "c": "composition", "s": "structure",
        "ps": "phase_stability", "calc": "calculation", "cp": "calculated_property",
        "pd": "prototype_definition", "et": "elastic_tensor",
        "tp": "thermal_property", "mp": "magnetic_property",
        "se": "surface_energy", "gb": "grain_boundary",
        "bs": "band_structure", "dos": "density_of_states",
        "e": "element", "ep": "element_property",
        "md": "material_defect", "dt": "defect_type",
        "ms": "material_synthesis", "sm": "synthesis_method",
        "lr": "literature_reference", "mr": "material_reference",
        "ad": "application_domain", "ma": "material_application",
        "em": "experimental_measurement", "mpr": "measured_property",
        "pde": "phase_diagram_entry", "als": "alloy_system",
        "mas": "material_alloy_system", "sg": "space_group",
    }
    # Dynamic aliases override static ones (e.g. cp_bm, c_ni)
    alias_to_table.update(_extract_aliases_from_sql(sql))

    # Pattern: alias.column op 'string_value'
    string_compare = re.compile(
        r"(\w+)\.(\w+)\s*(?:=|!=|<>|<|>|<=|>=)\s*'([^']*)'", re.IGNORECASE
    )
    # Pattern: alias.column op numeric_value
    numeric_compare = re.compile(
        r"(\w+)\.(\w+)\s*(?:=|!=|<>|<|>|<=|>=)\s*(-?\d+\.?\d*)", re.IGNORECASE
    )

    clean = _strip_literals(sql)
    # For string comparisons, we need original SQL (with literals)
    for m in string_compare.finditer(sql):
        alias = m.group(1).lower()
        col = m.group(2).lower()
        table = alias_to_table.get(alias, alias)
        canonical = f"{table}.{col}"
        expected_type = type_registry.get(canonical)
        if expected_type == "numeric":
            warnings.append(
                f"Type mismatch: {canonical} is numeric but compared with string '{m.group(3)}'"
            )

    for m in numeric_compare.finditer(clean):
        alias = m.group(1).lower()
        col = m.group(2).lower()
        table = alias_to_table.get(alias, alias)
        canonical = f"{table}.{col}"
        expected_type = type_registry.get(canonical)
        if expected_type == "text":
            warnings.append(
                f"Type mismatch: {canonical} is text but compared with number {m.group(3)}"
            )

    return warnings


@lru_cache(maxsize=1)
def _load_column_synonyms() -> dict[str, str]:
    """Load column synonym mappings from material_terms.yaml."""
    path = Path(__file__).parent.parent / "llm" / "material_terms.yaml"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        return {}
    raw = data.get("column_synonyms")
    if not isinstance(raw, dict):
        return {}
    return {k.lower(): v.lower() for k, v in raw.items()}


def suggest_column_correction(
    bad_column: str,
    allowed_columns: list[str],
) -> str | None:
    """Suggest the correct column for a hallucinated column reference.

    Checks column_synonyms first, then falls back to fuzzy matching
    (same table, shortest edit distance).
    """
    synonyms = _load_column_synonyms()
    bad_lower = bad_column.lower()
    allowed_lower = {c.lower() for c in allowed_columns}

    # Check synonym mapping (only if target is in allowed_columns)
    if bad_lower in synonyms:
        target = synonyms[bad_lower]
        if target in allowed_lower:
            return target
        # Cross-table synonym: target table not in current context
        # Fall through to fuzzy matching within same table

    # Fuzzy match: find closest column in the same table
    parts = bad_lower.split(".")
    if len(parts) != 2:
        return None
    table, col = parts
    same_table = [c for c in allowed_columns if c.lower().startswith(f"{table}.")]
    if not same_table:
        return None

    # Simple edit distance for short strings
    def _edit_distance(a: str, b: str) -> int:
        if len(a) < len(b):
            a, b = b, a
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            curr = [i] + [0] * len(b)
            for j, cb in enumerate(b, 1):
                curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb))
            prev = curr
        return prev[-1]

    best_col = None
    best_dist = 999
    for candidate in same_table:
        cand_col = candidate.lower().split(".", 1)[1]
        dist = _edit_distance(col, cand_col)
        if dist < best_dist:
            best_dist = dist
            best_col = candidate
    # Only suggest if reasonably close (edit distance <= half the column name length)
    if best_col and best_dist <= max(3, len(col) // 2):
        return best_col.lower()
    return None


def _extract_unqualified_columns(sql: str) -> list[str]:
    """Extract unqualified column names from SELECT/WHERE clauses only.

    Uses sqlglot AST when available; returns column names that have no
    table qualifier (e.g. ``SELECT col1 FROM t`` yields ``["col1"]``).
    Excludes SELECT aliases (used in ORDER BY/HAVING) and CTE names.
    """
    if HAS_SQLGLOT:
        try:
            parsed = sqlglot.parse(sql, dialect="postgres")
            cols: set[str] = set()
            select_aliases: set[str] = set()
            cte_names: set[str] = set()
            for stmt in parsed:
                if stmt is None:
                    continue
                # Collect CTE names
                for cte in stmt.find_all(sqlglot_exp.CTE):
                    alias_node = cte.args.get("alias")
                    if alias_node:
                        cte_names.add(alias_node.name.lower())
                # Collect SELECT aliases (e.g. COUNT(*) AS cnt → "cnt")
                for alias in stmt.find_all(sqlglot_exp.Alias):
                    select_aliases.add(alias.alias.lower())
                # Collect unqualified columns from WHERE clauses only
                for where in stmt.find_all(sqlglot_exp.Where):
                    for col in where.find_all(sqlglot_exp.Column):
                        table_node = col.args.get("table")
                        if not table_node and col.name:
                            cols.add(col.name)
                # Collect from SELECT expressions (but not from ORDER BY)
                select_node = stmt.find(sqlglot_exp.Select)
                if select_node:
                    for expr in (select_node.expressions or []):
                        for col in expr.find_all(sqlglot_exp.Column):
                            table_node = col.args.get("table")
                            if not table_node and col.name:
                                cols.add(col.name)
            # Remove SELECT aliases and CTE names
            skip = select_aliases | cte_names | {
                "count", "sum", "avg", "min", "max", "coalesce",
                "case", "when", "then", "else", "end", "as",
                "true", "false", "null", "distinct",
            }
            return sorted(c for c in cols if c.lower() not in skip)
        except Exception:
            # sqlglot parse failure; fall back to empty column list
            logger.debug("sqlglot failed to parse SQL for column extraction")
    return []


def _get_from_tables(sql: str) -> list[str]:
    """Extract actual table names from FROM/JOIN clauses (not aliases).

    Excludes CTE names so that ``WITH data AS (...) SELECT ... FROM data``
    doesn't treat ``data`` as a real table for column validation.
    """
    alias_map = _extract_aliases_from_sql(sql)
    tables = {v for v in alias_map.values()}
    # Remove CTE names
    if HAS_SQLGLOT:
        try:
            parsed = sqlglot.parse(sql, dialect="postgres")
            for stmt in parsed:
                if stmt is None:
                    continue
                for cte in stmt.find_all(sqlglot_exp.CTE):
                    alias_node = cte.args.get("alias")
                    if alias_node:
                        tables.discard(alias_node.name.lower())
        except Exception:
            # sqlglot parse failure; proceed with alias-based table set
            logger.debug("sqlglot failed to parse SQL for CTE name extraction")
    return sorted(tables)


def check_allowed_columns(
    sql: str,
    allowed_columns: list[str] | None = None,
    schema_path: Path | None = None,
) -> list[str]:
    """Return list of disallowed column references found in SQL.

    Each entry includes a correction suggestion if available, e.g.:
    "c.fractional_amount (did you mean: composition.atomic_fraction?)"

    Also checks unqualified columns when the query has a single FROM table.
    """
    if allowed_columns is None:
        schema = _load_allowed_schema(schema_path)
        allowed_columns = schema.get("allowed_columns", [])
    if not allowed_columns:
        return []
    used = extract_columns_from_sql(sql)
    allowed_lower = {c.lower() for c in allowed_columns}
    # Static well-known aliases
    alias_to_table = {
        "m": "material_entry", "c": "composition", "s": "structure",
        "ps": "phase_stability", "calc": "calculation", "cp": "calculated_property",
        "pd": "prototype_definition", "et": "elastic_tensor",
        "tp": "thermal_property", "mp": "magnetic_property",
        "se": "surface_energy", "gb": "grain_boundary",
        "bs": "band_structure", "dos": "density_of_states",
        "e": "element", "ep": "element_property",
        "md": "material_defect", "dt": "defect_type",
        "ms": "material_synthesis", "sm": "synthesis_method",
        "lr": "literature_reference", "mr": "material_reference",
        "ad": "application_domain", "ma": "material_application",
        "em": "experimental_measurement", "mpr": "measured_property",
        "pde": "phase_diagram_entry", "als": "alloy_system",
        "mas": "material_alloy_system", "sg": "space_group",
    }
    # Merge dynamic aliases extracted from the SQL itself (e.g. cp_bm, c_ni).
    # Dynamic aliases override static ones since they reflect actual SQL declarations.
    dynamic_aliases = _extract_aliases_from_sql(sql)
    alias_to_table.update(dynamic_aliases)
    disallowed: list[str] = []
    for col_ref in used:
        parts = col_ref.split(".")
        if len(parts) != 2:
            continue
        alias, col = parts
        table = alias_to_table.get(alias.lower(), alias.lower())
        canonical = f"{table}.{col}".lower()
        if canonical not in allowed_lower:
            suggestion = suggest_column_correction(canonical, allowed_columns)
            if suggestion:
                disallowed.append(f"{col_ref} (use {suggestion} instead)")
            else:
                disallowed.append(col_ref)

    # Check unqualified columns against FROM tables
    unqualified = _extract_unqualified_columns(sql)
    if unqualified:
        from_tables = _get_from_tables(sql)
        if from_tables:
            for col in unqualified:
                found = False
                for tbl in from_tables:
                    if f"{tbl}.{col}".lower() in allowed_lower:
                        found = True
                        break
                if not found:
                    disallowed.append(col)
    return disallowed


def _extract_aliases_from_sql(sql: str) -> dict[str, str]:
    """Extract alias→table mappings from FROM/JOIN clauses in SQL.

    Handles: FROM table alias, JOIN table alias ON ..., FROM table AS alias.
    Falls back to identity (alias=table name) if no alias is found.
    """
    alias_map: dict[str, str] = {}
    clean = _strip_literals(sql)
    # FROM table [AS] alias
    for m in re.finditer(
        r"\bFROM\s+(\w+)(?:\s+AS\s+(\w+)|\s+(\w+))?",
        clean, re.IGNORECASE,
    ):
        table = m.group(1).lower()
        alias = (m.group(2) or m.group(3) or table).lower()
        # Skip if "alias" is a SQL keyword
        if alias.upper() in {"WHERE", "ON", "JOIN", "INNER", "LEFT", "RIGHT",
                              "FULL", "CROSS", "ORDER", "GROUP", "HAVING",
                              "LIMIT", "UNION", "EXCEPT", "INTERSECT"}:
            alias = table
        alias_map[alias] = table
        alias_map[table] = table  # table name also resolves to itself
    # JOIN table [AS] alias
    for m in re.finditer(
        r"\bJOIN\s+(\w+)(?:\s+AS\s+(\w+)|\s+(\w+))?\s+ON\b",
        clean, re.IGNORECASE,
    ):
        table = m.group(1).lower()
        alias = (m.group(2) or m.group(3) or table).lower()
        alias_map[alias] = table
        alias_map[table] = table
    return alias_map


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
    alias_to_table = _extract_aliases_from_sql(sql)
    warnings: list[str] = []
    clean = _strip_literals(sql)
    # Match all ON conditions including AND-chained ones
    for m in re.finditer(
        r"\bON\s+(.*?)(?=\s+(?:JOIN|WHERE|ORDER|GROUP|HAVING|LIMIT|UNION)\b|;|\)|$)",
        clean, re.IGNORECASE | re.DOTALL,
    ):
        on_clause = m.group(1)
        for cond in re.finditer(
            r"(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)", on_clause
        ):
            alias1 = cond.group(1).lower()
            col1 = cond.group(2).lower()
            alias2 = cond.group(3).lower()
            col2 = cond.group(4).lower()
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
    if any(re.search(pat, clean) for pat in patterns):
        warnings.append("Tautological condition detected")
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

    # Layer: System/catalog table access prevention
    sys_tables = check_system_tables(sql)
    if sys_tables:
        errors.append(f"System table access blocked: {', '.join(sys_tables)}")
        _escalate("rejected_security")

    # Layer: CTE body must be pure SELECT (no writable CTEs)
    cte_violations = check_cte_bodies_select_only(sql)
    if cte_violations:
        errors.append(f"Writable CTE detected: {'; '.join(cte_violations)}")
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
    else:
        warnings.append(
            "sqlglot not installed — SQL syntax validation degraded. "
            "Install with: pip install sqlglot>=20.0"
        )

    # Layer: Column type safety (warnings, not errors — does not reject)
    type_warnings = check_column_type_safety(sql)
    for tw in type_warnings:
        warnings.append(tw)

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "classification": classification,
        "sql": sql,
    }
