"""Evaluation metrics for Text-to-SQL comparison.

Fixes applied (audit 2026-06-02):
- B1: Added precision/F1 alongside recall
- B3: Position-based fallback when common columns are empty
- B4: Unified type normalization (str-cast all values)
- B5: AST-based alias resolution for JOIN hallucination
- B15: syntax_validity handles WITH/CTE queries
- B16: ON-clause regex word-boundary fix (composition/calculation false match)
- B17: CTE names excluded from hallucinated-join detection
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import sqlglot
    import sqlglot.expressions
    HAS_SQLGLOT = True
else:
    try:
        import sqlglot
        import sqlglot.expressions
        HAS_SQLGLOT = True
    except ImportError:
        sqlglot = None
        HAS_SQLGLOT = False


def syntax_validity(sql: str) -> bool:
    """Check whether SQL parses without error.

    Fix B15: Accept WITH/CTE queries (not just SELECT-starting).
    """
    if not sql or not sql.strip():
        return False
    if HAS_SQLGLOT:
        try:
            sqlglot.parse(sql, dialect="postgres")
            return True
        except Exception:
            return False
    # Fallback: accept SELECT or WITH (CTE)
    upper = sql.strip().upper()
    return upper.startswith("SELECT") or upper.startswith("WITH")


def execution_validity(result: dict[str, Any]) -> bool:
    """Check whether execution succeeded."""
    return result.get("success", False)


def _normalize_value(v: Any) -> str:
    """Normalize a cell value to string for comparison.

    Fix B4: Unified type normalization across evaluations.
    Handles Decimal, int, float, None consistently.
    """
    if v is None:
        return "__NULL__"
    s = str(v).strip()
    # Normalize numeric representations: "180.0" == "180" == "180.00"
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
        return f"{f:.6g}"
    except (ValueError, OverflowError):
        return s.lower()


def execution_accuracy_full(
    result_rows: list[list[Any]],
    expected_rows: list[list[Any]],
    result_columns: list[str] | None = None,
    expected_columns: list[str] | None = None,
) -> dict[str, float]:
    """Compute recall, precision, and F1 for row-set overlap.

    Fix B1: Returns all three metrics (recall/precision/F1).
    Fix B3: Falls back to position-based matching when column names
            have no overlap (e.g., 'count' vs 'l12_count').
    Fix B4: All values are str-normalized before comparison.

    Returns dict with keys: recall, precision, f1.
    """
    if not expected_rows:
        if not result_rows:
            # Both empty → perfect match
            return {"recall": 1.0, "precision": 1.0, "f1": 1.0}
        # Expected is empty but result is non-empty → wrong (should return nothing)
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0}

    if not result_rows:
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0}

    def _build_set(rows: list[list[Any]], indices: list[int] | None = None) -> set[tuple[str, ...]]:
        result = set()
        for r in rows:
            if indices is not None:
                vals = tuple(_normalize_value(r[i]) for i in indices if i < len(r))
            else:
                vals = tuple(_normalize_value(v) for v in r)
            result.add(vals)
        return result

    result_set: set[tuple[str, ...]]
    expected_set: set[tuple[str, ...]]

    if result_columns and expected_columns:
        rc = [c.lower() for c in result_columns]
        ec = [c.lower() for c in expected_columns]
        common = [c for c in ec if c in rc]

        if common:
            ri = [rc.index(c) for c in common]
            ei = [ec.index(c) for c in common]
            result_set = _build_set(result_rows, ri)
            expected_set = _build_set(expected_rows, ei)
        else:
            # Fix B3: No common column names — fall back to position-based
            # matching using min(len(result_cols), len(expected_cols)) columns
            min_cols = min(len(rc), len(ec))
            if min_cols > 0:
                result_set = _build_set(result_rows, list(range(min_cols)))
                expected_set = _build_set(expected_rows, list(range(min_cols)))
            else:
                return {"recall": 0.0, "precision": 0.0, "f1": 0.0}
    else:
        result_set = _build_set(result_rows)
        expected_set = _build_set(expected_rows)

    if not expected_set:
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0}

    matched = len(result_set & expected_set)
    recall = matched / len(expected_set) if expected_set else 0.0
    precision = matched / len(result_set) if result_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"recall": recall, "precision": precision, "f1": f1}


def common_column_exact_overlap(
    result_rows: list[list[Any]],
    expected_rows: list[list[Any]],
    result_columns: list[str] | None = None,
    expected_columns: list[str] | None = None,
) -> float:
    """Diagnostic overlap metric: 1.0 when precision and recall are both 1.0
    on the common-column projection used by execution_accuracy_full().

    This is NOT an exact result-set match: a generated query that returns
    only a subset of the gold columns can still score 1.0 here.  The
    canonical exact metric is
    evaluation.metrics_strict.exact_result_set_match (exact gold column
    list + row multiset + row order only when the expected result's
    ``semantic_ordered`` flag is set, i.e. the question itself asks for an
    ordered answer -- not whenever the gold SQL merely has an ORDER BY).
    """
    m = execution_accuracy_full(
        result_rows, expected_rows, result_columns, expected_columns,
    )
    return 1.0 if m["recall"] == 1.0 and m["precision"] == 1.0 else 0.0


def execution_accuracy(
    result_rows: list[list[Any]],
    expected_rows: list[list[Any]],
    result_columns: list[str] | None = None,
    expected_columns: list[str] | None = None,
) -> float:
    """Backward-compatible wrapper returning recall only.

    For new code, prefer execution_accuracy_full() which returns all metrics.
    """
    metrics = execution_accuracy_full(
        result_rows, expected_rows, result_columns, expected_columns,
    )
    return metrics["recall"]


def hallucinated_table_rate(
    generated_tables: list[str],
    allowed_tables: list[str],
) -> float:
    """Fraction of generated tables not in allowed list."""
    if not generated_tables:
        return 0.0
    allowed_lower = {t.lower() for t in allowed_tables}
    bad = [t for t in generated_tables if t.lower() not in allowed_lower]
    return len(bad) / len(generated_tables)


def hallucinated_column_rate(
    generated_columns: list[str],
    allowed_columns: list[str],
    sql: str = "",
) -> float:
    """Fraction of generated columns not in allowed list.

    Resolves alias-qualified columns (e.g. ``m.formula``) to their
    full ``table.column`` form using the alias map extracted from *sql*
    so that comparison against the ``table.column`` allowed list is
    meaningful.
    """
    if not generated_columns:
        return 0.0
    alias_map = _extract_aliases_from_sql(sql) if sql else {}
    allowed_lower = {c.lower() for c in allowed_columns}

    def _resolve_col(col: str) -> str:
        parts = col.strip().lower().split(".")
        if len(parts) == 2:
            table = alias_map.get(parts[0], parts[0])
            return f"{table}.{parts[1]}"
        return col.lower()

    resolved = [_resolve_col(c) for c in generated_columns]
    bad = [c for c in resolved if c not in allowed_lower]
    return len(bad) / len(resolved)


def _extract_aliases_from_sql(sql: str) -> dict[str, str]:
    """Extract alias-to-table mapping from SQL using AST when possible.

    Fix B5: Build alias table from actual SQL instead of a hardcoded dict.
    Falls back to regex when sqlglot is unavailable.
    """
    alias_map: dict[str, str] = {}

    if HAS_SQLGLOT and sqlglot is not None:
        try:
            parsed = sqlglot.parse(sql, dialect="postgres")
            for stmt in parsed:
                if stmt is None:
                    continue
                for table in stmt.find_all(sqlglot.expressions.Table):
                    table_name = table.name
                    alias_node = table.args.get("alias")
                    if alias_node:
                        alias_map[alias_node.name.lower()] = table_name.lower()
                    else:
                        alias_map[table_name.lower()] = table_name.lower()
            if alias_map:
                return alias_map
        except Exception:
            pass

    # Regex fallback: match FROM/JOIN table alias patterns
    # Pattern: FROM tablename alias / JOIN tablename alias
    for m in re.finditer(
        r"(?:FROM|JOIN)\s+(\w+)\s+(?:AS\s+)?(\w+)",
        sql, re.IGNORECASE,
    ):
        table, alias = m.group(1).lower(), m.group(2).lower()
        # Skip if "alias" is actually ON/WHERE/etc.
        if alias.upper() in ("ON", "WHERE", "AND", "OR", "LEFT", "RIGHT",
                              "INNER", "OUTER", "CROSS", "FULL", "JOIN",
                              "GROUP", "ORDER", "HAVING", "LIMIT", "UNION"):
            continue
        alias_map[alias] = table

    # Also add identity mapping for unaliased tables
    for m in re.finditer(r"(?:FROM|JOIN)\s+(\w+)(?:\s|$|;)", sql, re.IGNORECASE):
        t = m.group(1).lower()
        if t not in alias_map:
            alias_map[t] = t

    return alias_map


def hallucinated_join_rate(
    sql: str,
    allowed_joins: list[str],
) -> float:
    """Fraction of joins that are not in the allowed list.

    Fix B5: Resolve aliases from actual SQL AST, not a hardcoded dict.
    Extracts ALL join conditions (including AND-combined ON clauses).
    """
    if not sql or not sql.strip():
        return 0.0

    alias_map = _extract_aliases_from_sql(sql)

    # Fix B17: add CTE names as identity mappings so they pass alias resolution
    _cte_names: set[str] = set()
    for _cte_m in re.finditer(r"\bWITH\b\s+(\w+)\s+AS\b", sql, re.IGNORECASE):
        _cte_names.add(_cte_m.group(1).lower())
    for _cte_m in re.finditer(r",\s*(\w+)\s+AS\s*\(", sql, re.IGNORECASE):
        _cte_names.add(_cte_m.group(1).lower())
    for cte in _cte_names:
        alias_map.setdefault(cte, cte)

    def _resolve(ref: str) -> str:
        parts = ref.strip().split(".")
        if len(parts) == 2:
            table = alias_map.get(parts[0].lower(), parts[0].lower())
            return f"{table}.{parts[1].lower()}"
        return ref.lower()

    def normalize(j: str) -> tuple[str, str]:
        j = re.sub(r"\s+", " ", j.strip().lower())
        sides = j.split("=")
        if len(sides) == 2:
            left = _resolve(sides[0].strip())
            right = _resolve(sides[1].strip())
            return (min(left, right), max(left, right))
        return (j, "")

    # Fix B17: collect CTE names from WITH clause so JOINs to CTEs are not
    # treated as hallucinated (CTE names are valid virtual tables).
    cte_names: set[str] = set()
    for cte_m in re.finditer(
        r"\bWITH\b\s+(\w+)\s+AS\b", sql, re.IGNORECASE,
    ):
        cte_names.add(cte_m.group(1).lower())
    # Also handle comma-separated CTEs: WITH a AS (...), b AS (...)
    for cte_m in re.finditer(
        r",\s*(\w+)\s+AS\s*\(", sql, re.IGNORECASE,
    ):
        cte_names.add(cte_m.group(1).lower())

    # Extract join conditions from SQL, including AND-combined ON clauses
    gen_joins: list[str] = []
    # Fix B16: \bON\b prevents matching the suffix "on" in words like
    # "composition" or "calculation" (34/100 queries affected without \b)
    for m in re.finditer(
        r"\bON\b\s+(.*?)(?=\s+(?:JOIN|WHERE|GROUP|ORDER|HAVING|LIMIT|UNION)\b|;|\)|$)",
        sql, re.IGNORECASE | re.DOTALL,
    ):
        on_clause = m.group(1).strip()
        # Split by AND to get individual conditions
        parts = re.split(r"\bAND\b", on_clause, flags=re.IGNORECASE)
        for p in parts:
            p = p.strip()
            if "=" in p and "." in p:
                gen_joins.append(p)

    if not gen_joins:
        return 0.0

    allowed_set = {normalize(j) for j in allowed_joins}
    # Fix B17: JOINs where either side references a CTE name are valid
    # (the CTE is defined in the query itself, not a hallucinated table).
    bad = []
    for j in gen_joins:
        nj = normalize(j)
        if nj in allowed_set:
            continue
        # Check if either side of the join references a CTE
        parts = j.lower().split("=")
        is_cte_join = False
        for part in parts:
            table = part.strip().split(".")[0].strip()
            resolved = alias_map.get(table, table)
            if resolved in cte_names:
                is_cte_join = True
                break
        if not is_cte_join:
            bad.append(j)
    return len(bad) / len(gen_joins) if gen_joins else 0.0


def multi_hop_success(n_tables: int, is_correct: bool) -> dict[str, Any]:
    """Tag multi-hop success.

    Multi-hop defined as n_tables >= 3 (3+ tables referenced in gold SQL).
    With current evaluation set: 28 Medium + 22 Hard + 23 Very Hard = 73 queries.

    Args:
        n_tables: Number of tables referenced in the gold SQL.
        is_correct: Whether the query was judged correct (F1 >= 0.8).
    """
    return {
        "n_tables": n_tables,
        "is_multi_hop": n_tables >= 3,
        "correct": is_correct,
    }


def normalize_limit(sql: str) -> str:
    """Normalize LIMIT for evaluation.

    Fix B2: Only ADD LIMIT 10000 when no LIMIT exists.
    Preserve existing LIMIT values (LIMIT 1, LIMIT 10 etc.).
    """
    if not sql:
        return sql
    if not re.search(r"\bLIMIT\b", sql, re.IGNORECASE):
        sql = sql.rstrip().rstrip(";") + "\nLIMIT 10000;"
    return sql
