"""Evaluation metrics for Text-to-SQL comparison.

Fixes applied (audit 2026-06-02):
- B1: Added precision/F1 alongside recall
- B3: Position-based fallback when common columns are empty
- B4: Unified type normalization (str-cast all values)
- B5: AST-based alias resolution for JOIN hallucination
- B15: syntax_validity handles WITH/CTE queries
"""
from __future__ import annotations

import re
from typing import Any

try:
    import sqlglot
    HAS_SQLGLOT = True
except ImportError:
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
            return {"recall": 1.0, "precision": 1.0, "f1": 1.0}
        return {"recall": 1.0, "precision": 0.0, "f1": 0.0}

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
) -> float:
    """Fraction of generated columns not in allowed list."""
    if not generated_columns:
        return 0.0
    allowed_lower = {c.lower() for c in allowed_columns}
    bad = [c for c in generated_columns if c.lower() not in allowed_lower]
    return len(bad) / len(generated_columns)


def _extract_aliases_from_sql(sql: str) -> dict[str, str]:
    """Extract alias-to-table mapping from SQL using AST when possible.

    Fix B5: Build alias table from actual SQL instead of a hardcoded dict.
    Falls back to regex when sqlglot is unavailable.
    """
    alias_map: dict[str, str] = {}

    if HAS_SQLGLOT:
        try:
            parsed = sqlglot.parse(sql, dialect="postgres")
            for stmt in parsed:
                if stmt is None:
                    continue
                for table in stmt.find_all(sqlglot.exp.Table):
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

    # Extract join conditions from SQL, including AND-combined ON clauses
    gen_joins: list[str] = []
    # Match ON ... conditions up to next JOIN/WHERE/GROUP/ORDER/LIMIT/;/)
    for m in re.finditer(
        r"ON\s+(.*?)(?=\s+(?:JOIN|WHERE|GROUP|ORDER|HAVING|LIMIT|UNION)\b|;|\)|$)",
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
    bad = [j for j in gen_joins if normalize(j) not in allowed_set]
    return len(bad) / len(gen_joins)


def multi_hop_success(hop_count: int, is_correct: bool) -> dict[str, Any]:
    """Tag multi-hop success.

    Fix: multi-hop is hop_count >= 2, matching 5.1.1 definition of 84 queries.
    """
    return {
        "hop_count": hop_count,
        "is_multi_hop": hop_count >= 2,
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
