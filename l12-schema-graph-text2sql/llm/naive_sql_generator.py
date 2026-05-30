"""Naive Text-to-SQL generator (Level 0) — no Schema Graph, no Few-Shot.

This module intentionally skips the Schema Graph Traversal Engine and
few-shot retrieval.  It demonstrates the problems that arise when SQL
is generated from extracted conditions alone:

- Unnecessary JOINs (all tables are always joined)
- Missing or incorrect JOIN paths
- No LIMIT enforcement
- No allowed-table / allowed-column validation
- No safety guard

It serves as a baseline to quantify the value of Schema Graph and
Few-Shot enhancements.
"""
from __future__ import annotations

from typing import Any

from llm.entity_extractor import extract_conditions


def naive_pipeline(user_query: str) -> dict[str, Any]:
    """Generate SQL with minimal processing — conditions only, no graph."""
    conditions = extract_conditions(user_query)

    select_cols = ["m.entry_id", "m.formula"]
    where_clauses: list[str] = []
    order_by = ""

    # Always join ALL tables regardless of query needs (wasteful)
    joins = [
        "JOIN composition c ON c.entry_id = m.entry_id",
        "JOIN structure s ON s.entry_id = m.entry_id",
        "JOIN phase_stability ps ON ps.entry_id = m.entry_id",
        "JOIN calculation calc ON calc.entry_id = m.entry_id",
        "JOIN calculated_property cp ON cp.calculation_id = calc.calculation_id",
    ]

    # Prototype
    proto = conditions.get("prototype")
    if proto:
        if isinstance(proto, list):
            parts = " OR ".join(f"s.prototype = '{p}'" for p in proto)
            where_clauses.append(f"({parts})")
        else:
            where_clauses.append(f"s.prototype = '{proto}'")
        select_cols.append("s.prototype")

    # Elements — simple AND, no EXISTS subquery (can produce wrong results
    # when multiple elements are joined via single composition row)
    elems = conditions.get("contains_elements", [])
    if len(elems) == 1:
        where_clauses.append(f"c.element = '{elems[0]}'")
    elif len(elems) > 1:
        # Naive approach: AND on same table — this is WRONG for multi-element
        # because a single composition row can only have one element.
        # The correct approach uses EXISTS subqueries.
        for e in elems:
            where_clauses.append(f"c.element = '{e}'")

    # Stability
    stab = conditions.get("stability")
    if stab == "stable":
        where_clauses.append("ps.energy_above_hull <= 0.001")
    elif stab == "metastable":
        where_clauses.append("ps.energy_above_hull <= 0.05")

    # Sort
    sort_by = conditions.get("sort_by")
    if sort_by:
        order = conditions.get("sort_order", "asc").upper()
        order_by = f"ORDER BY {sort_by} {order}"

    # Build SQL — no LIMIT (unsafe), no DISTINCT
    sql_parts = [f"SELECT {', '.join(select_cols)}"]
    sql_parts.append("FROM material_entry m")
    for j in joins:
        sql_parts.append(f"    {j}")
    if where_clauses:
        sql_parts.append("WHERE " + " AND ".join(where_clauses))
    if order_by:
        sql_parts.append(order_by)
    # No LIMIT — intentionally unsafe
    sql = "\n".join(sql_parts) + ";"

    return {
        "sql": sql,
        "conditions": conditions,
        "model": "naive_no_graph",
        "issues": _identify_issues(conditions, joins, where_clauses, elems),
    }


def _identify_issues(
    conditions: dict[str, Any],
    joins: list[str],
    where_clauses: list[str],
    elements: list[str],
) -> list[str]:
    """Identify known problems in the naive approach."""
    issues: list[str] = []

    # Issue 1: Always joins all tables
    needed = set()
    if conditions.get("prototype"):
        needed.add("structure")
    if conditions.get("contains_elements"):
        needed.add("composition")
    if conditions.get("stability") or conditions.get("sort_by", "").startswith("phase_stability"):
        needed.add("phase_stability")
    unnecessary = {"composition", "structure", "phase_stability", "calculation", "calculated_property"} - needed
    if unnecessary:
        issues.append(f"Unnecessary JOINs: {', '.join(sorted(unnecessary))}")

    # Issue 2: Multi-element AND on same table
    if len(elements) > 1:
        issues.append(
            f"Multi-element AND on same composition row — will return 0 rows "
            f"(should use EXISTS subqueries for {elements})"
        )

    # Issue 3: No LIMIT
    issues.append("No LIMIT clause — unbounded result set")

    # Issue 4: No DISTINCT
    issues.append("No DISTINCT — may return duplicate rows from JOINs")

    # Issue 5: No SQL safety validation
    issues.append("No SQL safety validation (sqlglot/forbidden keyword check)")

    return issues
