"""Naive Text-to-SQL baseline.

Generates SQL from keywords without schema graph constraints.
This is the weakest baseline: keyword matching → direct table access,
no JOIN path validation, no coverage scoring.
"""
from __future__ import annotations

from .entity_extractor import _normalize, _load_terms


def generate_naive_sql(query: str) -> str:
    """Generate SQL using simple keyword matching (no schema graph)."""
    q = _normalize(query).lower()
    terms = _load_terms()

    where: list[str] = []
    tables_used = {"material_entry"}
    joins: list[str] = []

    # Prototype
    for proto, aliases in terms.get("prototype_aliases", {}).items():
        for alias in aliases:
            if _normalize(alias).lower() in q:
                tables_used.add("structure")
                joins.append("JOIN structure s ON s.entry_id = material_entry.entry_id")
                where.append(f"s.prototype = '{proto}'")
                break

    # Elements (naive row-level AND — known to produce wrong results for multi-element)
    found_elems: list[str] = []
    for elem, info in terms.get("elements", {}).items():
        for alias in info.get("aliases", []):
            if alias.lower() in q:
                found_elems.append(elem)
                break
    if found_elems:
        tables_used.add("composition")
        if "composition" not in " ".join(joins):
            joins.append("JOIN composition c ON c.entry_id = material_entry.entry_id")
        elem_conds = " OR ".join(f"c.element = '{e}'" for e in found_elems)
        if len(found_elems) > 1:
            where.append(f"({elem_conds})")
        else:
            where.append(elem_conds)

    # Stability
    if "安定" in query or "stable" in q:
        tables_used.add("phase_stability")
        if "phase_stability" not in " ".join(joins):
            joins.append("JOIN phase_stability ps ON ps.entry_id = material_entry.entry_id")
        if "準安定" in query or "metastable" in q:
            where.append("ps.energy_above_hull <= 0.05")
        else:
            where.append("ps.energy_above_hull <= 0.001")

    sql_parts = ["SELECT material_entry.entry_id, material_entry.formula"]
    sql_parts.append("FROM material_entry")
    for j in joins:
        sql_parts.append(f"  {j}")
    if where:
        sql_parts.append("WHERE " + " AND ".join(where))
    sql_parts.append("LIMIT 100;")

    return "\n".join(sql_parts)
