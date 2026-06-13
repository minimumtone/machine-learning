"""Generate SQL JOIN clauses from graph-derived join paths."""
from __future__ import annotations

import networkx as nx

from graph.traversal_engine import extract_join_edges, find_join_subgraph


TABLE_ALIASES: dict[str, str] = {
    "material_entry": "m",
    "composition": "c",
    "structure": "s",
    "calculation": "calc",
    "calculated_property": "cp",
    "phase_stability": "ps",
    "prototype_definition": "pd",
    "elastic_tensor": "et",
    "thermal_property": "tp",
    "magnetic_property": "mp",
    "surface_energy": "se",
    "grain_boundary": "gb",
    "band_structure": "bs",
    "density_of_states": "dos",
    "element": "e",
    "element_property": "ep",
    "material_defect": "md",
    "defect_type": "dt",
    "material_synthesis": "ms",
    "synthesis_method": "sm",
    "literature_reference": "lr",
    "material_reference": "mr",
    "application_domain": "ad",
    "material_application": "ma",
    "experimental_measurement": "em",
    "measured_property": "mpr",
    "phase_diagram_entry": "pde",
    "alloy_system": "als",
    "material_alloy_system": "mas",
    "space_group": "sg",
}


def _alias(table: str, used: set[str] | None = None) -> str:
    """Return alias for table, ensuring uniqueness when *used* is provided."""
    base = TABLE_ALIASES.get(table, table[:3] if len(table) > 2 else table)
    if used is None:
        return base
    if base not in used:
        used.add(base)
        return base
    for i in range(2, 100):
        candidate = f"{base}{i}"
        if candidate not in used:
            used.add(candidate)
            return candidate
    return base


def generate_join_clause(join_path: list[dict[str, str]], base_table: str = "material_entry") -> str:
    """Generate SQL JOIN clauses from a list of join edge dicts.

    Skips edges that would JOIN the base_table itself (self-join bug).
    """
    parts: list[str] = []
    used_aliases: set[str] = set()
    base_alias = _alias(base_table, used_aliases)

    for edge in join_path:
        src_t = edge["source_table"]
        tgt_t = edge["target_table"]
        src_c = edge["source_column"]
        tgt_c = edge["target_column"]
        # Determine which table to JOIN (skip if it would be the base table)
        if src_t == base_table:
            join_table = tgt_t
            sa = base_alias
            ta = _alias(join_table, used_aliases)
            parts.append(f"JOIN {join_table} {ta} ON {ta}.{tgt_c} = {sa}.{src_c}")
        elif tgt_t == base_table:
            join_table = src_t
            sa = _alias(join_table, used_aliases)
            ta = base_alias
            parts.append(f"JOIN {join_table} {sa} ON {sa}.{src_c} = {ta}.{tgt_c}")
        else:
            sa = _alias(src_t, used_aliases)
            ta = _alias(tgt_t, used_aliases)
            parts.append(f"JOIN {tgt_t} {ta} ON {ta}.{tgt_c} = {sa}.{src_c}")
    return "\n".join(parts)


def generate_joins_for_tables(
    table_graph: nx.Graph,
    required_tables: list[str],
    base_table: str = "material_entry",
) -> str:
    """Generate complete JOIN clause for a set of required tables.

    The *base_table* is used as the FROM table; JOINs are generated for the
    remaining tables.
    """
    subgraph = find_join_subgraph(table_graph, required_tables)
    all_edges: list[dict[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for path in subgraph.get("paths", []):
        edges = extract_join_edges(table_graph, path)
        for e in edges:
            pair = (e["source_table"], e["target_table"])
            rev = (e["target_table"], e["source_table"])
            if pair not in seen_pairs and rev not in seen_pairs:
                seen_pairs.add(pair)
                all_edges.append(e)

    # Build an adjacency structure from collected edges so we can BFS from
    # base_table and emit JOINs in parent-first order.  This guarantees
    # that the alias referenced on the right side of every ON clause has
    # already been introduced by a prior JOIN (or is the FROM table).
    adj: dict[str, list[dict[str, str]]] = {}
    for edge in all_edges:
        s, t = edge["source_table"], edge["target_table"]
        adj.setdefault(s, []).append(edge)
        adj.setdefault(t, []).append(edge)

    visited: set[str] = {base_table}
    queue: list[str] = [base_table]
    ordered_edges: list[dict[str, str]] = []
    while queue:
        node = queue.pop(0)
        for edge in adj.get(node, []):
            s, t = edge["source_table"], edge["target_table"]
            neighbour = t if s == node else s
            if neighbour not in visited:
                visited.add(neighbour)
                ordered_edges.append(edge)
                queue.append(neighbour)

    used_aliases: set[str] = set()
    table_to_alias: dict[str, str] = {}
    base_alias = _alias(base_table, used_aliases)
    table_to_alias[base_table] = base_alias
    parts: list[str] = []

    for edge in ordered_edges:
        src_t = edge["source_table"]
        src_c = edge["source_column"]
        tgt_t = edge["target_table"]
        tgt_c = edge["target_column"]
        # Determine which side is already joined (parent) vs new (child)
        if src_t in table_to_alias and tgt_t not in table_to_alias:
            join_table = tgt_t
            ja = _alias(join_table, used_aliases)
            table_to_alias[join_table] = ja
            parent_alias = table_to_alias[src_t]
            on_clause = f"{ja}.{tgt_c} = {parent_alias}.{src_c}"
        elif tgt_t in table_to_alias and src_t not in table_to_alias:
            join_table = src_t
            ja = _alias(join_table, used_aliases)
            table_to_alias[join_table] = ja
            parent_alias = table_to_alias[tgt_t]
            on_clause = f"{ja}.{src_c} = {parent_alias}.{tgt_c}"
        else:
            # Both already visited (cycle edge) — skip
            continue
        parts.append(f"JOIN {join_table} {ja} ON {on_clause}")

    return "\n".join(parts)


def get_allowed_join_list(
    table_graph: nx.Graph,
) -> list[str]:
    """Return a list of allowed join conditions from the table graph."""
    joins: list[str] = []
    # Fix B12: Emit both directions so that join matching works regardless of
    # which table appears on each side of the = in generated SQL.
    for u, v, data in table_graph.edges(data=True):
        sc = data.get("source_column", "")
        tc = data.get("target_column", "")
        if sc and tc:
            joins.append(f"{u}.{sc} = {v}.{tc}")
            joins.append(f"{v}.{tc} = {u}.{sc}")  # reverse
    return joins
