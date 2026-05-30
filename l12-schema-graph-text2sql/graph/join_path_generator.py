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
}


def _alias(table: str) -> str:
    return TABLE_ALIASES.get(table, table[:2])


def generate_join_clause(join_path: list[dict[str, str]]) -> str:
    """Generate SQL JOIN clauses from a list of join edge dicts."""
    parts: list[str] = []
    for edge in join_path:
        src_t = edge["source_table"]
        src_c = edge["source_column"]
        tgt_t = edge["target_table"]
        tgt_c = edge["target_column"]
        sa = _alias(src_t)
        ta = _alias(tgt_t)
        parts.append(f"JOIN {src_t} {sa} ON {sa}.{src_c} = {ta}.{tgt_c}")
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

    parts: list[str] = []
    for edge in all_edges:
        src_t = edge["source_table"]
        src_c = edge["source_column"]
        tgt_t = edge["target_table"]
        tgt_c = edge["target_column"]
        if src_t == base_table:
            join_table = tgt_t
            on_clause = (
                f"{_alias(join_table)}.{tgt_c} = {_alias(base_table)}.{src_c}"
            )
        elif tgt_t == base_table:
            join_table = src_t
            on_clause = (
                f"{_alias(join_table)}.{src_c} = {_alias(base_table)}.{tgt_c}"
            )
        else:
            join_table = tgt_t
            on_clause = (
                f"{_alias(tgt_t)}.{tgt_c} = {_alias(src_t)}.{src_c}"
            )
        parts.append(f"JOIN {join_table} {_alias(join_table)} ON {on_clause}")

    return "\n".join(parts)


def get_allowed_join_list(
    table_graph: nx.Graph,
) -> list[str]:
    """Return a list of allowed join conditions from the table graph."""
    joins: list[str] = []
    for u, v, data in table_graph.edges(data=True):
        sc = data.get("source_column", "")
        tc = data.get("target_column", "")
        if sc and tc:
            joins.append(f"{u}.{sc} = {v}.{tc}")
    return joins
