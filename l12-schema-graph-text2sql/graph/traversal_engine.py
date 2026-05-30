"""Graph traversal for finding JOIN paths between tables."""
from __future__ import annotations

from itertools import combinations

import networkx as nx

from graph.schema_parser import ForeignKeyMetadata


def find_shortest_table_path(
    graph: nx.Graph,
    source_table: str,
    target_table: str,
) -> list[str]:
    """Return the shortest join path between two tables."""
    try:
        return nx.shortest_path(graph, source_table, target_table)
    except (nx.NodeNotFound, nx.NetworkXNoPath):
        return []


def find_join_subgraph(
    graph: nx.Graph,
    required_tables: list[str],
) -> dict[str, list[str]]:
    """Return join paths connecting all required tables via a Steiner-tree approximation."""
    if len(required_tables) <= 1:
        return {"paths": [required_tables]}

    paths: list[list[str]] = []
    covered: set[str] = set()

    for src, tgt in combinations(required_tables, 2):
        if src in covered and tgt in covered:
            continue
        path = find_shortest_table_path(graph, src, tgt)
        if path:
            paths.append(path)
            covered.update(path)

    for t in required_tables:
        if t not in covered:
            for c in covered:
                p = find_shortest_table_path(graph, t, c)
                if p:
                    paths.append(p)
                    covered.update(p)
                    break

    return {"paths": paths}


def rank_join_paths(paths: list[list[str]]) -> list[list[str]]:
    """Rank candidate join paths by length (shorter is better)."""
    return sorted(paths, key=len)


def extract_join_edges(
    graph: nx.Graph,
    path: list[str],
) -> list[dict[str, str]]:
    """Extract the join column info for each edge in a table path."""
    edges: list[dict[str, str]] = []
    for i in range(len(path) - 1):
        data = graph.edges[path[i], path[i + 1]]
        edges.append({
            "source_table": path[i],
            "source_column": data.get("source_column", ""),
            "target_table": path[i + 1],
            "target_column": data.get("target_column", ""),
        })
    return edges
