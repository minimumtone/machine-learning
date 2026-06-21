"""Graph traversal for finding JOIN paths between tables."""
from __future__ import annotations

from itertools import combinations

import networkx as nx


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
) -> dict[str, list[list[str]]]:
    """Return join paths connecting all required tables via a Steiner-tree approximation.

    Uses a union-find to track connected components so that pairs already
    connected by prior paths are correctly skipped while pairs in separate
    components are still joined.
    """
    if len(required_tables) <= 1:
        return {"paths": [required_tables]}

    # Simple union-find for connectivity tracking
    parent: dict[str, str] = {}

    def _find(x: str) -> str:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def _union(a: str, b: str) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[ra] = rb

    paths: list[list[str]] = []

    for src, tgt in combinations(required_tables, 2):
        if _find(src) == _find(tgt):
            continue
        path = find_shortest_table_path(graph, src, tgt)
        if path:
            paths.append(path)
            for i in range(len(path) - 1):
                _union(path[i], path[i + 1])

    # Ensure all required tables are connected
    for t in required_tables:
        root = _find(required_tables[0])
        if _find(t) != root:
            for other in required_tables:
                if _find(other) == root:
                    p = find_shortest_table_path(graph, t, other)
                    if p:
                        paths.append(p)
                        for i in range(len(p) - 1):
                            _union(p[i], p[i + 1])
                        break

    return {"paths": paths}


def rank_join_paths(paths: list[list[str]]) -> list[list[str]]:
    """Rank candidate join paths by length (shorter is better)."""
    return sorted(paths, key=len)


def extract_join_edges(
    graph: nx.Graph,
    path: list[str],
) -> list[dict[str, str]]:
    """Extract the join column info for each edge in a table path.

    In an undirected graph, edge data is stored with the original source/target
    from edge creation. When traversing in the reverse direction, we must swap
    the column assignments to match the actual traversal direction.
    """
    edges: list[dict[str, str]] = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        data = graph.edges[u, v]
        # Determine if we are traversing in the original FK direction.
        # nx.Graph stores edges keyed by the first two args of add_edge.
        # graph_builder.build_table_graph calls add_edge(src, tgt, ...),
        # so the canonical node order may be (src, tgt) or (tgt, src).
        # We check which node was the original source_table.
        edge_src = data.get("_edge_source", None)
        if edge_src is not None and edge_src != u:
            # Reverse direction — swap columns
            edges.append({
                "source_table": u,
                "source_column": data.get("target_column", ""),
                "target_table": v,
                "target_column": data.get("source_column", ""),
            })
        else:
            edges.append({
                "source_table": u,
                "source_column": data.get("source_column", ""),
                "target_table": v,
                "target_column": data.get("target_column", ""),
            })
    return edges
