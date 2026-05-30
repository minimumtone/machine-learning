"""Build a NetworkX graph from parsed DB schema information."""
from __future__ import annotations

from typing import Any

import networkx as nx

from graph.schema_parser import ColumnMetadata, ForeignKeyMetadata


def build_schema_graph(
    tables: list[str],
    columns: dict[str, list[ColumnMetadata]],
    foreign_keys: list[ForeignKeyMetadata],
) -> nx.DiGraph:
    """Construct a directed graph with table and column nodes plus FK edges."""
    g = nx.DiGraph()

    for table in tables:
        g.add_node(f"table:{table}", node_type="table", table=table)
        for col in columns.get(table, []):
            col_id = f"column:{table}.{col.column_name}"
            g.add_node(
                col_id,
                node_type="column",
                table=table,
                column=col.column_name,
                data_type=col.data_type,
                is_primary_key=col.is_primary_key,
            )
            g.add_edge(
                f"table:{table}", col_id,
                edge_type="HAS_COLUMN",
            )

    for fk in foreign_keys:
        src = f"column:{fk.source_table}.{fk.source_column}"
        tgt = f"column:{fk.target_table}.{fk.target_column}"
        g.add_edge(src, tgt, edge_type="FOREIGN_KEY")
        g.add_edge(tgt, src, edge_type="FOREIGN_KEY")
        g.add_edge(
            f"table:{fk.source_table}",
            f"table:{fk.target_table}",
            edge_type="JOINABLE_WITH",
            join_on=f"{fk.source_table}.{fk.source_column} = "
                     f"{fk.target_table}.{fk.target_column}",
        )
        g.add_edge(
            f"table:{fk.target_table}",
            f"table:{fk.source_table}",
            edge_type="JOINABLE_WITH",
            join_on=f"{fk.target_table}.{fk.target_column} = "
                     f"{fk.source_table}.{fk.source_column}",
        )

    return g


def build_table_graph(
    foreign_keys: list[ForeignKeyMetadata],
) -> nx.Graph:
    """Build an undirected graph containing only table-level join edges."""
    g = nx.Graph()
    for fk in foreign_keys:
        g.add_edge(
            fk.source_table,
            fk.target_table,
            source_column=fk.source_column,
            target_column=fk.target_column,
        )
    return g


def get_joinable_tables(graph: nx.DiGraph, table: str) -> list[str]:
    """Return tables directly joinable with the given table."""
    node = f"table:{table}"
    if node not in graph:
        return []
    result: list[str] = []
    for _, target, data in graph.edges(node, data=True):
        if data.get("edge_type") == "JOINABLE_WITH":
            result.append(data.get("join_on", target))
    return result


def schema_graph_summary(graph: nx.DiGraph) -> dict[str, Any]:
    """Return a summary of the schema graph."""
    tables = [n for n, d in graph.nodes(data=True) if d.get("node_type") == "table"]
    columns = [n for n, d in graph.nodes(data=True) if d.get("node_type") == "column"]
    fk_edges = [
        (u, v) for u, v, d in graph.edges(data=True)
        if d.get("edge_type") == "FOREIGN_KEY"
    ]
    join_edges = [
        (u, v, d.get("join_on"))
        for u, v, d in graph.edges(data=True)
        if d.get("edge_type") == "JOINABLE_WITH"
    ]
    return {
        "num_tables": len(tables),
        "num_columns": len(columns),
        "num_fk_edges": len(fk_edges),
        "num_join_edges": len(join_edges),
        "tables": [t.replace("table:", "") for t in tables],
    }
