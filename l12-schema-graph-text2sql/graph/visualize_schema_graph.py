"""Visualize the schema graph using matplotlib."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from graph.schema_parser import ForeignKeyMetadata
from graph.graph_builder import build_schema_graph, build_table_graph, ColumnMetadata


def draw_table_graph(
    foreign_keys: list[ForeignKeyMetadata],
    output_path: str | Path = "schema_graph.png",
) -> None:
    """Draw a simplified table-level graph and save to *output_path*."""
    g = build_table_graph(foreign_keys)
    fig, ax = plt.subplots(figsize=(10, 7))
    pos = nx.spring_layout(g, seed=42, k=2.0)
    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=2000, node_color="#4FC3F7")
    nx.draw_networkx_labels(g, pos, ax=ax, font_size=9, font_weight="bold")
    edge_labels = {
        (u, v): f"{d['source_column']}\n= {d['target_column']}"
        for u, v, d in g.edges(data=True)
    }
    nx.draw_networkx_edges(g, pos, ax=ax, arrows=True, arrowstyle="-|>",
                           connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_edge_labels(g, pos, edge_labels, ax=ax, font_size=7)
    ax.set_title("L1$_2$ Materials Database Schema Graph")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    print(f"Schema graph saved to {output_path}")


def draw_full_schema_graph(
    tables: list[str],
    columns: dict[str, list[ColumnMetadata]],
    foreign_keys: list[ForeignKeyMetadata],
    output_path: str | Path = "full_schema_graph.png",
) -> None:
    """Draw the full schema graph (tables + columns + FK edges)."""
    g = build_schema_graph(tables, columns, foreign_keys)
    fig, ax = plt.subplots(figsize=(16, 12))
    color_map = []
    for n, d in g.nodes(data=True):
        if d.get("node_type") == "table":
            color_map.append("#4FC3F7")
        else:
            color_map.append("#C8E6C9")
    pos = nx.spring_layout(g, seed=42, k=0.8, iterations=80)
    sizes = [
        1500 if d.get("node_type") == "table" else 400
        for _, d in g.nodes(data=True)
    ]
    labels = {}
    for n, d in g.nodes(data=True):
        if d.get("node_type") == "table":
            labels[n] = n.replace("table:", "")
        else:
            labels[n] = d.get("column", n.split(".")[-1])
    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=sizes, node_color=color_map,
                           alpha=0.9)
    nx.draw_networkx_labels(g, pos, labels=labels, ax=ax, font_size=6)
    fk_edges = [(u, v) for u, v, d in g.edges(data=True)
                if d.get("edge_type") == "FOREIGN_KEY"]
    has_col_edges = [(u, v) for u, v, d in g.edges(data=True)
                     if d.get("edge_type") == "HAS_COLUMN"]
    nx.draw_networkx_edges(g, pos, edgelist=has_col_edges, ax=ax,
                           edge_color="#B0BEC5", style="solid", alpha=0.5)
    nx.draw_networkx_edges(g, pos, edgelist=fk_edges, ax=ax,
                           edge_color="#E53935", style="dashed", width=2)
    ax.set_title("Full Schema Graph")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    print(f"Full schema graph saved to {output_path}")
