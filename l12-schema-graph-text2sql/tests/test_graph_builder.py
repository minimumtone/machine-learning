"""Tests for graph.graph_builder."""
from graph.schema_parser import ColumnMetadata, ForeignKeyMetadata
from graph.graph_builder import (
    build_schema_graph,
    build_table_graph,
    get_joinable_tables,
    schema_graph_summary,
)


def _sample_data():
    tables = ["material_entry", "composition", "structure"]
    columns = {
        "material_entry": [
            ColumnMetadata("material_entry", "entry_id", "text", False, True),
            ColumnMetadata("material_entry", "formula", "text", False, False),
        ],
        "composition": [
            ColumnMetadata("composition", "composition_id", "text", False, True),
            ColumnMetadata("composition", "entry_id", "text", False, False),
            ColumnMetadata("composition", "element", "text", False, False),
        ],
        "structure": [
            ColumnMetadata("structure", "structure_id", "text", False, True),
            ColumnMetadata("structure", "entry_id", "text", False, False),
            ColumnMetadata("structure", "prototype", "text", True, False),
        ],
    }
    foreign_keys = [
        ForeignKeyMetadata("composition", "entry_id", "material_entry", "entry_id"),
        ForeignKeyMetadata("structure", "entry_id", "material_entry", "entry_id"),
    ]
    return tables, columns, foreign_keys


def test_build_schema_graph():
    tables, columns, fks = _sample_data()
    g = build_schema_graph(tables, columns, fks)
    assert "table:material_entry" in g
    assert "column:composition.entry_id" in g
    assert g.has_edge("table:composition", "table:material_entry")


def test_build_table_graph():
    _, _, fks = _sample_data()
    g = build_table_graph(fks)
    assert g.has_edge("composition", "material_entry")
    assert g.has_edge("structure", "material_entry")


def test_schema_graph_summary():
    tables, columns, fks = _sample_data()
    g = build_schema_graph(tables, columns, fks)
    summary = schema_graph_summary(g)
    assert summary["num_tables"] == 3
    assert "material_entry" in summary["tables"]


def test_get_joinable_tables():
    tables, columns, fks = _sample_data()
    g = build_schema_graph(tables, columns, fks)
    joinable = get_joinable_tables(g, "composition")
    assert len(joinable) > 0
