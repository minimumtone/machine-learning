"""Tests for graph.join_path_generator."""
from graph.schema_parser import ForeignKeyMetadata
from graph.graph_builder import build_table_graph
from graph.join_path_generator import (
    generate_joins_for_tables,
    get_allowed_join_list,
)


def _sample_fks():
    return [
        ForeignKeyMetadata("composition", "entry_id", "material_entry", "entry_id"),
        ForeignKeyMetadata("structure", "entry_id", "material_entry", "entry_id"),
        ForeignKeyMetadata("phase_stability", "entry_id", "material_entry", "entry_id"),
        ForeignKeyMetadata("calculation", "entry_id", "material_entry", "entry_id"),
        ForeignKeyMetadata("calculated_property", "calculation_id", "calculation", "calculation_id"),
    ]


def test_generate_joins_for_tables():
    g = build_table_graph(_sample_fks())
    result = generate_joins_for_tables(
        g,
        ["material_entry", "composition", "structure", "phase_stability"],
    )
    assert "JOIN" in result
    assert "composition" in result
    assert "structure" in result
    assert "phase_stability" in result


def test_get_allowed_join_list():
    g = build_table_graph(_sample_fks())
    joins = get_allowed_join_list(g)
    assert len(joins) == 5
    assert any("composition" in j for j in joins)
