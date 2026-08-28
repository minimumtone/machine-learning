"""Tests for graph.join_path_generator."""
from graph.schema_parser import ForeignKeyMetadata
from graph.graph_builder import build_table_graph
from graph.join_path_generator import (
    generate_join_clause,
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


def test_multi_hop_join_order():
    """Sorted required_tables must not cause undefined alias references."""
    g = build_table_graph(_sample_fks())
    # sorted() puts calculated_property before material_entry — this was the bug
    tables = sorted(["material_entry", "calculation", "calculated_property"])
    assert tables[0] != "material_entry"  # confirm sort triggers the edge case
    result = generate_joins_for_tables(g, tables)
    lines = result.strip().split("\n")
    # Every alias used on the right of ON must be defined by an earlier JOIN
    introduced = {"m"}  # base_table alias
    for line in lines:
        # Extract the alias being JOINed (e.g. "JOIN calculation calc ON ...")
        parts = line.split()
        new_alias = parts[2] if len(parts) > 2 else ""
        # Check right side of ON — the alias referenced must be already introduced
        on_idx = line.index(" ON ") + 4
        on_clause = line[on_idx:]
        rhs = on_clause.split("=")[1].strip()
        rhs_alias = rhs.split(".")[0].strip()
        assert rhs_alias in introduced, (
            f"Alias '{rhs_alias}' used before introduction in: {line}"
        )
        introduced.add(new_alias)


def test_generate_join_clause_multi_hop_alias_reuse():
    """A multi-hop path must reuse the alias introduced for a table."""
    path = [
        {"source_table": "material_entry", "source_column": "entry_id",
         "target_table": "calculation", "target_column": "entry_id"},
        {"source_table": "calculation", "source_column": "calculation_id",
         "target_table": "calculated_property",
         "target_column": "calculation_id"},
    ]
    result = generate_join_clause(path)
    lines = result.strip().split("\n")
    assert lines[0] == "JOIN calculation calc ON calc.entry_id = m.entry_id"
    assert lines[1] == ("JOIN calculated_property cp "
                        "ON cp.calculation_id = calc.calculation_id")
    assert "calc2" not in result
    # Every alias referenced on the right of an ON must already exist.
    introduced = {"m"}
    for line in lines:
        parts = line.split()
        rhs_alias = line.split("=")[1].strip().split(".")[0]
        assert rhs_alias in introduced, line
        introduced.add(parts[2])


def test_get_allowed_join_list():
    g = build_table_graph(_sample_fks())
    joins = get_allowed_join_list(g)
    # Fix B12: both directions emitted. 5 FK edges * 2 = 10
    # (_SEMANTIC_JOINS is empty: composition->element is a physical FK now)
    assert len(joins) == 10
    assert any("composition" in j for j in joins)
