"""Tests for evaluation.metrics."""
from evaluation.metrics import (
    syntax_validity,
    execution_validity,
    execution_accuracy,
    hallucinated_table_rate,
    hallucinated_column_rate,
    hallucinated_join_rate,
    multi_hop_success,
)


def test_syntax_validity_valid():
    assert syntax_validity("SELECT * FROM t LIMIT 10;")


def test_execution_validity_true():
    assert execution_validity({"success": True})


def test_execution_validity_false():
    assert not execution_validity({"success": False})


def test_execution_accuracy_exact():
    result = [[1, "a"], [2, "b"]]
    expected = [[1, "a"], [2, "b"]]
    assert execution_accuracy(result, expected) == 1.0


def test_execution_accuracy_partial():
    result = [[1, "a"]]
    expected = [[1, "a"], [2, "b"]]
    assert execution_accuracy(result, expected) == 0.5


def test_hallucinated_table_rate_none():
    rate = hallucinated_table_rate(
        ["material_entry", "structure"],
        ["material_entry", "structure", "composition"],
    )
    assert rate == 0.0


def test_hallucinated_table_rate_one():
    rate = hallucinated_table_rate(
        ["material_entry", "secret"],
        ["material_entry", "structure"],
    )
    assert rate == 0.5


def test_hallucinated_column_rate():
    rate = hallucinated_column_rate(
        ["m.formula", "m.fake"],
        ["m.formula"],
    )
    assert rate == 0.5


def test_hallucinated_join_rate():
    rate = hallucinated_join_rate(
        ["a.x = b.y"],
        ["a.x = b.y", "c.z = d.w"],
    )
    assert rate == 0.0


def test_hallucinated_join_rate_alias_resolution():
    """Test that alias-form joins are resolved to table-form before comparison."""
    # Generated SQL uses aliases: c.entry_id = m.entry_id
    # Allowed list uses full table names: composition.entry_id = material_entry.entry_id
    rate = hallucinated_join_rate(
        ["c.entry_id = m.entry_id", "ps.entry_id = m.entry_id"],
        ["composition.entry_id = material_entry.entry_id",
         "phase_stability.entry_id = material_entry.entry_id"],
    )
    assert rate == 0.0

    # Reversed order should also match (canonical sorting)
    rate2 = hallucinated_join_rate(
        ["m.entry_id = c.entry_id"],
        ["composition.entry_id = material_entry.entry_id"],
    )
    assert rate2 == 0.0

    # Truly hallucinated join should still be caught
    rate3 = hallucinated_join_rate(
        ["c.entry_id = m.entry_id", "x.foo = y.bar"],
        ["composition.entry_id = material_entry.entry_id"],
    )
    assert rate3 == 0.5


def test_multi_hop_success():
    result = multi_hop_success(3, True)
    assert result["is_multi_hop"]
    assert result["correct"]
