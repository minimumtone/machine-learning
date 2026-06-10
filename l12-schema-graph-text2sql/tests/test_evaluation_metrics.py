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
    # Fix B5: first arg is now a SQL string, not a list of joins
    sql = "SELECT * FROM a JOIN b ON a.x = b.y"
    rate = hallucinated_join_rate(
        sql,
        ["a.x = b.y", "c.z = d.w"],
    )
    assert rate == 0.0


def test_hallucinated_join_rate_alias_resolution():
    """Test that alias-form joins are resolved to table-form before comparison."""
    # Fix B5: first arg is now a SQL string with aliases
    sql = ("SELECT * FROM material_entry AS m "
           "JOIN structure AS s ON s.entry_id = m.entry_id "
           "JOIN phase_stability AS ps ON ps.entry_id = m.entry_id")
    rate = hallucinated_join_rate(
        sql,
        ["structure.entry_id = material_entry.entry_id",
         "phase_stability.entry_id = material_entry.entry_id"],
    )
    assert rate == 0.0

    # Reversed order should also match (canonical sorting)
    sql2 = "SELECT * FROM material_entry AS m JOIN structure AS s ON m.entry_id = s.entry_id"
    rate2 = hallucinated_join_rate(
        sql2,
        ["structure.entry_id = material_entry.entry_id"],
    )
    assert rate2 == 0.0

    # Truly hallucinated join should still be caught
    sql3 = ("SELECT * FROM material_entry AS m "
            "JOIN structure AS s ON s.entry_id = m.entry_id "
            "JOIN x ON x.foo = y.bar")
    rate3 = hallucinated_join_rate(
        sql3,
        ["structure.entry_id = material_entry.entry_id"],
    )
    assert rate3 == 0.5


def test_multi_hop_success():
    result = multi_hop_success(3, True)
    assert result["is_multi_hop"]
    assert result["correct"]
