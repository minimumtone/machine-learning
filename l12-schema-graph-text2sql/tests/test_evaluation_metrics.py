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


def test_multi_hop_success():
    result = multi_hop_success(3, True)
    assert result["is_multi_hop"]
    assert result["correct"]
