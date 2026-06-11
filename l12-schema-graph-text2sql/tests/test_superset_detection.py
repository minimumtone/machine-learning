"""Tests for superset / condition-insufficiency detection in repair_loop."""
from llm.repair_loop import (
    count_sql_conditions,
    count_expected_conditions,
    detect_superset,
    _build_superset_msg,
)


def test_count_sql_conditions_simple():
    sql = "SELECT * FROM material_entry WHERE formula = 'Ni3Al'"
    assert count_sql_conditions(sql) == 1  # WHERE only


def test_count_sql_conditions_multiple():
    sql = (
        "SELECT * FROM material_entry m "
        "JOIN structure s ON s.entry_id = m.entry_id "
        "WHERE s.prototype = 'L12' AND m.formula LIKE '%Ni%'"
    )
    assert count_sql_conditions(sql) == 2  # WHERE + AND


def test_count_sql_conditions_complex():
    sql = (
        "SELECT * FROM material_entry m "
        "JOIN structure s ON s.entry_id = m.entry_id "
        "JOIN phase_stability ps ON ps.entry_id = m.entry_id "
        "WHERE s.prototype = 'L12' AND ps.is_stable = TRUE "
        "AND m.formula LIKE '%Ni%' OR m.formula LIKE '%Co%'"
    )
    # Fix B13: OR expands result set, not counted as restriction
    assert count_sql_conditions(sql) == 3  # WHERE + 2 AND (OR not counted)


def test_count_sql_conditions_no_where():
    sql = "SELECT * FROM material_entry LIMIT 100"
    assert count_sql_conditions(sql) == 0


def test_count_sql_conditions_ignores_string_literals():
    sql = "SELECT * FROM t WHERE name = 'AND OR WHERE'"
    # String literal replaced, so only 1 WHERE condition
    assert count_sql_conditions(sql) == 1


def test_count_expected_conditions_empty():
    assert count_expected_conditions({}) == 0


def test_count_expected_conditions_elements():
    conditions = {"contains_elements": ["Ni", "Al"]}
    assert count_expected_conditions(conditions) == 2


def test_count_expected_conditions_multiple():
    conditions = {
        "prototype": "L12",
        "contains_elements": ["Ni", "Al"],
        "stability": "stable",
        "numeric_conditions": [{"column": "band_gap", "op": ">", "value": 1.0}],
    }
    assert count_expected_conditions(conditions) == 5  # 1 proto + 2 elem + 1 stab + 1 num


def test_detect_superset_true_missing_conditions():
    sql = "SELECT * FROM material_entry m JOIN structure s ON s.entry_id = m.entry_id WHERE s.prototype = 'L12' LIMIT 10000"
    conditions = {
        "prototype": "L12",
        "contains_elements": ["Ni", "Al"],
        "stability": "stable",
    }
    result = detect_superset(sql, 500, conditions)
    assert result["is_superset"] is True
    assert result["sql_conditions"] == 1
    assert result["expected_conditions"] == 4


def test_detect_superset_false_sufficient():
    sql = (
        "SELECT * FROM material_entry m "
        "JOIN structure s ON s.entry_id = m.entry_id "
        "WHERE s.prototype = 'L12' AND m.formula LIKE '%Ni%'"
    )
    conditions = {"prototype": "L12", "contains_elements": ["Ni"]}
    result = detect_superset(sql, 50, conditions)
    assert result["is_superset"] is False


def test_detect_superset_high_row_ratio():
    sql = "SELECT * FROM material_entry LIMIT 10000"
    conditions = {"prototype": "L12", "contains_elements": ["Ni"]}
    result = detect_superset(sql, 1000, conditions)
    assert result["is_superset"] is True
    assert "insufficient filtering" in result["reason"]


def test_detect_superset_no_conditions():
    sql = "SELECT * FROM material_entry LIMIT 10000"
    conditions = {}
    result = detect_superset(sql, 1471, conditions)
    assert result["is_superset"] is False


def test_build_superset_msg_content():
    conditions = {
        "prototype": "L12",
        "contains_elements": ["Ni", "Al"],
    }
    superset_info = {
        "is_superset": True,
        "reason": "SQL has 0 conditions but query mentions 3 constraints",
        "sql_conditions": 0,
        "expected_conditions": 3,
        "row_ratio": 0.7,
    }
    msg = _build_superset_msg(
        "SELECT * FROM material_entry", "L12型でNiとAlを含む化合物",
        1000, conditions, superset_info,
    )
    assert "SUPERSET" in msg
    assert "1000 rows" in msg
    assert "Ni" in msg and "Al" in msg
    assert "L12" in msg
    assert "missing WHERE conditions" in msg.lower() or "Missing conditions" in msg
