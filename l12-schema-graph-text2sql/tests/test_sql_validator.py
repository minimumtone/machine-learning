"""Tests for safety.sql_validator."""
from safety.sql_validator import (
    check_forbidden_keywords,
    check_multiple_statements,
    check_select_only,
    check_limit,
    check_allowed_tables,
    validate_sql,
)


ALLOWED = [
    "material_entry", "composition", "structure",
    "calculation", "calculated_property", "phase_stability",
    "prototype_definition",
]


def test_forbidden_drop():
    assert "DROP" in check_forbidden_keywords("DROP TABLE material_entry;")


def test_forbidden_delete():
    assert "DELETE" in check_forbidden_keywords("DELETE FROM material_entry;")


def test_no_forbidden_in_select():
    assert check_forbidden_keywords("SELECT * FROM material_entry LIMIT 10;") == []


def test_multiple_statements():
    assert check_multiple_statements("SELECT 1; DELETE FROM x;")
    assert not check_multiple_statements("SELECT 1;")


def test_select_only():
    assert check_select_only("SELECT * FROM t")
    assert not check_select_only("INSERT INTO t VALUES (1)")


def test_limit_present():
    has, sql = check_limit("SELECT * FROM t LIMIT 10;")
    assert has


def test_limit_added():
    has, sql = check_limit("SELECT * FROM t")
    assert not has
    assert "LIMIT" in sql


def test_allowed_tables():
    bad = check_allowed_tables("SELECT * FROM secret_table", ALLOWED)
    assert "secret_table" in bad


def test_validate_valid_sql():
    result = validate_sql(
        "SELECT formula FROM material_entry LIMIT 10;"
    )
    assert result["valid"]


def test_validate_drop_rejected():
    result = validate_sql("DROP TABLE material_entry;")
    assert not result["valid"]
    assert any("Forbidden" in e for e in result["errors"])


def test_validate_multi_statement_rejected():
    result = validate_sql("SELECT * FROM material_entry; DELETE FROM material_entry;")
    assert not result["valid"]


def test_validate_unknown_table_rejected():
    result = validate_sql("SELECT * FROM secret_table LIMIT 10;")
    assert not result["valid"]


def test_validate_imaginary_column():
    sql = "SELECT imaginary_column FROM material_entry LIMIT 10;"
    result = validate_sql(sql)
    assert result["valid"]
