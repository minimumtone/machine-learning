"""Tests for safety.sql_validator."""
from safety.sql_validator import (
    check_allowed_columns,
    check_forbidden_keywords,
    check_multiple_statements,
    check_select_only,
    check_limit,
    check_allowed_tables,
    check_system_tables,
    check_cte_bodies_select_only,
    check_column_type_safety,
    suggest_column_correction,
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
    """Unqualified imaginary columns should be rejected when FROM table is known."""
    sql = "SELECT imaginary_column FROM material_entry LIMIT 10;"
    result = validate_sql(sql)
    assert not result["valid"]
    assert any("imaginary_column" in e for e in result["errors"])


# ── System table blocking tests ──


def test_system_table_pg_shadow():
    violations = check_system_tables("SELECT * FROM pg_shadow LIMIT 10")
    assert "pg_shadow" in violations


def test_system_table_pg_authid():
    violations = check_system_tables("SELECT * FROM pg_authid LIMIT 10")
    assert "pg_authid" in violations


def test_system_table_pg_stat_activity():
    violations = check_system_tables(
        "SELECT * FROM pg_stat_activity LIMIT 10"
    )
    assert "pg_stat_activity" in violations


def test_system_table_prefix_pg():
    violations = check_system_tables(
        "SELECT * FROM pg_some_new_table LIMIT 10"
    )
    assert len(violations) == 1


def test_system_table_normal_table_ok():
    violations = check_system_tables(
        "SELECT * FROM material_entry LIMIT 10"
    )
    assert violations == []


def test_system_table_validate_rejects():
    result = validate_sql("SELECT * FROM pg_shadow LIMIT 10")
    assert not result["valid"]
    assert any("System table" in e for e in result["errors"])


# ── CTE body SELECT-only tests ──


def test_cte_select_only_valid():
    sql = """
    WITH stable AS (
        SELECT entry_id FROM phase_stability
        WHERE energy_above_hull <= 0.001
    )
    SELECT m.formula FROM material_entry m
    JOIN stable s ON s.entry_id = m.entry_id
    LIMIT 10000
    """
    violations = check_cte_bodies_select_only(sql)
    assert violations == []


def test_cte_writable_insert_blocked():
    sql = """
    WITH inserted AS (
        INSERT INTO material_entry (formula) VALUES ('HACK') RETURNING entry_id
    )
    SELECT * FROM inserted
    """
    violations = check_cte_bodies_select_only(sql)
    assert len(violations) > 0
    assert any("INSERT" in v.upper() for v in violations)


def test_cte_writable_delete_blocked():
    sql = """
    WITH deleted AS (
        DELETE FROM material_entry WHERE entry_id = 1 RETURNING entry_id
    )
    SELECT * FROM deleted
    """
    violations = check_cte_bodies_select_only(sql)
    assert len(violations) > 0
    assert any("DELETE" in v.upper() for v in violations)


def test_cte_writable_update_blocked():
    sql = """
    WITH updated AS (
        UPDATE material_entry SET formula = 'HACK' WHERE entry_id = 1 RETURNING *
    )
    SELECT * FROM updated
    """
    violations = check_cte_bodies_select_only(sql)
    assert len(violations) > 0
    assert any("UPDATE" in v.upper() for v in violations)


def test_cte_validate_rejects_writable():
    sql = """
    WITH inserted AS (
        INSERT INTO material_entry (formula) VALUES ('HACK') RETURNING entry_id
    )
    SELECT * FROM inserted LIMIT 10
    """
    result = validate_sql(sql)
    assert not result["valid"]
    assert any("CTE" in e or "Writable" in e for e in result["errors"])


# ── Column type safety tests ──


def test_type_safety_numeric_vs_string():
    sql = "SELECT * FROM phase_stability ps WHERE ps.energy_above_hull = 'stable' LIMIT 10"
    warnings = check_column_type_safety(sql)
    assert len(warnings) > 0
    assert any("numeric" in w and "string" in w for w in warnings)


def test_type_safety_text_vs_number():
    sql = "SELECT * FROM composition c WHERE c.element > 100 LIMIT 10"
    warnings = check_column_type_safety(sql)
    assert len(warnings) > 0
    assert any("text" in w and "number" in w for w in warnings)


def test_type_safety_correct_usage_no_warning():
    sql = "SELECT * FROM phase_stability ps WHERE ps.energy_above_hull <= 0.001 LIMIT 10"
    warnings = check_column_type_safety(sql)
    assert warnings == []


def test_type_safety_string_for_text_ok():
    sql = "SELECT * FROM composition c WHERE c.element = 'Ni' LIMIT 10"
    warnings = check_column_type_safety(sql)
    assert warnings == []


def test_type_safety_in_validate_as_warning():
    sql = "SELECT * FROM phase_stability ps WHERE ps.energy_above_hull = 'stable' LIMIT 10"
    result = validate_sql(sql)
    # Type mismatch is a warning, not an error — SQL still passes validation
    assert result["valid"]
    assert any("Type mismatch" in w for w in result["warnings"])


# ── Column hallucination prevention tests ──

ALLOWED_COLS = [
    "composition.entry_id", "composition.element", "composition.atomic_fraction",
    "structure.entry_id", "structure.prototype", "structure.lattice_a",
    "calculation.entry_id", "calculation.functional",
    "phase_stability.entry_id", "phase_stability.is_stable",
]


def test_column_synonym_correction_fractional_amount():
    """Synonym map: fractional_amount -> atomic_fraction."""
    result = suggest_column_correction("composition.fractional_amount", ALLOWED_COLS)
    assert result == "composition.atomic_fraction"


def test_column_synonym_correction_xc_functional():
    """Synonym map: xc_functional -> functional."""
    result = suggest_column_correction("calculation.xc_functional", ALLOWED_COLS)
    assert result == "calculation.functional"


def test_column_fuzzy_match():
    """Fuzzy match: atomc_fraction (typo) -> atomic_fraction."""
    result = suggest_column_correction("composition.atomc_fraction", ALLOWED_COLS)
    assert result == "composition.atomic_fraction"


def test_column_no_suggestion_for_unrelated():
    """No suggestion for completely unrelated column names."""
    result = suggest_column_correction("composition.xyz_zzz_unknown", ALLOWED_COLS)
    assert result is None


def test_check_allowed_columns_with_suggestion():
    """check_allowed_columns includes correction suggestion in error message."""
    sql = "SELECT c.element, c.fractional_amount FROM composition c LIMIT 10"
    bad = check_allowed_columns(sql, ALLOWED_COLS)
    assert len(bad) == 1
    assert "fractional_amount" in bad[0]
    assert "composition.atomic_fraction" in bad[0]


def test_check_allowed_columns_correct():
    """No errors for SQL using only valid column names."""
    sql = "SELECT c.element, c.atomic_fraction FROM composition c LIMIT 10"
    bad = check_allowed_columns(sql, ALLOWED_COLS)
    assert bad == []


def test_validate_sql_rejects_hallucinated_column():
    """validate_sql() rejects SQL with hallucinated columns via allowed_schema.yaml."""
    sql = "SELECT c.fractional_amount FROM composition c LIMIT 10"
    result = validate_sql(sql)
    assert not result["valid"]
    assert result["classification"] == "rejected_schema"
    assert any("fractional_amount" in e for e in result["errors"])
