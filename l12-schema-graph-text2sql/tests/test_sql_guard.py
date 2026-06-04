"""Tests for safety.sql_guard — empty-result diagnosis helpers."""
from safety.sql_guard import _extract_referenced_entities


def test_extract_elements():
    sql = "SELECT * FROM composition c WHERE c.element = 'Xe'"
    ents = _extract_referenced_entities(sql)
    assert ents["elements"] == ["Xe"]


def test_extract_multiple_elements():
    sql = (
        "SELECT * FROM material_entry m "
        "WHERE EXISTS (SELECT 1 FROM composition c_ni WHERE c_ni.entry_id = m.entry_id AND c_ni.element = 'Ni') "
        "AND EXISTS (SELECT 1 FROM composition c_al WHERE c_al.entry_id = m.entry_id AND c_al.element = 'Al')"
    )
    ents = _extract_referenced_entities(sql)
    assert set(ents["elements"]) == {"Ni", "Al"}


def test_extract_prototype():
    sql = "SELECT * FROM structure s WHERE s.prototype = 'AuCu3' OR s.strukturbericht = 'L12'"
    ents = _extract_referenced_entities(sql)
    assert "AuCu3" in ents["prototypes"]
    assert "L12" in ents["prototypes"]


def test_extract_formula():
    sql = "SELECT * FROM material_entry m WHERE m.formula = 'Ni3Al'"
    ents = _extract_referenced_entities(sql)
    assert ents["formulas"] == ["Ni3Al"]


def test_extract_no_entities():
    sql = "SELECT COUNT(*) FROM material_entry m LIMIT 10000"
    ents = _extract_referenced_entities(sql)
    assert ents["elements"] == []
    assert ents["prototypes"] == []
    assert ents["formulas"] == []


def test_extract_mixed():
    sql = (
        "SELECT m.formula FROM material_entry m "
        "JOIN composition c ON c.entry_id = m.entry_id "
        "JOIN structure s ON s.entry_id = m.entry_id "
        "WHERE c.element = 'Pt' AND s.prototype = 'AuCu3' "
        "AND m.formula = 'Pt3Ti'"
    )
    ents = _extract_referenced_entities(sql)
    assert ents["elements"] == ["Pt"]
    assert ents["prototypes"] == ["AuCu3"]
    assert ents["formulas"] == ["Pt3Ti"]
