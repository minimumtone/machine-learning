"""Tests for coverage score, numeric condition parser, and chemical formula parser."""
from __future__ import annotations


from llm.entity_extractor import (
    extract_conditions,
    extract_formula,
    extract_numeric_conditions,
)


class TestNumericParser:
    def test_band_gap_gt(self):
        results = extract_numeric_conditions("band gap > 1.0 eV")
        assert len(results) == 1
        assert results[0]["column"] == "phase_stability.band_gap"
        assert results[0]["operator"] == ">"
        assert results[0]["value"] == 1.0

    def test_band_gap_ja_ijo(self):
        results = extract_numeric_conditions("band gapが1 eV以上")
        assert len(results) == 1
        assert results[0]["operator"] == ">="
        assert results[0]["value"] == 1.0

    def test_formation_energy_negative(self):
        results = extract_numeric_conditions("形成エネルギーが負の化合物")
        assert len(results) == 1
        assert results[0]["column"] == "phase_stability.formation_energy_per_atom"
        assert results[0]["operator"] == "<"
        assert results[0]["value"] == 0

    def test_ehull_lt(self):
        results = extract_numeric_conditions("Ehull < 0.05 eV/atom")
        assert len(results) == 1
        assert results[0]["column"] == "phase_stability.energy_above_hull"
        assert results[0]["value"] == 0.05

    def test_lattice_ja_ijo(self):
        results = extract_numeric_conditions("格子定数が3.5 Å以上")
        assert len(results) == 1
        assert results[0]["column"] == "structure.lattice_a"
        assert results[0]["operator"] == ">="
        assert results[0]["value"] == 3.5


class TestFormulaParser:
    def test_nial_contains_elements(self):
        result = extract_formula("NiAl L12")
        assert result is not None
        assert result["interpretation"] == "contains_elements"
        assert sorted(result["elements"]) == ["Al", "Ni"]

    def test_ni3al_exact_formula(self):
        result = extract_formula("Ni3Al")
        assert result is not None
        assert result["interpretation"] == "exact_formula"
        assert result["composition"]["Ni"] == 3.0
        assert result["composition"]["Al"] == 1.0

    def test_no_formula(self):
        result = extract_formula("B2化合物を出して")
        assert result is None

    def test_ternary_formula(self):
        result = extract_formula("FeCoNi")
        assert result is not None
        assert sorted(result["elements"]) == ["Co", "Fe", "Ni"]


class TestCoverageScore:
    def test_known_elements_high_coverage(self):
        conds = extract_conditions("Feを含むB2化合物を出して")
        cov = conds["_coverage"]
        assert cov["coverage_score"] > 0.5
        assert not cov["unknown_elements"]

    def test_unknown_element_detected(self):
        conds = extract_conditions("Xeを含むB2化合物を出して")
        cov = conds["_coverage"]
        assert "Xe" in cov["unknown_elements"]
        assert cov["action"] == "fallback_to_llm"

    def test_irrelevant_query_low_coverage(self):
        conds = extract_conditions("今日の天気を教えて")
        cov = conds["_coverage"]
        assert cov["coverage_score"] < 0.3
        assert cov["action"] == "clarification_required"

    def test_full_coverage(self):
        conds = extract_conditions("band gap > 1.0 eVのB2化合物")
        cov = conds["_coverage"]
        assert cov["coverage_score"] >= 0.8
        assert cov["action"] == "execute_rule_based"

    def test_numeric_conditions_in_extract(self):
        conds = extract_conditions("band gap > 1.0 eVのB2化合物")
        assert "numeric_conditions" in conds
        assert conds["numeric_conditions"][0]["column"] == "phase_stability.band_gap"

    def test_formula_in_extract(self):
        conds = extract_conditions("Ni3Al")
        assert "formula" in conds
        assert conds["formula"]["interpretation"] == "exact_formula"
