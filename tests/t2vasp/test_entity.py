"""Tests for t2vasp.entity — element / formula / prototype extraction."""

import pytest
from t2vasp.entity import extract, parse_formula, EntityResult


# ── parse_formula ────────────────────────────────────────────────────

@pytest.mark.parametrize("formula, expected", [
    ("Ni3Al", {"Ni": 3.0, "Al": 1.0}),
    ("BaTiO3", {"Ba": 1.0, "Ti": 1.0, "O": 3.0}),
    ("Fe", {"Fe": 1.0}),
    ("Cu2O", {"Cu": 2.0, "O": 1.0}),
    ("SrTiO3", {"Sr": 1.0, "Ti": 1.0, "O": 3.0}),
    ("CrFeCoNi", {"Cr": 1.0, "Fe": 1.0, "Co": 1.0, "Ni": 1.0}),
])
def test_parse_formula(formula: str, expected: dict) -> None:
    assert parse_formula(formula) == expected


def test_parse_formula_parenthesized() -> None:
    result = parse_formula("(CrFeCoNi)0.25")
    assert abs(result["Cr"] - 0.25) < 1e-10
    assert abs(result["Fe"] - 0.25) < 1e-10


# ── extract: formula from query ──────────────────────────────────────

def test_extract_formula_ni3al() -> None:
    result = extract("Ni3AlのL12構造を最適化して")
    assert result.formula_str == "Ni3Al"
    assert result.composition == {"Ni": 3.0, "Al": 1.0}
    assert result.elements == ["Ni", "Al"]


def test_extract_formula_batio3() -> None:
    result = extract("BaTiO3のペロブスカイト構造")
    assert result.composition == {"Ba": 1.0, "Ti": 1.0, "O": 3.0}


def test_extract_formula_crfeconi() -> None:
    result = extract("CrFeCoNiのBCC SQSを作って")
    assert set(result.elements) == {"Cr", "Fe", "Co", "Ni"}


# ── extract: prototype ───────────────────────────────────────────────

@pytest.mark.parametrize("query, expected_proto", [
    ("Ni3AlのL12構造", "L12"),
    ("NiAlのB2構造", "B2"),
    ("BaTiO3のペロブスカイト", "perovskite"),
    ("NaClの岩塩型構造", "rocksalt"),
    ("TiO2のルチル型", "rutile"),
    ("Fe BCC構造", "BCC"),
    ("Cu FCC構造", "FCC"),
    ("Zn wurtzite structure", "wurtzite"),
    ("GaAs zincblende", "zincblende"),
])
def test_extract_prototype(query: str, expected_proto: str) -> None:
    result = extract(query)
    assert result.prototype == expected_proto


# ── extract: Japanese element names ──────────────────────────────────

def test_extract_japanese_element() -> None:
    result = extract("ニッケルの構造を計算して")
    assert "Ni" in result.elements


def test_extract_japanese_element_iron() -> None:
    result = extract("鉄の構造を最適化")
    assert "Fe" in result.elements


# ── extract: VASP parameters ────────────────────────────────────────

def test_extract_encut() -> None:
    result = extract("ENCUT=600でNi3Alを計算")
    assert result.encut == 600


def test_extract_kpoints() -> None:
    result = extract("k点8×8×8でFe計算")
    assert result.kpoints == (8, 8, 8)


def test_extract_spin() -> None:
    result = extract("スピン偏極計算でFe")
    assert result.spin_polarized is True


# ── EntityResult.species_list ────────────────────────────────────────

def test_species_list() -> None:
    result = extract("Ni3Al")
    assert result.species_list == ["Ni", "Al"]
