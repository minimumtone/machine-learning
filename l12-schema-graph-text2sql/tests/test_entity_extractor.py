"""Tests for llm.entity_extractor."""
from llm.entity_extractor import (
    extract_prototype,
    extract_elements,
    extract_stability,
    extract_conditions,
    extract_lattice_reference,
)


def test_extract_prototype_l12():
    assert extract_prototype("L1₂構造を持つ化合物") == "L12"


def test_extract_prototype_gamma_prime():
    assert extract_prototype("γ'相候補を探して") == "L12"


def test_extract_prototype_cu3au():
    assert extract_prototype("Cu3Au型構造") == "L12"


def test_extract_elements_ni():
    assert "Ni" in extract_elements("Niを含む化合物")


def test_extract_elements_multi():
    elems = extract_elements("NiとAlの両方を含む")
    assert "Ni" in elems
    assert "Al" in elems


def test_extract_stability_stable():
    assert extract_stability("安定なL1₂型化合物") == "stable"


def test_extract_stability_metastable():
    assert extract_stability("準安定な化合物") == "metastable"


def test_extract_conditions_full():
    cond = extract_conditions(
        "Niを含む安定なL1₂型化合物を形成エネルギーが低い順に出して"
    )
    assert cond["prototype"] == "L12"
    assert "Ni" in cond["contains_elements"]
    assert cond["stability"] == "stable"
    assert cond["sort_order"] == "asc"


def test_extract_lattice_reference():
    ref = extract_lattice_reference("Ni₃Alに近い格子定数を持つ候補を探して")
    assert ref is not None
    assert ref["reference_formula"] == "Ni3Al"
    assert abs(ref["reference_lattice_a"] - 3.572) < 0.01


def test_conditions_l12_list():
    cond = extract_conditions("L1₂構造を持つ化合物を一覧にして")
    assert cond["prototype"] == "L12"


def test_conditions_ni_stable_l12():
    cond = extract_conditions("Niを含む安定なL1₂化合物を抽出して")
    assert cond["prototype"] == "L12"
    assert "Ni" in cond["contains_elements"]
    assert cond["stability"] == "stable"


def test_conditions_al_metastable():
    cond = extract_conditions("Alを含む準安定L1₂化合物を形成エネルギー順に出して")
    assert cond["prototype"] == "L12"
    assert "Al" in cond["contains_elements"]
    assert cond["stability"] == "metastable"


def test_conditions_lattice_near_ni3al():
    cond = extract_conditions("Ni₃Alに近い格子定数を持つL1₂候補を探して")
    assert cond["prototype"] == "L12"
    assert "lattice_reference" in cond
