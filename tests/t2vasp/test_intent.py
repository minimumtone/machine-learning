"""Tests for t2vasp.intent — natural language → calculation type."""

import pytest
from t2vasp.intent import classify, IntentResult


# ── Japanese queries ─────────────────────────────────────────────────

@pytest.mark.parametrize("query, expected_type", [
    ("Ni3Alの構造を最適化して", "relax"),
    ("Ni3Alを緩和計算", "relax"),
    ("BaTiO3のDOSを計算して", "dos"),
    ("Cu酸化物の状態密度を調べたい", "dos"),
    ("バンド構造を計算して", "band"),
    ("BaTiO3の自発分極をBerry phaseで計算して", "polarization"),
    ("誘電定数を求めたい", "dielectric"),
    ("フォノン分散を計算して", "phonon"),
    ("弾性定数を計算", "elastic"),
    ("結晶場分裂を調べて", "crystal_field"),
    ("ヤーン・テラー効果", "crystal_field"),
    ("磁気モーメントを計算", "magnetic"),
    ("NEB計算で反応経路を探索", "neb"),
    ("分子動力学をAIMDで実行", "md"),
    ("SQSランダム固溶体を作って", "sqs"),
])
def test_classify_japanese(query: str, expected_type: str) -> None:
    result = classify(query)
    assert result.calc_type == expected_type


# ── English queries ──────────────────────────────────────────────────

@pytest.mark.parametrize("query, expected_type", [
    ("optimize the crystal structure of Ni3Al", "relax"),
    ("relax Fe BCC", "relax"),
    ("calculate DOS for Cu", "dos"),
    ("band structure of GaAs", "band"),
    ("Berry phase polarization of BaTiO3", "polarization"),
    ("dielectric constant", "dielectric"),
    ("phonon dispersion of Si", "phonon"),
    ("elastic constants of Fe", "elastic"),
    ("crystal field splitting", "crystal_field"),
    ("Jahn-Teller distortion", "crystal_field"),
    ("magnetic ordering", "magnetic"),
    ("NEB transition state", "neb"),
    ("molecular dynamics at 300 K", "md"),
    ("SQS solid solution", "sqs"),
])
def test_classify_english(query: str, expected_type: str) -> None:
    result = classify(query)
    assert result.calc_type == expected_type


# ── Multi-step detection ─────────────────────────────────────────────

def test_multistep_relax_dos() -> None:
    result = classify("構造最適化して、DOSも計算して")
    assert result.is_multi_step
    types = [result.calc_type] + result.secondary_types
    assert "relax" in types
    assert "dos" in types


def test_multistep_relax_band() -> None:
    result = classify("optimize structure and compute band structure")
    assert result.is_multi_step


# ── Default fallback ─────────────────────────────────────────────────

def test_default_relax() -> None:
    result = classify("Ni3Al")
    assert result.calc_type == "relax"
    assert result.confidence < 0.5


# ── IntentResult fields ──────────────────────────────────────────────

def test_intent_result_fields() -> None:
    result = classify("Ni3Alの構造を最適化して")
    assert isinstance(result, IntentResult)
    assert result.raw_query == "Ni3Alの構造を最適化して"
    assert len(result.matched_keywords) > 0
