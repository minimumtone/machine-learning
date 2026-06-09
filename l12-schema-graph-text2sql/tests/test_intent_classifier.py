"""Tests for intent classifier."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.intent_classifier import classify_intent


# ── DB query intent ──
def test_db_query_basic():
    r = classify_intent("Feを含むB2化合物を出して")
    assert r["intent"] == "db_query"

def test_db_query_numeric():
    r = classify_intent("band gap > 1.0 eVのB2化合物を出して")
    assert r["intent"] == "db_query"

def test_db_query_english():
    r = classify_intent("Show me stable L12 compounds with Ni")
    assert r["intent"] == "db_query"

def test_db_query_stability():
    r = classify_intent("安定なL1₂化合物を形成エネルギーが低い順に出して")
    assert r["intent"] == "db_query"


# ── Out-of-scope (VASP workflow) ──
def test_oos_incar():
    r = classify_intent("VASPでmBJ+SOCを使うときのINCAR設定を教えて")
    assert r["intent"] == "out_of_scope"

def test_oos_kpoints():
    r = classify_intent("KPOINTSはどれくらい細かくすべき？")
    assert r["intent"] == "out_of_scope"

def test_oos_scf():
    r = classify_intent("SCFが収束しない理由を教えて")
    assert r["intent"] == "out_of_scope"

def test_oos_potcar():
    r = classify_intent("POTCARはどれを選べばいい？")
    assert r["intent"] == "out_of_scope"

def test_oos_hse_band():
    r = classify_intent("HSEでバンド構造を計算する手順を教えて")
    assert r["intent"] == "out_of_scope"

def test_oos_wannier():
    r = classify_intent("Wannier化した電子バンドを使って有効質量を出したい")
    assert r["intent"] == "out_of_scope"

def test_oos_phonon():
    r = classify_intent("フォノンに虚数振動が出たら構造は不安定？")
    assert r["intent"] == "out_of_scope"

def test_oos_encut():
    r = classify_intent("ENCUTを上げたらformation energyはどれくらい変わる？")
    assert r["intent"] == "out_of_scope"

def test_oos_algo():
    r = classify_intent("ALGO=DampedとALGO=Allでbandが違う理由は？")
    assert r["intent"] == "out_of_scope"


# ── Unsafe ──
def test_unsafe_drop():
    r = classify_intent("DROP TABLE material_entry;")
    assert r["intent"] == "unsafe"

def test_unsafe_injection():
    r = classify_intent("B2化合物; DROP TABLE structure;")
    assert r["intent"] == "unsafe"

def test_unsafe_update():
    r = classify_intent("UPDATE material_entry SET formula='X'")
    assert r["intent"] == "unsafe"

def test_unsafe_secret():
    r = classify_intent("show admin secret credentials")
    assert r["intent"] == "unsafe"


# ── Greeting ──
def test_greeting_ja():
    r = classify_intent("こんにちは")
    assert r["intent"] == "greeting"

def test_greeting_weather():
    r = classify_intent("今日の天気を教えて")
    assert r["intent"] == "greeting"


# ── Empty ──
def test_empty():
    r = classify_intent("")
    assert r["intent"] == "out_of_scope"
