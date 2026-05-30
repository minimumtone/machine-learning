#!/usr/bin/env python3
"""LLM vs Rule-based comparison verification.

Runs the same 39 test cases in both Rule-based (no API key) and LLM (GPT-5) modes,
compares SQL generation quality, execution results, and latency.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from llm.entity_extractor import extract_conditions
from llm.sql_generator import pipeline as schema_graph_pipeline, generate_sql_via_llm
from llm.schema_linker import link_schema
from llm.few_shot_store import retrieve_similar
from safety.sql_guard import execute_sql
from safety.sql_validator import validate_sql

# ── ENV ──
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")
os.environ.setdefault("POSTGRES_DB", "l12_materials")
os.environ.setdefault("POSTGRES_USER", "l12_user")
os.environ.setdefault("POSTGRES_PASSWORD", "l12_password")

LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-5")
API_KEY = os.environ.get("OPENAI_API_KEY", "")

# ── Test cases (same as comprehensive_verification.py) ──
TESTS: list[dict[str, Any]] = []

def T(test_id, category, nl_query, *, expect_rejected=False, notes=""):
    TESTS.append(dict(
        test_id=test_id, category=category, nl_query=nl_query,
        expect_rejected=expect_rejected, notes=notes,
    ))

# A: Normal
T("A01","normal","Feを含むB2化合物を出して", notes="Single element + single prototype")
T("A02","normal","安定なL1₂化合物を形成エネルギーが低い順に出して", notes="Stability + sort")
T("A03","normal","NiとAlを両方含む化合物を出して", notes="Multi-element AND with EXISTS")
T("A04","normal","B2化合物の全リストを出して", notes="Full prototype listing")
T("A05","normal","準安定なB2化合物を出して", notes="Metastable filter")
T("A06","normal","Coを含む安定なL1₂化合物を出して", notes="Element + stability")
T("A07","normal","FeとAlを含むB2とL1₂化合物を出して", notes="Multi-element + multi-prototype")
T("A08","normal","γ'化合物のリストを出して", notes="gamma prime notation → L12")
T("A09","normal","ニッケルを含むL1₂型化合物を出して", notes="Japanese element name")
T("A10","normal","L1₂化合物の全データを出して", notes="Full L12 listing")
T("A11","normal","Tiを含むB2化合物を格子定数が大きい順に出して", notes="Sort by lattice_a DESC")
T("A12","normal","CsCl型化合物を出して", notes="CsCl alias for B2")
T("A13","normal","Cu₃Au型化合物を出して", notes="Cu3Au alias for L12")
T("A14","normal","鉄を含むB2化合物を出して", notes="Japanese element name (iron)")
T("A15","normal","安定なL1₂化合物でNiを含むものを出して", notes="Stability + element")
# B: No results
T("B01","no_results","Xeを含むB2化合物を出して", notes="Xe not in dictionary")
T("B02","no_results","UとPuを含むL1₂化合物を出して", notes="U/Pu not in dictionary")
T("B03","no_results","RnとOgを含むB2化合物を出して", notes="Rn/Og not in dictionary")
T("B04","no_results","ScとIrを含む安定なB2化合物を出して", notes="Sc+Ir B2 stable")
T("B05","no_results","PtとGaを含む安定なB2化合物を出して", notes="PtGa B2 stable: likely 0")
# C: Sloppy
T("C01","sloppy","安定な化合物を出して", notes="Missing prototype")
T("C02","sloppy","B2", notes="Minimal input")
T("C03","sloppy","なにか安定なものを出して", notes="Vague input")
T("C04","sloppy","", notes="Empty input")
T("C05","sloppy","今日の天気を教えて", notes="Irrelevant query")
T("C06","sloppy","Fe", notes="Single element only")
T("C07","sloppy","格子定数", notes="Property name only")
T("C08","sloppy","安定なB2化合物でband gapが大きいもの", notes="Vague band gap")
T("C09","sloppy","NiAlのL12", notes="Compressed shorthand")
T("C10","sloppy","B2でもL12でもいいからFeを含むやつ", notes="Casual Japanese")
# D: Contradictory
T("D01","contradictory","安定かつ準安定なB2化合物", notes="Contradictory stability")
T("D02","contradictory","Feを含まないFeB2化合物を出して", notes="Self-contradictory")
# E: Rejection
T("E01","rejection","DROP TABLE material_entry;", expect_rejected=True, notes="SQL injection: DROP")
T("E02","rejection","SELECT * FROM material_entry; DELETE FROM composition;", expect_rejected=True, notes="Multiple statements")
T("E03","rejection","Feを含むB2; DROP TABLE structure;", expect_rejected=True, notes="Piggyback injection")
T("E04","rejection","SELECT * FROM secret_passwords", expect_rejected=True, notes="Forbidden table")
T("E05","rejection","INSERT INTO material_entry VALUES ('x','x','x','x',0)", expect_rejected=True, notes="INSERT injection")
# F: Safety
T("F01","safety","SELECT entry_id FROM material_entry", notes="Valid SELECT")
T("F02","safety","UPDATE material_entry SET formula='X'", expect_rejected=True, notes="UPDATE rejected")


JOIN_LIST = [
    "composition.entry_id = material_entry.entry_id",
    "structure.entry_id = material_entry.entry_id",
    "phase_stability.entry_id = material_entry.entry_id",
    "calculation.entry_id = material_entry.entry_id",
    "calculated_property.calculation_id = calculation.calculation_id",
]

ALL_COLUMNS = [
    "material_entry.entry_id", "material_entry.formula",
    "material_entry.reduced_formula", "material_entry.chemical_system",
    "composition.element", "composition.atomic_fraction", "composition.site_label",
    "structure.prototype", "structure.strukturbericht", "structure.lattice_a",
    "structure.lattice_b", "structure.lattice_c", "structure.volume_per_atom",
    "structure.formula_type", "structure.space_group_number", "structure.space_group",
    "phase_stability.formation_energy_per_atom",
    "phase_stability.energy_above_hull", "phase_stability.is_stable",
    "phase_stability.band_gap",
    "calculation.method", "calculation.functional",
    "calculated_property.property_name", "calculated_property.value",
    "calculated_property.unit",
]


def run_rule_based(nl_query: str) -> dict[str, Any]:
    """Run query in Rule-based mode (no API key)."""
    saved_key = os.environ.pop("OPENAI_API_KEY", None)
    try:
        result = schema_graph_pipeline(nl_query)
    finally:
        if saved_key:
            os.environ["OPENAI_API_KEY"] = saved_key
    return result


def run_llm(nl_query: str) -> dict[str, Any]:
    """Run query in LLM mode (GPT-5)."""
    conditions = extract_conditions(nl_query)
    linked = link_schema(conditions)
    result = generate_sql_via_llm(
        user_query=nl_query,
        allowed_tables=linked["required_tables"],
        allowed_columns=[
            c for c in ALL_COLUMNS
            if c.split(".")[0] in linked["required_tables"]
        ],
        allowed_joins=[
            j for j in JOIN_LIST
            if any(t in j for t in linked["required_tables"])
        ],
        model=LLM_MODEL,
        api_key=API_KEY,
    )
    result["conditions"] = conditions
    result["linked_schema"] = linked
    return result


def execute_and_compare(test: dict) -> dict[str, Any]:
    """Run a single test in both modes and compare."""
    entry = {
        "test_id": test["test_id"],
        "category": test["category"],
        "nl_query": test["nl_query"],
        "notes": test["notes"],
    }

    # Skip rejection/safety tests from LLM SQL generation comparison
    if test["expect_rejected"] or test["category"] == "safety":
        val = validate_sql(test["nl_query"])
        entry["rule_based"] = {"skipped": True, "reason": "rejection/safety test"}
        entry["llm"] = {"skipped": True, "reason": "rejection/safety test"}
        entry["validation"] = {"valid": val["valid"], "errors": val["errors"]}
        if test["expect_rejected"]:
            entry["passed_rb"] = not val["valid"]
            entry["passed_llm"] = not val["valid"]
        else:
            entry["passed_rb"] = val["valid"]
            entry["passed_llm"] = val["valid"]
        return entry

    # ── Rule-based ──
    try:
        t0 = time.time()
        rb_res = run_rule_based(test["nl_query"])
        rb_latency = int((time.time() - t0) * 1000)
        rb_sql = rb_res["sql"]
        entry["rule_based"] = {
            "sql": rb_sql,
            "model": rb_res["model"],
            "latency_ms": rb_latency,
            "conditions": rb_res["conditions"],
        }
        # Execute
        if rb_sql:
            db_res = execute_sql(rb_sql)
            entry["rule_based"]["db_result"] = {
                "success": db_res["success"],
                "row_count": db_res.get("row_count", 0),
                "errors": db_res.get("errors", []),
                "sample_formulas": [],
            }
            if db_res["success"] and db_res.get("columns") and "formula" in db_res["columns"]:
                fi = db_res["columns"].index("formula")
                entry["rule_based"]["db_result"]["sample_formulas"] = sorted(set(
                    row[fi] for row in db_res.get("rows", [])
                ))
            entry["passed_rb"] = db_res["success"] and db_res.get("row_count", 0) >= 0
        else:
            entry["rule_based"]["db_result"] = {"success": False, "row_count": 0}
            entry["passed_rb"] = False
    except Exception as e:
        entry["rule_based"] = {"error": str(e), "trace": traceback.format_exc()}
        entry["passed_rb"] = False

    # ── LLM (GPT-5) ──
    try:
        t0 = time.time()
        llm_res = run_llm(test["nl_query"])
        llm_latency = int((time.time() - t0) * 1000)
        llm_sql = llm_res["sql"]
        entry["llm"] = {
            "sql": llm_sql,
            "model": llm_res["model"],
            "tokens": llm_res.get("tokens", 0),
            "latency_ms": llm_latency,
            "few_shot_count": llm_res.get("few_shot_count", 0),
            "few_shot_queries": llm_res.get("few_shot_queries", []),
        }
        # Validate LLM SQL
        if llm_sql:
            val = validate_sql(llm_sql)
            entry["llm"]["validation"] = {
                "valid": val["valid"],
                "errors": val["errors"],
            }
            if val["valid"]:
                db_res = execute_sql(val.get("sql", llm_sql))
                entry["llm"]["db_result"] = {
                    "success": db_res["success"],
                    "row_count": db_res.get("row_count", 0),
                    "errors": db_res.get("errors", []),
                    "sample_formulas": [],
                }
                if db_res["success"] and db_res.get("columns") and "formula" in db_res["columns"]:
                    fi = db_res["columns"].index("formula")
                    entry["llm"]["db_result"]["sample_formulas"] = sorted(set(
                        row[fi] for row in db_res.get("rows", [])
                    ))
                entry["passed_llm"] = db_res["success"]
            else:
                entry["llm"]["db_result"] = {
                    "success": False, "row_count": 0,
                    "errors": val["errors"],
                }
                entry["passed_llm"] = False
        else:
            entry["llm"]["db_result"] = {"success": False, "row_count": 0}
            entry["passed_llm"] = False
    except Exception as e:
        entry["llm"] = {"error": str(e), "trace": traceback.format_exc()}
        entry["passed_llm"] = False

    # ── Compare results ──
    rb_formulas = set(entry.get("rule_based", {}).get("db_result", {}).get("sample_formulas", []))
    llm_formulas = set(entry.get("llm", {}).get("db_result", {}).get("sample_formulas", []))
    if rb_formulas and llm_formulas:
        intersection = rb_formulas & llm_formulas
        union = rb_formulas | llm_formulas
        entry["comparison"] = {
            "rb_count": len(rb_formulas),
            "llm_count": len(llm_formulas),
            "intersection": len(intersection),
            "jaccard": round(len(intersection) / max(len(union), 1), 3),
            "rb_only": sorted(rb_formulas - llm_formulas)[:10],
            "llm_only": sorted(llm_formulas - rb_formulas)[:10],
        }
    else:
        entry["comparison"] = {
            "rb_count": len(rb_formulas),
            "llm_count": len(llm_formulas),
            "note": "Cannot compare (one or both produced no formula results)",
        }

    return entry


def run_all_comparisons() -> dict[str, Any]:
    """Run all tests in both modes."""
    results = []
    rb_passed = 0
    llm_passed = 0

    print("=" * 70)
    print(f"  LLM vs Rule-based Comparison (model: {LLM_MODEL})")
    print("=" * 70)

    for i, test in enumerate(TESTS):
        print(f"  [{i+1:2d}/{len(TESTS)}] {test['test_id']} {test['category']:15s} | "
              f"{test['nl_query'][:45]:45s} ... ", end="", flush=True)
        t0 = time.time()
        try:
            entry = execute_and_compare(test)
        except Exception as e:
            entry = {
                "test_id": test["test_id"],
                "category": test["category"],
                "nl_query": test["nl_query"],
                "passed_rb": False,
                "passed_llm": False,
                "error": str(e),
            }
        elapsed = int((time.time() - t0) * 1000)
        entry["elapsed_ms"] = elapsed
        results.append(entry)
        rb_ok = entry.get("passed_rb", False)
        llm_ok = entry.get("passed_llm", False)
        if rb_ok: rb_passed += 1
        if llm_ok: llm_passed += 1
        rb_tag = "RB:PASS" if rb_ok else "RB:FAIL"
        llm_tag = "LLM:PASS" if llm_ok else "LLM:FAIL"
        print(f"{rb_tag} {llm_tag} ({elapsed}ms)")

    # Summary
    total = len(TESTS)
    total_tokens = sum(
        r.get("llm", {}).get("tokens", 0) for r in results
        if not r.get("llm", {}).get("skipped")
    )
    total_llm_latency = sum(
        r.get("llm", {}).get("latency_ms", 0) for r in results
        if not r.get("llm", {}).get("skipped")
    )
    llm_test_count = sum(1 for r in results if not r.get("llm", {}).get("skipped"))

    summary = {
        "total": total,
        "rule_based": {"passed": rb_passed, "pass_rate": round(rb_passed / total * 100, 1)},
        "llm": {
            "model": LLM_MODEL,
            "passed": llm_passed,
            "pass_rate": round(llm_passed / total * 100, 1),
            "total_tokens": total_tokens,
            "avg_latency_ms": int(total_llm_latency / max(llm_test_count, 1)),
            "test_count": llm_test_count,
        },
        "categories": {},
    }
    for cat in ["normal", "no_results", "sloppy", "contradictory", "rejection", "safety"]:
        cat_results = [r for r in results if r.get("category") == cat]
        cat_rb = sum(1 for r in cat_results if r.get("passed_rb"))
        cat_llm = sum(1 for r in cat_results if r.get("passed_llm"))
        summary["categories"][cat] = {
            "total": len(cat_results),
            "rb_passed": cat_rb,
            "llm_passed": cat_llm,
        }

    print("\n" + "=" * 70)
    print(f"  Rule-based: {rb_passed}/{total} ({summary['rule_based']['pass_rate']}%)")
    print(f"  LLM ({LLM_MODEL}):  {llm_passed}/{total} ({summary['llm']['pass_rate']}%)")
    print(f"  Total tokens: {total_tokens}, Avg latency: {summary['llm']['avg_latency_ms']}ms")
    print("=" * 70)

    return {"summary": summary, "results": results}


def generate_comparison_html(data: dict) -> str:
    """Generate HTML comparison report."""
    S = data["summary"]
    R = data["results"]

    def h(s):
        return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    parts: list[str] = []
    W = parts.append

    W(f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LLM vs Rule-based 比較検証レポート</title>
<style>
:root {{ --pass:#2e7d32; --fail:#c62828; --warn:#ef6c00; --bg:#fafafa; }}
*{{ margin:0; padding:0; box-sizing:border-box; }}
body{{ font-family:'Segoe UI',Roboto,sans-serif; background:var(--bg); color:#333;
       line-height:1.85; padding:24px 32px; max-width:1400px; margin:auto; }}
h1{{ color:#1a237e; border-bottom:3px solid #1a237e; padding-bottom:8px;
     margin:30px 0 16px; font-size:1.8em; }}
h2{{ color:#283593; border-bottom:2px solid #c5cae9; padding-bottom:6px;
     margin:36px 0 14px; font-size:1.4em; }}
h3{{ color:#3949ab; margin:24px 0 10px; font-size:1.15em; }}
p{{ margin:8px 0 10px; }}
table{{ border-collapse:collapse; width:100%; margin:14px 0; font-size:0.92em; }}
th,td{{ border:1px solid #bbb; padding:8px 10px; text-align:left; }}
th{{ background:#e8eaf6; font-weight:600; }}
tr:nth-child(even){{ background:#f5f5f5; }}
.pass{{ color:var(--pass); font-weight:bold; }} .fail{{ color:var(--fail); font-weight:bold; }}
.cards{{ display:flex; gap:18px; flex-wrap:wrap; margin:18px 0; }}
.card{{ background:#fff; border-radius:10px; box-shadow:0 2px 8px rgba(0,0,0,.1);
        padding:18px; flex:1; min-width:190px; text-align:center; }}
.big{{ font-size:2.5em; font-weight:bold; }}
.sql{{ background:#263238; color:#e0e0e0; padding:14px; border-radius:6px;
       overflow-x:auto; font-family:'Fira Code',monospace; font-size:0.87em;
       white-space:pre-wrap; margin:10px 0; }}
.note{{ background:#e3f2fd; border-left:4px solid #1565c0; padding:12px 16px;
        margin:12px 0; border-radius:0 6px 6px 0; }}
.warn-box{{ background:#fff3e0; border-left:4px solid #ef6c00; padding:12px 16px;
            margin:12px 0; border-radius:0 6px 6px 0; }}
details{{ margin:10px 0; }} details summary{{ cursor:pointer; font-weight:600; color:#1565c0; }}
.tag{{ display:inline-block; padding:2px 8px; border-radius:4px;
       font-size:0.85em; font-weight:600; color:#fff; }}
.tag-normal{{ background:#1976d2; }} .tag-no_results{{ background:#7b1fa2; }}
.tag-sloppy{{ background:#ef6c00; }} .tag-contradictory{{ background:#c62828; }}
.tag-rejection{{ background:#d32f2f; }} .tag-safety{{ background:#388e3c; }}
</style>
</head>
<body>

<h1>LLM (GPT-5) vs Rule-based 比較検証レポート</h1>
<p style="color:#666;">Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
<p style="color:#666;">対象読者：材料工学を専門とする研究者・エンジニア</p>

<div class="note">
<b>このレポートについて：</b>
同じ39件のテストケースを Rule-based モード（APIなし）と LLM モード（{h(S['llm']['model'])}）の
両方で実行し、SQL 生成品質・実行結果・レイテンシを比較したものです。
</div>

<h2>1. 全体サマリ</h2>
<div class="cards">
<div class="card"><div class="big">{S['total']}</div>テスト総数</div>
<div class="card"><div class="big pass">{S['rule_based']['passed']}/{S['total']}</div>Rule-based 合格</div>
<div class="card"><div class="big {'pass' if S['llm']['passed'] >= S['rule_based']['passed'] else 'fail'}">{S['llm']['passed']}/{S['total']}</div>{h(S['llm']['model'])} 合格</div>
<div class="card"><div class="big">{S['llm']['total_tokens']:,}</div>LLM 消費トークン</div>
<div class="card"><div class="big">{S['llm']['avg_latency_ms']:,}ms</div>LLM 平均レイテンシ</div>
</div>

<h3>1.1 カテゴリ別比較</h3>
<table>
<tr><th>カテゴリ</th><th>件数</th><th>Rule-based 合格</th><th>{h(S['llm']['model'])} 合格</th><th>差分</th></tr>
""")

    for cat, info in S["categories"].items():
        diff = info["llm_passed"] - info["rb_passed"]
        diff_str = f"+{diff}" if diff > 0 else str(diff)
        diff_cls = "pass" if diff > 0 else ("fail" if diff < 0 else "")
        W(f'<tr><td><span class="tag tag-{cat}">{cat}</span></td>'
          f'<td>{info["total"]}</td>'
          f'<td>{info["rb_passed"]}/{info["total"]}</td>'
          f'<td>{info["llm_passed"]}/{info["total"]}</td>'
          f'<td class="{diff_cls}">{diff_str}</td></tr>')

    W('</table>')

    # ── Section 2: Test-by-test comparison ──
    W('<h2>2. テスト別詳細比較</h2>')
    W('<table>')
    W('<tr><th>ID</th><th>カテゴリ</th><th>クエリ</th>'
      '<th>RB結果</th><th>RB行数</th>'
      '<th>LLM結果</th><th>LLM行数</th>'
      '<th>LLMトークン</th><th>LLMレイテンシ</th>'
      '<th>結果一致</th></tr>')

    for r in R:
        rb_ok = r.get("passed_rb", False)
        llm_ok = r.get("passed_llm", False)
        rb_rows = r.get("rule_based", {}).get("db_result", {}).get("row_count", "—")
        llm_rows = r.get("llm", {}).get("db_result", {}).get("row_count", "—")
        tokens = r.get("llm", {}).get("tokens", "—")
        latency = r.get("llm", {}).get("latency_ms", "—")
        comp = r.get("comparison", {})
        jaccard = comp.get("jaccard")
        match_str = f"{jaccard*100:.0f}%" if jaccard is not None else "—"

        if r.get("rule_based", {}).get("skipped") or r.get("llm", {}).get("skipped"):
            rb_rows = "N/A"
            llm_rows = "N/A"
            tokens = "—"
            latency = "—"
            match_str = "N/A (validation test)"

        W(f'<tr>')
        W(f'<td><b>{r["test_id"]}</b></td>')
        W(f'<td><span class="tag tag-{r["category"]}">{r["category"]}</span></td>')
        W(f'<td>{h(r["nl_query"][:50])}</td>')
        W(f'<td class="{"pass" if rb_ok else "fail"}">{"PASS" if rb_ok else "FAIL"}</td>')
        W(f'<td>{rb_rows}</td>')
        W(f'<td class="{"pass" if llm_ok else "fail"}">{"PASS" if llm_ok else "FAIL"}</td>')
        W(f'<td>{llm_rows}</td>')
        W(f'<td>{tokens}</td>')
        W(f'<td>{latency}ms</td>' if latency != "—" else f'<td>{latency}</td>')
        W(f'<td>{match_str}</td>')
        W('</tr>')

    W('</table>')

    # ── Section 3: SQL comparison details ──
    W('<h2>3. SQL 生成比較（詳細）</h2>')
    W('<p>各テストの Rule-based と LLM が生成した SQL を並べて比較。</p>')

    for r in R:
        if r.get("rule_based", {}).get("skipped"):
            continue
        rb_sql = r.get("rule_based", {}).get("sql", "N/A")
        llm_sql = r.get("llm", {}).get("sql", "N/A")
        llm_err = r.get("llm", {}).get("error", "")
        llm_val_err = r.get("llm", {}).get("validation", {}).get("errors", [])

        W(f'<details><summary>{r["test_id"]}: {h(r["nl_query"][:60])}</summary>')
        W(f'<p><b>Rule-based SQL:</b></p>')
        W(f'<div class="sql">{h(rb_sql)}</div>')
        W(f'<p><b>LLM ({h(S["llm"]["model"])}) SQL:</b></p>')
        W(f'<div class="sql">{h(llm_sql)}</div>')
        if llm_err:
            W(f'<p class="fail"><b>LLM Error:</b> {h(llm_err)}</p>')
        if llm_val_err:
            W(f'<p class="fail"><b>Validation Errors:</b> {h("; ".join(llm_val_err))}</p>')
        comp = r.get("comparison", {})
        if comp.get("jaccard") is not None:
            W(f'<p><b>結果一致率 (Jaccard):</b> {comp["jaccard"]*100:.1f}%'
              f' (RB: {comp["rb_count"]}件, LLM: {comp["llm_count"]}件, 共通: {comp["intersection"]}件)</p>')
            if comp.get("rb_only"):
                W(f'<p>RBのみ: {", ".join(comp["rb_only"][:5])}</p>')
            if comp.get("llm_only"):
                W(f'<p>LLMのみ: {", ".join(comp["llm_only"][:5])}</p>')
        W('</details>')

    # ── Section 4: Analysis ──
    W('<h2>4. 分析と考察</h2>')

    # Count meaningful comparisons
    both_pass = sum(1 for r in R if r.get("passed_rb") and r.get("passed_llm")
                    and not r.get("rule_based", {}).get("skipped"))
    rb_only_pass = sum(1 for r in R if r.get("passed_rb") and not r.get("passed_llm")
                       and not r.get("rule_based", {}).get("skipped"))
    llm_only_pass = sum(1 for r in R if not r.get("passed_rb") and r.get("passed_llm")
                        and not r.get("rule_based", {}).get("skipped"))
    both_fail = sum(1 for r in R if not r.get("passed_rb") and not r.get("passed_llm")
                    and not r.get("rule_based", {}).get("skipped"))

    W(f"""
<h3>4.1 パス/フェイル マトリックス</h3>
<table>
<tr><th></th><th>LLM PASS</th><th>LLM FAIL</th></tr>
<tr><td><b>RB PASS</b></td><td class="pass">{both_pass}</td><td class="warn">{rb_only_pass}</td></tr>
<tr><td><b>RB FAIL</b></td><td class="pass">{llm_only_pass}</td><td class="fail">{both_fail}</td></tr>
</table>

<h3>4.2 コスト分析</h3>
<table>
<tr><th>項目</th><th>Rule-based</th><th>LLM ({h(S['llm']['model'])})</th></tr>
<tr><td>合格率</td><td>{S['rule_based']['pass_rate']}%</td><td>{S['llm']['pass_rate']}%</td></tr>
<tr><td>平均レイテンシ</td><td>&lt;100ms</td><td>{S['llm']['avg_latency_ms']:,}ms</td></tr>
<tr><td>消費トークン (全{S['llm']['test_count']}件)</td><td>0</td><td>{S['llm']['total_tokens']:,}</td></tr>
<tr><td>API コスト概算</td><td>¥0</td><td>〜¥{int(S['llm']['total_tokens'] * 0.006):,}</td></tr>
<tr><td>オフライン動作</td><td class="pass">可能</td><td class="fail">不可（API必須）</td></tr>
</table>

<h3>4.3 LLM モードの利点</h3>
<ul>
<li><b>数値条件の生成：</b>Rule-based では不可能な「band gap &gt; 1.0 eV」のような
数値比較 WHERE 句を LLM は生成できる可能性がある。</li>
<li><b>自由文への対応力：</b>辞書に未登録の表現でも、LLM は文脈から
意図を推定して適切な SQL を生成できる場合がある。</li>
<li><b>Few-Shot の効果：</b>類似の成功事例がプロンプトに注入されることで、
スキーマリンクの精度が向上する。</li>
</ul>

<h3>4.4 LLM モードのリスク</h3>
<div class="warn-box">
<ul>
<li><b>ハルシネーション：</b>存在しないテーブル名やカラム名を生成する可能性がある。
Schema 制約でプロンプトに許可テーブル/カラムを明示しているが、完全には防げない。</li>
<li><b>SQL構文エラー：</b>LLM が生成した SQL が PostgreSQL の構文として不正な場合がある。
SQL Guard で検証し、repair_loop で修復を試みるが、修復に失敗するケースもある。</li>
<li><b>レイテンシ：</b>Rule-based の &lt;100ms に対し、LLM は数秒〜数十秒かかる。
インタラクティブな利用には不向きな場合がある。</li>
<li><b>コスト：</b>API 呼び出しごとに課金が発生する。大量クエリの場合はコスト管理が必要。</li>
<li><b>セキュリティ：</b>ユーザーのクエリ内容が外部 API に送信される。
機密データを含むクエリには不適切。</li>
</ul>
</div>
""")

    W(f"""
<footer style="margin-top:40px; padding-top:16px; border-top:1px solid #ccc; color:#888; font-size:0.85em;">
L1<sub>2</sub>/B2 Schema-Graph-Assisted Text-to-SQL System &mdash;
LLM Comparison Report &mdash;
Generated by llm_comparison.py &mdash;
{time.strftime('%Y-%m-%d %H:%M UTC')}
</footer>
</body>
</html>
""")

    return "\n".join(parts)


if __name__ == "__main__":
    print(f"\nUsing model: {LLM_MODEL}")
    print(f"API key: {'SET' if API_KEY and API_KEY != 'your_api_key_here' else 'NOT SET'}\n")

    data = run_all_comparisons()

    # Save JSON
    json_path = Path(__file__).parent / "llm_comparison_results.json"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nJSON results: {json_path}")

    # Save HTML
    html = generate_comparison_html(data)
    html_path = Path(__file__).parent / "llm_comparison_report.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"HTML report:  {html_path}")
