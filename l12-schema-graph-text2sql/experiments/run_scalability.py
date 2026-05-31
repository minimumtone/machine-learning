#!/usr/bin/env python3
"""Scalability measurement: latency vs query complexity and DB size.

Measures:
1. Latency by query type (element, prototype, numeric, multi-element, sorting)
2. Latency by mode (rule-based vs LLM)
3. Component-level latency breakdown (extract, link, generate, validate, execute)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.entity_extractor import extract_conditions
from llm.intent_classifier import classify_intent
from llm.schema_linker import link_schema
from llm.sql_generator import (
    generate_sql_via_llm,
    pipeline as schema_graph_pipeline,
    _rule_based_fallback,
)
from safety.sql_guard import execute_sql
from safety.sql_validator import validate_sql

os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")
os.environ.setdefault("POSTGRES_DB", "l12_materials")
os.environ.setdefault("POSTGRES_USER", "l12_user")
os.environ.setdefault("POSTGRES_PASSWORD", "l12_password")

LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-5.5")
API_KEY = os.environ.get("OPENAI_API_KEY", "")
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Queries organized by complexity type
SCALABILITY_QUERIES = {
    "simple_element": [
        "Feを含むB2化合物を出して",
        "Niを含むL12化合物を出して",
        "Coを含む化合物を出して",
    ],
    "simple_prototype": [
        "B2化合物の全リストを出して",
        "L1₂化合物の全データを出して",
    ],
    "numeric_condition": [
        "band gap > 1.0 eVのB2化合物を出して",
        "形成エネルギーが負のL1₂化合物を出して",
        "Ehull < 0.05 eV/atomのB2化合物",
    ],
    "multi_element": [
        "NiとAlを両方含む化合物を出して",
        "FeとAlを含むB2化合物を出して",
        "NiとCoを含むL1₂化合物を出して",
    ],
    "sorting_limit": [
        "安定なL1₂化合物を形成エネルギーが低い順に出して",
        "安定なB2化合物を形成エネルギーが低い順に出して",
    ],
    "compound_query": [
        "Feを含む安定なB2型合金の一覧",
        "band gapが正のB2化合物を大きい順に",
        "energy above hullが0.1 eV/atom以下のB2化合物",
    ],
}

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

ALL_JOINS = [
    "composition.entry_id = material_entry.entry_id",
    "structure.entry_id = material_entry.entry_id",
    "phase_stability.entry_id = material_entry.entry_id",
    "calculation.entry_id = material_entry.entry_id",
    "calculated_property.calculation_id = calculation.calculation_id",
]


def _measure_pipeline_latency(query: str, mode: str = "rule_based",
                                n_runs: int = 3) -> dict[str, Any]:
    """Measure detailed latency breakdown for a single query."""
    timings: list[dict[str, Any]] = []

    for _ in range(n_runs):
        t_total = time.time()

        # Intent classification
        t0 = time.time()
        intent = classify_intent(query)
        t_intent = int((time.time() - t0) * 1000)

        # Entity extraction
        t0 = time.time()
        conditions = extract_conditions(query)
        t_extract = int((time.time() - t0) * 1000)

        # Schema linking
        t0 = time.time()
        linked = link_schema(conditions)
        t_link = int((time.time() - t0) * 1000)

        # SQL generation
        t0 = time.time()
        if mode == "llm" and API_KEY:
            result = generate_sql_via_llm(
                user_query=query,
                allowed_tables=linked["required_tables"],
                allowed_columns=[c for c in ALL_COLUMNS
                                 if c.split(".")[0] in linked["required_tables"]],
                allowed_joins=[j for j in ALL_JOINS
                               if any(t in j for t in linked["required_tables"])],
                model=LLM_MODEL,
                api_key=API_KEY,
            )
            sql = result["sql"]
        else:
            sql = _rule_based_fallback(
                query,
                linked["required_tables"],
                [c for c in ALL_COLUMNS
                 if c.split(".")[0] in linked["required_tables"]],
                [j for j in ALL_JOINS
                 if any(t in j for t in linked["required_tables"])],
            )
        t_generate = int((time.time() - t0) * 1000)

        # Validation
        t0 = time.time()
        validation = validate_sql(sql)
        t_validate = int((time.time() - t0) * 1000)

        # Execution
        t0 = time.time()
        if validation["valid"]:
            exec_result = execute_sql(validation["sql"], validate=False)
            row_count = exec_result.get("row_count", 0)
        else:
            row_count = 0
        t_execute = int((time.time() - t0) * 1000)

        t_total_ms = int((time.time() - t_total) * 1000)

        timings.append({
            "intent_ms": t_intent,
            "extract_ms": t_extract,
            "link_ms": t_link,
            "generate_ms": t_generate,
            "validate_ms": t_validate,
            "execute_ms": t_execute,
            "total_ms": t_total_ms,
            "row_count": row_count,
        })

    avg = lambda key: round(sum(t[key] for t in timings) / len(timings), 1)
    return {
        "query": query,
        "mode": mode,
        "n_runs": n_runs,
        "avg_intent_ms": avg("intent_ms"),
        "avg_extract_ms": avg("extract_ms"),
        "avg_link_ms": avg("link_ms"),
        "avg_generate_ms": avg("generate_ms"),
        "avg_validate_ms": avg("validate_ms"),
        "avg_execute_ms": avg("execute_ms"),
        "avg_total_ms": avg("total_ms"),
        "row_count": timings[0]["row_count"],
        "runs": timings,
    }


def main():
    print("=== Scalability Measurement ===\n")

    all_results: dict[str, Any] = {
        "experiment_date": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "llm_model": LLM_MODEL,
        "db_entries": 909,
        "tables": 7,
    }

    # Phase 1: Rule-based latency by query type
    print("--- Rule-based Latency by Query Type ---")
    rb_results: dict[str, list[dict[str, Any]]] = {}
    for qtype, queries in SCALABILITY_QUERIES.items():
        rb_results[qtype] = []
        for q in queries:
            print(f"  [{qtype}] {q[:40]}...", flush=True)
            m = _measure_pipeline_latency(q, mode="rule_based", n_runs=5)
            rb_results[qtype].append(m)

    all_results["rule_based_by_type"] = rb_results

    # Summary table
    print("\n  Type                  | Avg Total (ms) | Extract | Link | Generate | Execute")
    print("  " + "-" * 78)
    for qtype, measures in rb_results.items():
        avg_t = round(sum(m["avg_total_ms"] for m in measures) / len(measures), 1)
        avg_e = round(sum(m["avg_extract_ms"] for m in measures) / len(measures), 1)
        avg_l = round(sum(m["avg_link_ms"] for m in measures) / len(measures), 1)
        avg_g = round(sum(m["avg_generate_ms"] for m in measures) / len(measures), 1)
        avg_x = round(sum(m["avg_execute_ms"] for m in measures) / len(measures), 1)
        print(f"  {qtype:23s} | {avg_t:14.1f} | {avg_e:7.1f} | {avg_l:4.1f} | {avg_g:8.1f} | {avg_x:7.1f}")

    # Phase 2: LLM latency by query type
    if API_KEY:
        print("\n--- LLM Latency by Query Type ---")
        llm_results: dict[str, list[dict[str, Any]]] = {}
        for qtype, queries in SCALABILITY_QUERIES.items():
            llm_results[qtype] = []
            for q in queries:
                print(f"  [{qtype}] {q[:40]}...", flush=True)
                m = _measure_pipeline_latency(q, mode="llm", n_runs=3)
                llm_results[qtype].append(m)

        all_results["llm_by_type"] = llm_results

        print("\n  Type                  | Avg Total (ms) | Generate (LLM)")
        print("  " + "-" * 55)
        for qtype, measures in llm_results.items():
            avg_t = round(sum(m["avg_total_ms"] for m in measures) / len(measures), 1)
            avg_g = round(sum(m["avg_generate_ms"] for m in measures) / len(measures), 1)
            print(f"  {qtype:23s} | {avg_t:14.1f} | {avg_g:14.1f}")

    # Phase 3: RB vs LLM comparison
    print("\n--- Rule-based vs LLM Comparison ---")
    comparison_queries = [
        "Feを含むB2化合物を出して",
        "band gap > 1.0 eVのB2化合物を出して",
        "NiとAlを両方含む化合物を出して",
        "安定なL1₂化合物を形成エネルギーが低い順に出して",
    ]
    comparison_results = []
    for q in comparison_queries:
        rb = _measure_pipeline_latency(q, mode="rule_based", n_runs=5)
        llm_m = _measure_pipeline_latency(q, mode="llm", n_runs=3) if API_KEY else None
        entry = {
            "query": q,
            "rb_total_ms": rb["avg_total_ms"],
            "rb_generate_ms": rb["avg_generate_ms"],
        }
        if llm_m:
            entry["llm_total_ms"] = llm_m["avg_total_ms"]
            entry["llm_generate_ms"] = llm_m["avg_generate_ms"]
            entry["speedup"] = round(llm_m["avg_total_ms"] / max(rb["avg_total_ms"], 0.1), 1)
        comparison_results.append(entry)
        print(f"  {q[:40]}... RB={rb['avg_total_ms']:.0f}ms"
              + (f" LLM={llm_m['avg_total_ms']:.0f}ms ({entry.get('speedup',0)}x)" if llm_m else ""))

    all_results["rb_vs_llm"] = comparison_results

    # Save
    out_path = RESULTS_DIR / "scalability.json"
    out_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ Scalability results saved to {out_path}")


if __name__ == "__main__":
    main()
