#!/usr/bin/env python3
"""Graph Traversal Ablation Experiment.

Compares SQL generation quality with and without FK-based graph traversal
across queries requiring different numbers of table JOINs.

Metrics:
- JOIN count (number of JOINs in generated SQL)
- Unnecessary JOIN count (JOINs not needed for the query)
- Result correctness (does the result match expected output)
- Result row count
- Execution success
- Jaccard similarity between traversal and no-traversal results
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.entity_extractor import extract_conditions
from llm.sql_generator import (
    _rule_based_fallback,
    generate_sql_via_llm,
    pipeline as schema_graph_pipeline,
)
from llm.schema_linker import link_schema
from safety.sql_guard import execute_sql
from safety.sql_validator import validate_sql

# ── ENV ──
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")
os.environ.setdefault("POSTGRES_DB", "l12_materials")
os.environ.setdefault("POSTGRES_USER", "l12_user")
os.environ.setdefault("POSTGRES_PASSWORD", "l12_password")

LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-5.5")
API_KEY = os.environ.get("OPENAI_API_KEY", "")
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Full table/column/join lists (no traversal = use everything) ──
ALL_TABLES = [
    "material_entry", "composition", "structure",
    "phase_stability", "calculation", "calculated_property",
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

ALL_JOINS = [
    "composition.entry_id = material_entry.entry_id",
    "structure.entry_id = material_entry.entry_id",
    "phase_stability.entry_id = material_entry.entry_id",
    "calculation.entry_id = material_entry.entry_id",
    "calculated_property.calculation_id = calculation.calculation_id",
]

# ── Test queries designed to stress graph traversal ──
# Categorized by number of tables needed (beyond material_entry)
TRAVERSAL_QUERIES: list[dict[str, Any]] = [
    # --- Category 1: Single table JOIN (1 extra table) ---
    {"id": "T01", "query": "B2化合物の全リストを出して",
     "category": "1-table", "min_tables": ["material_entry", "structure"],
     "min_joins": 1, "description": "structure only"},
    {"id": "T02", "query": "Feを含む化合物を出して",
     "category": "1-table", "min_tables": ["material_entry", "composition"],
     "min_joins": 1, "description": "composition only"},
    {"id": "T03", "query": "安定な化合物を出して",
     "category": "1-table", "min_tables": ["material_entry", "phase_stability"],
     "min_joins": 1, "description": "phase_stability only"},

    # --- Category 2: Two table JOINs (2 extra tables) ---
    {"id": "T04", "query": "Feを含むB2化合物を出して",
     "category": "2-table", "min_tables": ["material_entry", "composition", "structure"],
     "min_joins": 2, "description": "composition + structure"},
    {"id": "T05", "query": "安定なL1₂化合物を形成エネルギー順に出して",
     "category": "2-table", "min_tables": ["material_entry", "structure", "phase_stability"],
     "min_joins": 2, "description": "structure + phase_stability"},
    {"id": "T06", "query": "Coを含む安定な化合物を出して",
     "category": "2-table", "min_tables": ["material_entry", "composition", "phase_stability"],
     "min_joins": 2, "description": "composition + phase_stability"},
    {"id": "T07", "query": "NiとAlを両方含む化合物を出して",
     "category": "2-table", "min_tables": ["material_entry", "composition"],
     "min_joins": 1, "description": "composition (EXISTS subquery)"},

    # --- Category 3: Three table JOINs (3 extra tables) ---
    {"id": "T08", "query": "Feを含む安定なB2化合物を出して",
     "category": "3-table", "min_tables": ["material_entry", "composition", "structure", "phase_stability"],
     "min_joins": 3, "description": "composition + structure + phase_stability"},
    {"id": "T09", "query": "NiとAlを含む安定なL1₂化合物を形成エネルギー順に出して",
     "category": "3-table", "min_tables": ["material_entry", "composition", "structure", "phase_stability"],
     "min_joins": 3, "description": "composition + structure + phase_stability + sort"},
    {"id": "T10", "query": "band gap > 1.0 eVの安定なB2化合物を出して",
     "category": "3-table", "min_tables": ["material_entry", "structure", "phase_stability"],
     "min_joins": 2, "description": "structure + phase_stability (numeric)"},

    # --- Category 4: Indirect JOIN path (calculated_property via calculation) ---
    {"id": "T11", "query": "bulk modulusが計算されている化合物を出して",
     "category": "indirect", "min_tables": ["material_entry", "calculation", "calculated_property"],
     "min_joins": 2, "description": "calculation -> calculated_property (indirect FK)"},
    {"id": "T12", "query": "B2化合物のbulk modulusを出して",
     "category": "indirect", "min_tables": ["material_entry", "structure", "calculation", "calculated_property"],
     "min_joins": 3, "description": "structure + calculation -> calculated_property"},

    # --- Category 5: Multi-element AND (complex subquery) ---
    {"id": "T13", "query": "FeとAlを含むB2化合物を出して",
     "category": "multi-element", "min_tables": ["material_entry", "composition", "structure"],
     "min_joins": 2, "description": "multi-element EXISTS + structure"},
    {"id": "T14", "query": "NiとCoを含む安定なL1₂化合物を出して",
     "category": "multi-element", "min_tables": ["material_entry", "composition", "structure", "phase_stability"],
     "min_joins": 3, "description": "multi-element + structure + phase_stability"},
    {"id": "T15", "query": "TiとAlを含むB2化合物の格子定数を出して",
     "category": "multi-element", "min_tables": ["material_entry", "composition", "structure"],
     "min_joins": 2, "description": "multi-element + structure lattice_a"},

    # --- Category 6: Format/sort stress ---
    {"id": "T16", "query": "全B2化合物を格子定数の大きい順に出して",
     "category": "sort", "min_tables": ["material_entry", "structure"],
     "min_joins": 1, "description": "structure + ORDER BY"},
    {"id": "T17", "query": "安定なL1₂化合物をband gapの大きい順に出して",
     "category": "sort", "min_tables": ["material_entry", "structure", "phase_stability"],
     "min_joins": 2, "description": "structure + phase_stability + ORDER BY"},
    {"id": "T18", "query": "形成エネルギーが負のB2化合物を出して",
     "category": "numeric", "min_tables": ["material_entry", "structure", "phase_stability"],
     "min_joins": 2, "description": "structure + phase_stability (numeric filter)"},

    # --- Category 7: Queries requiring NO extra tables ---
    {"id": "T19", "query": "全化合物のリストを出して",
     "category": "0-table", "min_tables": ["material_entry"],
     "min_joins": 0, "description": "material_entry only"},
    {"id": "T20", "query": "NiAlの化合物を出して",
     "category": "0-table", "min_tables": ["material_entry"],
     "min_joins": 0, "description": "formula search only"},

    # --- Additional multi-table queries for stronger evidence ---
    {"id": "T21", "query": "Ptを含むL1₂化合物の形成エネルギーを出して",
     "category": "3-table", "min_tables": ["material_entry", "composition", "structure", "phase_stability"],
     "min_joins": 3, "description": "composition + structure + phase_stability"},
    {"id": "T22", "query": "Ehull < 0.05 eV/atomのB2化合物の格子定数を出して",
     "category": "2-table", "min_tables": ["material_entry", "structure", "phase_stability"],
     "min_joins": 2, "description": "structure + phase_stability (numeric + select)"},
    {"id": "T23", "query": "安定なB2化合物でFeを含むものの形成エネルギーを出して",
     "category": "3-table", "min_tables": ["material_entry", "composition", "structure", "phase_stability"],
     "min_joins": 3, "description": "all three + element filter"},
    {"id": "T24", "query": "GaとGeを含む化合物の安定性を出して",
     "category": "multi-element", "min_tables": ["material_entry", "composition", "phase_stability"],
     "min_joins": 2, "description": "multi-element + phase_stability"},
    {"id": "T25", "query": "CsCl型化合物の全物性を出して",
     "category": "1-table", "min_tables": ["material_entry", "structure"],
     "min_joins": 1, "description": "structure only (alias)"},
    {"id": "T26", "query": "band gapが0の安定なL1₂化合物でNiを含むものを出して",
     "category": "3-table", "min_tables": ["material_entry", "composition", "structure", "phase_stability"],
     "min_joins": 3, "description": "composition + structure + phase_stability (complex)"},
    {"id": "T27", "query": "Irを含む化合物の格子定数とband gapを出して",
     "category": "3-table", "min_tables": ["material_entry", "composition", "structure", "phase_stability"],
     "min_joins": 3, "description": "composition + structure + phase_stability (multi-select)"},
    {"id": "T28", "query": "WとTiを含むB2化合物の安定性を出して",
     "category": "multi-element", "min_tables": ["material_entry", "composition", "structure", "phase_stability"],
     "min_joins": 3, "description": "multi-element + structure + phase_stability"},
    {"id": "T29", "query": "形成エネルギーが-0.5 eV/atom以下のL1₂化合物を出して",
     "category": "numeric", "min_tables": ["material_entry", "structure", "phase_stability"],
     "min_joins": 2, "description": "structure + phase_stability (strict numeric)"},
    {"id": "T30", "query": "Cu₃Au型の安定な化合物でCoを含むものを出して",
     "category": "3-table", "min_tables": ["material_entry", "composition", "structure", "phase_stability"],
     "min_joins": 3, "description": "composition + structure + phase_stability (alias Cu3Au)"},
]


def _count_joins(sql: str) -> int:
    """Count number of JOIN clauses in SQL."""
    if not sql:
        return 0
    return len(re.findall(r'\bJOIN\b', sql, re.IGNORECASE))


def _count_tables(sql: str) -> set[str]:
    """Extract table names referenced in SQL."""
    if not sql:
        return set()
    tables = set()
    for t in ALL_TABLES:
        if re.search(rf'\b{t}\b', sql, re.IGNORECASE):
            tables.add(t)
    return tables


def _execute_query(sql: str) -> dict[str, Any]:
    """Execute SQL and return result dict with formula set."""
    if not sql or not sql.strip():
        return {"success": False, "errors": ["Empty SQL"], "rows": [],
                "row_count": 0, "formulas": set()}
    validation = validate_sql(sql)
    if not validation["valid"]:
        return {"success": False, "errors": validation["errors"], "rows": [],
                "row_count": 0, "formulas": set()}
    try:
        result = execute_sql(validation["sql"], validate=False)
        # Extract formula set for Jaccard comparison
        formulas = set()
        cols = result.get("columns", [])
        if "formula" in cols:
            fi = cols.index("formula")
            for row in result.get("rows", []):
                if len(row) > fi and row[fi]:
                    formulas.add(row[fi])
        result["formulas"] = formulas
        return result
    except Exception as e:
        return {"success": False, "errors": [str(e)], "rows": [],
                "row_count": 0, "formulas": set()}


def _jaccard(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity."""
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def _generate_naive_all_join(query: str) -> str:
    """Generate SQL with ALL tables joined (no graph traversal).
    
    Simulates a naive approach that joins all tables regardless of query needs.
    """
    conditions = extract_conditions(query)
    linked = link_schema(conditions)

    select_cols = ["m.entry_id", "m.formula"]
    where_clauses: list[str] = []
    order_by = ""

    # Always join ALL tables (naive approach)
    joins = [
        "JOIN composition c ON c.entry_id = m.entry_id",
        "JOIN structure s ON s.entry_id = m.entry_id",
        "JOIN phase_stability ps ON ps.entry_id = m.entry_id",
        "JOIN calculation calc ON calc.entry_id = m.entry_id",
        "JOIN calculated_property cp ON cp.calculation_id = calc.calculation_id",
    ]

    has_exists_elements = False
    for frag in linked["sql_fragments"]:
        sql_f = frag["sql_fragment"]
        if frag["type"] == "sort":
            order_by = sql_f
        elif frag["type"] == "element_exists":
            has_exists_elements = True
            where_clauses.append(sql_f)
        else:
            where_clauses.append(sql_f)

    if has_exists_elements:
        joins = [j for j in joins if "JOIN composition" not in j]

    select_cols.extend(["s.prototype", "s.lattice_a", "s.space_group"])
    select_cols.extend(["ps.formation_energy_per_atom", "ps.energy_above_hull", "ps.band_gap"])

    sql_parts = [f"SELECT DISTINCT\n    {', '.join(select_cols)}"]
    sql_parts.append("FROM material_entry m")
    for j in joins:
        sql_parts.append(f"    {j}")
    if where_clauses:
        sql_parts.append("WHERE\n    " + "\n    AND ".join(where_clauses))
    if order_by:
        sql_parts.append(order_by)
    sql_parts.append("LIMIT 100;")
    return "\n".join(sql_parts)


def _generate_llm_no_traversal(query: str, model: str, api_key: str) -> dict[str, Any]:
    """Generate SQL via LLM with schema info but WITHOUT graph traversal.

    Provides all tables/columns/joins to LLM without filtering.
    """
    t0 = time.time()
    result = generate_sql_via_llm(
        user_query=query,
        allowed_tables=ALL_TABLES,
        allowed_columns=ALL_COLUMNS,
        allowed_joins=ALL_JOINS,
    )
    latency_ms = int((time.time() - t0) * 1000)
    result["latency_ms"] = latency_ms
    return result


def _generate_llm_with_traversal(query: str, model: str, api_key: str) -> dict[str, Any]:
    """Generate SQL via LLM WITH graph traversal (schema_graph_pipeline).

    Only provides traversal-selected tables/columns/joins.
    """
    t0 = time.time()
    result = schema_graph_pipeline(query, skip_intent_check=True)
    latency_ms = int((time.time() - t0) * 1000)
    result["latency_ms"] = latency_ms
    return result


def run_traversal_ablation() -> list[dict[str, Any]]:
    """Run the full traversal ablation experiment."""
    results = []
    print(f"=" * 70)
    print(f"Graph Traversal Ablation Experiment")
    print(f"Model: {LLM_MODEL}, Queries: {len(TRAVERSAL_QUERIES)}")
    print(f"=" * 70)

    for test in TRAVERSAL_QUERIES:
        qid = test["id"]
        query = test["query"]
        category = test["category"]
        min_joins = test["min_joins"]
        min_tables = test["min_tables"]
        print(f"\n  [{qid}] ({category}, min_joins={min_joins}) {query[:50]}...")

        entry: dict[str, Any] = {
            "id": qid,
            "query": query,
            "category": category,
            "min_joins": min_joins,
            "min_tables": min_tables,
            "description": test["description"],
        }

        # ── Condition A: Naive (all tables joined, rule-based) ──
        try:
            naive_sql = _generate_naive_all_join(query)
            naive_result = _execute_query(naive_sql)
            naive_joins = _count_joins(naive_sql)
            naive_tables = _count_tables(naive_sql)
            unnecessary_naive = naive_joins - min_joins
            entry["naive_all_join"] = {
                "sql": naive_sql,
                "success": naive_result.get("success", False),
                "row_count": naive_result.get("row_count", 0),
                "join_count": naive_joins,
                "tables_used": sorted(naive_tables),
                "unnecessary_joins": max(0, unnecessary_naive),
                "formulas": sorted(naive_result.get("formulas", set())),
            }
            print(f"    [A] Naive-all:  joins={naive_joins}, unnecessary={max(0, unnecessary_naive)}, "
                  f"rows={naive_result.get('row_count', 0)}, ok={naive_result.get('success', False)}")
        except Exception as e:
            entry["naive_all_join"] = {"sql": "", "success": False, "error": str(e)}
            print(f"    [A] Naive-all:  ERROR: {e}")

        # ── Condition B: Schema Graph + Rule-based (with traversal) ──
        try:
            sg_result = schema_graph_pipeline(query, skip_intent_check=True)
            sg_sql = sg_result.get("sql", "")
            sg_exec = _execute_query(sg_sql)
            sg_joins = _count_joins(sg_sql)
            sg_tables = _count_tables(sg_sql)
            unnecessary_sg = sg_joins - min_joins
            entry["sg_rb"] = {
                "sql": sg_sql,
                "success": sg_exec.get("success", False),
                "row_count": sg_exec.get("row_count", 0),
                "join_count": sg_joins,
                "tables_used": sorted(sg_tables),
                "unnecessary_joins": max(0, unnecessary_sg),
                "formulas": sorted(sg_exec.get("formulas", set())),
            }
            print(f"    [B] SG+RB:      joins={sg_joins}, unnecessary={max(0, unnecessary_sg)}, "
                  f"rows={sg_exec.get('row_count', 0)}, ok={sg_exec.get('success', False)}")
        except Exception as e:
            entry["sg_rb"] = {"sql": "", "success": False, "error": str(e)}
            print(f"    [B] SG+RB:      ERROR: {e}")

        # ── Condition C: LLM + all schema (no traversal) ──
        if API_KEY:
            try:
                llm_no_trav = _generate_llm_no_traversal(query, LLM_MODEL, API_KEY)
                llm_no_sql = llm_no_trav.get("sql", "")
                llm_no_exec = _execute_query(llm_no_sql)
                llm_no_joins = _count_joins(llm_no_sql)
                llm_no_tables = _count_tables(llm_no_sql)
                unnecessary_llm_no = llm_no_joins - min_joins
                entry["llm_no_traversal"] = {
                    "sql": llm_no_sql,
                    "success": llm_no_exec.get("success", False),
                    "row_count": llm_no_exec.get("row_count", 0),
                    "join_count": llm_no_joins,
                    "tables_used": sorted(llm_no_tables),
                    "unnecessary_joins": max(0, unnecessary_llm_no),
                    "latency_ms": llm_no_trav.get("latency_ms", 0),
                    "formulas": sorted(llm_no_exec.get("formulas", set())),
                }
                print(f"    [C] LLM-noSG:   joins={llm_no_joins}, unnecessary={max(0, unnecessary_llm_no)}, "
                      f"rows={llm_no_exec.get('row_count', 0)}, ok={llm_no_exec.get('success', False)}")
            except Exception as e:
                entry["llm_no_traversal"] = {"sql": "", "success": False, "error": str(e)}
                print(f"    [C] LLM-noSG:   ERROR: {e}")

            # ── Condition D: LLM + graph traversal ──
            try:
                llm_trav = _generate_llm_with_traversal(query, LLM_MODEL, API_KEY)
                llm_trav_sql = llm_trav.get("sql", "")
                llm_trav_exec = _execute_query(llm_trav_sql)
                llm_trav_joins = _count_joins(llm_trav_sql)
                llm_trav_tables = _count_tables(llm_trav_sql)
                unnecessary_llm_trav = llm_trav_joins - min_joins
                entry["llm_with_traversal"] = {
                    "sql": llm_trav_sql,
                    "success": llm_trav_exec.get("success", False),
                    "row_count": llm_trav_exec.get("row_count", 0),
                    "join_count": llm_trav_joins,
                    "tables_used": sorted(llm_trav_tables),
                    "unnecessary_joins": max(0, unnecessary_llm_trav),
                    "latency_ms": llm_trav.get("latency_ms", 0),
                    "formulas": sorted(llm_trav_exec.get("formulas", set())),
                }
                print(f"    [D] LLM+SG:     joins={llm_trav_joins}, unnecessary={max(0, unnecessary_llm_trav)}, "
                      f"rows={llm_trav_exec.get('row_count', 0)}, ok={llm_trav_exec.get('success', False)}")
            except Exception as e:
                entry["llm_with_traversal"] = {"sql": "", "success": False, "error": str(e)}
                print(f"    [D] LLM+SG:     ERROR: {e}")

        # ── Compute Jaccard similarities ──
        conditions_present = []
        for cond_key in ["naive_all_join", "sg_rb", "llm_no_traversal", "llm_with_traversal"]:
            if cond_key in entry and "formulas" in entry[cond_key]:
                conditions_present.append(cond_key)

        if "sg_rb" in conditions_present and "naive_all_join" in conditions_present:
            j_naive_sg = _jaccard(
                set(entry["naive_all_join"]["formulas"]),
                set(entry["sg_rb"]["formulas"])
            )
            entry["jaccard_naive_vs_sg"] = round(j_naive_sg, 4)

        if "llm_no_traversal" in conditions_present and "llm_with_traversal" in conditions_present:
            j_llm = _jaccard(
                set(entry["llm_no_traversal"]["formulas"]),
                set(entry["llm_with_traversal"]["formulas"])
            )
            entry["jaccard_llm_nosg_vs_sg"] = round(j_llm, 4)

        if "sg_rb" in conditions_present and "llm_with_traversal" in conditions_present:
            j_rb_llm = _jaccard(
                set(entry["sg_rb"]["formulas"]),
                set(entry["llm_with_traversal"]["formulas"])
            )
            entry["jaccard_rb_vs_llm_traversal"] = round(j_rb_llm, 4)

        results.append(entry)

    return results


def _compute_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute summary statistics."""
    summary: dict[str, Any] = {"total_queries": len(results)}

    for cond_key, cond_name in [
        ("naive_all_join", "Naive (all JOINs)"),
        ("sg_rb", "SG + Rule-based"),
        ("llm_no_traversal", "LLM (no traversal)"),
        ("llm_with_traversal", "LLM + SG traversal"),
    ]:
        valid = [r for r in results if cond_key in r and "join_count" in r[cond_key]]
        if not valid:
            continue
        total_joins = sum(r[cond_key]["join_count"] for r in valid)
        total_unnecessary = sum(r[cond_key].get("unnecessary_joins", 0) for r in valid)
        success_count = sum(1 for r in valid if r[cond_key].get("success", False))

        summary[cond_key] = {
            "name": cond_name,
            "count": len(valid),
            "exec_success": success_count,
            "exec_success_rate": round(success_count / len(valid) * 100, 1),
            "total_joins": total_joins,
            "avg_joins": round(total_joins / len(valid), 2),
            "total_unnecessary_joins": total_unnecessary,
            "avg_unnecessary_joins": round(total_unnecessary / len(valid), 2),
        }

    # Category-level breakdown
    categories: dict[str, list] = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)

    cat_summary = {}
    for cat, cat_results in sorted(categories.items()):
        cat_data: dict[str, Any] = {"count": len(cat_results)}
        for cond_key in ["naive_all_join", "sg_rb", "llm_no_traversal", "llm_with_traversal"]:
            valid_cat = [r for r in cat_results if cond_key in r and "join_count" in r[cond_key]]
            if valid_cat:
                cat_data[f"{cond_key}_avg_joins"] = round(
                    sum(r[cond_key]["join_count"] for r in valid_cat) / len(valid_cat), 2
                )
                cat_data[f"{cond_key}_avg_unnecessary"] = round(
                    sum(r[cond_key].get("unnecessary_joins", 0) for r in valid_cat) / len(valid_cat), 2
                )
        cat_summary[cat] = cat_data

    summary["by_category"] = cat_summary

    # Jaccard summaries
    for jkey in ["jaccard_naive_vs_sg", "jaccard_llm_nosg_vs_sg", "jaccard_rb_vs_llm_traversal"]:
        vals = [r[jkey] for r in results if jkey in r]
        if vals:
            summary[f"mean_{jkey}"] = round(sum(vals) / len(vals), 4)
            summary[f"perfect_{jkey}"] = sum(1 for v in vals if v == 1.0)

    return summary


def main():
    print("\n" + "=" * 70)
    print("  GRAPH TRAVERSAL ABLATION EXPERIMENT")
    print("=" * 70)

    results = run_traversal_ablation()
    summary = _compute_summary(results)

    # Clean formulas from output (too verbose for JSON)
    for r in results:
        for k in ["naive_all_join", "sg_rb", "llm_no_traversal", "llm_with_traversal"]:
            if k in r and "formulas" in r[k]:
                r[k]["formula_count"] = len(r[k]["formulas"])
                del r[k]["formulas"]

    output = {"summary": summary, "results": results}
    outfile = RESULTS_DIR / "traversal_ablation.json"
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nResults saved to: {outfile}")


if __name__ == "__main__":
    main()
