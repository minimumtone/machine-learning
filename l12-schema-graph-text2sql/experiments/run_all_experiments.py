#!/usr/bin/env python3
"""Comprehensive experiment suite for STAM-Methods paper.

Runs:
1. Baseline comparison (7 conditions)
2. RAG ablation (4 conditions)
3. LLM reproducibility (20 queries × 5 runs)
4. Scalability measurement
5. Failure mode analysis
6. Blind query evaluation
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.entity_extractor import extract_conditions
from llm.sql_generator import (
    build_constrained_prompt,
    extract_sql_from_response,
    generate_sql_via_llm,
    pipeline as schema_graph_pipeline,
    _rule_based_fallback,
)
from llm.schema_linker import link_schema
from llm.few_shot_store import load_store, retrieve_similar, format_few_shot_block
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

# Default column/join lists
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

ALL_TABLES = [
    "material_entry", "composition", "structure",
    "phase_stability", "calculation", "calculated_property",
]

ALL_JOINS = [
    "composition.entry_id = material_entry.entry_id",
    "structure.entry_id = material_entry.entry_id",
    "phase_stability.entry_id = material_entry.entry_id",
    "calculation.entry_id = material_entry.entry_id",
    "calculated_property.calculation_id = calculation.calculation_id",
]

# ── Test queries ──
CURATED_TESTS: list[dict[str, str]] = [
    {"id": "A01", "query": "Feを含むB2化合物を出して"},
    {"id": "A02", "query": "安定なL1₂化合物を形成エネルギーが低い順に出して"},
    {"id": "A03", "query": "NiとAlを両方含む化合物を出して"},
    {"id": "A04", "query": "B2化合物の全リストを出して"},
    {"id": "A05", "query": "準安定なB2化合物を出して"},
    {"id": "A06", "query": "Coを含む安定なL1₂化合物を出して"},
    {"id": "A07", "query": "Ptを含む化合物の格子定数を出して"},
    {"id": "A08", "query": "B2とL1₂の両方を出して"},
    {"id": "A09", "query": "NiとCoを含むL1₂化合物を出して"},
    {"id": "A10", "query": "L1₂化合物の全データを出して"},
    {"id": "A11", "query": "FeとAlを含むB2化合物を出して"},
    {"id": "A12", "query": "安定なB2化合物を形成エネルギーが低い順に出して"},
    {"id": "A13", "query": "Tiを含むL1₂化合物を出して"},
    {"id": "A14", "query": "NiとTiを含む安定な化合物を出して"},
    {"id": "A15", "query": "安定なL1₂化合物でNiを含むものを出して"},
]

NUMERIC_TESTS: list[dict[str, str]] = [
    {"id": "N01", "query": "band gap > 1.0 eVのB2化合物を出して"},
    {"id": "N02", "query": "形成エネルギーが負のL1₂化合物を出して"},
    {"id": "N03", "query": "Ehull < 0.05 eV/atomのB2化合物"},
    {"id": "N04", "query": "格子定数が3.5 Å以上のB2化合物を出して"},
    {"id": "N05", "query": "band gapが0のL1₂化合物"},
]

ADVERSARIAL_TESTS: list[dict[str, str]] = [
    {"id": "B01", "query": "Xeを含むB2化合物を出して"},
    {"id": "B02", "query": ""},
    {"id": "B03", "query": "今日の天気を教えて"},
    {"id": "B04", "query": "NiAl L12"},
    {"id": "B05", "query": "Ni3Al"},
    {"id": "B06", "query": "FeなしのB2化合物"},
    {"id": "B07", "query": "ヘスラー合金を出して"},
    {"id": "B08", "query": "band gapが大きい化合物"},
]

BLIND_QUERIES: list[dict[str, str]] = [
    {"id": "BL01", "query": "Which B2 compounds contain iron?"},
    {"id": "BL02", "query": "NiAlの安定性を教えて"},
    {"id": "BL03", "query": "Show me all L12 alloys with Co and Ni"},
    {"id": "BL04", "query": "Feを含む安定なB2型合金の一覧"},
    {"id": "BL05", "query": "lattice constant > 3.6のB2を出して"},
    {"id": "BL06", "query": "energy above hullが最も低いL12化合物top10"},
    {"id": "BL07", "query": "CuAu型構造の化合物は？"},
    {"id": "BL08", "query": "Ti-Al系B2合金で安定なもの"},
    {"id": "BL09", "query": "band gapが正のB2化合物を大きい順に"},
    {"id": "BL10", "query": "Ni3Alの形成エネルギー"},
    {"id": "BL11", "query": "Pt containing stable compounds"},
    {"id": "BL12", "query": "Scを含むL12化合物はありますか"},
    {"id": "BL13", "query": "形成エネルギーが-0.5 eV/atom以下の化合物"},
    {"id": "BL14", "query": "B2とL12のbandgap比較"},
    {"id": "BL15", "query": "GaとGeを含む化合物"},
    {"id": "BL16", "query": "ordered FCCの安定な化合物"},
    {"id": "BL17", "query": "NbとTaを含むB2化合物のリスト"},
    {"id": "BL18", "query": "formation energy < -0.3のL12"},
    {"id": "BL19", "query": "Irを含む全ての化合物の格子定数"},
    {"id": "BL20", "query": "metastable B2 compounds with Fe"},
    {"id": "BL21", "query": "WとTiのB2化合物"},
    {"id": "BL22", "query": "L12 compounds sorted by band gap descending"},
    {"id": "BL23", "query": "CuとNiを含むL12型化合物の安定性"},
    {"id": "BL24", "query": "全ての化合物を格子定数の大きい順に"},
    {"id": "BL25", "query": "energy above hullが0.1 eV/atom以下のB2化合物"},
    {"id": "BL26", "query": "Alを含まないB2化合物"},
    {"id": "BL27", "query": "bulk modulusが100 GPa以上の化合物"},
    {"id": "BL28", "query": "CoAlのB2型合金の物性"},
    {"id": "BL29", "query": "最も格子定数が小さいL12化合物"},
    {"id": "BL30", "query": "formation energyが正のB2化合物"},
]


def _execute_query(sql: str) -> dict[str, Any]:
    """Execute SQL and return result dict."""
    if not sql or not sql.strip():
        return {"success": False, "errors": ["Empty SQL"], "rows": [], "columns": [], "row_count": 0}
    validation = validate_sql(sql)
    if not validation["valid"]:
        return {"success": False, "errors": validation["errors"], "rows": [], "columns": [],
                "row_count": 0, "classification": validation.get("classification", "rejected")}
    try:
        result = execute_sql(validation["sql"], validate=False)
        result["classification"] = validation.get("classification", "accepted")
        return result
    except Exception as e:
        return {"success": False, "errors": [str(e)], "rows": [], "columns": [], "row_count": 0}


def _generate_llm_only(query: str, model: str, api_key: str,
                        schema_prompt: bool = False,
                        few_shot: bool = False) -> dict[str, Any]:
    """Generate SQL using LLM only (no schema graph constraint)."""
    import openai
    client = openai.OpenAI(api_key=api_key)

    system_msg = "You are a PostgreSQL expert for materials databases."
    user_msg = f"Generate a PostgreSQL SELECT query for the following request.\n\nUser query: {query}\n\nReturn ONLY the SQL query, no explanation."

    if schema_prompt:
        schema_yaml = """
Database schema:
Tables and columns:
- material_entry: entry_id, formula, reduced_formula, chemical_system
- composition: composition_id, entry_id (FK→material_entry), element, atomic_fraction, site_label
- structure: structure_id, entry_id (FK→material_entry), prototype, strukturbericht, lattice_a, lattice_b, lattice_c, volume_per_atom, formula_type, space_group_number, space_group
- phase_stability: stability_id, entry_id (FK→material_entry), formation_energy_per_atom, energy_above_hull, is_stable, band_gap
- calculation: calculation_id, entry_id (FK→material_entry), method, functional
- calculated_property: property_id, calculation_id (FK→calculation), property_name, value, unit

Important notes:
- Multi-element queries require EXISTS subqueries (one per element) against composition table
- prototype column uses Strukturbericht notation (B2, L12)
- Stable compounds: energy_above_hull <= 0.001
- Always use DISTINCT and LIMIT 100
"""
        user_msg = schema_yaml + "\n\n" + user_msg

    if few_shot:
        examples = retrieve_similar(query, top_k=3)
        if examples:
            user_msg = format_few_shot_block(examples) + "\n" + user_msg

    t0 = time.time()
    create_kwargs: dict[str, Any] = dict(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )
    _is_new = any(t in model for t in ("gpt-5", "o1", "o3", "o4"))
    if _is_new:
        create_kwargs["max_completion_tokens"] = 4096
    else:
        create_kwargs["temperature"] = 0.0
        create_kwargs["max_tokens"] = 512

    resp = client.chat.completions.create(**create_kwargs)
    latency_ms = int((time.time() - t0) * 1000)
    raw = resp.choices[0].message.content or ""
    sql = extract_sql_from_response(raw)
    usage = resp.usage
    tokens = (usage.prompt_tokens + usage.completion_tokens) if usage else 0
    return {
        "sql": sql, "model": model, "tokens": tokens,
        "latency_ms": latency_ms, "raw_response": raw,
    }


# ====================================================================
# Experiment 1: Baseline comparison (7 conditions)
# ====================================================================
def run_baseline_comparison(queries: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Compare 7 conditions: naive_rb, llm_only, llm_schema, llm_schema_fs,
    sg_rb, sg_llm_no_rag, sg_llm_rag."""
    results = []
    for test in queries:
        qid = test["id"]
        query = test["query"]
        if not query:
            continue
        print(f"  [{qid}] {query[:50]}...", flush=True)
        entry: dict[str, Any] = {"id": qid, "query": query, "conditions": {}}

        conditions = extract_conditions(query)
        entry["conditions"] = {k: v for k, v in conditions.items() if not k.startswith("_")}
        cov = conditions.get("_coverage", {})
        entry["coverage"] = {
            "coverage_score": cov.get("coverage_score", 0),
            "action": cov.get("action", ""),
            "unknown_elements": cov.get("unknown_elements", []),
            "unrecognized_terms": cov.get("unrecognized_terms", []),
        }

        # Condition 1: Naive rule-based (no schema graph)
        from llm.naive_sql_generator import generate_naive_sql
        try:
            naive_sql = generate_naive_sql(query)
            naive_result = _execute_query(naive_sql)
            entry["naive_rb"] = {
                "sql": naive_sql,
                "success": naive_result.get("success", False),
                "row_count": naive_result.get("row_count", 0),
                "latency_ms": naive_result.get("latency_ms", 0),
            }
        except Exception as e:
            entry["naive_rb"] = {"sql": "", "success": False, "error": str(e), "row_count": 0}

        # Condition 5: Schema Graph + Rule-based
        try:
            sg_rb = schema_graph_pipeline(query)
            sg_rb_result = _execute_query(sg_rb["sql"])
            entry["sg_rb"] = {
                "sql": sg_rb["sql"],
                "success": sg_rb_result.get("success", False),
                "row_count": sg_rb_result.get("row_count", 0),
                "latency_ms": sg_rb_result.get("latency_ms", 0),
            }
        except Exception as e:
            entry["sg_rb"] = {"sql": "", "success": False, "error": str(e), "row_count": 0}

        # LLM-based conditions (2-4, 6-7) require API key
        if API_KEY and API_KEY != "your_api_key_here":
            # Condition 2: LLM-only (no schema info)
            try:
                llm_only = _generate_llm_only(query, LLM_MODEL, API_KEY,
                                               schema_prompt=False, few_shot=False)
                llm_only_result = _execute_query(llm_only["sql"])
                entry["llm_only"] = {
                    "sql": llm_only["sql"],
                    "success": llm_only_result.get("success", False),
                    "row_count": llm_only_result.get("row_count", 0),
                    "tokens": llm_only.get("tokens", 0),
                    "latency_ms": llm_only.get("latency_ms", 0),
                }
            except Exception as e:
                entry["llm_only"] = {"sql": "", "success": False, "error": str(e)}

            # Condition 3: LLM + schema prompt
            try:
                llm_sp = _generate_llm_only(query, LLM_MODEL, API_KEY,
                                             schema_prompt=True, few_shot=False)
                llm_sp_result = _execute_query(llm_sp["sql"])
                entry["llm_schema_prompt"] = {
                    "sql": llm_sp["sql"],
                    "success": llm_sp_result.get("success", False),
                    "row_count": llm_sp_result.get("row_count", 0),
                    "tokens": llm_sp.get("tokens", 0),
                    "latency_ms": llm_sp.get("latency_ms", 0),
                }
            except Exception as e:
                entry["llm_schema_prompt"] = {"sql": "", "success": False, "error": str(e)}

            # Condition 4: LLM + schema prompt + few-shot examples
            try:
                llm_sp_fs = _generate_llm_only(query, LLM_MODEL, API_KEY,
                                                schema_prompt=True, few_shot=True)
                llm_sp_fs_result = _execute_query(llm_sp_fs["sql"])
                entry["llm_schema_fs"] = {
                    "sql": llm_sp_fs["sql"],
                    "success": llm_sp_fs_result.get("success", False),
                    "row_count": llm_sp_fs_result.get("row_count", 0),
                    "tokens": llm_sp_fs.get("tokens", 0),
                    "latency_ms": llm_sp_fs.get("latency_ms", 0),
                }
            except Exception as e:
                entry["llm_schema_fs"] = {"sql": "", "success": False, "error": str(e)}

            # Condition 6: Schema Graph + LLM without RAG
            try:
                linked = link_schema(conditions)
                llm_no_rag = generate_sql_via_llm(
                    user_query=query,
                    allowed_tables=linked["required_tables"],
                    allowed_columns=[c for c in ALL_COLUMNS if c.split(".")[0] in linked["required_tables"]],
                    allowed_joins=[j for j in ALL_JOINS if any(t in j for t in linked["required_tables"])],
                    model=LLM_MODEL,
                    api_key=API_KEY,
                )
                llm_no_rag_result = _execute_query(llm_no_rag["sql"])
                entry["sg_llm_no_rag"] = {
                    "sql": llm_no_rag["sql"],
                    "success": llm_no_rag_result.get("success", False),
                    "row_count": llm_no_rag_result.get("row_count", 0),
                    "tokens": llm_no_rag.get("tokens", 0),
                    "latency_ms": llm_no_rag.get("latency_ms", 0),
                }
            except Exception as e:
                entry["sg_llm_no_rag"] = {"sql": "", "success": False, "error": str(e)}

            # Condition 7: Schema Graph + LLM with RAG (full pipeline)
            try:
                sg_llm_rag = schema_graph_pipeline(query)
                sg_llm_rag_result = _execute_query(sg_llm_rag["sql"])
                entry["sg_llm_rag"] = {
                    "sql": sg_llm_rag["sql"],
                    "success": sg_llm_rag_result.get("success", False),
                    "row_count": sg_llm_rag_result.get("row_count", 0),
                    "tokens": sg_llm_rag.get("tokens", 0),
                    "latency_ms": sg_llm_rag.get("latency_ms", 0),
                }
            except Exception as e:
                entry["sg_llm_rag"] = {"sql": "", "success": False, "error": str(e)}
        else:
            for key in ["llm_only", "llm_schema_prompt", "llm_schema_fs",
                        "sg_llm_no_rag", "sg_llm_rag"]:
                entry[key] = {"sql": "", "success": False, "error": "No API key", "row_count": 0}

        results.append(entry)
    return results


# ====================================================================
# Experiment 2: LLM reproducibility
# ====================================================================
def run_reproducibility_test(queries: list[dict[str, str]],
                              n_runs: int = 5) -> list[dict[str, Any]]:
    """Run same queries multiple times to measure LLM output variance."""
    if not API_KEY or API_KEY == "your_api_key_here":
        print("  [SKIP] No API key for reproducibility test")
        return []

    results = []
    for test in queries[:20]:
        qid = test["id"]
        query = test["query"]
        if not query:
            continue
        print(f"  [{qid}] {query[:40]}... x{n_runs}", flush=True)
        runs: list[dict[str, Any]] = []
        for run_idx in range(n_runs):
            try:
                r = schema_graph_pipeline(query)
                exec_result = _execute_query(r["sql"])
                runs.append({
                    "run": run_idx + 1,
                    "sql": r["sql"],
                    "success": exec_result.get("success", False),
                    "row_count": exec_result.get("row_count", 0),
                    "latency_ms": r.get("latency_ms", 0),
                    "tokens": r.get("tokens", 0),
                })
            except Exception as e:
                runs.append({"run": run_idx + 1, "sql": "", "error": str(e)})

        # Compute variance metrics
        sqls = [r["sql"] for r in runs if r.get("sql")]
        unique_sqls = len(set(sqls))
        row_counts = [r.get("row_count", 0) for r in runs if r.get("success")]
        results.append({
            "id": qid, "query": query, "n_runs": n_runs,
            "unique_sql_count": unique_sqls,
            "sql_consistency_rate": 1.0 if unique_sqls <= 1 else round(max(sqls.count(s) for s in set(sqls)) / len(sqls), 3),
            "result_consistency": all(rc == row_counts[0] for rc in row_counts) if row_counts else False,
            "row_count_range": [min(row_counts), max(row_counts)] if row_counts else [],
            "mean_latency_ms": round(sum(r.get("latency_ms", 0) for r in runs) / len(runs)),
            "runs": runs,
        })
    return results


# ====================================================================
# Experiment 2b: RAG Ablation (4 conditions)
# ====================================================================
def run_rag_ablation(queries: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Compare 4 RAG conditions:
    1. No examples (schema info only)
    2. Manual examples only (source=manual/seed)
    3. Paper-extracted examples only (source=paper)
    4. All examples (manual + paper + runtime)
    """
    if not API_KEY or API_KEY == "your_api_key_here":
        print("  [SKIP] No API key for RAG ablation")
        return []

    all_examples = load_store()
    manual_examples = [e for e in all_examples if e.get("source") in ("manual", "seed", "pipeline")]
    paper_examples = [e for e in all_examples if e.get("source") == "paper"]

    print(f"  Store: {len(all_examples)} total, {len(manual_examples)} manual/seed, {len(paper_examples)} paper")

    results = []
    for test in queries:
        qid = test["id"]
        query = test["query"]
        if not query:
            continue
        print(f"  [{qid}] {query[:40]}...", flush=True)
        entry: dict[str, Any] = {"id": qid, "query": query}

        conditions = extract_conditions(query)
        linked = link_schema(conditions)
        allowed_cols = [c for c in ALL_COLUMNS if c.split(".")[0] in linked["required_tables"]]
        allowed_joins_filtered = [j for j in ALL_JOINS if any(t in j for t in linked["required_tables"])]

        # Condition 1: No examples
        try:
            prompt_no_ex = build_constrained_prompt(
                query, linked["required_tables"], allowed_cols, allowed_joins_filtered,
                few_shot_examples=None,
            )
            import openai
            client = openai.OpenAI(api_key=API_KEY)
            t0 = time.time()
            create_kwargs: dict[str, Any] = dict(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a PostgreSQL expert for materials databases."},
                    {"role": "user", "content": prompt_no_ex},
                ],
            )
            _is_new = any(t in LLM_MODEL for t in ("gpt-5", "o1", "o3", "o4"))
            if _is_new:
                create_kwargs["max_completion_tokens"] = 4096
            else:
                create_kwargs["temperature"] = 0.0
                create_kwargs["max_tokens"] = 512
            resp = client.chat.completions.create(**create_kwargs)
            latency = int((time.time() - t0) * 1000)
            sql = extract_sql_from_response(resp.choices[0].message.content or "")
            result = _execute_query(sql)
            entry["no_examples"] = {
                "sql": sql, "success": result.get("success", False),
                "row_count": result.get("row_count", 0),
                "latency_ms": latency,
                "tokens": (resp.usage.prompt_tokens + resp.usage.completion_tokens) if resp.usage else 0,
            }
        except Exception as e:
            entry["no_examples"] = {"sql": "", "success": False, "error": str(e)}

        # Condition 2: Manual examples only
        try:
            from llm.few_shot_store import _tokenize, _tf, _idf, _cosine
            query_tokens = _tokenize(query)
            if manual_examples:
                corpus_tokens = [_tokenize(e["nl_query"]) for e in manual_examples]
                idf = _idf(corpus_tokens + [query_tokens])
                def _tfidf(tokens):
                    tf = _tf(tokens)
                    return {t: tf[t] * idf.get(t, 1.0) for t in tf}
                q_vec = _tfidf(query_tokens)
                scored = []
                for i, doc_tokens in enumerate(corpus_tokens):
                    d_vec = _tfidf(doc_tokens)
                    sim = _cosine(q_vec, d_vec)
                    scored.append((sim, i))
                scored.sort(reverse=True)
                manual_fs = [{**manual_examples[idx], "similarity": sim}
                             for sim, idx in scored[:3] if sim > 0.05]
            else:
                manual_fs = []

            prompt_manual = build_constrained_prompt(
                query, linked["required_tables"], allowed_cols, allowed_joins_filtered,
                few_shot_examples=manual_fs,
            )
            t0 = time.time()
            create_kwargs2: dict[str, Any] = dict(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a PostgreSQL expert for materials databases."},
                    {"role": "user", "content": prompt_manual},
                ],
            )
            if _is_new:
                create_kwargs2["max_completion_tokens"] = 4096
            else:
                create_kwargs2["temperature"] = 0.0
                create_kwargs2["max_tokens"] = 512
            resp2 = client.chat.completions.create(**create_kwargs2)
            latency2 = int((time.time() - t0) * 1000)
            sql2 = extract_sql_from_response(resp2.choices[0].message.content or "")
            result2 = _execute_query(sql2)
            entry["manual_only"] = {
                "sql": sql2, "success": result2.get("success", False),
                "row_count": result2.get("row_count", 0),
                "latency_ms": latency2, "few_shot_count": len(manual_fs),
                "tokens": (resp2.usage.prompt_tokens + resp2.usage.completion_tokens) if resp2.usage else 0,
            }
        except Exception as e:
            entry["manual_only"] = {"sql": "", "success": False, "error": str(e)}

        # Condition 3: Paper-extracted examples only
        try:
            if paper_examples:
                corpus_tokens_p = [_tokenize(e["nl_query"]) for e in paper_examples]
                idf_p = _idf(corpus_tokens_p + [query_tokens])
                def _tfidf_p(tokens):
                    tf = _tf(tokens)
                    return {t: tf[t] * idf_p.get(t, 1.0) for t in tf}
                q_vec_p = _tfidf_p(query_tokens)
                scored_p = []
                for i, doc_tokens in enumerate(corpus_tokens_p):
                    d_vec = _tfidf_p(doc_tokens)
                    sim = _cosine(q_vec_p, d_vec)
                    scored_p.append((sim, i))
                scored_p.sort(reverse=True)
                paper_fs = [{**paper_examples[idx], "similarity": sim}
                            for sim, idx in scored_p[:3] if sim > 0.05]
            else:
                paper_fs = []

            prompt_paper = build_constrained_prompt(
                query, linked["required_tables"], allowed_cols, allowed_joins_filtered,
                few_shot_examples=paper_fs,
            )
            t0 = time.time()
            create_kwargs3: dict[str, Any] = dict(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a PostgreSQL expert for materials databases."},
                    {"role": "user", "content": prompt_paper},
                ],
            )
            if _is_new:
                create_kwargs3["max_completion_tokens"] = 4096
            else:
                create_kwargs3["temperature"] = 0.0
                create_kwargs3["max_tokens"] = 512
            resp3 = client.chat.completions.create(**create_kwargs3)
            latency3 = int((time.time() - t0) * 1000)
            sql3 = extract_sql_from_response(resp3.choices[0].message.content or "")
            result3 = _execute_query(sql3)
            entry["paper_only"] = {
                "sql": sql3, "success": result3.get("success", False),
                "row_count": result3.get("row_count", 0),
                "latency_ms": latency3, "few_shot_count": len(paper_fs),
                "tokens": (resp3.usage.prompt_tokens + resp3.usage.completion_tokens) if resp3.usage else 0,
            }
        except Exception as e:
            entry["paper_only"] = {"sql": "", "success": False, "error": str(e)}

        # Condition 4: All examples (full RAG)
        try:
            all_fs = retrieve_similar(query, top_k=3)
            prompt_all = build_constrained_prompt(
                query, linked["required_tables"], allowed_cols, allowed_joins_filtered,
                few_shot_examples=all_fs,
            )
            t0 = time.time()
            create_kwargs4: dict[str, Any] = dict(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a PostgreSQL expert for materials databases."},
                    {"role": "user", "content": prompt_all},
                ],
            )
            if _is_new:
                create_kwargs4["max_completion_tokens"] = 4096
            else:
                create_kwargs4["temperature"] = 0.0
                create_kwargs4["max_tokens"] = 512
            resp4 = client.chat.completions.create(**create_kwargs4)
            latency4 = int((time.time() - t0) * 1000)
            sql4 = extract_sql_from_response(resp4.choices[0].message.content or "")
            result4 = _execute_query(sql4)
            entry["all_examples"] = {
                "sql": sql4, "success": result4.get("success", False),
                "row_count": result4.get("row_count", 0),
                "latency_ms": latency4, "few_shot_count": len(all_fs),
                "tokens": (resp4.usage.prompt_tokens + resp4.usage.completion_tokens) if resp4.usage else 0,
            }
        except Exception as e:
            entry["all_examples"] = {"sql": "", "success": False, "error": str(e)}

        results.append(entry)
    return results


# ====================================================================
# Experiment 3: Failure mode analysis
# ====================================================================
def run_failure_analysis(queries: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Classify each query result into failure mode categories."""
    results = []
    for test in queries:
        qid = test["id"]
        query = test["query"]
        print(f"  [{qid}] {query[:50]}...", flush=True)

        conditions = extract_conditions(query)
        coverage = conditions.get("_coverage", {})

        # Try RB mode
        try:
            rb = schema_graph_pipeline(query)
            rb_result = _execute_query(rb["sql"])
        except Exception as e:
            rb = {"sql": ""}
            rb_result = {"success": False, "errors": [str(e)], "row_count": 0}

        failure_mode = "exact_success"
        if not rb_result.get("success"):
            failure_mode = "sql_error"
        elif coverage.get("coverage_score", 1.0) < 0.5:
            failure_mode = "unsafe_overbroad"
        elif coverage.get("unknown_elements"):
            failure_mode = "silent_constraint_drop" if rb_result.get("row_count", 0) > 0 else "safe_rejection"
        elif coverage.get("unrecognized_terms") and len(coverage.get("unrecognized_terms", [])) > len([k for k in conditions if not k.startswith("_")]):
            failure_mode = "partial_success"

        results.append({
            "id": qid, "query": query,
            "coverage_score": coverage.get("coverage_score", 0),
            "coverage_action": coverage.get("action", ""),
            "unknown_elements": coverage.get("unknown_elements", []),
            "unrecognized_terms": coverage.get("unrecognized_terms", []),
            "rb_success": rb_result.get("success", False),
            "rb_row_count": rb_result.get("row_count", 0),
            "rb_sql": rb.get("sql", ""),
            "failure_mode": failure_mode,
        })
    return results


# ====================================================================
# Main
# ====================================================================
def main():
    all_results: dict[str, Any] = {
        "experiment_date": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "llm_model": LLM_MODEL,
        "has_api_key": bool(API_KEY and API_KEY != "your_api_key_here"),
    }

    all_queries = CURATED_TESTS + NUMERIC_TESTS + ADVERSARIAL_TESTS + BLIND_QUERIES

    # Experiment 1: Baseline comparison
    print("\n=== Experiment 1: Baseline Comparison (7 conditions) ===")
    baseline_results = run_baseline_comparison(all_queries)
    all_results["baseline_comparison"] = baseline_results
    # Save intermediate
    (RESULTS_DIR / "baseline_comparison.json").write_text(
        json.dumps(baseline_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  → {len(baseline_results)} queries tested")

    # Experiment 2: Reproducibility
    print("\n=== Experiment 2: LLM Reproducibility ===")
    repro_results = run_reproducibility_test(CURATED_TESTS + NUMERIC_TESTS, n_runs=5)
    all_results["reproducibility"] = repro_results
    if repro_results:
        (RESULTS_DIR / "reproducibility.json").write_text(
            json.dumps(repro_results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  → {len(repro_results)} queries × 5 runs")

    # Experiment 2b: RAG Ablation
    print("\n=== Experiment 2b: RAG Ablation (4 conditions) ===")
    rag_queries = CURATED_TESTS + NUMERIC_TESTS + BLIND_QUERIES  # skip adversarial for RAG
    rag_results = run_rag_ablation(rag_queries)
    all_results["rag_ablation"] = rag_results
    if rag_results:
        (RESULTS_DIR / "rag_ablation.json").write_text(
            json.dumps(rag_results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  → {len(rag_results)} queries × 4 RAG conditions")

    # Experiment 3: Failure mode analysis
    print("\n=== Experiment 3: Failure Mode Analysis ===")
    failure_results = run_failure_analysis(all_queries)
    all_results["failure_analysis"] = failure_results
    (RESULTS_DIR / "failure_analysis.json").write_text(
        json.dumps(failure_results, ensure_ascii=False, indent=2), encoding="utf-8")

    # Summary stats
    modes = {}
    for r in failure_results:
        m = r["failure_mode"]
        modes[m] = modes.get(m, 0) + 1
    print(f"  → Failure mode distribution: {json.dumps(modes)}")

    # Save all
    (RESULTS_DIR / "all_experiments.json").write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ All results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
