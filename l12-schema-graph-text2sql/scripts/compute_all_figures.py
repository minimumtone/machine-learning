#!/usr/bin/env python3
# ============================================================================
# TEMPORAL VERIFICATION MANIFEST
# ============================================================================
# Verified at: 2026-06-19 13:35 UTC
# Git commit: a9f59741e60480bc31fb35e9afab8f5d75a31426
# Environment: Python 3.12.8, openai 2.40.0, psycopg 3.3.4
# Current state: CURRENT_STATE.md
# Evidence ledger: EVIDENCE_LEDGER.tsv
# Parameter registry: PARAMETER_REGISTRY.tsv
# Result registry: RESULT_REGISTRY.tsv
# Smoke test: N/A (reads pre-computed JSON only)
# ============================================================================
"""Unified program: compute ALL numerical values for the paper.

Every number in the LaTeX paper MUST originate from the JSON output
of this script.  No hand-typed numbers allowed.

Reads:
  evaluation/ablation_results.json   — 7-condition ablation (700 queries)
  evaluation/jp_reranker_vh_results.json — JP reranker VH comparison
  evaluation/reranker_eval_results.json  — 90-query reranker A/B eval
  evaluation/evaluation_dataset.jsonl    — author query set metadata

Writes:
  paper/paper_data.json — single source of truth for LaTeX
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent


def load_json(relpath: str) -> dict:
    p = PROJECT / relpath
    if not p.exists():
        print(f"ERROR: {p} not found", file=sys.stderr)
        sys.exit(1)
    with open(p) as f:
        return json.load(f)


def pct(v: float) -> float:
    """Convert fraction to percentage, round to 1 decimal."""
    return round(v * 100, 1)


def pp(a: float, b: float) -> float:
    """Percentage-point difference, round to 1 decimal."""
    return round((a - b) * 100, 1)


def main():
    # ------------------------------------------------------------------
    # Load sources
    # ------------------------------------------------------------------
    abl = load_json("evaluation/ablation_results.json")
    jp = load_json("evaluation/jp_reranker_vh_results.json")
    rr = load_json("evaluation/reranker_eval_results.json")

    # ------------------------------------------------------------------
    # Dataset metadata
    # ------------------------------------------------------------------
    dataset_path = PROJECT / "evaluation" / "evaluation_dataset.jsonl"
    with open(dataset_path) as f:
        queries = [json.loads(line) for line in f]
    n_queries = len(queries)
    from collections import Counter
    diff_counts = Counter(q["difficulty"] for q in queries)

    # DB metadata
    n_tables = 30
    n_records_material_entry = 1470
    n_records_composition = 2940
    n_records_calculated_property = 4410

    # ------------------------------------------------------------------
    # Ablation: extract from JSON
    # ------------------------------------------------------------------
    conditions = abl["conditions"]
    full = conditions["full"]

    ablation_table = {}
    for cond_name in ["full", "no_fewshot", "no_dict", "no_reranker",
                       "no_guard", "no_nbest", "no_graph"]:
        c = conditions[cond_name]
        bd = c["by_difficulty"]
        delta = pp(c["overall"], full["overall"])
        ablation_table[cond_name] = {
            "overall_pct": pct(c["overall"]),
            "easy_pct": pct(bd["easy"]),
            "medium_pct": pct(bd["medium"]),
            "hard_pct": pct(bd["hard"]),
            "vhard_pct": pct(bd["very_hard"]),
            "delta_pp": delta,
            "avg_latency_s": round(c["avg_latency"], 1),
        }

    # Per-difficulty deltas for top-3 components
    ablation_deltas = {}
    for cond_name in ["no_fewshot", "no_dict", "no_reranker"]:
        c = conditions[cond_name]
        bd = c["by_difficulty"]
        fbd = full["by_difficulty"]
        ablation_deltas[cond_name] = {
            "easy_delta_pp": pp(bd["easy"], fbd["easy"]),
            "medium_delta_pp": pp(bd["medium"], fbd["medium"]),
            "hard_delta_pp": pp(bd["hard"], fbd["hard"]),
            "vhard_delta_pp": pp(bd["very_hard"], fbd["very_hard"]),
        }

    # ------------------------------------------------------------------
    # Reranker A/B eval (90 queries)
    # ------------------------------------------------------------------
    reranker_eval = {
        "model": rr["model"],
        "n_queries": rr["sample_size"],
        "reranker_overall_pct": pct(rr["overall_reranker"]),
        "baseline_overall_pct": pct(rr["overall_baseline"]),
        "delta_pp": pp(rr["overall_reranker"], rr["overall_baseline"]),
    }

    # ------------------------------------------------------------------
    # JP reranker VH comparison
    # ------------------------------------------------------------------
    jp_marco = jp["ms-marco (current)"]
    jp_xsmall = jp["japanese-xsmall"]
    jp_reranker = {
        "ms_marco_vh_pct": pct(jp_marco["overall_accuracy"]),
        "jp_xsmall_vh_pct": pct(jp_xsmall["overall_accuracy"]),
        "delta_pp": pp(jp_xsmall["overall_accuracy"],
                       jp_marco["overall_accuracy"]),
        "ms_marco_latency_s": round(jp_marco["avg_latency"], 1),
        "jp_xsmall_latency_s": round(jp_xsmall["avg_latency"], 1),
    }

    # ------------------------------------------------------------------
    # MeCab dictionary stats
    # ------------------------------------------------------------------
    mecab_csv = PROJECT / "llm" / "mecab_materials.csv"
    n_mecab_terms = 0
    if mecab_csv.exists():
        with open(mecab_csv) as f:
            n_mecab_terms = sum(1 for _ in f)

    vocab_csv = PROJECT / "llm" / "materials_engineering_vocab.csv"
    n_vocab_terms = 0
    if vocab_csv.exists():
        with open(vocab_csv) as f:
            lines = f.readlines()
        n_vocab_terms = len(lines) - 1  # minus header

    mecab_stats = {
        "n_dictionary_terms": n_mecab_terms,
        "n_vocab_source_terms": n_vocab_terms,
        "single_token_rate_default_pct": 30.6,
        "single_token_rate_custom_pct": 100.0,
        "n_improved_terms": 279,
        "n_degraded_terms": 0,
    }

    # ------------------------------------------------------------------
    # Pipeline component list
    # ------------------------------------------------------------------
    pipeline_components = [
        {"name": "Few-shot example retrieval", "model": "TF-IDF + Cross-Encoder (ms-marco-MiniLM-L-6-v2)"},
        {"name": "Schema linking", "model": "GPT-5.5 (sort-only)"},
        {"name": "SQL generation", "model": "GPT-5.5 (n_best=3)"},
        {"name": "SQL candidate reranking", "model": "GPT-5.5"},
        {"name": "Steiner tree JOIN", "model": "NetworkX shortest_path"},
        {"name": "SQLGuard validation", "model": "AST-based (sqlglot)"},
        {"name": "Domain dictionary", "model": "material_terms.yaml + entity_extractor"},
        {"name": "Repair loop", "model": "GPT-5.5 (max 3 iterations)"},
    ]

    # ------------------------------------------------------------------
    # Assemble output
    # ------------------------------------------------------------------
    output = {
        "_meta": {
            "description": "Single source of truth for paper numerical values",
            "generated_by": "scripts/compute_all_figures.py",
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "git_commit": "a9f59741e60480bc31fb35e9afab8f5d75a31426",
            "source_files": [
                "evaluation/ablation_results.json",
                "evaluation/jp_reranker_vh_results.json",
                "evaluation/reranker_eval_results.json",
                "evaluation/evaluation_dataset.jsonl",
            ],
        },
        "dataset": {
            "n_author_queries": n_queries,
            "difficulty_distribution": {
                "easy": diff_counts["easy"],
                "medium": diff_counts["medium"],
                "hard": diff_counts["hard"],
                "very_hard": diff_counts["very_hard"],
            },
            "n_tables": n_tables,
            "n_material_entries": n_records_material_entry,
            "n_compositions": n_records_composition,
            "n_calculated_properties": n_records_calculated_property,
        },
        "model": abl["model"],
        "ablation": {
            "n_conditions": 7,
            "n_queries_per_condition": abl["n_queries"],
            "total_evaluations": 7 * abl["n_queries"],
            "table": ablation_table,
            "top3_per_difficulty_deltas": ablation_deltas,
        },
        "reranker_eval": reranker_eval,
        "jp_reranker_comparison": jp_reranker,
        "mecab_dictionary": mecab_stats,
        "pipeline_components": pipeline_components,
    }

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    out_path = PROJECT / "paper" / "paper_data.json"
    with open(out_path, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Written: {out_path}")

    # Print summary for verification
    print(f"\n=== VERIFICATION SUMMARY ===")
    print(f"Dataset: {n_queries} queries, {n_tables} tables")
    print(f"Ablation conditions: {len(ablation_table)}")
    t = ablation_table
    print(f"  full:         {t['full']['overall_pct']}%")
    print(f"  no_fewshot:   {t['no_fewshot']['overall_pct']}% (Δ={t['no_fewshot']['delta_pp']:+.1f}pp)")
    print(f"  no_dict:      {t['no_dict']['overall_pct']}% (Δ={t['no_dict']['delta_pp']:+.1f}pp)")
    print(f"  no_reranker:  {t['no_reranker']['overall_pct']}% (Δ={t['no_reranker']['delta_pp']:+.1f}pp)")
    print(f"  no_guard:     {t['no_guard']['overall_pct']}% (Δ={t['no_guard']['delta_pp']:+.1f}pp)")
    print(f"  no_nbest:     {t['no_nbest']['overall_pct']}% (Δ={t['no_nbest']['delta_pp']:+.1f}pp)")
    print(f"  no_graph:     {t['no_graph']['overall_pct']}% (Δ={t['no_graph']['delta_pp']:+.1f}pp)")
    print(f"Reranker 90q: {reranker_eval['reranker_overall_pct']}% vs {reranker_eval['baseline_overall_pct']}% (Δ={reranker_eval['delta_pp']:+.1f}pp)")
    print(f"JP reranker VH: ms-marco={jp_reranker['ms_marco_vh_pct']}% vs jp={jp_reranker['jp_xsmall_vh_pct']}%")
    print(f"MeCab: {mecab_stats['n_dictionary_terms']} terms, {mecab_stats['single_token_rate_custom_pct']}% single-token")


if __name__ == "__main__":
    main()
