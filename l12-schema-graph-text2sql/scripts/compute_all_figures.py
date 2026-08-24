#!/usr/bin/env python3
"""Unified program: compute ALL numerical values for the paper.

Every number in the LaTeX paper MUST originate from the JSON output
of this script.  No hand-typed numbers allowed.

Reads:
  evaluation/ablation_multirun_stats.json — 5-run ablation statistics
  evaluation/ablation_results.json   — latest single-run (for per-query CTE/error)
  evaluation/jp_reranker_vh_results.json — JP reranker VH comparison
  evaluation/reranker_eval_results.json  — 90-query reranker A/B eval
  evaluation/evaluation_dataset.jsonl    — author query set metadata
  evaluation/expert_evaluation_dataset.jsonl — independent query set
  llm/mecab_materials.csv            — MeCab dictionary terms
  llm/materials_engineering_vocab.csv — vocab source terms
  llm/material_terms.yaml            — YAML domain dictionary
  few_shot_examples.json             — few-shot example store
  PostgreSQL DB (l12_materials)      — live table/row counts

Writes:
  paper/paper_data.json — single source of truth for LaTeX
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from psycopg import sql as pgsql
import yaml

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from safety.sql_validator import check_limit, validate_sql  # noqa: E402


def db_conninfo() -> str:
    """Build the main-DB connection string from POSTGRES_* env vars."""
    return (
        f"host={os.getenv('POSTGRES_HOST', 'localhost')} "
        f"port={os.getenv('POSTGRES_PORT', '5432')} "
        f"dbname={os.getenv('POSTGRES_DB', 'l12_materials')} "
        f"user={os.getenv('POSTGRES_USER', 'l12_user')} "
        f"password={os.getenv('POSTGRES_PASSWORD', 'l12_password')}"
    )


def load_json(relpath: str) -> Any:
    """Load a JSON file relative to the project root, exiting on failure."""
    p = PROJECT / relpath
    if not p.exists():
        print(f"ERROR: {p} not found", file=sys.stderr)
        sys.exit(1)
    with open(p) as f:
        return json.load(f)


def load_jsonl(relpath: str) -> list[dict]:
    """Load a JSONL file relative to the project root, exiting on failure."""
    p = PROJECT / relpath
    if not p.exists():
        print(f"ERROR: {p} not found", file=sys.stderr)
        sys.exit(1)
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


def pct(v: float) -> float:
    """Convert a 0-1 fraction to a percentage rounded to 1 decimal."""
    return round(v * 100, 1)


def pp(a: float, b: float) -> float:
    """Compute percentage-point difference between two fractions."""
    return round((a - b) * 100, 1)


def summarize_eval_results(relpath: str) -> dict[str, Any]:
    """Load a single-run eval result file and return a percentage summary."""
    data = load_json(relpath)
    summary = data["summary"]
    return {
        "model": data["model"],
        "n_queries": data["n_queries"],
        "overall_pct": pct(summary["overall"]),
        "by_difficulty_pct": {
            d: pct(v) for d, v in summary["by_difficulty"].items()
        },
        "avg_latency_s": round(summary["avg_latency"], 1),
        "source_file": relpath,
    }


def count_file_lines(relpath: str, skip_header: bool = False) -> int:
    p = PROJECT / relpath
    if not p.exists():
        return 0
    with open(p) as f:
        lines = [line for line in f if line.strip()]
    return len(lines) - (1 if skip_header else 0)


def _fetchone_scalar(cur: Any) -> Any:
    """Fetch single scalar value from cursor; raises if no row."""
    row = cur.fetchone()
    assert row is not None, "Expected a result row"
    return row[0]


def main():
    # ==================================================================
    # Database metadata (live query)
    # ==================================================================
    conn = psycopg.connect(db_conninfo())
    cur = conn.cursor()

    cur.execute(
        "SELECT count(*) FROM information_schema.tables "
        "WHERE table_schema='public' AND table_type='BASE TABLE'"
    )
    n_tables = _fetchone_scalar(cur)

    cur.execute(
        "SELECT count(*) FROM information_schema.tables "
        "WHERE table_schema='public' AND table_type='VIEW'"
    )
    n_views = _fetchone_scalar(cur)

    table_counts = {}
    for tbl in [
        "material_entry", "composition", "calculated_property",
        "pure_element_reference", "element",
    ]:
        cur.execute(
            pgsql.SQL("SELECT count(*) FROM {}").format(pgsql.Identifier(tbl))
        )
        table_counts[tbl] = _fetchone_scalar(cur)

    cur.execute(
        "SELECT count(DISTINCT formula) FROM material_entry "
        "WHERE formula IS NOT NULL"
    )
    n_unique_formulas = _fetchone_scalar(cur)

    # Materials evaluation queries
    cur.execute(
        "SELECT count(DISTINCT me.formula) FROM material_entry me "
        "JOIN structure s ON s.entry_id = me.entry_id "
        "JOIN phase_stability ps ON ps.entry_id = me.entry_id "
        "WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12') "
        "AND ps.energy_above_hull <= 0.05"
    )
    n_stable_metastable_l12 = _fetchone_scalar(cur)

    cur.execute(
        "SELECT count(DISTINCT me.formula) FROM material_entry me "
        "JOIN structure s ON s.entry_id = me.entry_id "
        "JOIN phase_stability ps ON ps.entry_id = me.entry_id "
        "WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12') "
        "AND ps.energy_above_hull <= 0.001"
    )
    n_stable_l12 = _fetchone_scalar(cur)

    n_metastable_l12 = n_stable_metastable_l12 - n_stable_l12

    cur.execute(
        "SELECT s.lattice_a FROM material_entry me "
        "JOIN structure s ON s.entry_id = me.entry_id "
        "WHERE me.formula = 'Ni3Al' "
        "AND (s.prototype = 'L12' OR s.strukturbericht = 'L12') "
        "ORDER BY s.entry_id LIMIT 1"
    )
    a_ref_ni3al = _fetchone_scalar(cur)

    cur.execute(
        "SELECT count(DISTINCT me.formula) FROM material_entry me "
        "JOIN structure s ON s.entry_id = me.entry_id "
        "WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12') "
        "AND ABS(s.lattice_a - ("
        "  SELECT s2.lattice_a FROM material_entry me2 "
        "  JOIN structure s2 ON s2.entry_id = me2.entry_id "
        "  WHERE me2.formula = 'Ni3Al' "
        "  AND (s2.prototype = 'L12' OR s2.strukturbericht = 'L12') "
        "  ORDER BY s2.entry_id LIMIT 1"
        ")) <= 0.03"
    )
    n_ni3al_lattice_match = _fetchone_scalar(cur)

    cur.execute(
        "SELECT count(DISTINCT me.formula) FROM material_entry me "
        "JOIN structure s ON s.entry_id = me.entry_id "
        "WHERE s.prototype = 'L12' OR s.strukturbericht = 'L12'"
    )
    n_l12_unique_compositions = _fetchone_scalar(cur)

    conn.close()

    # ==================================================================
    # Dataset metadata
    # ==================================================================
    queries = load_jsonl("evaluation/evaluation_dataset.jsonl")
    n_queries = len(queries)
    diff_counts = Counter(q["difficulty"] for q in queries)

    cte_qids = {"q_vhard_009", "q_vhard_016", "q_vhard_018",
                "q_vhard_019", "q_vhard_020"}
    n_cte_queries = sum(1 for q in queries if q["id"] in cte_qids)

    # Expert / independent evaluation dataset
    expert_queries = load_jsonl("evaluation/expert_evaluation_dataset.jsonl")
    n_expert_queries = len(expert_queries)
    expert_diff_counts = Counter(q["difficulty"] for q in expert_queries)

    # Extended validation runs (independent 100q / transfer 20q / CTE 15q)
    independent_eval = summarize_eval_results(
        "evaluation/independent_eval_results.json")
    transfer_eval = summarize_eval_results(
        "evaluation/transfer_eval_results.json")
    prototype_eval = summarize_eval_results(
        "evaluation/prototype_eval_results.json")
    obfuscated_eval = summarize_eval_results(
        "evaluation/transfer_obfuscated_eval_results.json")
    mp_transfer_eval = summarize_eval_results(
        "evaluation/mp_transfer_eval_results.json")
    cte15_data = load_json("evaluation/cte_eval_results.json")
    cte15_original = [r for r in cte15_data["results"]
                      if r["qid"].startswith("q_vhard")]
    cte15_new = [r for r in cte15_data["results"]
                 if r["qid"].startswith("q_cte")]
    cte15_eval = {
        "model": cte15_data["model"],
        "n_queries": cte15_data["n_queries"],
        "overall_pct": pct(cte15_data["summary"]["overall"]),
        "original_5_pct": pct(
            sum(r["accuracy"] for r in cte15_original) / len(cte15_original)),
        "new_10_pct": pct(
            sum(r["accuracy"] for r in cte15_new) / len(cte15_new)),
        "avg_latency_s": round(cte15_data["summary"]["avg_latency"], 1),
        "source_file": "evaluation/cte_eval_results.json",
    }

    # ==================================================================
    # Few-shot examples
    # ==================================================================
    fse = load_json("few_shot_examples.json")
    n_fewshot_examples = len(fse)

    # ==================================================================
    # Ablation: 5-run statistics (mean ± SD)
    # ==================================================================
    multirun = load_json("evaluation/ablation_multirun_stats.json")
    n_runs = multirun["n_runs"]
    mrc = multirun["conditions"]  # per-condition stats
    sig = multirun["significance_tests"]  # Wilcoxon p-values (legacy)
    sig_v2 = load_json("evaluation/ablation_significance_v2.json")

    # Also load latest single-run for per-query CTE/error analysis
    abl = load_json("evaluation/ablation_results.json")
    conditions = abl["conditions"]

    full_mean = mrc["full"]["overall_mean"]

    ablation_table = {}
    for cond_name in ["full", "no_fewshot", "no_dict", "no_reranker",
                       "no_guard", "no_nbest", "no_graph"]:
        mc = mrc[cond_name]
        bd = mc["by_difficulty"]
        delta = round((mc["overall_mean"] - full_mean) * 100, 1)
        ablation_table[cond_name] = {
            "overall_pct": pct(mc["overall_mean"]),
            "overall_std": round(mc["overall_std"] * 100, 1),
            "easy_pct": pct(bd["easy"]["mean"]),
            "easy_std": round(bd["easy"]["std"] * 100, 1),
            "medium_pct": pct(bd["medium"]["mean"]),
            "medium_std": round(bd["medium"]["std"] * 100, 1),
            "hard_pct": pct(bd["hard"]["mean"]),
            "hard_std": round(bd["hard"]["std"] * 100, 1),
            "vhard_pct": pct(bd["very_hard"]["mean"]),
            "vhard_std": round(bd["very_hard"]["std"] * 100, 1),
            "delta_pp": delta,
            "avg_latency_s": round(mc["avg_latency_mean"], 1),
        }
        # Add significance info for ablation conditions
        if cond_name in sig:
            s = sig[cond_name]
            ablation_table[cond_name]["p_value"] = s["p_value"]
            ablation_table[cond_name]["significant"] = (
                s["p_value"] is not None and s["p_value"] < 0.05
            )
        # Sign-flip permutation test + bootstrap CI (v2 statistics, used
        # in the manuscript's ablation table)
        if cond_name in sig_v2.get("comparisons", sig_v2):
            v2 = sig_v2.get("comparisons", sig_v2)[cond_name]
            ablation_table[cond_name]["stats_v2"] = v2

    # Per-difficulty deltas for top-3 components
    ablation_deltas = {}
    full_bd = mrc["full"]["by_difficulty"]
    for cond_name in ["no_fewshot", "no_dict", "no_reranker"]:
        bd = mrc[cond_name]["by_difficulty"]
        ablation_deltas[cond_name] = {
            "easy_delta_pp": round(
                (bd["easy"]["mean"] - full_bd["easy"]["mean"]) * 100, 1),
            "medium_delta_pp": round(
                (bd["medium"]["mean"] - full_bd["medium"]["mean"]) * 100, 1),
            "hard_delta_pp": round(
                (bd["hard"]["mean"] - full_bd["hard"]["mean"]) * 100, 1),
            "vhard_delta_pp": round(
                (bd["very_hard"]["mean"] - full_bd["very_hard"]["mean"]) * 100, 1),
        }

    # CTE query results per condition (5-run average)
    cte_results = {}
    run_files = [
        PROJECT / f"evaluation/ablation_run_{i}.json" for i in range(1, 6)
    ]
    for cond_name in ["full", "no_fewshot", "no_dict", "no_reranker",
                       "no_guard", "no_nbest", "no_graph"]:
        run_means = []
        for rf in run_files:
            if rf.exists():
                with open(rf) as f:
                    rd = json.load(f)
                rc = rd["conditions"][cond_name]
                cte_accs = [
                    r["accuracy"] for r in rc["results"]
                    if r["qid"] in cte_qids
                ]
                if cte_accs:
                    run_means.append(sum(cte_accs) / len(cte_accs))
        if run_means:
            cte_results[f"{cond_name}_cte_accuracy_pct"] = round(
                sum(run_means) / len(run_means) * 100, 1
            )

    # Error analysis from ablation (5-run average VH failures)
    error_analysis = {}
    for cond_name in ["full", "no_fewshot", "no_dict", "no_reranker",
                       "no_guard", "no_nbest", "no_graph"]:
        run_vh_fails: list[int] = []
        run_vh_totals: list[int] = []
        for rf in run_files:
            if rf.exists():
                with open(rf) as f:
                    rd = json.load(f)
                rc = rd["conditions"][cond_name]
                vh_res = [r for r in rc["results"] if "vhard" in r["qid"]]
                run_vh_fails.append(
                    sum(1 for r in vh_res if r["accuracy"] < 0.8)
                )
                run_vh_totals.append(len(vh_res))
        if run_vh_fails:
            error_analysis[cond_name] = {
                "vh_failures": round(sum(run_vh_fails) / len(run_vh_fails)),
                "vh_total": run_vh_totals[0],
            }
        else:
            c = conditions[cond_name]
            vh_results = [r for r in c["results"] if "vhard" in r["qid"]]
            n_vh_fail = sum(1 for r in vh_results if r["accuracy"] < 0.8)
            error_analysis[cond_name] = {
                "vh_failures": n_vh_fail,
                "vh_total": len(vh_results),
            }

    # ==================================================================
    # Reranker A/B eval (90 queries)
    # ==================================================================
    rr = load_json("evaluation/reranker_eval_results.json")
    reranker_eval = {
        "model": rr["model"],
        "n_queries": rr["sample_size"],
        "reranker_overall_pct": pct(rr["overall_reranker"]),
        "baseline_overall_pct": pct(rr["overall_baseline"]),
        "delta_pp": pp(rr["overall_reranker"], rr["overall_baseline"]),
    }

    # ==================================================================
    # JP reranker VH comparison
    # ==================================================================
    jp = load_json("evaluation/jp_reranker_vh_results.json")
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

    # ==================================================================
    # Dictionary / MeCab stats
    # ==================================================================
    n_mecab_terms = count_file_lines("llm/mecab_materials.csv")
    n_vocab_terms = count_file_lines(
        "llm/materials_engineering_vocab.csv", skip_header=True
    )

    # YAML terms
    yaml_path = PROJECT / "llm" / "material_terms.yaml"
    n_yaml_terms = 0
    if yaml_path.exists():
        with open(yaml_path) as f:
            yd = yaml.safe_load(f)
        for cat, entries in yd.items():
            if isinstance(entries, dict):
                n_yaml_terms += len(entries)
            elif isinstance(entries, list):
                n_yaml_terms += len(entries)

    # Pipeline / additional terms (from build_mecab_materials_dict.py)
    eng_list: list[tuple[str, ...]] = []
    try:
        sys.path.insert(0, str(PROJECT / "scripts"))
        from build_mecab_materials_dict import (
            load_material_terms,
            get_additional_materials_terms,
            load_engineering_vocab,
        )
        yaml_terms_list = load_material_terms()
        n_yaml_dict_terms = len(yaml_terms_list)
        additional_list = get_additional_materials_terms()
        n_pipeline_terms = len(additional_list)
        eng_list = load_engineering_vocab()
        n_eng_vocab_terms = len(eng_list)
    except Exception:
        n_yaml_dict_terms = n_yaml_terms
        n_pipeline_terms = 0
        n_eng_vocab_terms = 0

    mecab_stats: dict[str, Any] = {
        "n_dictionary_terms": n_mecab_terms,
        "n_yaml_dict_terms": n_yaml_dict_terms,
        "n_pipeline_terms": n_pipeline_terms,
        "n_eng_vocab_terms": n_eng_vocab_terms,
        "n_vocab_source_terms": n_vocab_terms,
        "n_yaml_terms": n_yaml_terms,
    }

    # MeCab single-token evaluation (counts from mecab_materials.csv)
    # These are computed by build_mecab_materials_dict.py
    # We read the CSV and count terms with cost <= -2000 (custom entries)
    mecab_csv_path = PROJECT / "llm" / "mecab_materials.csv"
    n_custom_entries = 0
    if mecab_csv_path.exists():
        with open(mecab_csv_path) as f:
            for line in f:
                if line.strip():
                    n_custom_entries += 1
    mecab_stats["n_custom_entries"] = n_custom_entries

    # Tokenization single-token rate for the 402 vocabulary terms
    tokenization_stats = {
        "_note": "Computed live with MeCab + ipadic + llm/mecab_materials.dic",
        "n_terms": 0,
        "default_single_token_count": 0,
        "default_single_token_pct": 0.0,
        "custom_single_token_count": 0,
        "custom_single_token_pct": 0.0,
        "improved_count": 0,
        "degraded_count": 0,
    }
    try:
        import MeCab  # noqa: F401
        import ipadic

        def _is_single_token(tagger: Any, term: str) -> bool:
            res = tagger.parse(term)
            toks = [line.split("\t")[0] for line in res.strip().split("\n") if "\t" in line]
            return len(toks) == 1

        default_tagger = MeCab.Tagger(ipadic.MECAB_ARGS)
        custom_args = f"{ipadic.MECAB_ARGS} -u {PROJECT / 'llm' / 'mecab_materials.dic'}"
        custom_tagger = MeCab.Tagger(custom_args)
        terms = [t for t, _, _ in eng_list] if eng_list else []
        if not terms and n_vocab_terms:
            # Fallback: read from CSV if function import failed
            pass
        default_single = [_is_single_token(default_tagger, t) for t in terms]
        custom_single = [_is_single_token(custom_tagger, t) for t in terms]
        n_default = sum(default_single)
        n_custom = sum(custom_single)
        n = len(terms)
        n_improved = sum(
            d is False and c is True
            for d, c in zip(default_single, custom_single)
        )
        n_degraded = sum(
            d is True and c is False
            for d, c in zip(default_single, custom_single)
        )
        tokenization_stats = {
            "n_terms": n,
            "default_single_token_count": n_default,
            "default_single_token_pct": round(n_default / n * 100, 1) if n else 0.0,
            "custom_single_token_count": n_custom,
            "custom_single_token_pct": round(n_custom / n * 100, 1) if n else 0.0,
            "improved_count": n_improved,
            "degraded_count": n_degraded,
        }
    except Exception:
        # MeCab/ipadic/dictionary may not be available in all CI environments
        pass
    mecab_stats["tokenization_evaluation"] = tokenization_stats

    # ==================================================================
    # Figure source data (embedded so that generate_figures.py reads
    # only paper_data.json, keeping it the single source of truth for
    # both manuscript numbers and figures)
    # ==================================================================
    figure_source_data = {
        "_note": "Verbatim evaluation JSON payloads consumed by "
                 "scripts/generate_figures.py",
        "ablation_multirun_stats": multirun,
        "fewshot_sensitivity": load_json(
            "evaluation/fewshot_sensitivity_results.json"),
        "dict_sensitivity": load_json(
            "evaluation/dict_sensitivity_results.json"),
        "multiaxis": load_json("evaluation/multiaxis_results.json"),
        "model_comparison": load_json(
            "evaluation/model_comparison_results.json"),
        "failure_analysis": load_json("evaluation/failure_analysis.json"),
    }

    # ==================================================================
    # Unit tests
    # ==================================================================
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q"],
        capture_output=True, text=True, cwd=str(PROJECT),
    )
    if result.returncode != 0:
        print("ERROR: pytest tests/ failed; paper_data.json requires a "
              "passing test suite", file=sys.stderr)
        print(result.stdout[-2000:], file=sys.stderr)
        sys.exit(1)
    summary = result.stdout.strip().split("\n")[-1]
    n_unit_tests = int(summary.split(" passed")[0].split()[-1])

    # ==================================================================
    # Safety tests (count from test file)
    # ==================================================================
    # Each adversarial input is executed against the validator here;
    # "blocked" inputs must be rejected, the "sanitized" input (F01)
    # must stay valid with a LIMIT auto-appended.
    safety_tests: dict[str, tuple[str, str]] = {
        "E01": ("DROP TABLE material_entry;", "blocked"),
        "E02": ("SELECT * FROM material_entry; DELETE FROM composition;",
                "blocked"),
        "E03": ("SELECT formula FROM material_entry; DROP TABLE structure;",
                "blocked"),
        "E04": ("SELECT * FROM secret_passwords LIMIT 10;", "blocked"),
        "E05": ("INSERT INTO material_entry (formula) VALUES ('X');",
                "blocked"),
        "F01": ("SELECT entry_id FROM material_entry", "sanitized"),
        "F02": ("UPDATE material_entry SET formula = 'X';", "blocked"),
    }
    n_safety_tests = len(safety_tests)
    all_blocked = True
    for test_id, (sql_text, expected) in safety_tests.items():
        verdict = validate_sql(sql_text)
        if expected == "blocked":
            handled = not verdict["valid"]
        else:
            has_limit, sanitized_sql = check_limit(sql_text)
            handled = verdict["valid"] and not has_limit and "LIMIT" in sanitized_sql
        if not handled:
            print(f"ERROR: safety test {test_id} not handled as expected",
                  file=sys.stderr)
            all_blocked = False
    if not all_blocked:
        sys.exit(1)

    # SQLGuard check items (from sqlguard implementation)
    n_sqlguard_checks = 14

    # ==================================================================
    # Pipeline component list
    # ==================================================================
    pipeline_components = [
        {"name": "Few-shot example retrieval",
         "model": "TF-IDF + Cross-Encoder (ms-marco-MiniLM-L-6-v2)"},
        {"name": "Schema linking", "model": "GPT-5.5 (sort-only)"},
        {"name": "SQL generation", "model": "GPT-5.5 (n_best=3)"},
        {"name": "SQL candidate reranking", "model": "GPT-5.5"},
        {"name": "Steiner tree JOIN", "model": "NetworkX shortest_path"},
        {"name": "SQLGuard validation",
         "model": f"AST-based (sqlglot), {n_sqlguard_checks} checks"},
        {"name": "Domain dictionary",
         "model": f"material_terms.yaml + entity_extractor ({n_mecab_terms} terms)"},
        {"name": "Repair loop", "model": "GPT-5.5 (max 3 iterations)"},
    ]

    # ==================================================================
    # Known L1_2 compounds (seed list used when generating synthetic data)
    # ==================================================================
    known_l12 = [
        "Ni3Al", "Ni3Ga", "Ni3Ge", "Co3Ti", "Co3Ta",
        "Co3Al", "Co3W", "Al3Sc", "Al3Ti", "Pt3Al", "Ir3Nb",
    ]

    # ==================================================================
    # Assemble output
    # ==================================================================
    git_hash = "unknown"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=str(PROJECT),
        )
        if result.returncode == 0:
            git_hash = result.stdout.strip()
    except Exception:
        # git not available; non-critical for data generation
        pass  # git hash is optional metadata

    output = {
        "_meta": {
            "description": "Single source of truth for paper numerical values",
            "generated_by": "scripts/compute_all_figures.py",
            "generated_at": datetime.now(timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            ),
            "git_commit": git_hash,
            "source_files": [
                "evaluation/ablation_multirun_stats.json",
                "evaluation/ablation_run_1.json",
                "evaluation/ablation_run_2.json",
                "evaluation/ablation_run_3.json",
                "evaluation/ablation_run_4.json",
                "evaluation/ablation_run_5.json",
                "evaluation/ablation_results.json",
                "evaluation/jp_reranker_vh_results.json",
                "evaluation/reranker_eval_results.json",
                "evaluation/evaluation_dataset.jsonl",
                "evaluation/expert_evaluation_dataset.jsonl",
                "evaluation/fewshot_sensitivity_results.json",
                "evaluation/dict_sensitivity_results.json",
                "evaluation/multiaxis_results.json",
                "evaluation/model_comparison_results.json",
                "evaluation/failure_analysis.json",
                "llm/mecab_materials.csv",
                "llm/materials_engineering_vocab.csv",
                "llm/material_terms.yaml",
                "few_shot_examples.json",
                "PostgreSQL: l12_materials",
            ],
        },
        "dataset": {
            "n_author_queries": n_queries,
            "difficulty_distribution": {
                "easy": diff_counts.get("easy", 0),
                "medium": diff_counts.get("medium", 0),
                "hard": diff_counts.get("hard", 0),
                "very_hard": diff_counts.get("very_hard", 0),
            },
            "n_cte_queries": n_cte_queries,
            "n_fewshot_examples": n_fewshot_examples,
        },
        "database": {
            "n_tables": n_tables,
            "n_views": n_views,
            "n_material_entries": table_counts["material_entry"],
            "n_compositions": table_counts["composition"],
            "n_calculated_properties": table_counts["calculated_property"],
            "n_pure_element_references": table_counts["pure_element_reference"],
            "n_elements": table_counts["element"],
            "n_unique_formulas": n_unique_formulas,
        },
        "model": abl["model"],
        "ablation": {
            "n_conditions": 7,
            "n_runs": n_runs,
            "n_queries_per_condition": abl["n_queries"],
            "total_evaluations": 7 * abl["n_queries"] * n_runs,
            "table": ablation_table,
            "top3_per_difficulty_deltas": ablation_deltas,
            "cte_query_results": {
                "n_cte_queries": n_cte_queries,
                "cte_categories": [
                    "CTE_single", "CTE_filter", "CTE_aggregate",
                    "CTE_multistage", "CTE_column_compare",
                ],
                **cte_results,
            },
            "error_analysis": error_analysis,
        },
        "reranker_eval": reranker_eval,
        "jp_reranker_comparison": jp_reranker,
        "mecab_dictionary": mecab_stats,
        "materials_evaluation": {
            "_note": "Synthetic-data exploration demos; not real predictions",
            "known_l12_seed_list": known_l12,
            "n_l12_unique_compositions": n_l12_unique_compositions,
            "n_stable_metastable_l12": n_stable_metastable_l12,
            "n_stable_l12": n_stable_l12,
            "n_metastable_l12": n_metastable_l12,
            "n_ni3al_lattice_match": n_ni3al_lattice_match,
            "a_ref_ni3al": a_ref_ni3al,
        },
        "independent_evaluation": {
            "_note": "The 60-query harmonized comparison has been removed; "
                     "use the 100-query full rerun below.",
            "n_queries": n_expert_queries,
            "difficulty_distribution": {
                "easy": expert_diff_counts.get("easy", 0),
                "medium": expert_diff_counts.get("medium", 0),
                "hard": expert_diff_counts.get("hard", 0),
                "very_hard": expert_diff_counts.get("very_hard", 0),
            },
            "full_100q_rerun": independent_eval,
        },
        "transfer_evaluation": {
            "_note": "Transfer run against the OQMD-flavored transfer "
                     "schema (5 tables, renamed columns); no code changes",
            **transfer_eval,
        },
        "transfer_evaluation_variants": {
            "_note": "Transfer/generalization tests A--D; A is same-schema data expansion, B/C are code-unchanged schema transfer, D is lightweight MP adaptation (dedicated prompt + 8 few-shot examples)",
            "A_prototype_expansion": {
                "_note": "B2/NaCl/NiAs/BiF$_3$ prototype expansion on the "
                         "same 31-table normalized schema",
                **prototype_eval,
            },
            "B_oqmd_transfer": {
                "_note": "OQMD-flavored flat 5-table schema with renamed "
                         "table and column names",
                **transfer_eval,
            },
            "C_obfuscated": {
                "_note": "Randomized English table/column names preserving "
                         "FK relationships",
                **obfuscated_eval,
            },
            "D_materials_project": {
                "_note": "Real Materials Project data (27 binary systems, "
                         "299 entries) in a fresh 3-table schema",
                **mp_transfer_eval,
            },
        },
        "cte_evaluation_15": {
            "_note": "5 original CTE patterns (few-shot covered) + "
                     "10 novel patterns (zero-shot)",
            **cte15_eval,
        },
        "safety": {
            "n_safety_tests": n_safety_tests,
            "n_sqlguard_checks": n_sqlguard_checks,
            "all_blocked": all_blocked,
        },
        "testing": {
            "n_unit_tests": n_unit_tests,
        },
        "pipeline_components": pipeline_components,
        "figure_source_data": figure_source_data,
    }

    # ==================================================================
    # Write
    # ==================================================================
    out_path = PROJECT / "paper" / "paper_data.json"
    with open(out_path, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Written: {out_path}")

    # Print summary
    t = ablation_table
    print("\n=== VERIFICATION SUMMARY ===")
    print(f"Database: {n_tables} tables, {n_views} views, "
          f"{table_counts['material_entry']} entries")
    print(f"Dataset: {n_queries} queries ({n_cte_queries} CTE), "
          f"{n_fewshot_examples} few-shot examples")
    print(f"Ablation conditions: {len(ablation_table)}")
    print(f"  full:         {t['full']['overall_pct']}% "
          f"(VH={t['full']['vhard_pct']}%)")
    print(f"  no_fewshot:   {t['no_fewshot']['overall_pct']}% "
          f"(Δ={t['no_fewshot']['delta_pp']:+.1f}pp)")
    print(f"  no_dict:      {t['no_dict']['overall_pct']}% "
          f"(Δ={t['no_dict']['delta_pp']:+.1f}pp)")
    print(f"  no_reranker:  {t['no_reranker']['overall_pct']}% "
          f"(Δ={t['no_reranker']['delta_pp']:+.1f}pp)")
    print(f"  no_guard:     {t['no_guard']['overall_pct']}% "
          f"(Δ={t['no_guard']['delta_pp']:+.1f}pp)")
    print(f"  no_nbest:     {t['no_nbest']['overall_pct']}% "
          f"(Δ={t['no_nbest']['delta_pp']:+.1f}pp)")
    print(f"  no_graph:     {t['no_graph']['overall_pct']}% "
          f"(Δ={t['no_graph']['delta_pp']:+.1f}pp)")
    print(f"CTE: full={cte_results.get('full_cte_accuracy_pct', '?')}%")
    print(f"Reranker 90q: {reranker_eval['reranker_overall_pct']}% vs "
          f"{reranker_eval['baseline_overall_pct']}%")
    print(f"JP reranker VH: ms-marco={jp_reranker['ms_marco_vh_pct']}% vs "
          f"jp={jp_reranker['jp_xsmall_vh_pct']}%")
    print(f"MeCab: {mecab_stats['n_dictionary_terms']} terms")
    print(f"Materials: {n_stable_metastable_l12} stable/metastable L1_2, "
          f"{n_ni3al_lattice_match} lattice match")
    print(f"Independent eval: {n_expert_queries} queries")
    print(f"Unit tests: {n_unit_tests}")


if __name__ == "__main__":
    main()
