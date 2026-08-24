#!/usr/bin/env python3
"""Dictionary size sensitivity analysis: measure the effect of reducing the
domain-specific materials dictionary on pipeline accuracy.

Runs the full pipeline with dictionary sizes:
  - full   (61 schema-link entries + 93 term entries = 100%)
  - 50%    (top ~30 schema-link + ~46 term entries)
  - 25%    (top ~15 schema-link + ~23 term entries)
  - 0%     (no domain dictionary at all, same as ablation no_dict)

Reduction strategy: keep the most frequently used entries first (core
materials concepts like prototype, elements, stability, lattice).

Output: evaluation/dict_sensitivity_results.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import psycopg  # noqa: E402

from evaluation.metrics import execution_accuracy_full, normalize_limit  # noqa: E402
from graph.graph_builder import build_table_graph  # noqa: E402
from graph.join_path_generator import get_allowed_join_list  # noqa: E402
from graph.schema_parser import get_foreign_keys, get_tables, get_columns  # noqa: E402
from llm.sql_generator import pipeline as sql_pipeline  # noqa: E402

EVAL_DIR = PROJECT / "evaluation"
RESULTS_DIR = EVAL_DIR / "expected_results"

CONNINFO = (
    f"host={os.getenv('POSTGRES_HOST', 'localhost')} "
    f"port={os.getenv('POSTGRES_PORT', '5432')} "
    f"dbname={os.getenv('POSTGRES_DB', 'l12_materials')} "
    f"user={os.getenv('POSTGRES_USER', 'l12_user')} "
    f"password={os.getenv('POSTGRES_PASSWORD', 'l12_password')}"
)

# Priority order of schema-linker keys (most important first).
# Keys in the evaluation queries are weighted by frequency of use.
_PRIORITY_KEYS = [
    # Tier 1: Core (used in >50% of queries)
    "prototype", "contains_elements", "formula",
    "lattice_reference", "lattice_constant",
    "stability", "formation_energy",
    # Tier 2: Frequent (used in 20-50%)
    "crystal_system", "space_group", "volume",
    "bulk_modulus", "shear_modulus", "youngs_modulus",
    "band_gap", "total_magnetization", "magnetic_ordering",
    "debye_temperature", "source_db", "number_of_elements",
    # Tier 3: Moderate (used in 5-20%)
    "surface_energy", "work_function", "miller_index",
    "vacancy_formation", "defect", "dopant",
    "thermal_conductivity", "gruneisen_parameter",
    "elastic_stability", "poisson_ratio",
    "dos_at_fermi", "is_metallic", "spin_polarized",
    "curie_temperature", "magnetic_anisotropy",
    "lattice_c", "chemical_system", "atomic_number",
    "electronegativity", "element_property",
    "direct_gap", "surface_reconstruction",
    "grain_boundary_energy", "interstitial",
    # Tier 4: Rare
    "site_label", "calculation_method", "functional",
    "synthesis", "ball_milling", "arc_melting", "experimental",
    "doi", "literature", "reference",
    "application", "phase_diagram", "alloy_system",
    "pure_element", "reference_energy", "ground_state",
    "polymorph", "formation_enthalpy",
    "band_structure",
]


def _subset_dict(
    full_dict: dict[str, list[str]], fraction: float,
) -> dict[str, list[str]]:
    """Return a subset of `full_dict` keeping the top `fraction` of keys."""
    n_keep = max(1, int(len(full_dict) * fraction))
    kept: dict[str, list[str]] = {}
    for key in _PRIORITY_KEYS:
        if key in full_dict and len(kept) < n_keep:
            kept[key] = full_dict[key]
    # Fill remaining with keys not in priority list
    for key in full_dict:
        if key not in kept and len(kept) < n_keep:
            kept[key] = full_dict[key]
    return kept


def _subset_terms(
    full_terms: dict[str, Any], fraction: float,
) -> dict[str, Any]:
    """Return a subset of material_terms keeping top `fraction` of entries."""
    result: dict[str, Any] = {}
    for section, content in full_terms.items():
        if isinstance(content, dict):
            n_keep = max(1, int(len(content) * fraction))
            items = list(content.items())[:n_keep]
            result[section] = dict(items)
        elif isinstance(content, list):
            n_keep = max(1, int(len(content) * fraction))
            result[section] = content[:n_keep]
        else:
            result[section] = content
    return result


# Dict size configurations: (label, fraction)
DICT_CONFIGS = [
    ("full", 1.0),
    ("50%", 0.5),
    ("25%", 0.25),
    ("10%", 0.1),
    ("0%", 0.0),
]


def load_queries():
    queries = []
    with open(EVAL_DIR / "evaluation_dataset.jsonl") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    return queries


def load_expected(qid):
    path = RESULTS_DIR / f"{qid}.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        return data.get("rows", []), data.get("columns", [])
    return [], []


def execute_sql(conn, sql):
    try:
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout = '10s'")
            cur.execute(sql)
            columns = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchall()
        return {
            "success": True, "columns": columns,
            "rows": [list(r) for r in rows], "row_count": len(rows),
        }
    except Exception as e:
        conn.rollback()
        return {
            "success": False, "error": str(e),
            "rows": [], "row_count": 0, "columns": [],
        }


def compute_accuracy(conn, sql, qid):
    expected_rows, expected_columns = load_expected(qid)
    if not sql:
        return 0.0
    exec_result = execute_sql(conn, sql)
    if not exec_result.get("success"):
        return 0.0
    metrics = execution_accuracy_full(
        exec_result["rows"], expected_rows,
        exec_result["columns"], expected_columns,
    )
    return metrics.get("recall", 0.0)


def run_dict_condition(
    conn, queries, fraction,
    allowed_joins, allowed_columns, table_graph, exec_fn,
):
    """Run the full pipeline with a reduced dictionary."""
    import llm.schema_linker as sl_mod
    import llm.entity_extractor as ee_mod
    import llm.condition_mapper as cm_mod

    # Save originals
    orig_ctm = sl_mod.CONDITION_TABLE_MAP
    orig_ccm = sl_mod.CONDITION_COLUMN_MAP
    orig_map = sl_mod.map_conditions
    orig_terms_ee = ee_mod._load_terms
    orig_terms_cm = cm_mod._load_terms

    if fraction <= 0:
        # Full disable (same as no_dict ablation)
        sl_mod.CONDITION_TABLE_MAP = {}
        sl_mod.CONDITION_COLUMN_MAP = {}
        sl_mod.map_conditions = lambda c: {}
    else:
        # Partial reduction
        sl_mod.CONDITION_TABLE_MAP = _subset_dict(orig_ctm, fraction)
        sl_mod.CONDITION_COLUMN_MAP = _subset_dict(orig_ccm, fraction)

        # Also reduce material_terms
        import yaml
        terms_path = Path(__file__).resolve().parent.parent / "llm" / "material_terms.yaml"
        with open(terms_path) as f:
            full_terms = yaml.safe_load(f)
        reduced_terms = _subset_terms(full_terms, fraction)

        def patched_load_terms_ee(path=None):
            return reduced_terms

        def patched_load_terms_cm(path=None):
            return reduced_terms

        ee_mod._load_terms = patched_load_terms_ee  # type: ignore[assignment]
        cm_mod._load_terms = patched_load_terms_cm  # type: ignore[assignment]
        # Clear LRU caches
        if hasattr(ee_mod._load_terms, "cache_clear"):
            ee_mod._load_terms.cache_clear()  # type: ignore[union-attr]
        if hasattr(cm_mod._load_terms, "cache_clear"):
            cm_mod._load_terms.cache_clear()  # type: ignore[union-attr]

    n_schema = len(sl_mod.CONDITION_TABLE_MAP)
    print(f"  Schema-link entries: {n_schema}/{len(orig_ctm)}")

    results = []
    try:
        for i, q in enumerate(queries):
            qid = q["id"]
            question = q["question"]
            difficulty = q["difficulty"]

            print(
                f"  [{i+1}/{len(queries)}] {qid} ({difficulty})...",
                end=" ", flush=True,
            )

            t0 = time.time()
            try:
                pipe_result = sql_pipeline(
                    user_query=question,
                    join_list=allowed_joins,
                    all_columns=allowed_columns,
                    skip_intent_check=True,
                    n_best=3,
                    execute_fn=exec_fn,
                    table_graph=table_graph,
                )
                elapsed = time.time() - t0
                sql = pipe_result.get("sql", "")
                if sql:
                    sql = normalize_limit(sql)
                acc = compute_accuracy(conn, sql, qid)
                print(f"acc={acc:.1%}  {elapsed:.1f}s")

                results.append({
                    "qid": qid,
                    "difficulty": difficulty,
                    "accuracy": acc,
                    "latency_s": round(elapsed, 1),
                })
            except Exception as e:
                elapsed = time.time() - t0
                print(f"ERROR: {type(e).__name__}: {e!s:.80s}  {elapsed:.1f}s")
                results.append({
                    "qid": qid,
                    "difficulty": difficulty,
                    "accuracy": 0.0,
                    "latency_s": round(elapsed, 1),
                })
    finally:
        sl_mod.CONDITION_TABLE_MAP = orig_ctm
        sl_mod.CONDITION_COLUMN_MAP = orig_ccm
        sl_mod.map_conditions = orig_map
        ee_mod._load_terms = orig_terms_ee  # type: ignore[assignment]
        cm_mod._load_terms = orig_terms_cm  # type: ignore[assignment]
        # Restore caches
        if hasattr(orig_terms_ee, "cache_clear"):
            orig_terms_ee.cache_clear()  # type: ignore[union-attr]
        if hasattr(orig_terms_cm, "cache_clear"):
            orig_terms_cm.cache_clear()  # type: ignore[union-attr]

    return results


def main():
    model = os.getenv("LLM_MODEL", "gpt-5.5")
    out_path = PROJECT / "evaluation" / "dict_sensitivity_results.json"

    # Allow resuming from a specific config
    start_from = os.getenv("DICT_START", "")
    configs = DICT_CONFIGS
    if start_from:
        idx = next((i for i, c in enumerate(DICT_CONFIGS) if c[0] == start_from), 0)
        configs = DICT_CONFIGS[idx:]

    print(f"Model: {model}")
    print(f"Configs: {[c[0] for c in configs]}")
    print("Connecting to PostgreSQL...")
    conn = psycopg.connect(CONNINFO)

    print("Loading schema...")
    tables = get_tables(conn)
    columns = {}
    for t in tables:
        columns[t] = get_columns(conn, t)
    fks = get_foreign_keys(conn)
    table_graph = build_table_graph(fks)
    if not table_graph.has_edge("composition", "element"):
        table_graph.add_edge(
            "composition", "element",
            source_column="element", target_column="symbol",
        )
    allowed_columns = [
        f"{t}.{c.column_name}" for t, cols in columns.items() for c in cols
    ]
    allowed_joins = get_allowed_join_list(table_graph)

    print("Loading queries...")
    all_queries = load_queries()
    print(f"Total queries: {len(all_queries)}")

    def exec_fn(sql):
        return execute_sql(conn, sql)

    # Load existing results if resuming
    all_results: dict[str, Any] = {}
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
        all_results = existing.get("conditions", {})
        print(f"Loaded existing results: {list(all_results.keys())}")

    for label, fraction in configs:
        cond_name = f"dict_{label}"
        if cond_name in all_results:
            print(f"\nSkipping {cond_name} (already exists)")
            continue

        print(f"\n{'='*70}")
        print(f"CONDITION: dictionary={label} (fraction={fraction})")
        print(f"{'='*70}")

        results = run_dict_condition(
            conn, all_queries, fraction,
            allowed_joins, allowed_columns, table_graph, exec_fn,
        )

        # Compute summary
        total_acc = sum(r["accuracy"] for r in results) / len(results)
        by_diff: dict[str, list[float]] = {}
        for r in results:
            by_diff.setdefault(r["difficulty"], []).append(r["accuracy"])
        diff_summary = {
            d: sum(accs) / len(accs) for d, accs in by_diff.items()
        }
        avg_latency = sum(r["latency_s"] for r in results) / len(results)

        print(f"\n  Overall: {total_acc:.1%}")
        for d in ["easy", "medium", "hard", "very_hard"]:
            if d in diff_summary:
                print(f"  {d:12s}: {diff_summary[d]:.1%}")
        print(f"  Avg latency: {avg_latency:.1f}s")

        all_results[cond_name] = {
            "label": label,
            "fraction": fraction,
            "overall": total_acc,
            "by_difficulty": diff_summary,
            "avg_latency": avg_latency,
            "results": results,
        }

        # Save after each config (incremental)
        with open(out_path, "w") as f:
            json.dump({
                "model": model,
                "n_queries": len(all_queries),
                "configs": [c[0] for c in DICT_CONFIGS],
                "conditions": all_results,
            }, f, ensure_ascii=False, indent=2)
        print(f"  Saved to {out_path}")

    # Final summary
    print(f"\n{'='*70}")
    print("DICTIONARY SENSITIVITY SUMMARY")
    print(f"{'='*70}")
    print(
        f"{'Dict':>8s} {'Overall':>8s} {'Easy':>8s} {'Medium':>8s} "
        f"{'Hard':>8s} {'VHard':>8s} {'Latency':>8s}"
    )
    print("-" * 60)

    for label, _frac in DICT_CONFIGS:
        cond_name = f"dict_{label}"
        if cond_name not in all_results:
            continue
        r = all_results[cond_name]
        diff = r["by_difficulty"]
        print(
            f"{label:>8s} {r['overall']:7.1%} {diff.get('easy',0):7.1%} "
            f"{diff.get('medium',0):7.1%} {diff.get('hard',0):7.1%} "
            f"{diff.get('very_hard',0):7.1%} {r['avg_latency']:6.1f}s"
        )

    conn.close()
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
