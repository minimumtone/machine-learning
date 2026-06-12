#!/usr/bin/env python3
"""Validate that paper_figures.json values match TeX hardcoded numbers.

Usage:
    python scripts/validate_paper_numbers.py

Checks key numeric values in paper/t2sql_materials_paper.tex against
paper/paper_figures.json to detect drift between the two.
Additionally validates a hardcoded list of critical values that must appear in TeX.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEX = ROOT / "paper" / "t2sql_materials_paper.tex"
JSON_PATH = ROOT / "paper" / "paper_figures.json"


def _tex_contains(tex: str, value: str | int | float) -> bool:
    """Check if value appears in TeX, considering LaTeX number formatting."""
    s = str(value)
    if s in tex:
        return True
    # LaTeX thousand separator: 1{,}470
    if s.isdigit() and len(s) >= 4:
        latex_num = re.sub(r"(\d)(?=(\d{3})+$)", r"\1{,}", s)
        if latex_num in tex:
            return True
    return False


def main() -> int:
    if not JSON_PATH.exists():
        print(f"ERROR: {JSON_PATH} not found")
        return 1
    if not TEX.exists():
        print(f"ERROR: {TEX} not found")
        return 1

    with open(JSON_PATH, encoding="utf-8") as f:
        fig = json.load(f)
    tex = TEX.read_text(encoding="utf-8")

    errors: list[str] = []
    warnings: list[str] = []
    checked = 0

    def check(label: str, value, *, required: bool = True):
        nonlocal checked
        checked += 1
        if _tex_contains(tex, value):
            return
        msg = f"{label}: value={value} not found in TeX"
        if required:
            errors.append(msg)
        else:
            warnings.append(msg)

    # ================================================================
    # Part 1: JSON-driven checks (keys must match paper_figures.json)
    # ================================================================

    # Proposed overall
    proposed = fig.get("proposed_overall", {})
    if proposed.get("exec_accuracy_pct") is not None:
        check("Proposed accuracy", proposed["exec_accuracy_pct"])
    else:
        errors.append("KEY MISSING: proposed_overall.exec_accuracy_pct")

    # 3-run stats
    stats = fig.get("proposed_3run_stats", {})
    if stats.get("mean_accuracy_pct") is not None:
        check("3-run mean", stats["mean_accuracy_pct"])
    else:
        errors.append("KEY MISSING: proposed_3run_stats.mean_accuracy_pct")
    if stats.get("stdev_pp") is not None:
        check("3-run stdev", stats["stdev_pp"])
    else:
        errors.append("KEY MISSING: proposed_3run_stats.stdev_pp")

    # DB stats
    db = fig.get("db_stats", {})
    proto = db.get("prototype_distribution", {})
    for pname in ["L12", "B2", "NaCl", "NiAs", "BiF3"]:
        if proto.get(pname) is not None:
            check(f"Prototype {pname}", proto[pname], required=(pname == "L12"))

    n_entries = db.get("n_entries")
    if n_entries is not None:
        check("DB total entries", n_entries)
    n_tables = db.get("n_tables")
    if n_tables is not None:
        check("DB n_tables", n_tables, required=False)

    # Materials engineering
    mat = fig.get("materials_engineering", {})
    for key in ["stable_l12_screened_total", "stable_candidates", "metastable_candidates",
                "gamma_prime_ranking_total"]:
        val = mat.get(key)
        if val is not None:
            check(f"materials_engineering.{key}", val)

    # L12 recovery
    recovery = mat.get("known_l12_recovery")
    if recovery:
        total = recovery.get("total")
        recovered = recovery.get("recovered")
        if total is not None and recovered is not None:
            check("L12 recovery", f"{recovered}/{total}")

    # Baselines
    bl_comp = fig.get("baseline_comparison", {})
    for bl_key in ["B1", "B2", "B3", "B4"]:
        bl = bl_comp.get(bl_key, {})
        bl_acc = bl.get("exec_accuracy")
        if bl_acc is not None:
            check(f"Baseline {bl_key} accuracy", bl_acc, required=False)

    # Independent evaluation
    indep = fig.get("independent_eval", {})
    if indep:
        for key in ["binary_correct_rate", "mean_exec_accuracy"]:
            val = indep.get(key)
            if val is not None:
                check(f"independent_eval.{key}", val)

    # Latency
    latency = proposed.get("avg_latency_ms")
    if latency is not None:
        check("Avg latency (ms)", int(latency))
    avg_tokens = proposed.get("avg_token_usage")
    if avg_tokens is not None:
        check("Avg token usage", int(avg_tokens), required=False)

    # Difficulty breakdown (by_difficulty from proposed)
    by_diff = fig.get("proposed_by_difficulty", {})
    for diff_key in ["Easy", "Medium", "Hard", "Very Hard"]:
        d = by_diff.get(diff_key, {})
        acc = d.get("exec_accuracy_pct")
        if acc is not None:
            check(f"Difficulty {diff_key} accuracy", acc, required=False)
        n = d.get("n")
        if n is not None:
            check(f"Difficulty {diff_key} count", n, required=False)

    # ================================================================
    # Part 2: Explicit critical values (safety net)
    # These MUST appear in TeX regardless of JSON structure.
    # ================================================================
    critical_values = {
        "Proposed representative accuracy": 70.6,
        "3-run mean accuracy": 70.9,
        "3-run stdev": 1.7,
        "Very Hard query count": 23,
        "L12 prototype count": 392,
        "DB total entries": 1470,
        "Stable+metastable L12 total": 162,
        "Stable L12 count": 8,
        "Metastable L12 count": 154,
        "Gamma-prime ranking total": 259,
        "Independent binary correct rate": 67.0,
        "Independent mean accuracy": 76.6,
    }
    for label, val in critical_values.items():
        check(f"[CRITICAL] {label}", val)

    # Independent eval difficulty breakdown
    indep_by_diff = indep.get("by_difficulty", {})
    for diff_key in ["easy", "medium", "hard", "very_hard"]:
        d = indep_by_diff.get(diff_key, {})
        acc = d.get("accuracy")
        if acc is not None:
            check(f"Independent {diff_key} accuracy", acc, required=False)

    # ================================================================
    # Part 3: TeX unresolved reference detection
    # ================================================================
    unresolved = []
    for m in re.finditer(r'(?<![\\%])(\?\?)', tex):
        line_num = tex[:m.start()].count('\n') + 1
        unresolved.append(f"Line {line_num}: unresolved reference '??'")
    for m in re.finditer(r'\[\\?\?\]', tex):
        line_num = tex[:m.start()].count('\n') + 1
        unresolved.append(f"Line {line_num}: unresolved citation '[?]'")
    if unresolved:
        for u in unresolved[:10]:
            errors.append(f"[UNRESOLVED] {u}")
        if len(unresolved) > 10:
            errors.append(f"[UNRESOLVED] ... and {len(unresolved) - 10} more")

    # ================================================================
    # Report
    # ================================================================
    print(f"Checked {checked} values.")
    if errors:
        print(f"\n{'='*60}")
        print(f"VALIDATION FAILED: {len(errors)} error(s), {len(warnings)} warning(s)")
        print(f"{'='*60}")
        for e in errors:
            print(f"  ERROR: {e}")
        for w in warnings:
            print(f"  WARN:  {w}")
        return 1

    print(f"All {checked} values verified in TeX ({len(warnings)} warning(s))")
    for w in warnings:
        print(f"  WARN: {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
