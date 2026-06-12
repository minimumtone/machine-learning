#!/usr/bin/env python3
"""Validate that paper_figures.json values match TeX hardcoded numbers.

Usage:
    python scripts/validate_paper_numbers.py

Checks key numeric values in paper/t2sql_materials_paper.tex against
paper/paper_figures.json to detect drift between the two.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEX = ROOT / "paper" / "t2sql_materials_paper.tex"
JSON_PATH = ROOT / "paper" / "paper_figures.json"


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

    def check(label: str, json_val: str, tex_pattern: str, *, required: bool = True):
        """Check if json_val appears in TeX matching the pattern."""
        if re.search(tex_pattern, tex):
            return
        # Try plain containment
        if str(json_val) in tex:
            return
        # Try LaTeX number format (e.g. 1{,}470 for 1470)
        val_str = str(json_val)
        if val_str.isdigit() and len(val_str) >= 4:
            latex_num = re.sub(r'(\d)(?=(\d{3})+$)', r'\1{,}', val_str)
            if latex_num in tex:
                return
        msg = f"{label}: JSON={json_val}, pattern not found in TeX"
        if required:
            errors.append(msg)
        else:
            warnings.append(msg)

    # --- Proposed accuracy ---
    proposed = fig.get("proposed_overall", {})
    acc = proposed.get("exec_accuracy_pct")
    if acc is not None:
        check("Proposed accuracy", acc, rf"{acc}\s*\\?%")

    # --- 3-run stats ---
    stats = fig.get("proposed_3run_stats", {})
    mean_acc = stats.get("mean_accuracy_pct")
    std_acc = stats.get("stdev_pp")
    if mean_acc is not None:
        check("3-run mean", mean_acc, rf"{mean_acc}")
    if std_acc is not None:
        check("3-run std", std_acc, rf"{std_acc}")

    # --- DB stats ---
    db = fig.get("db_stats", {})
    proto = db.get("prototype_distribution", {})
    l12_count = proto.get("L12")
    if l12_count is not None:
        check("L12 count", l12_count, rf"\b{l12_count}\b")

    n_entries = db.get("n_entries")
    if n_entries is not None:
        check("DB entries", n_entries, rf"\b{n_entries}\b", required=False)

    # --- Materials engineering ---
    mat = fig.get("materials_engineering", {})
    screened = mat.get("stable_l12_screened_total")
    if screened is not None:
        check("Stable+metastable L12", screened, rf"\b{screened}\b")

    # --- Baselines ---
    bl_comp = fig.get("baseline_comparison", {})
    for bl_key, bl_label in [("B1", "LLM-only"), ("B2", "Full-schema"), ("B3", "Rule-based"), ("B4", "FK-list")]:
        bl = bl_comp.get(bl_key, {})
        bl_acc = bl.get("exec_accuracy")
        if bl_acc is not None:
            check(f"{bl_label} ({bl_key}) accuracy", bl_acc, rf"{bl_acc}", required=False)

    # --- Independent evaluation ---
    expert = fig.get("independent_eval", {})
    if expert:
        binary = expert.get("binary_correct_rate")
        if binary is not None:
            check("Expert binary correct", binary, rf"{binary}")
        expert_acc = expert.get("mean_exec_accuracy")
        if expert_acc is not None:
            check("Expert mean accuracy", expert_acc, rf"{expert_acc}")

    # --- Latency ---
    latency = proposed.get("avg_latency_ms")
    if latency is not None:
        # check() handles LaTeX {,} format for integers >= 4 digits
        lat_int = int(latency) if isinstance(latency, (int, float)) else latency
        check("Avg latency", lat_int, rf"\b{lat_int}\b", required=False)

    # --- Report ---
    if errors:
        print(f"\n{'='*60}")
        print(f"VALIDATION FAILED: {len(errors)} error(s), {len(warnings)} warning(s)")
        print(f"{'='*60}")
        for e in errors:
            print(f"  ERROR: {e}")
        for w in warnings:
            print(f"  WARN:  {w}")
        return 1

    print(f"✓ All key values in paper_figures.json found in TeX ({len(warnings)} warning(s))")
    for w in warnings:
        print(f"  WARN: {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
