#!/usr/bin/env python3
"""Paired statistics for the ja/en language-dependence evaluation.

Reads evaluation/language_paired_{ja,en}_run{1..3}.json, pairs the per-query
3-run mean recalls by query id, and computes:

* exact Wilcoxon signed-rank test on the non-zero paired differences
  (same method as the ablation significance analysis),
* a seeded bootstrap 95% percentile CI for the mean EN-JA difference (pp),
* per-difficulty exploratory differences with the number of non-tied pairs.

Writes evaluation/language_paired_stats.json.  Deterministic (fixed seed);
does not touch the saved run files.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

PROJECT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT / "evaluation"
SEED = 20260602
N_BOOT = 100_000


def per_query_means(lang: str) -> tuple[dict[str, float], dict[str, str]]:
    paths = sorted(EVAL_DIR.glob(f"language_paired_{lang}_run*.json"))
    acc: dict[str, list[float]] = {}
    diff_map: dict[str, str] = {}
    for p in paths:
        run = json.loads(p.read_text())
        for res in run["results"]:
            acc.setdefault(res["qid"], []).append(res["recall"])
            diff_map[res["qid"]] = res["difficulty"]
    return {q: statistics.mean(v) for q, v in acc.items()}, diff_map


def main() -> None:
    ja, diff_map = per_query_means("ja")
    en, _ = per_query_means("en")
    qids = sorted(set(ja) & set(en))
    diffs = np.array([en[q] - ja[q] for q in qids])
    nonzero = diffs[diffs != 0]
    p_value = float(wilcoxon(nonzero, method="auto").pvalue) if len(nonzero) else 1.0

    rng = np.random.default_rng(SEED)
    n = len(diffs)
    boot = np.empty(N_BOOT)
    for i in range(N_BOOT):
        boot[i] = diffs[rng.integers(0, n, n)].mean()
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5]) * 100

    by_diff: dict[str, dict] = {}
    for level in ("easy", "medium", "hard", "very_hard"):
        qs = [q for q in qids if diff_map[q] == level]
        d = np.array([en[q] - ja[q] for q in qs])
        by_diff[level] = {
            "n_queries": len(qs),
            "mean_diff_pp": round(float(d.mean() * 100), 1),
            "n_nonzero_pairs": int((d != 0).sum()),
        }

    out = {
        "n_queries": n,
        "mean_diff_pp": round(float(diffs.mean() * 100), 1),
        "wilcoxon_p_value": round(p_value, 3),
        "n_nonzero_pairs": int(len(nonzero)),
        "bootstrap_ci95_pp": [round(float(ci_lo), 1), round(float(ci_hi), 1)],
        "bootstrap_n_resamples": N_BOOT,
        "bootstrap_seed": SEED,
        "test": "wilcoxon-signed-rank-exact on per-query 3-run mean recall (en - ja)",
        "by_difficulty": by_diff,
    }
    dst = EVAL_DIR / "language_paired_stats.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"Saved: {dst}")


if __name__ == "__main__":
    main()
