#!/usr/bin/env python3
"""Multi-run ablation: run eval_ablation N times, save each run separately.

Usage:
    python scripts/eval_ablation_multirun.py [--n-runs 5] [--start-run 1] [--condition COND]

Each run saves to evaluation/ablation_run_{i}.json
After all runs, computes mean +/- SD and saves to evaluation/ablation_multirun_stats.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from scripts.provenance import build_provenance  # noqa: E402
from scripts.sign_permutation import (  # noqa: E402
    TEST_LABEL, sign_permutation_pvalue, stats_meta)

EVAL_DIR = PROJECT / "evaluation"
RESULTS_FILE = EVAL_DIR / "ablation_results.json"


def run_single(run_id: int, condition: str | None = None) -> dict:
    """Run a single ablation pass and return results."""
    # Back up existing results
    backup = RESULTS_FILE.with_suffix(f".backup_run{run_id}.json")
    if RESULTS_FILE.exists():
        shutil.copy2(RESULTS_FILE, backup)
        # eval_ablation resumes from an existing results file
        RESULTS_FILE.unlink()

    # Set environment to start from specific condition if requested
    env = os.environ.copy()
    if condition:
        env["ABLATION_START"] = condition

    # Import and run the ablation main function
    import importlib
    import scripts.eval_ablation as ablation_mod
    importlib.reload(ablation_mod)

    ablation_mod.main()

    # Read and return results
    with open(RESULTS_FILE) as f:
        return json.load(f)


def compute_stats(runs: list[dict]) -> dict:
    """Compute mean, SD, min, max across runs for each condition."""
    import numpy as np

    conditions = list(runs[0]["conditions"].keys())
    stats = {}

    for cond in conditions:
        cond_stats: dict = {"n_runs": len(runs)}

        # Overall accuracy
        overalls = [r["conditions"][cond]["overall"] for r in runs]
        cond_stats["overall_mean"] = float(np.mean(overalls))
        cond_stats["overall_std"] = float(np.std(overalls, ddof=1))
        cond_stats["overall_min"] = float(np.min(overalls))
        cond_stats["overall_max"] = float(np.max(overalls))

        # Per-difficulty
        diff_stats = {}
        for diff in ["easy", "medium", "hard", "very_hard"]:
            vals = [r["conditions"][cond]["by_difficulty"].get(diff, 0) for r in runs]
            diff_stats[diff] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
        cond_stats["by_difficulty"] = diff_stats

        # Latency
        lats = [r["conditions"][cond]["avg_latency"] for r in runs]
        cond_stats["avg_latency_mean"] = float(np.mean(lats))
        cond_stats["avg_latency_std"] = float(np.std(lats, ddof=1))

        # Per-query accuracy across runs (for McNemar test later)
        per_query = {}
        for r in runs:
            for qr in r["conditions"][cond]["results"]:
                qid = qr["qid"]
                per_query.setdefault(qid, []).append(qr["accuracy"])
        cond_stats["per_query_mean"] = {
            qid: float(np.mean(accs)) for qid, accs in per_query.items()
        }

        stats[cond] = cond_stats

    # Compute deltas relative to full
    full_mean = stats["full"]["overall_mean"]
    for cond in conditions:
        if cond != "full":
            stats[cond]["delta_mean"] = stats[cond]["overall_mean"] - full_mean

    return stats


def _holm(pvalues: dict) -> dict:
    """Holm-Bonferroni step-down adjusted p-values."""
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(ordered)
    adjusted: dict = {}
    running = 0.0
    for i, (name, p) in enumerate(ordered):
        running = min(1.0, max(running, (m - i) * p))
        adjusted[name] = running
    return adjusted


def compute_significance(stats: dict) -> dict:
    """Sign-permutation test of the signed-rank statistic, full vs each condition.

    Paired samples are the per-query mean accuracies across runs.  Zero
    differences are dropped and the two-sided p-value is the exact
    sign-permutation p-value of the Wilcoxon signed-rank statistic with
    midranks for tied |diff| (``scripts/sign_permutation.py``).  The per-query
    means take only the values {0, 0.2, ..., 1.0}, so the differences are
    heavily tied and SciPy's untied ``method="exact"`` table is not exact for
    them; the enumeration is deterministic and SciPy-version independent, and
    it is the same test used for the EN/JA language comparison.  Because every
    ablated condition is compared against the same ``full`` baseline, a
    Holm-Bonferroni correction is applied across conditions and
    ``significant`` refers to the corrected p-value.

    ``p_value`` is the uncorrected p-value and ``p_value_holm`` the corrected
    one; report the corrected value when claiming significance.
    """

    full_pq = stats["full"]["per_query_mean"]
    raw: dict = {}
    significance: dict = {}
    for cond, cond_stats in stats.items():
        if cond == "full":
            continue
        cond_pq = cond_stats["per_query_mean"]
        qids = sorted(set(full_pq) & set(cond_pq))
        diffs = [full_pq[q] - cond_pq[q] for q in qids]
        nonzero = [d for d in diffs if d != 0]
        delta_pp = float(sum(diffs) / len(diffs) * 100)
        p_value = sign_permutation_pvalue(nonzero)
        raw[cond] = p_value
        significance[cond] = {
            "delta_pp": delta_pp,
            "p_value": p_value,
            "n_nonzero": len(nonzero),
            "n_queries": len(qids),
            "test": TEST_LABEL,
        }

    adjusted = _holm(raw)
    for cond, entry in significance.items():
        entry["p_value_holm"] = adjusted[cond]
        entry["significant"] = adjusted[cond] < 0.05
        entry["correction"] = "holm-bonferroni"
    return significance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--start-run", type=int, default=1)
    parser.add_argument("--condition", type=str, default=None,
                        help="Run only this condition (default: all)")
    args = parser.parse_args()

    runs = []

    # Load any previously completed runs
    for i in range(1, args.start_run):
        run_file = EVAL_DIR / f"ablation_run_{i}.json"
        if run_file.exists():
            with open(run_file) as f:
                runs.append(json.load(f))
            print(f"Loaded existing run {i}")

    for run_id in range(args.start_run, args.n_runs + 1):
        run_file = EVAL_DIR / f"ablation_run_{run_id}.json"

        # Skip if already completed
        if run_file.exists():
            print(f"\n{'='*70}")
            print(f"RUN {run_id}/{args.n_runs}: Already completed, loading")
            print(f"{'='*70}")
            with open(run_file) as f:
                runs.append(json.load(f))
            continue

        print(f"\n{'='*70}")
        print(f"RUN {run_id}/{args.n_runs}")
        print(f"{'='*70}")

        t0 = time.time()
        result = run_single(run_id, args.condition)
        elapsed = time.time() - t0

        result["run_id"] = run_id
        result["elapsed_s"] = round(elapsed, 1)

        # Save individual run
        with open(run_file, "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nRun {run_id} saved to {run_file} ({elapsed:.0f}s)")

        runs.append(result)

    # Compute and save statistics
    if len(runs) >= 2:
        stats = compute_stats(runs)
        significance = compute_significance(stats)
        stats_file = EVAL_DIR / "ablation_multirun_stats.json"
        with open(stats_file, "w") as f:
            json.dump({"_meta": stats_meta(len(runs)),
                       "n_runs": len(runs),
                       "provenance": build_provenance(
                           EVAL_DIR / "evaluation_dataset.jsonl",
                           rescore_note=(
                               "aggregated deterministically from stored "
                               "ablation_run_*.json; the per-run provenance "
                               "in those files identifies the inference "
                               "revision")),
                       "conditions": stats,
                       "significance_tests": significance}, f,
                      ensure_ascii=False, indent=2)
        print(f"\nStatistics saved to {stats_file}")

        # Print summary
        print(f"\n{'='*70}")
        print(f"MULTI-RUN SUMMARY ({len(runs)} runs)")
        print(f"{'='*70}")
        print(f"{'Condition':15s} {'Mean':>8s} {'SD':>6s} {'Min':>8s} {'Max':>8s} {'Delta':>8s}")
        print("-" * 55)
        for cond in ["full", "no_fewshot", "no_dict", "no_reranker", "no_guard", "no_nbest", "no_graph"]:
            if cond not in stats:
                continue
            s = stats[cond]
            delta = f"{s.get('delta_mean', 0):+.1%}" if cond != "full" else "---"
            print(f"{cond:15s} {s['overall_mean']:7.1%} {s['overall_std']:5.1%} "
                  f"{s['overall_min']:7.1%} {s['overall_max']:7.1%} {delta:>8s}")


if __name__ == "__main__":
    main()
