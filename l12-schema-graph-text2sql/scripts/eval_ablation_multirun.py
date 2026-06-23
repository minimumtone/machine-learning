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

EVAL_DIR = PROJECT / "evaluation"
RESULTS_FILE = EVAL_DIR / "ablation_results.json"


def run_single(run_id: int, condition: str | None = None) -> dict:
    """Run a single ablation pass and return results."""
    # Back up existing results
    backup = RESULTS_FILE.with_suffix(f".backup_run{run_id}.json")
    if RESULTS_FILE.exists():
        shutil.copy2(RESULTS_FILE, backup)

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
        cond_stats["overall_sd"] = float(np.std(overalls, ddof=1))
        cond_stats["overall_min"] = float(np.min(overalls))
        cond_stats["overall_max"] = float(np.max(overalls))

        # Per-difficulty
        diff_stats = {}
        for diff in ["easy", "medium", "hard", "very_hard"]:
            vals = [r["conditions"][cond]["by_difficulty"].get(diff, 0) for r in runs]
            diff_stats[diff] = {
                "mean": float(np.mean(vals)),
                "sd": float(np.std(vals, ddof=1)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
        cond_stats["by_difficulty"] = diff_stats

        # Latency
        lats = [r["conditions"][cond]["avg_latency"] for r in runs]
        cond_stats["avg_latency_mean"] = float(np.mean(lats))
        cond_stats["avg_latency_sd"] = float(np.std(lats, ddof=1))

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
        stats_file = EVAL_DIR / "ablation_multirun_stats.json"
        with open(stats_file, "w") as f:
            json.dump({"n_runs": len(runs), "conditions": stats}, f,
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
            print(f"{cond:15s} {s['overall_mean']:7.1%} {s['overall_sd']:5.1%} "
                  f"{s['overall_min']:7.1%} {s['overall_max']:7.1%} {delta:>8s}")


if __name__ == "__main__":
    main()
