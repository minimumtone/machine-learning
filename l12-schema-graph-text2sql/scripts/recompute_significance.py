#!/usr/bin/env python3
"""Recompute the ablation significance tests correctly.

Problems with the test as implemented in ``scripts/eval_ablation_multirun.py``:

1. **The normal approximation is forced.**  ``wilcoxon(..., method="approx")``
   is used regardless of sample size.  For ``no_graph`` only 3 query-level
   differences are non-zero and SciPy itself emits
   "Sample size too small for normal approximation".  SciPy's default
   (``method="auto"``) selects the exact distribution for small samples and is
   what should be used.

2. **No multiple-comparison correction.**  Six ablated conditions are each
   tested against the same ``full`` baseline at alpha = 0.05, and the result
   drives the significance stars in Figure ``ablation_bar``.  A Holm-Bonferroni
   step-down correction is applied here.

3. **Run-to-run variance is discarded.**  Each query's accuracy is averaged
   over the 5 runs first, and the test is then applied across queries.  The
   per-query mean can only take the values {0, .2, .4, .6, .8, 1}, so the
   paired differences are heavily tied, which the signed-rank test does not
   assume.  Two alternative analyses that do not discard the run structure are
   reported alongside: a paired test over the 5 run-level overall accuracies,
   and a cluster (per-query) bootstrap confidence interval for the delta.

Nothing is overwritten unless ``--apply`` is passed.  The default writes
``evaluation/significance_recomputed.json`` and prints a comparison so that the
effect of the correction on the reported numbers is visible before adopting it.

Usage:
    python scripts/recompute_significance.py
    python scripts/recompute_significance.py --apply     # update the stats file
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest, wilcoxon

PROJECT = Path(__file__).resolve().parent.parent
EVAL = PROJECT / "evaluation"
STATS_FILE = EVAL / "ablation_multirun_stats.json"

BOOTSTRAP_N = 10000
SEED = 20260821


def load_runs() -> list[dict[str, Any]]:
    runs = []
    for i in range(1, 100):
        f = EVAL / f"ablation_run_{i}.json"
        if not f.exists():
            break
        runs.append(json.load(open(f)))
    if not runs:
        sys.exit(f"No ablation_run_*.json found in {EVAL}")
    return runs


def per_query_matrix(runs: list[dict], cond: str) -> dict[str, list[float]]:
    """qid -> list of accuracies, one per run."""
    out: dict[str, list[float]] = {}
    for r in runs:
        for qr in r["conditions"][cond]["results"]:
            out.setdefault(qr["qid"], []).append(float(qr["accuracy"]))
    return out


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni step-down adjusted p-values."""
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for i, (name, p) in enumerate(ordered):
        running = min(1.0, max(running, (m - i) * p))
        adjusted[name] = running
    return adjusted


def stars(p: float | None) -> str:
    if p is None:
        return "---"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def bootstrap_ci(diffs: list[float], rng: np.random.Generator) -> tuple[float, float]:
    """Percentile CI of the mean paired difference, resampling queries."""
    arr = np.asarray(diffs, dtype=float)
    idx = rng.integers(0, len(arr), size=(BOOTSTRAP_N, len(arr)))
    means = arr[idx].mean(axis=1) * 100
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true",
                    help="overwrite significance_tests in ablation_multirun_stats.json")
    ap.add_argument("--out", default=str(EVAL / "significance_recomputed.json"))
    args = ap.parse_args()

    runs = load_runs()
    conds = list(runs[0]["conditions"].keys())
    rng = np.random.default_rng(SEED)

    pq = {c: per_query_matrix(runs, c) for c in conds}
    pq_mean = {c: {q: float(np.mean(v)) for q, v in d.items()} for c, d in pq.items()}
    full_mean = pq_mean["full"]

    shipped = {}
    if STATS_FILE.exists():
        shipped = json.load(open(STATS_FILE)).get("significance_tests", {})

    raw: dict[str, float] = {}
    detail: dict[str, dict[str, Any]] = {}

    for cond in conds:
        if cond == "full":
            continue
        qids = sorted(set(full_mean) & set(pq_mean[cond]))
        diffs = [full_mean[q] - pq_mean[cond][q] for q in qids]
        nonzero = [d for d in diffs if d != 0]
        delta_pp = float(np.mean(diffs) * 100)

        if nonzero:
            p_exact = float(wilcoxon(nonzero, method="auto").pvalue)
            with warnings.catch_warnings():
                # Deliberately reproducing the legacy call, which SciPy warns
                # about precisely because the sample is too small for it.
                warnings.simplefilter("ignore")
                p_approx = float(wilcoxon(nonzero, method="approx").pvalue)
        else:
            p_exact = p_approx = 1.0
        raw[cond] = p_exact

        # Run-level paired analysis (n = number of runs), which keeps the run
        # structure instead of collapsing it.
        run_full = [r["conditions"]["full"]["overall"] for r in runs]
        run_cond = [r["conditions"][cond]["overall"] for r in runs]
        run_diffs = [a - b for a, b in zip(run_full, run_cond)]
        run_nonzero = [d for d in run_diffs if d != 0]
        p_run = float(wilcoxon(run_nonzero, method="auto").pvalue) if run_nonzero else 1.0
        n_pos = sum(1 for d in run_nonzero if d > 0)
        p_sign = (binomtest(n_pos, len(run_nonzero), 0.5).pvalue
                  if run_nonzero else 1.0)

        lo, hi = bootstrap_ci(diffs, rng)

        detail[cond] = {
            "delta_pp": delta_pp,
            "n_queries": len(qids),
            "n_nonzero": len(nonzero),
            "p_value_exact": p_exact,
            "p_value_approx_legacy": p_approx,
            "run_level": {
                "n_runs": len(runs),
                "delta_pp": float(np.mean(run_diffs) * 100),
                "p_wilcoxon": p_run,
                "p_sign_test": float(p_sign),
            },
            "bootstrap_ci95_pp": [lo, hi],
            "legacy_p_value": shipped.get(cond, {}).get("p_value"),
        }

    adjusted = holm(raw)
    for cond, d in detail.items():
        d["p_value_holm"] = adjusted[cond]
        d["significant_holm"] = adjusted[cond] < 0.05
        d["stars_holm"] = stars(adjusted[cond])

    hdr = (f"{'condition':14s}{'n_nz':>5s}{'Δpp':>8s}{'legacy p':>12s}{'':>5s}"
           f"{'exact p':>12s}{'Holm p':>12s}{'':>6s}{'run-level p':>13s}"
           f"{'bootstrap 95% CI (pp)':>26s}")
    print(hdr)
    print("-" * len(hdr))
    changed = []
    for cond, d in detail.items():
        legacy = d["legacy_p_value"]
        lo, hi = d["bootstrap_ci95_pp"]
        print(f"{cond:14s}{d['n_nonzero']:5d}{d['delta_pp']:8.2f}"
              f"{(f'{legacy:.3e}' if legacy is not None else '--'):>12s}"
              f"{stars(legacy):>5s}"
              f"{d['p_value_exact']:12.3e}{d['p_value_holm']:12.3e}"
              f"{d['stars_holm']:>6s}"
              f"{d['run_level']['p_wilcoxon']:13.3f}"
              f"{f'[{lo:+.2f}, {hi:+.2f}]':>26s}")
        if legacy is not None and (legacy < 0.05) != d["significant_holm"]:
            changed.append(cond)

    print()
    if changed:
        print(f"Conclusions that change after correction: {', '.join(changed)}")
    else:
        print("No conclusion changes: the same conditions are significant before "
              "and after switching to the exact test and applying Holm.")
    print("Note: 'run-level p' has n = number of runs (5), so its smallest "
          "attainable value is 0.0625; it is reported as a sanity check on the "
          "sign of the effect, not as the primary test.")

    payload = {
        "_meta": {
            "generated_by": "scripts/recompute_significance.py",
            "n_runs": len(runs),
            "test": "Wilcoxon signed-rank on per-query mean accuracy, "
                    "exact distribution (SciPy method='auto')",
            "correction": "Holm-Bonferroni across the ablated conditions",
            "bootstrap": {"n_resamples": BOOTSTRAP_N, "unit": "query", "seed": SEED},
        },
        "conditions": detail,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {args.out}")

    if args.apply:
        if not STATS_FILE.exists():
            sys.exit(f"{STATS_FILE} not found; cannot apply")
        stats = json.load(open(STATS_FILE))
        stats["significance_tests"] = {
            cond: {
                "delta_pp": d["delta_pp"],
                "p_value": d["p_value_exact"],
                "p_value_holm": d["p_value_holm"],
                "significant": d["significant_holm"],
                "n_nonzero": d["n_nonzero"],
                "test": "wilcoxon-exact",
                "correction": "holm",
            }
            for cond, d in detail.items()
        }
        STATS_FILE.write_text(json.dumps(stats, indent=2))
        print(f"Applied to {STATS_FILE}. Re-run scripts/compute_all_figures.py "
              f"and scripts/generate_figures.py, and update the p-values quoted "
              f"in the manuscript.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
