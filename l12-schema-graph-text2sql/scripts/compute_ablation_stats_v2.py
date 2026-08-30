"""Extended significance statistics for the 7-condition ablation study.

For each full-vs-ablated comparison, using per-query accuracies averaged
over the 5 runs (n = 100 paired samples):

- n_eff: number of non-tied pairs (query-level mean difference != 0)
- exact paired sign-flip permutation p-value for the mean difference
  (full enumeration when n_eff <= 20, otherwise 200,000 Monte-Carlo
  sign flips with a fixed seed)
- matched-pairs rank-biserial correlation as effect size
- query-level bootstrap 95% CI (percentile, 20,000 resamples, n = 100)
  for the mean difference in percentage points

Outputs evaluation/ablation_significance_v2.json.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "evaluation"
N_RUNS = 5
CONDITIONS = [
    "no_fewshot", "no_dict", "no_reranker",
    "no_guard", "no_nbest", "no_graph",
]


def load_per_query() -> dict[str, dict[str, float]]:
    acc: dict[str, dict[str, list[float]]] = {}
    for i in range(1, N_RUNS + 1):
        data = json.loads((EVAL / f"ablation_run_{i}.json").read_text())
        for cond, cd in data["conditions"].items():
            for qr in cd["results"]:
                acc.setdefault(cond, {}).setdefault(qr["qid"], []).append(
                    qr["accuracy"]
                )
    return {
        cond: {qid: float(np.mean(v)) for qid, v in qmap.items()}
        for cond, qmap in acc.items()
    }


def sign_flip_p(diffs: np.ndarray, rng: np.random.Generator) -> tuple[float, str]:
    nz = diffs[diffs != 0]
    n = len(nz)
    obs = abs(diffs.mean())
    denom = len(diffs)
    if n == 0:
        return 1.0, "exact"
    if n <= 20:
        count = 0
        total = 2 ** n
        for signs in itertools.product((1.0, -1.0), repeat=n):
            stat = abs((nz * np.array(signs)).sum() / denom)
            if stat >= obs - 1e-12:
                count += 1
        return count / total, "exact"
    n_mc = 200_000
    signs = rng.choice((1.0, -1.0), size=(n_mc, n))
    stats = np.abs((signs * nz).sum(axis=1) / denom)
    p = (np.sum(stats >= obs - 1e-12) + 1) / (n_mc + 1)
    return float(p), "monte_carlo_200k"


def rank_biserial(diffs: np.ndarray) -> float:
    nz = diffs[diffs != 0]
    if len(nz) == 0:
        return 0.0
    ranks = np.argsort(np.argsort(np.abs(nz))) + 1.0
    # handle ties with average ranks
    from scipy.stats import rankdata
    ranks = rankdata(np.abs(nz))
    w_pos = ranks[nz > 0].sum()
    w_neg = ranks[nz < 0].sum()
    return float((w_pos - w_neg) / ranks.sum())


def main() -> None:
    pq = load_per_query()
    full = pq["full"]
    qids = sorted(full)
    rng = np.random.default_rng(20260602)
    out: dict[str, dict] = {}
    for cond in CONDITIONS:
        cpq = pq[cond]
        diffs = np.array([full[q] - cpq[q] for q in qids])
        n_eff = int(np.sum(diffs != 0))
        p, method = sign_flip_p(diffs, rng)
        boots = rng.choice(diffs, size=(20_000, len(diffs)), replace=True)
        means = boots.mean(axis=1) * 100
        ci = np.percentile(means, [2.5, 97.5])
        min_p = 2.0 / (2 ** n_eff) if n_eff > 0 else 1.0
        out[cond] = {
            "delta_pp": float(diffs.mean() * 100),
            "n_eff": n_eff,
            "p_exact_sign_flip": p,
            "p_method": method,
            "min_attainable_p": min_p,
            "test_powered_at_0.05": min_p <= 0.05,
            "rank_biserial": rank_biserial(diffs),
            "bootstrap_ci95_pp": [float(ci[0]), float(ci[1])],
            "n_bootstrap": 20_000,
            "seed": 20260602,
        }
    doc = {
        "_meta": {
            "generated_by": "scripts/compute_ablation_stats_v2.py",
            "status": (
                "alternative-method statistics (exact sign-flip test + "
                "rank-biserial) used only by the paper figure pipeline; "
                "the canonical significance results are "
                "evaluation/significance_recomputed.json"),
        },
        "conditions": out,
    }
    (EVAL / "ablation_significance_v2.json").write_text(
        json.dumps(doc, indent=2) + "\n"
    )
    for cond, s in out.items():
        print(
            f"{cond:12s} d={s['delta_pp']:+6.2f}pp n_eff={s['n_eff']:2d} "
            f"p={s['p_exact_sign_flip']:.4g} ({s['p_method']}) "
            f"min_p={s['min_attainable_p']:.4g} r_rb={s['rank_biserial']:+.2f} "
            f"CI=[{s['bootstrap_ci95_pp'][0]:+.1f},{s['bootstrap_ci95_pp'][1]:+.1f}]"
        )


if __name__ == "__main__":
    main()
