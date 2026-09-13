#!/usr/bin/env python3
"""Assign lattice constants to the Chen et al. (2023) stable single-phase HEAs.

Chen et al., Nat. Commun. 14, 2856 (2023), DOI: 10.5281/zenodo.7633180 predict
phase stability but publish no lattice constants.  This script attaches the
present work's Omega_sf prediction to their stable equiatomic quinaries,
producing a combined stability + lattice-constant screening table.

Each structure uses the reference/q combination the paper actually validated:
BCC takes the SQS Omega_sf on the DFT Vegard reference with q=1, while FCC
takes the L1_2 Omega_sf on the King reference with q_FCC calibrated by
optimize_gamma() on ALONSO_TABLE2.  The databases, the calibration and the
prediction formula are imported from the paper's single-source modules;
nothing is re-derived here.  HCP is out of scope (no HCP database).
"""
from pathlib import Path
import itertools
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PAPER))

from chen2023_crosscheck import DOI, chen_data_root  # noqa: E402
from generate_all_figures import (  # noqa: E402
    EXCLUDE_ELEMENTS,
    compute_omega_sf_pairwise,
    load_compounds,
    load_sqs_data,
    optimize_gamma,
)
from hea_lattice_xgboost import (  # noqa: E402
    ALONSO_TABLE2,
    KING_ATOMIC_VOLUMES,
    compute_eq10_scaled,
    compute_vegard,
)

# Chen's 30,201 single-phase count corresponds to the 0.9*Tm annealing dataset.
DATASET = "quinaries_0.9Tm.csv.gz"
STRUCTS = ("BCC", "FCC")
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "font.size": 22})


def load_stable():
    path = chen_data_root() / "dataset" / DATASET
    d = pd.read_csv(path)
    stable = d[d.stability == "stable"].copy()
    stable["elements"] = stable.system.str.split("-")
    return d, stable


def adopted_models():
    """Return {structure: (Omega_sf database, q, label)} as validated in the paper.

    BCC uses the SQS database on the DFT Vegard reference at q=1 (the adopted
    model).  FCC has no calibrated q for the SQS database, so it uses the
    paper's FCC model: L1_2 Omega_sf on the King reference with q_FCC from
    optimize_gamma() on the Alonso training HEAs.
    """
    sqs = load_sqs_data()
    if sqs is None:
        raise RuntimeError("load_sqs_data() returned no SQS database")
    ob2, ol12 = compute_omega_sf_pairwise(load_compounds())
    _, q_fcc = optimize_gamma(ALONSO_TABLE2, ob2, ol12)
    return {
        "BCC": (sqs["omega_dft"], 1.0, "SQS, DFT Vegard reference"),
        "FCC": (ol12, float(q_fcc), "L1_2, King reference"),
    }


def screen(stable, models):
    """Attach Vegard and Omega_sf-corrected constants where the pairs allow."""
    rows, skipped = [], {"phase_out_of_scope": 0, "excluded_element": 0,
                         "no_king_volume": 0, "missing_pairs": 0}
    for _, r in stable.iterrows():
        if r.phase not in STRUCTS:
            skipped["phase_out_of_scope"] += 1
            continue
        els = list(r.elements)
        if any(e in EXCLUDE_ELEMENTS for e in els):
            skipped["excluded_element"] += 1
            continue
        if any(e not in KING_ATOMIC_VOLUMES for e in els):
            skipped["no_king_volume"] += 1
            continue
        omega, q_used, omega_source = models[r.phase]
        pairs = [tuple(sorted(p)) for p in itertools.combinations(els, 2)]
        n_covered = sum(1 for p in pairs if p in omega)
        if n_covered < len(pairs):
            skipped["missing_pairs"] += 1
            continue
        comp = {e: 1.0 / len(els) for e in els}
        a_veg = compute_vegard(comp, r.phase)
        a_pred = compute_eq10_scaled(comp, r.phase, omega, q_used)
        rows.append({
            "system": r.system,
            "phase": r.phase,
            "e_above_eV": r.e_above,
            "hmix_eV": r.hmix,
            "delta_r_percent": r.delta_r,
            "vec": r.vec,
            "a_vegard_A": a_veg,
            "a_pred_A": a_pred,
            "q_used": q_used,
            "omega_source": omega_source,
            "shift_A": a_pred - a_veg,
            "shift_percent": 100.0 * (a_pred - a_veg) / a_veg,
        })
    return pd.DataFrame(rows), skipped


def pair_coverage(stable, omega_by_struct):
    """Distribution of covered binary pairs per stable BCC/FCC alloy."""
    counts = []
    for _, r in stable.iterrows():
        if r.phase not in STRUCTS:
            continue
        els = list(r.elements)
        omega = omega_by_struct[r.phase]
        pairs = [tuple(sorted(p)) for p in itertools.combinations(els, 2)]
        counts.append(sum(1 for p in pairs if p in omega))
    series = pd.Series(counts)
    return {str(k): int(v) for k, v in series.value_counts().sort_index().items()}


def fig_distribution(d):
    fig, ax = plt.subplots(1, 2, figsize=(17, 6.5))
    for a, struct, color in zip(ax, STRUCTS, ("#1f77b4", "#d62728")):
        s = d[d.phase == struct]
        if s.empty:
            a.set_axis_off()
            continue
        q = float(s.q_used.iloc[0])
        a.hist(s.a_pred_A, bins=40, color=color, alpha=0.85,
               label=f"$q={q:.4g}$ 補正後 ($n$={len(s)})")
        a.hist(s.a_vegard_A, bins=40, histtype="step", linewidth=2.5,
               color="black", label="Vegard")
        a.set_xlabel("格子定数 $a$ (Å)")
        a.set_ylabel("合金数")
        a.set_title(f"{struct}（Chen安定単相）")
        a.legend(fontsize=17)
    fig.tight_layout()
    out = PAPER / "fig_chen2023_screening_lattice.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig_shift(d):
    fig, ax = plt.subplots(1, 2, figsize=(17, 6.5))
    for struct, color in zip(STRUCTS, ("#1f77b4", "#d62728")):
        s = d[d.phase == struct]
        if s.empty:
            continue
        q = float(s.q_used.iloc[0])
        ax[0].scatter(s.a_vegard_A, s.a_pred_A, s=8, alpha=0.35,
                      color=color, label=f"{struct} ($q={q:.4g}$, $n$={len(s)})")
        ax[1].hist(s.shift_percent, bins=40, alpha=0.6, color=color,
                   label=f"{struct} ($q={q:.4g}$)")
    lo = min(d.a_vegard_A.min(), d.a_pred_A.min()) - 0.05
    hi = max(d.a_vegard_A.max(), d.a_pred_A.max()) + 0.05
    ax[0].plot([lo, hi], [lo, hi], "k--", linewidth=2)
    ax[0].set_xlim(lo, hi)
    ax[0].set_ylim(lo, hi)
    ax[0].set_xlabel("Vegard $a$ (Å)")
    q_text = "; ".join(
        f"{struct} $q={float(d[d.phase == struct].q_used.iloc[0]):.4g}$"
        for struct in STRUCTS if not d[d.phase == struct].empty
    )
    ax[0].set_ylabel(f"$q$補正後 $a$ (Å) [{q_text}]")
    ax[0].legend(fontsize=17)
    ax[1].axvline(0.0, color="black", linewidth=2)
    ax[1].set_xlabel(r"$\Omega_{\mathrm{sf}}$ 補正量 (%)")
    ax[1].set_ylabel("合金数")
    ax[1].legend(fontsize=17)
    fig.tight_layout()
    out = PAPER / "fig_chen2023_screening_shift.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main():
    models = adopted_models()

    full, stable = load_stable()
    d, skipped = screen(stable, models)
    d = d.sort_values(["phase", "a_pred_A"]).reset_index(drop=True)
    csv_path = PAPER / "results_chen2023_screening.csv"
    d.to_csv(csv_path, index=False)

    n_stable_struct = int((stable.phase.isin(STRUCTS)).sum())
    metrics = {
        "_source": f"Chen et al. 2023 ({DOI}), dataset {DATASET}",
        "_generated_by": "paper/chen2023_lattice_screening.py",
        "models": {
            struct: {
                "omega_source": models[struct][2],
                "q": models[struct][1],
                "n_pairs": len(models[struct][0]),
            }
            for struct in STRUCTS
        },
        "n_quinaries_total": int(len(full)),
        "n_stable_total": int(len(stable)),
        "n_stable_by_phase": {k: int(v) for k, v in
                              stable.phase.value_counts().items()},
        "n_stable_bcc_fcc": n_stable_struct,
        "n_predicted": int(len(d)),
        "coverage_fraction_of_stable_bcc_fcc":
            round(len(d) / n_stable_struct, 6) if n_stable_struct else None,
        "n_skipped": {k: int(v) for k, v in skipped.items()},
        "pair_coverage_histogram": pair_coverage(
            stable, {k: v[0] for k, v in models.items()}
        ),
    }
    for struct in STRUCTS:
        s = d[d.phase == struct]
        if s.empty:
            continue
        metrics[f"{struct.lower()}_stats"] = {
            "n": int(len(s)),
            "a_pred_min_A": round(float(s.a_pred_A.min()), 4),
            "a_pred_max_A": round(float(s.a_pred_A.max()), 4),
            "a_pred_mean_A": round(float(s.a_pred_A.mean()), 4),
            "shift_mean_percent": round(float(s.shift_percent.mean()), 4),
            "shift_min_percent": round(float(s.shift_percent.min()), 4),
            "shift_max_percent": round(float(s.shift_percent.max()), 4),
            "shift_abs_median_percent":
                round(float(s.shift_percent.abs().median()), 4),
            "n_shift_abs_gt_1percent": int((s.shift_percent.abs() > 1.0).sum()),
        }
    top = d.reindex(d.shift_percent.abs().sort_values(ascending=False).index)
    metrics["largest_corrections"] = [
        {
            "system": row.system,
            "phase": row.phase,
            "shift_percent": round(float(row.shift_percent), 4),
            "a_pred_A": round(float(row.a_pred_A), 4),
        }
        for row in top.head(10).itertuples()
    ]

    figs = [fig_distribution(d), fig_shift(d)] if not d.empty else []
    json_path = PAPER / "chen2023_screening_metrics.json"
    json.dump(metrics, open(json_path, "w"), indent=2, ensure_ascii=False)

    print(f"stable quinaries: {len(stable)} / {len(full)}")
    print(f"stable BCC+FCC:   {n_stable_struct}")
    print(f"predicted:        {len(d)}  ({csv_path.name})")
    print(f"skipped:          {skipped}")
    for struct in STRUCTS:
        s = d[d.phase == struct]
        if s.empty:
            continue
        q = float(s.q_used.iloc[0])
        print(f"{struct}: n={len(s)}, q={q:.6g}, "
              f"a={s.a_pred_A.min():.3f}"
              f"-{s.a_pred_A.max():.3f} A, "
              f"median |shift|={s.shift_percent.abs().median():.3f}%")
    if not d.empty:
        top = d.reindex(d.shift_percent.abs().sort_values(ascending=False).index)
        print("largest corrections:")
        print(top.head(10)[["system", "phase", "a_vegard_A",
                            "a_pred_A", "q_used", "shift_percent"]].to_string(index=False))
    for f in figs:
        print(f"wrote {f.name}")
    print(f"wrote {json_path.name}")


if __name__ == "__main__":
    main()
