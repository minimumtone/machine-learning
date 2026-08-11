#!/usr/bin/env python3
"""Omega_sf usefulness and physical origin of q_BCC -> 1 / q_FCC -> 0.

Uses the existing 95-HEA dataset (64 calibration + 31 independent test) and
the DFT-derived structure-specific Omega_sf (B2 / L1_2, paper/results_omega_sf.csv)
plus the King(1966)/Alonso(2022) experimental Omega_sf to quantify:

 1. How useful the King/Alonso volume size factor Omega_sf is for HEA lattice
    constant prediction (RMSE vs Vegard, per structure).
 2. Why the calibrated scaling factor q_s approaches 1 for BCC and 0 for FCC:
    q_hat is a least-squares slope q = cov(dV_exp, C) / var(C)
                                    = r * (sigma_dV / sigma_C),
    so it decomposes into (a) the correlation r between the Omega correction C
    and the true non-Vegard deviation dV_exp, and (b) the amplitude ratio.
    For FCC HEAs sigma_dV is at the experimental noise floor -> q -> 0
    (regression shrinkage), while for BCC HEAs deviations are large and well
    correlated with the SQS/DFT Omega correction -> q -> 1.

Outputs (single pass):
  results_q_decomposition.json
  fig_q_scatter_bcc_fcc.png     dV_exp vs C scatter with fitted q slopes
  fig_q_decomposition.png       sigma_dV, sigma_C, r, q per structure
  fig_delta_mismatch.png        atomic size mismatch delta: BCC vs FCC HEAs
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "paper"))

from generate_all_figures import load_sqs_data

from hea_lattice_xgboost import (
    ALONSO_TABLE2,
    INDEPENDENT_TEST,
    KING_ATOMIC_VOLUMES,
)

plt.rcParams.update({"font.size": 16, "axes.grid": True, "grid.alpha": 0.3,
                     "font.family": ["Noto Sans CJK JP", "IPAGothic", "sans-serif"],
                     "axes.unicode_minus": False})

# --- load structure-specific DFT Omega_sf -----------------------------------
om = pd.read_csv(REPO / "paper" / "results_omega_sf.csv")
OMEGA = {"BCC": {}, "FCC": {}}
for _, r in om.iterrows():
    struct = "BCC" if r.structure == "B2" else "FCC"
    a, b = r["pair"].split("-")
    OMEGA[struct][(a, b)] = float(r.omega_sf)


def norm_comp(comp):
    els = list(comp)
    f = np.array([comp[e] for e in els], float)
    return els, f / f.sum()


def deviation_and_correction(h, omega_bcc=None):
    """Return per-atom volume deviation from Vegard and Omega correction C."""
    els, f = norm_comp(h["comp"])
    vols = np.array([KING_ATOMIC_VOLUMES.get(e, 15.0) for e in els])
    n_auc = 4 if h["struct"] == "FCC" else 2
    v_exp = h["a_exp"] ** 3 / n_auc
    v_veg = float(np.sum(f * vols))
    omega_map = OMEGA[h["struct"]]
    if omega_bcc is not None and h["struct"] == "BCC":
        omega_map = omega_bcc
    corr = 0.0
    for i, ei in enumerate(els):
        for j, ej in enumerate(els):
            if i == j:
                continue
            omega = omega_map.get(tuple(sorted((ei, ej))), 0.0)
            corr += f[i] * f[j] * vols[j] * omega
    return v_exp - v_veg, corr


def size_mismatch_delta(h):
    els, f = norm_comp(h["comp"])
    r = np.array([KING_ATOMIC_VOLUMES.get(e, 15.0) ** (1 / 3) for e in els])
    rbar = np.sum(f * r)
    return float(100 * np.sqrt(np.sum(f * (1 - r / rbar) ** 2)))


SQS = load_sqs_data()
OMEGA_BCC_SQS = SQS["omega_dft"] if SQS else {}

ALL = ALONSO_TABLE2 + INDEPENDENT_TEST
res = {"n_HEA": len(ALL)}
data = {}
for struct in ("BCC", "FCC", "BCC_SQS"):
    if struct == "BCC_SQS":
        hs = [h for h in ALL if h["struct"] == "BCC"]
        dv = np.array([deviation_and_correction(h, OMEGA_BCC_SQS)[0] for h in hs])
        c = np.array([deviation_and_correction(h, OMEGA_BCC_SQS)[1] for h in hs])
    else:
        hs = [h for h in ALL if h["struct"] == struct]
        dv = np.array([deviation_and_correction(h)[0] for h in hs])
        c = np.array([deviation_and_correction(h)[1] for h in hs])
    delta = np.array([size_mismatch_delta(h) for h in hs])
    # main estimator: through-origin regression slope of the model V = V_Vegard + q C
    #   q_hat = sum(dv*c)/sum(c^2)
    #         = (r*sigma_dv*sigma_c + mean_dv*mean_c) / (sigma_c^2 + mean_c^2)
    # slope_ols = cov(dv,c)/var(c) = r*sigma_dv/sigma_c is the slope WITH intercept
    # (they differ because mean(C) != 0).
    q_hat = float(np.sum(dv * c) / np.sum(c * c))
    r = float(np.corrcoef(dv, c)[0, 1])
    data[struct] = (dv, c, delta, q_hat, r)
    res[struct] = dict(
        n=len(hs), q_hat=round(q_hat, 3), r=round(r, 3),
        sigma_dV=round(float(dv.std()), 4), sigma_C=round(float(c.std()), 4),
        mean_dV=round(float(dv.mean()), 4), mean_C=round(float(c.mean()), 4),
        mean_absC=round(float(np.abs(c).mean()), 4),
        mean_abs_dV=round(float(np.abs(dv).mean()), 4),
        delta_mean=round(float(delta.mean()), 3),
        slope_ols_r_times_sigma_ratio=round(float(r * dv.std() / c.std()), 3),
        q_hat_check_decomposed=round(
            float((r * dv.std() * c.std() + dv.mean() * c.mean())
                  / (c.var() + c.mean() ** 2)), 3),
    )

# q profile: RMSE(q) plateau and 95% CI of q_hat -----------------------------
for struct in ("BCC", "BCC_SQS", "FCC"):
    dv, c, _, q_hat, _ = data[struct]
    n = len(dv)
    resid = dv - q_hat * c
    se = float(np.sqrt(np.sum(resid ** 2) / (n - 1) / np.sum(c ** 2)))
    qs = np.linspace(-0.5, 3.0, 351)
    rmse_q = np.array([np.sqrt(np.mean((dv - q * c) ** 2)) for q in qs])
    rmse_min = float(rmse_q.min())
    # plateau: q range where RMSE(q) <= 1.05 * min (5% flatness)
    inside = qs[rmse_q <= 1.05 * rmse_min]
    res[struct]["q_se"] = round(se, 3)
    res[struct]["q_CI95"] = [round(q_hat - 1.96 * se, 3), round(q_hat + 1.96 * se, 3)]
    res[struct]["rmse_plateau_5pct"] = [round(float(inside.min()), 2),
                                        round(float(inside.max()), 2)]
    res[struct]["rmse_at_q1_over_min"] = round(
        float(np.sqrt(np.mean((dv - 1.0 * c) ** 2)) / rmse_min), 4)
    data[struct] = (dv, c, data[struct][2], q_hat, data[struct][4], qs, rmse_q)

fig, ax = plt.subplots(figsize=(10, 7))
for struct, color in (("BCC", "tab:red"), ("BCC_SQS", "tab:orange"), ("FCC", "tab:blue")):
    dv, c, _, q_hat, _, qs, rmse_q = data[struct]
    ax.plot(qs, rmse_q, "-", lw=2.5, color=color, label=struct)
    ax.plot([q_hat], [rmse_q.min()], "o", ms=10, color=color)
ax.axvline(1.0, color="k", ls="--", lw=1.5, label="$q$ = 1")
ax.axvline(0.0, color="gray", ls=":", lw=1.5, label="$q$ = 0 (Vegard)")
ax.set_xlabel("$q$")
ax.set_ylabel(r"RMSE$(\Delta V_\mathrm{exp} - qC)$ (Å$^3$/atom)")
ax.set_title("$q$依存のRMSEプロファイル（プラトーの可視化）")
ax.legend(fontsize=13)
plt.tight_layout()
plt.savefig(HERE / "fig_q_profile.png", dpi=150)
plt.close()

# noise-floor comparison (sigma_a = 0.016 A from duplicate measurements)
SIGMA_A_NOISE = 0.016
for struct in ("BCC", "BCC_SQS", "FCC"):
    base = "BCC" if struct.startswith("BCC") else "FCC"
    hs = [h for h in ALL if h["struct"] == base]
    n_auc = 4 if base == "FCC" else 2
    a_mean = float(np.mean([h["a_exp"] for h in hs]))
    # per-atom volume noise: dV = 3 a^2 sigma_a / n_auc
    res[struct]["sigma_V_noise"] = round(3 * a_mean ** 2 * SIGMA_A_NOISE / n_auc, 4)
    res[struct]["sigma_dV_over_noise"] = round(
        res[struct]["sigma_dV"] / res[struct]["sigma_V_noise"], 2)

with open(HERE / "results_q_decomposition.json", "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res, indent=2))

# --- figures -----------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(20, 6.2))
for ax, struct, color in zip(axes, ("BCC", "BCC_SQS", "FCC"),
                             ("tab:red", "tab:orange", "tab:blue")):
    dv, c, delta, q_hat, r = data[struct][:5]
    ax.scatter(c, dv, s=60, alpha=0.7, color=color, edgecolor="k")
    xs = np.linspace(min(c.min(), 0) * 1.1, max(c.max(), 0) * 1.1, 10)
    ax.plot(xs, q_hat * xs, "k-", lw=2, label=f"$q$ = {q_hat:.2f} (fit)")
    ax.plot(xs, xs, "k--", lw=1.5, alpha=0.6, label="$q$ = 1")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel(r"$\Omega_\mathrm{sf}$補正 $C$ (Å$^3$/atom)")
    ax.set_ylabel(r"$\Delta V_\mathrm{exp}$ = $V_\mathrm{exp}-V_\mathrm{Vegard}$ (Å$^3$/atom)")
    label = {"BCC": "BCC (B2-$\\Omega$)", "BCC_SQS": "BCC (SQS-$\\Omega$)",
             "FCC": "FCC (L1$_2$-$\\Omega$)"}[struct]
    ax.set_title(f"{label} n={len(dv)}, r={r:.2f}")
    ax.legend(fontsize=14)
plt.tight_layout()
plt.savefig(HERE / "fig_q_scatter_bcc_fcc.png", dpi=150)
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(17, 5.6))
labels = ["BCC", "BCC_SQS", "FCC"]
w = 0.35
x = np.arange(3)
axes[0].bar(x - w / 2, [res[s]["sigma_dV"] for s in labels], w, label=r"$\sigma(\Delta V_\mathrm{exp})$")
axes[0].bar(x + w / 2, [res[s]["sigma_C"] for s in labels], w, label=r"$\sigma(C)$")
for i, s in enumerate(labels):
    axes[0].axhline(res[s]["sigma_V_noise"], xmin=i / 3 + 0.05, xmax=i / 3 + 0.28,
                    color="k", ls=":", lw=2)
axes[0].text(0.02, res["BCC"]["sigma_V_noise"] * 1.05, "実験ノイズ床", fontsize=13)
axes[0].set_xticks(x, labels)
axes[0].set_ylabel(r"標準偏差 (Å$^3$/atom)")
axes[0].set_title("偏差と補正の振幅")
axes[0].legend(fontsize=13)

axes[1].bar(x, [res[s]["r"] for s in labels], 0.5,
            color=["tab:red", "tab:orange", "tab:blue"])
axes[1].set_xticks(x, labels)
axes[1].set_ylim(0, 1)
axes[1].set_ylabel(r"相関係数 $r(\Delta V_\mathrm{exp}, C)$")
axes[1].set_title("補正と真の偏差の相関")

axes[2].bar(x, [res[s]["q_hat"] for s in labels], 0.5,
            color=["tab:red", "tab:orange", "tab:blue"])
axes[2].axhline(1, color="k", ls="--", lw=1.5)
axes[2].set_xticks(x, labels)
axes[2].set_ylabel(r"$\hat q = \Sigma \Delta V C\, /\, \Sigma C^2$")
axes[2].set_title(r"$q$因子（原点通過回帰の勾配）")
plt.tight_layout()
plt.savefig(HERE / "fig_q_decomposition.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(9, 6))
for struct, color in (("BCC", "tab:red"), ("FCC", "tab:blue")):
    ax.hist(data[struct][2], bins=12, alpha=0.6, color=color,
            label=f"{struct} (平均 {data[struct][2].mean():.2f}%)")
ax.set_xlabel(r"原子サイズミスマッチ $\delta$ (%)")
ax.set_ylabel("HEA数")
ax.legend(fontsize=14)
plt.tight_layout()
plt.savefig(HERE / "fig_delta_mismatch.png", dpi=150)
plt.close()
print("DONE")
