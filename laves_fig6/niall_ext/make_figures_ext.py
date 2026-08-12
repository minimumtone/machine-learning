#!/usr/bin/env python3
"""Figures for the extended Ni(Al) fcc sweep, vacancy representation,
BCC-Al limit, and B2 ordering-degree study.

Includes one cheap extra MLIP relaxation (bcc-Al 2x2x2) for the BCC-Al
reference lattice constant.
"""
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase.build import bulk
from ase.filters import FrechetCellFilter
from ase.optimize import LBFGS
from mace.calculators import mace_mp

BASE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.join(BASE, "analysis")
FIG = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({"font.size": 16, "axes.grid": True, "grid.alpha": 0.3,
                     "font.family": ["Noto Sans CJK JP", "IPAGothic", "sans-serif"],
                     "axes.unicode_minus": False})

B2AN = os.path.join(BASE, "..", "b2_offstoich", "analysis")

# --- bcc-Al reference ---------------------------------------------------------
bcc_al_json = os.path.join(AN, "bcc_al_ref.json")
if os.path.exists(bcc_al_json):
    with open(bcc_al_json) as f:
        bcc_al = json.load(f)
else:
    calc = mace_mp(model="medium", default_dtype="float64", device="cpu")
    at = bulk("Al", "bcc", a=3.2, cubic=True).repeat((2, 2, 2))
    at.calc = calc
    opt = LBFGS(FrechetCellFilter(at), logfile=None)
    t0 = time.time()
    opt.run(fmax=0.02, steps=500)
    v = float(at.get_volume()) / len(at)
    bcc_al = dict(V_per_atom_A3=v, a_bcc_A=(2 * v) ** (1 / 3),
                  converged=bool(opt.converged()))
    with open(bcc_al_json, "w") as f:
        json.dump(bcc_al, f, indent=2)
    print(f"bcc-Al: V={v:.4f} a={bcc_al['a_bcc_A']:.4f} ({time.time()-t0:.1f}s)")

V_NI, V_AL_FCC = 10.8133, 16.7356  # MACE fcc references

# --- Fig 1: extended fcc Ni(Al) vs Vegard -------------------------------------
ext = pd.read_csv(os.path.join(AN, "niall_fcc_ext.csv"))
old = pd.read_csv(os.path.join(BASE, "..", "05_analysis", "volumes.csv"))
old = old[old.parent_structure == "fcc-Ni(Al)"][["x_Al", "volume_per_atom_A3"]]
old = old.rename(columns={"volume_per_atom_A3": "V_per_atom_A3"})
allfcc = pd.concat([old, ext[["x_Al", "V_per_atom_A3"]]], ignore_index=True)
g = (allfcc.groupby("x_Al")
     .agg(V=("V_per_atom_A3", "mean"), Vstd=("V_per_atom_A3", "std"))
     .reset_index())

fig, ax = plt.subplots(figsize=(10.5, 7.5))
xs = np.array([0, 1])
sqs_path = os.path.join(AN, "niall_fcc_sqs.csv")
plot_sqs = os.path.exists(sqs_path)
if plot_sqs:
    sqs = pd.read_csv(sqs_path)
    gsqs = sqs.groupby("x_Al").agg(V=("V_per_atom_A3", "mean"),
                                     Vstd=("V_per_atom_A3", "std")).reset_index()

ax.plot(xs, V_NI + (V_AL_FCC - V_NI) * xs, "k--", lw=2,
        label="Vegard則 (fcc-Ni → fcc-Al)")
ax.errorbar(g.x_Al, g.V, yerr=g.Vstd, fmt="o-", ms=8, capsize=4,
            color="tab:purple", label="fcc Ni(Al) 乱数固溶体 (MLIP, ×3配置)")
if plot_sqs:
    ax.errorbar(gsqs.x_Al, gsqs.V, yerr=gsqs.Vstd, fmt="^-", ms=9,
                capsize=4, color="tab:green", alpha=0.9,
                label="SQS 32原子（理想ランダム固溶体）")
ax.set_xlabel(r"Al原子分率 $x_{\mathrm{Al}}$")
ax.set_ylabel(r"平均原子体積 $\bar V$ (Å$^3$/atom)")
ax.set_title(r"fcc Ni(Al)固溶体の全組成掃引とVegard則")
ax.legend(fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig_niall_fcc_vegard.png"), dpi=150)
plt.close()

dev = g.copy()
dev["vegard"] = V_NI + (V_AL_FCC - V_NI) * dev.x_Al
dev["dV"] = dev.V - dev.vegard

# --- Fig 2: vacancy concentration representation ------------------------------
b2 = pd.read_csv(os.path.join(B2AN, "b2_offstoich_volumes.csv"))
for f in ("b2_offstoich_volumes_extra_vac.csv", "b2_offstoich_volumes_wide_vac.csv"):
    p = os.path.join(B2AN, f)
    if os.path.exists(p):
        b2 = pd.concat([b2, pd.read_csv(p)], ignore_index=True)
vac = b2[b2.branch == "vacancy"].copy()
vac["c_vac"] = 1.0 - vac.n_atoms / vac.n_sites
vac["side"] = np.where(vac.x_Al_target < 0.5, "Ni過剰側 (Al空孔)", "Al過剰側 (Ni空孔)")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for side, color in (("Ni過剰側 (Al空孔)", "tab:blue"), ("Al過剰側 (Ni空孔)", "tab:red")):
    gg = (vac[vac.side == side].groupby("c_vac")
          .agg(a=("a_eff_A", "mean"), astd=("a_eff_A", "std"),
               V=("V_per_atom_A3", "mean"), Vstd=("V_per_atom_A3", "std"))
          .reset_index().sort_values("c_vac"))
    axes[0].errorbar(gg.c_vac * 100, gg.a, yerr=gg.astd, fmt="o-", ms=8,
                     capsize=4, color=color, label=side)
    axes[1].errorbar(gg.c_vac * 100, gg.V, yerr=gg.Vstd, fmt="o-", ms=8,
                     capsize=4, color=color, label=side)
axes[1].axhline(2 * 0 + (bcc_al["a_bcc_A"] ** 3) / 2, color="gray", ls=":",
                lw=2, label=f"bcc-Al (MLIP): {bcc_al['V_per_atom_A3']:.2f}")
axes[0].axhline(bcc_al["a_bcc_A"], color="gray", ls=":", lw=2,
                label=f"bcc-Al (MLIP): a={bcc_al['a_bcc_A']:.3f} Å")
axes[0].set_xlabel(r"空孔濃度 $c_{\mathrm{vac}}$ (サイト%)")
axes[0].set_ylabel(r"実効格子定数 $a$ (Å)")
axes[0].set_title("空孔濃度表現: 格子定数")
axes[1].set_xlabel(r"空孔濃度 $c_{\mathrm{vac}}$ (サイト%)")
axes[1].set_ylabel(r"平均原子体積 $\bar V$ (Å$^3$/atom)")
axes[1].set_title("空孔濃度表現: 平均原子体積")
for ax in axes:
    ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig_vacancy_representation.png"), dpi=150)
plt.close()

# --- Fig 3: B2 ordering degree ------------------------------------------------
op = pd.read_csv(os.path.join(AN, "b2_order_param.csv"))
go = (op.groupby("eta")
      .agg(a=("a_eff_A", "mean"), astd=("a_eff_A", "std"),
           V=("V_per_atom_A3", "mean"), Vstd=("V_per_atom_A3", "std"),
           E=("energy_eV", "mean"))
      .reset_index().sort_values("eta"))

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
axes[0].errorbar(go.eta, go.a, yerr=go.astd, fmt="o-", ms=9, capsize=4,
                 color="tab:green")
axes[0].set_xlabel(r"長距離秩序パラメータ $\eta$")
axes[0].set_ylabel(r"実効格子定数 $a$ (Å)")
axes[0].set_title(r"B2秩序度と格子定数 ($x_{\mathrm{Al}}=0.5$)")
axes[1].errorbar(go.eta, go.V, yerr=go.Vstd, fmt="o-", ms=9, capsize=4,
                 color="tab:green")
axes[1].set_xlabel(r"長距離秩序パラメータ $\eta$")
axes[1].set_ylabel(r"平均原子体積 $\bar V$ (Å$^3$/atom)")
axes[1].set_title("B2秩序度と平均原子体積")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig_b2_order_param.png"), dpi=150)
plt.close()

out = dict(
    bcc_al=bcc_al,
    fcc_vegard_deviation={f"{r.x_Al:.4f}": round(float(r.dV), 4)
                          for r in dev.itertuples()},
    order_param={f"{r.eta:.2f}": dict(a=round(float(r.a), 4),
                                      V=round(float(r.V), 4),
                                      dE_eV_atom=round(float((r.E - go.E.iloc[-1]) / 128), 4))
                 for r in go.itertuples()},
)
with open(os.path.join(AN, "niall_ext_summary.json"), "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
