#!/usr/bin/env python3
"""Compare SQS (ideal random fcc) vs random-seed relaxed fcc Ni(Al).

Answers the question: does a deliberately randomized/ideal fcc solution
(SQS) lie on Vegard's law, and is it lower or higher in energy than the
relaxed configurations (which may have Ni-Ni clustering or local order)?
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.join(BASE, "analysis")
FIG = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({"font.size": 18, "axes.grid": True, "grid.alpha": 0.3,
                     "font.family": ["Noto Sans CJK JP", "IPAGothic", "sans-serif"],
                     "axes.unicode_minus": False})

# load data
rand2 = pd.read_csv(os.path.join(AN, "niall_fcc_ext.csv"))
old = pd.read_csv(os.path.join(BASE, "..", "05_analysis", "volumes.csv"))
old = old[old.parent_structure == "fcc-Ni(Al)"][["x_Al", "volume_per_atom_A3", "energy_eV"]]
old = old.rename(columns={"volume_per_atom_A3": "V_per_atom_A3"})
rand2 = pd.concat([rand2, old], ignore_index=True)

sqs = pd.read_csv(os.path.join(AN, "niall_fcc_sqs.csv"))

v3_files = [f for f in [os.path.join(AN, "niall_fcc_3x3x3.csv")] if os.path.exists(f)]
rand3 = pd.concat([pd.read_csv(f) for f in v3_files], ignore_index=True) if v3_files else pd.DataFrame()

# pure-element references
pure_Ni = rand2[rand2.x_Al == 0.0].V_per_atom_A3.mean()
pure_Al = 16.7356  # corrected fcc-Al MLIP value from run_niall_ext
E_Ni = rand2[rand2.x_Al == 0.0].energy_eV.mean() / 32.0
E_Al = rand2[rand2.x_Al == 1.0].energy_eV.mean() / 32.0

def add_mix(df):
    df["E_mix_eV_atom"] = df.energy_eV / df.n_atoms - ((1 - df.x_Al) * E_Ni + df.x_Al * E_Al)
    return df

rand2 = add_mix(rand2.copy())
sqs = add_mix(sqs.copy())
if not rand3.empty:
    rand3 = add_mix(rand3.copy())

# aggregate
gr2 = rand2.groupby("x_Al").agg(V=("V_per_atom_A3", "mean"), Vstd=("V_per_atom_A3", "std"),
                                E=("E_mix_eV_atom", "mean"), Estd=("E_mix_eV_atom", "std")).reset_index()
gsqs = sqs.groupby("x_Al").agg(V=("V_per_atom_A3", "mean"), Vstd=("V_per_atom_A3", "std"),
                               E=("E_mix_eV_atom", "mean"), Estd=("E_mix_eV_atom", "std")).reset_index()
if not rand3.empty:
    gr3 = rand3.groupby("x_Al").agg(V=("V_per_atom_A3", "mean"), Vstd=("V_per_atom_A3", "std"),
                                    E=("E_mix_eV_atom", "mean"), Estd=("E_mix_eV_atom", "std")).reset_index()

fig, axes = plt.subplots(2, 1, figsize=(12, 14))
xs = np.linspace(0, 1, 200)
V_veg = pure_Ni + (pure_Al - pure_Ni) * xs
axes[0].plot(xs, V_veg, "k-", lw=2, label="Vegard則 (fcc-Ni → fcc-Al)")
axes[0].plot(xs, pure_Ni + (pure_Al - pure_Ni) * xs + 0 * xs, "k--", lw=1.5, alpha=0.7)
axes[0].errorbar(gr2.x_Al, gr2.V, yerr=gr2.Vstd, fmt="o-", ms=9, capsize=4,
                 color="tab:blue", label="2×2×2 乱数配置 (MLIP, ×3)")
if not rand3.empty:
    axes[0].errorbar(gr3.x_Al, gr3.V, yerr=gr3.Vstd, fmt="s-", ms=9, capsize=4,
                     color="tab:orange", label="3×3×3 乱数配置 (MLIP, ×3)")
axes[0].errorbar(gsqs.x_Al, gsqs.V, yerr=gsqs.Vstd, fmt="^-", ms=9, capsize=4,
                 color="tab:green", label="SQS 32原子 (icet, 理想ランダム固溶体)")
axes[0].set_xlabel(r"Al原子分率 $x_{\mathrm{Al}}$")
axes[0].set_ylabel(r"平均原子体積 $\bar V$ (Å$^3$/atom)")
axes[0].set_title(r"fcc Ni(Al)固溶体: SQS・乱数・Vegard則の比較")
axes[0].legend(fontsize=13)

axes[1].axhline(0, color="k", lw=1)
axes[1].errorbar(gr2.x_Al, gr2.E, yerr=gr2.Estd, fmt="o-", ms=9, capsize=4,
                 color="tab:blue", label="2×2×2 乱数配置")
if not rand3.empty:
    axes[1].errorbar(gr3.x_Al, gr3.E, yerr=gr3.Estd, fmt="s-", ms=9, capsize=4,
                     color="tab:orange", label="3×3×3 乱数配置")
axes[1].errorbar(gsqs.x_Al, gsqs.E, yerr=gsqs.Estd, fmt="^-", ms=9, capsize=4,
                 color="tab:green", label="SQS 32原子")
axes[1].set_xlabel(r"Al原子分率 $x_{\mathrm{Al}}$")
axes[1].set_ylabel(r"混合エネルギー $E_{\mathrm{mix}}$ (eV/atom)")
axes[1].set_title(r"SQS/乱数配置の混合エネルギー（純元素fcc基準）")
axes[1].legend(fontsize=13)

plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig_niall_sqs_stability.png"), dpi=150)
plt.close()

# text summary
summary = dict(
    pure_Ni_V=pure_Ni, pure_Al_V=pure_Al,
    n_sqs=len(sqs),
    n_rand2=len(rand2), n_rand3=len(rand3) if not rand3.empty else 0,
    sqs_vegard_deviation={f"{r.x_Al:.4f}": round(float(r.V - (pure_Ni + (pure_Al - pure_Ni) * r.x_Al)), 4) for r in gsqs.itertuples()},
    sqs_emix_min_eV=float(gsqs.E.min()),
    sqs_emix_max_eV=float(gsqs.E.max()),
)
with open(os.path.join(AN, "niall_sqs_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
