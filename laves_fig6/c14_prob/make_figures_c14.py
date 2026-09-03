#!/usr/bin/env python3
"""Post-processing for the probabilistic C14 sweep (Fig. 6(a) Laves branch).

Reads analysis/c14_prob_volumes.csv (dense random-occupation sweep) plus the
earlier 12-atom ordered/SQS results (laves_fig6/05_analysis/volumes.csv) and
compares with the digitized filled-triangle experimental points of
Yamanouchi Fig. 6(a).
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

plt.rcParams.update({"font.size": 16, "axes.grid": True, "grid.alpha": 0.3,
                     "font.family": ["Noto Sans CJK JP", "IPAGothic", "sans-serif"],
                     "axes.unicode_minus": False})

df = pd.read_csv(os.path.join(AN, "c14_prob_volumes.csv"))
tri = pd.read_csv(os.path.join(AN, "fig6a_digitized_triangles.csv"))
old = pd.read_csv(os.path.join(BASE, "..", "05_analysis", "volumes.csv"))
old_c14 = old[old.parent_structure == "C14-Nb(Ni,Al)2"]

agg = (df.groupby("x_Al")
       .agg(V=("V_per_atom_A3", "mean"), Vstd=("V_per_atom_A3", "std"),
            n=("V_per_atom_A3", "size"))
       .reset_index())

fig, ax = plt.subplots(figsize=(10.5, 7.5))
ax.errorbar(agg.x_Al, agg.V, yerr=agg.Vstd, fmt="o-", ms=8, capsize=4,
            color="tab:blue", label="C14乱数占有 2×2×1 (48原子, MLIP, ×3配置)")
g_old = (old_c14.groupby("x_Al")
         .agg(V=("volume_per_atom_A3", "mean"), Vstd=("volume_per_atom_A3", "std"))
         .reset_index())
ax.errorbar(g_old.x_Al, g_old.V, yerr=g_old.Vstd, fmt="s", ms=10, capsize=4,
            color="tab:green", alpha=0.8, label="既存 12原子 秩序/SQS (MLIP)")
ax.plot(tri.x_Al, tri.V_bar_A3, "^", ms=13, mfc="none", mec="k", mew=2,
        label="Yamanouchi実験 (Fig. 6(a) 黒三角)")
ax.set_xlabel(r"B副格子のAl分率 $x$ (Nb(Ni$_{1-x}$Al$_x$)$_2$)")
ax.set_ylabel(r"平均原子体積 $\bar V$ (Å$^3$/atom)")
ax.set_title(r"C14-Nb(Ni$_{1-x}$Al$_x$)$_2$ の確率論的組成掃引 (Fig. 6(a) Laves相データ)")
ax.legend(fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig_c14_prob_vbar.png"), dpi=150)
plt.close()

# quantitative comparison at the digitized experimental compositions
pred = np.interp(tri.x_Al, agg.x_Al, agg.V)
resid = pred - tri.V_bar_A3
comp = dict(
    n_exp=len(tri),
    RMSE_V_A3=float(np.sqrt(np.mean(resid ** 2))),
    MAPE_V_pct=float(np.mean(np.abs(resid) / tri.V_bar_A3) * 100),
    n_configs=len(df),
    n_compositions=int(agg.shape[0]),
    mean_Vstd_A3=float(agg.Vstd.mean()),
)
with open(os.path.join(AN, "c14_prob_comparison.json"), "w") as f:
    json.dump(comp, f, indent=2)
print(json.dumps(comp, indent=2))
