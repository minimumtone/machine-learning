#!/usr/bin/env python3
"""Plot the Helmholtz free energy per occupied atom (G_i) for B2 defect models.

G_i = (E_i - mu_Ni * N_Ni - mu_Al * N_Al) / N_atoms - k_B * 1473 K * ln(g_i) / N_atoms

This is an analytic configurational free energy at the annealing temperature
(1473 K) and corresponds to the quantity used for branch selection in
make_figures.py."""
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

plt.rcParams.update(
    {
        "font.size": 18,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.family": ["Noto Sans CJK JP", "IPAGothic", "sans-serif"],
        "axes.unicode_minus": False,
    }
)

bm = pd.read_csv(os.path.join(AN, "b2_offstoich_branch_means.csv"))
mix = pd.read_csv(os.path.join(AN, "b2_offstoich_boltzmann_mix.csv"))

fig, ax = plt.subplots(figsize=(10, 7.5))

colors = {"antisite": "tab:blue", "vacancy": "tab:red"}
for br, color in colors.items():
    sub = bm[bm.branch == br].sort_values("x_Al_target")
    ax.plot(
        sub.x_Al_target,
        sub.G,
        "o-",
        color=color,
        ms=7,
        label="欠陥モデル：" + ("空孔" if br == "vacancy" else "反サイト"),
    )
    # Optional error bars
    if sub.Gstd.notna().any():
        ax.errorbar(
            sub.x_Al_target,
            sub.G,
            yerr=sub.Gstd,
            fmt="none",
            ecolor=color,
            alpha=0.4,
            capsize=3,
        )

# Selected (minimum-G) branch envelope
mix = mix.sort_values("x_Al_target")
ax.plot(
    mix.x_Al_target,
    mix.G_atom_eV,
    "--",
    color="black",
    lw=2,
    label="最安定 Helmholtz 自由エネルギー",
)

# Perfect B2 marker at x=0.50
perfect = mix[mix.selected_branch == "perfect"]
if not perfect.empty:
    ax.plot(
        perfect.x_Al_target.values[0],
        perfect.G_atom_eV.values[0],
        "o",
        ms=12,
        mfc="tab:blue",
        mec="tab:red",
        mew=2.5,
        zorder=8,
        label="完全 B2",
    )

ax.set_xlabel(r"$x_{\mathrm{Al}}$")
ax.set_ylabel("Helmholtz 自由エネルギー $G$ (eV/atom)")
ax.set_title("B2-Ni$_{1-x}$Al$_x$：1473 K での 1 原子あたり Helmholtz 自由エネルギー")
ax.legend(loc="best", fontsize=13)
fig.tight_layout()

out = os.path.join(FIG, "fig_b2_gibbs_energy.png")
fig.savefig(out, dpi=300, bbox_inches="tight")
print(f"Wrote {out}")
