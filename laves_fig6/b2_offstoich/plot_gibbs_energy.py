#!/usr/bin/env python3
"""Plot the Helmholtz free energy per occupied atom (G_i) for B2 defect models.

G_i = (E_i - mu_Ni * N_Ni - mu_Al * N_Al) / N_atoms - k_B * 1473 K * ln(g_i) / N_atoms

This is an analytic configurational free energy at the annealing temperature
(1473 K) and corresponds to the quantity used for branch selection in
make_figures.py.  The lines shown here are smoothing splines through the
finite-supercell data points so that the finite-size discretisation kink near
x_Al ~ 0.55 is not visually exaggerated.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

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
perfect_G = float(mix[mix.selected_branch == "perfect"].G_atom_eV.iloc[0])

fig, ax = plt.subplots(figsize=(10, 7.5))

colors = {"antisite": "tab:blue", "vacancy": "tab:red"}
labels = {"antisite": "反サイト", "vacancy": "空孔"}

splines = {}
for br, color in colors.items():
    sub = bm[bm.branch == br].copy()
    # Both defect models collapse to a single perfect-B2 state at x=0.50.
    # Replace any existing stoichiometric branch point with a single exact
    # perfect-B2 point so the two PCHIP curves meet continuously there.
    sub = sub[np.abs(sub.x_Al_target - 0.5) > 1e-9].copy()
    sub = sub.sort_values("x_Al_target").drop_duplicates("x_Al_target")
    perfect_row = pd.DataFrame(
        [{"x_Al_target": 0.5, "G": perfect_G, "Gstd": 0.0}]
    )
    sub = pd.concat([sub, perfect_row], ignore_index=True).sort_values(
        "x_Al_target"
    ).reset_index(drop=True)
    x = sub.x_Al_target.values
    y = sub.G.values
    spl = PchipInterpolator(x, y)
    splines[br] = (spl, float(x.min()), float(x.max()))
    x_dense = np.linspace(x.min(), x.max(), 400)
    ax.plot(
        x_dense,
        spl(x_dense),
        "-",
        color=color,
        lw=2,
        label=f"欠陥モデル：{labels[br]}",
    )
    ax.plot(
        x,
        y,
        "o",
        color=color,
        ms=6,
        alpha=0.5,
        label="_nolegend_",
    )
    # Optional error bars (skip zero-width perfect point if it is at an edge)
    if sub.Gstd.notna().any():
        ax.errorbar(
            x,
            y,
            yerr=sub.Gstd,
            fmt="none",
            ecolor=color,
            alpha=0.4,
            capsize=3,
        )

# Selected (minimum-G) branch envelope from the interpolating curves
x_min_env = max(s[1] for s in splines.values())
x_max_env = min(s[2] for s in splines.values())
x_dense = np.linspace(x_min_env, x_max_env, 600)
y_env = np.full_like(x_dense, np.nan)
for xv in ["vacancy", "antisite"]:
    spl, xmin, xmax = splines[xv]
    mask = (x_dense >= xmin) & (x_dense <= xmax)
    vals = spl(x_dense[mask])
    idx = np.where(mask)[0]
    valid = np.isfinite(vals)
    for i, v in zip(idx[valid], vals[valid]):
        if np.isnan(y_env[i]) or v < y_env[i]:
            y_env[i] = v

ax.plot(
    x_dense,
    y_env,
    "--",
    color="black",
    lw=2,
    label="最安定 Helmholtz 自由エネルギー",
)

# Perfect B2 marker at x=0.50
ax.plot(
    0.5,
    perfect_G,
    "o",
    ms=12,
    mfc="tab:blue",
    mec="tab:red",
    mew=2.5,
    zorder=8,
    label="完全 B2",
)

ax.set_xlabel(r"$x_{\rm Al}$")
ax.set_ylabel("Helmholtz 自由エネルギー $G$ (eV/atom)")
ax.set_title("B2-Ni$_{1-x}$Al$_x$：1473 K での 1 原子あたり Helmholtz 自由エネルギー")
ax.set_xlim(0.40, 1.0)
ax.legend(loc="best", fontsize=13)
fig.tight_layout()

out = os.path.join(FIG, "fig_b2_gibbs_energy.png")
fig.savefig(out, dpi=300, bbox_inches="tight")
print(f"Wrote {out}")
