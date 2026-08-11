#!/usr/bin/env python3
"""Post-processing for the B2 off-stoichiometry pipeline (no re-relaxation).

Reads analysis/b2_offstoich_volumes.csv + the digitized Fig. 6(a) experimental
open circles and generates figures / a quantitative comparison summary.
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

df = pd.read_csv(os.path.join(AN, "b2_offstoich_volumes.csv"))
exp = pd.read_csv(os.path.join(AN, "fig6a_digitized_circles.csv"))
exp_b2 = exp[exp.region == "B2"]
KT_EV = 8.617333262e-5 * 1273.0

perfect = df[df.branch == "perfect"].iloc[0]

# aggregate per composition/branch
agg = (df[df.branch != "perfect"]
       .groupby(["x_Al_target", "branch"])
       .agg(x_Al=("x_Al", "mean"), V=("V_per_atom_A3", "mean"),
            Vstd=("V_per_atom_A3", "std"), a=("a_eff_A", "mean"),
            astd=("a_eff_A", "std"), Ef=("E_form_eV_atom", "mean"),
            n=("V_per_atom_A3", "size"))
       .reset_index())

# Boltzmann-weighted branch mixture (128-atom cells, 1273 K)
mix_rows = []
for x, g in agg.groupby("x_Al_target"):
    g = g.set_index("branch")
    if {"antisite", "vacancy"} <= set(g.index):
        dEf = g.loc["vacancy", "Ef"] - g.loc["antisite", "Ef"]
        w_vac = 1.0 / (1.0 + np.exp(dEf * 128 / KT_EV))
        mix_rows.append(dict(
            x_Al_target=x,
            x_Al=w_vac * g.loc["vacancy", "x_Al"] + (1 - w_vac) * g.loc["antisite", "x_Al"],
            V_mix=w_vac * g.loc["vacancy", "V"] + (1 - w_vac) * g.loc["antisite", "V"],
            a_mix=w_vac * g.loc["vacancy", "a"] + (1 - w_vac) * g.loc["antisite", "a"],
            w_vac=w_vac,
            preferred="vacancy" if dEf < 0 else "antisite",
            dEf_eV_atom=dEf,
        ))
mix = pd.DataFrame(mix_rows).sort_values("x_Al_target")

# --- Fig: V-bar vs x_Al ------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 7.5))
colors = {"antisite": "tab:blue", "vacancy": "tab:red"}
labels = {"antisite": "逆位相（antisite）配置", "vacancy": "空孔（vacancy）配置"}
for br, g in agg.groupby("branch"):
    g = g.sort_values("x_Al")
    ax.errorbar(g.x_Al, g.V, yerr=g.Vstd, fmt="o-", ms=9, capsize=4,
                color=colors[br], label=labels[br] + " (MLIP, SQS的乱数配置×3)")
ax.plot(mix.x_Al, mix.V_mix, "k--", lw=2, label="Boltzmann混合 (1273 K)")
ax.plot([0.5], [perfect.V_per_atom_A3], "s", ms=13, color="tab:green",
        label="完全B2 (MLIP)")
ax.plot(exp_b2.x_Al, exp_b2.V_bar_A3, "o", ms=11, mfc="none", mec="k", mew=2,
        label="実験 (Fig. 6(a) デジタイズ)")
ax.set_xlabel(r"Al原子分率 $x_{\mathrm{Al}}$")
ax.set_ylabel(r"平均原子体積 $\bar V$ (Å$^3$/atom)")
ax.set_title(r"B2-Ni$_{1-x}$Al$_x$ 不定比組成の平均原子体積")
ax.legend(fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig_b2_offstoich_vbar.png"), dpi=150)
plt.close()

# --- Fig: lattice constant ---------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 7))
for br, g in agg.groupby("branch"):
    g = g.sort_values("x_Al")
    ax.errorbar(g.x_Al, g.a, yerr=g.astd, fmt="o-", ms=9, capsize=4,
                color=colors[br], label=labels[br])
ax.plot(mix.x_Al, mix.a_mix, "k--", lw=2, label="Boltzmann混合 (1273 K)")
ax.plot([0.5], [perfect.a_eff_A], "s", ms=13, color="tab:green", label="完全B2")
ax.set_xlabel(r"Al原子分率 $x_{\mathrm{Al}}$")
ax.set_ylabel(r"実効格子定数 $a$ (Å)")
ax.set_title(r"B2-Ni$_{1-x}$Al$_x$ の格子定数（B2単位胞換算）")
ax.legend(fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig_b2_offstoich_a.png"), dpi=150)
plt.close()

# --- Fig: formation energies -------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 7))
for br, g in agg.groupby("branch"):
    g = g.sort_values("x_Al")
    ax.plot(g.x_Al, g.Ef, "o-", ms=9, color=colors[br], label=labels[br])
ax.set_xlabel(r"Al原子分率 $x_{\mathrm{Al}}$")
ax.set_ylabel(r"生成エネルギー $E_f$ (eV/atom)")
ax.set_title("欠陥様式ごとの生成エネルギー（純元素基準）")
ax.legend(fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig_b2_offstoich_eform.png"), dpi=150)
plt.close()

# --- quantitative comparison vs digitized experiment -------------------------
# interpolate preferred-branch (Boltzmann) model at experimental x, plus perfect B2 at 0.5
xs = np.concatenate([mix.x_Al.values, [0.5]])
vs = np.concatenate([mix.V_mix.values, [perfect.V_per_atom_A3]])
order = np.argsort(xs)
xs, vs = xs[order], vs[order]
mask = (exp_b2.x_Al >= xs.min()) & (exp_b2.x_Al <= xs.max())
ve = exp_b2[mask]
pred = np.interp(ve.x_Al, xs, vs)
resid = pred - ve.V_bar_A3.values
rmse = float(np.sqrt(np.mean(resid ** 2)))
mape = float(np.mean(np.abs(resid) / ve.V_bar_A3.values) * 100)

out = dict(
    n_exp_points_compared=int(mask.sum()),
    RMSE_V_A3=round(rmse, 4),
    MAPE_V_pct=round(mape, 3),
    V_B2_perfect=float(perfect.V_per_atom_A3),
    branch_preference={f"{r.x_Al_target:.2f}": dict(preferred=r.preferred,
                                                    w_vacancy_1273K=round(float(r.w_vac), 4),
                                                    dEf_eV_atom=round(float(r.dEf_eV_atom), 4))
                       for r in mix.itertuples()},
)
agg.to_csv(os.path.join(AN, "b2_offstoich_branch_means.csv"), index=False)
mix.to_csv(os.path.join(AN, "b2_offstoich_boltzmann_mix.csv"), index=False)
with open(os.path.join(AN, "b2_offstoich_comparison.json"), "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
print("DONE")
