#!/usr/bin/env python3
"""Post-processing for the B2 off-stoichiometry pipeline (no re-relaxation).

Reads analysis/b2_offstoich_volumes.csv (+ extra CSVs) and the digitized
Fig. 6(a) experimental points, then:

  1. averages over seeds for each composition / defect branch,
  2. computes semi-grand canonical (fixed lattice-site) free energies with
     analytic configurational entropy,
  3. evaluates the branch weight at the annealing temperature,
  4. generates volume-, lattice-constant-, and energy-composition figures.

All energies are written per **occupied atom** as before, but the Boltzmann
weight is evaluated using the total semi-grand potential for the 128-site B2
supercell:

    Ω_i = E_i - μ_Ni N_Ni - μ_Al N_Al - k_B T ln g_i

where g_i is the number of ways to place the point defect on its sublattice.
For a B2 4×4×4 supercell with 64 Ni-sites and 64 Al-sites:

    antisite branch : g = C(64, n_antisite)
    vacancy branch  : g = C(64, n_vacancies)

The lattice constant for the B2 branch is a = (V / 64)^(1/3).  The digitized
experimental average atomic volume is converted to an experimental lattice
constant assuming the triple-defect picture:

    x_Al ≤ 0.5 (Ni-antisite, no vacancies): a_exp = (2 * V_bar)^(1/3)
    x_Al ≥ 0.5 (Ni-vacancy):                a_exp = (V_bar / x_Al)^(1/3)
"""
import json
import math
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

# --- load data ---------------------------------------------------------------
df = pd.read_csv(os.path.join(AN, "b2_offstoich_volumes.csv"))
for extra in ("b2_offstoich_volumes_extra_vac.csv", "b2_offstoich_volumes_wide_vac.csv",
              "b2_offstoich_volumes_antisite_extra.csv",
              "b2_offstoich_volumes_50competition.csv",
              "b2_offstoich_volumes_antisite_alrich_dense.csv"):
    p = os.path.join(AN, extra)
    if os.path.exists(p):
        df = pd.concat([df, pd.read_csv(p)], ignore_index=True)

exp = pd.read_csv(os.path.join(AN, "fig6a_digitized_circles.csv"))
exp_b2 = exp[exp.region == "B2"].copy()
exp_ss = exp[exp.region != "B2"].copy()
vols_all = pd.read_csv(os.path.join(BASE, "..", "05_analysis", "volumes.csv"))
niall_ss = vols_all[vols_all.parent_structure == "fcc-Ni(Al)"]

T_ANNEAL_K = 1473.0  # Yamanouchi 2018: 1473 K / 168 h, water quench
KT_EV = 8.617333262e-5 * T_ANNEAL_K

with open(os.path.join(AN, "b2_offstoich_summary.json")) as f:
    summary = json.load(f)
MU_NI = summary["mu_Ni_eV"]
MU_AL = summary["mu_Al_eV"]
NCELL = 64           # 4×4×4 B2 supercell
N_SITES = 2 * NCELL  # 128 lattice sites

perfect = df[df.branch == "perfect"].iloc[0]

# Drop unrelaxed structures, keeping the perfect reference regardless.
df = df[df.converged | (df.branch == "perfect")].copy()

# --- semi-grand potential + configurational entropy --------------------------
def ln_comb(n, k):
    if k < 0 or k > n or n < 0:
        return -np.inf
    if k == 0 or k == n:
        return 0.0
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def n_defect(row):
    if row.branch == "antisite":
        if row.x_Al_target < 0.5:
            return int(row.n_Ni - NCELL)       # Ni on Al sublattice
        else:
            return int(row.n_Al - NCELL)       # Al on Ni sublattice
    elif row.branch == "vacancy":
        if row.x_Al_target < 0.5:
            return int(NCELL - row.n_Al)       # vacancies on Al sublattice
        else:
            return int(NCELL - row.n_Ni)       # vacancies on Ni sublattice
    return 0


def omega_per_atom(row):
    """
    Helmholtz free energy per occupied atom in semi-grand form.

    E_f = (E - mu_Ni N_Ni - mu_Al N_Al) / N_atoms  is the formation
    energy per atom relative to fcc elements.  The configurational
    entropy per atom is -k_B T ln(g) / N_atoms.  The branch with the
    lower value is thermodynamically preferred for a given composition
    at fixed temperature and pressure.
    """
    e = row.energy_eV
    g = n_defect(row)
    n = row.n_atoms
    return (e - MU_NI * row.n_Ni - MU_AL * row.n_Al) / n - KT_EV * ln_comb(NCELL, g) / n


df["n_defect"] = df.apply(n_defect, axis=1)
df["Omega_per_atom_eV"] = df.apply(omega_per_atom, axis=1)

# aggregate per composition/branch
agg_rows = []
for (x, br), g in df[df.branch != "perfect"].groupby(["x_Al_target", "branch"]):
    agg_rows.append({
        "x_Al_target": x, "branch": br,
        "x_Al": g.x_Al.mean(), "V": g.V_per_atom_A3.mean(),
        "Vstd": g.V_per_atom_A3.std(ddof=1) if len(g) > 1 else 0.0,
        "a": g.a_eff_A.mean(), "astd": g.a_eff_A.std(ddof=1) if len(g) > 1 else 0.0,
        "Ef": g.E_form_eV_atom.mean(),
        "Omega": g.Omega_per_atom_eV.mean(),
        "Omega_std": g.Omega_per_atom_eV.std(ddof=1) if len(g) > 1 else 0.0,
        "n_atoms": int(round(g.n_atoms.mean())),
        "n": len(g),
    })
agg = pd.DataFrame(agg_rows)

# branch mixture from per-atom semi-grand free energies
mix_rows = []
for x, g in agg.groupby("x_Al_target"):
    g = g.set_index("branch")
    avail = set(g.index)
    # If both branches available, pick the lower free-energy per atom branch.
    # A smooth crossover over ~k_B T / atom is retained by using the
    # Boltzmann factor for a representative sample size (128 atoms).
    dOmega = None
    if {"antisite", "vacancy"} <= avail:
        dOmega = g.loc["vacancy", "Omega"] - g.loc["antisite", "Omega"]
        N_bar = 0.5 * (g.loc["vacancy", "n_atoms"] + g.loc["antisite", "n_atoms"])
        w_vac = 1.0 / (1.0 + np.exp(dOmega * N_bar / KT_EV))
    elif "vacancy" in avail:
        w_vac = 1.0
    else:
        w_vac = 0.0
    if "vacancy" in avail and "antisite" in avail:
        x_Al_val = w_vac * g.loc["vacancy", "x_Al"] + (1 - w_vac) * g.loc["antisite", "x_Al"]
        V_val = w_vac * g.loc["vacancy", "V"] + (1 - w_vac) * g.loc["antisite", "V"]
        a_val = w_vac * g.loc["vacancy", "a"] + (1 - w_vac) * g.loc["antisite", "a"]
    elif "vacancy" in avail:
        x_Al_val = g.loc["vacancy", "x_Al"]
        V_val = g.loc["vacancy", "V"]
        a_val = g.loc["vacancy", "a"]
    else:
        x_Al_val = g.loc["antisite", "x_Al"]
        V_val = g.loc["antisite", "V"]
        a_val = g.loc["antisite", "a"]
    mix_rows.append(dict(
        x_Al_target=x,
        x_Al=x_Al_val,
        V_mix=V_val,
        a_mix=a_val,
        w_vac=w_vac,
        preferred="vacancy" if w_vac > 0.5 else "antisite",
        dOmega_eV=dOmega,
    ))
mix = pd.DataFrame(mix_rows).sort_values("x_Al_target")

# experimental lattice constant (triple-defect conversion)
def exp_a(row):
    if row.x_Al <= 0.5:
        return (2.0 * row.V_bar_A3) ** (1.0 / 3.0)
    else:
        return (row.V_bar_A3 / row.x_Al) ** (1.0 / 3.0)


exp_b2["a_exp_A"] = exp_b2.apply(exp_a, axis=1)

# --- Fig: V-bar vs x_Al (B2 branch only) ------------------------------------
fig, ax = plt.subplots(figsize=(10, 7.5))
colors = {"antisite": "tab:blue", "vacancy": "tab:red"}
labels = {"antisite": "反サイト（antisite）配置 (MLIP)",
          "vacancy": "空孔（vacancy）配置 (MLIP)"}
for br, g in agg.groupby("branch"):
    g = g.sort_values("x_Al")
    ax.errorbar(g.x_Al, g.V, yerr=g.Vstd, fmt="o-", ms=9, capsize=4,
                color=colors[br], label=labels[br])
ax.plot(mix.x_Al, mix.V_mix, "k--", lw=2.5,
        label=f"半巨視正準 Boltzmann 混合 ({T_ANNEAL_K:.0f} K, 配置エントロピー込み)")
ax.plot([0.5], [perfect.V_per_atom_A3], "s", ms=13, color="tab:green",
        label="完全B2 (MLIP)")
ax.plot(exp_b2.x_Al, exp_b2.V_bar_A3, "o", ms=11, mfc="none", mec="k", mew=2,
        label="Yamanouchi実験 B2枝 (Fig. 6(a)), 室温")
ax.set_xlabel(r"Al原子分率 $x_{\mathrm{Al}}$")
ax.set_ylabel(r"平均原子体積 $\bar V$ (Å$^3$/atom)")
ax.set_title(r"B2-Ni$_{1-x}$Al$_x$ 不定比組成の平均原子体積")
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig_b2_offstoich_vbar.png"), dpi=150)
plt.close()

# --- Fig: full Fig.6(a) ------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 8))
g_ss = (niall_ss.groupby("x_Al")
        .agg(V=("volume_per_atom_A3", "mean"), Vstd=("volume_per_atom_A3", "std"))
        .reset_index())
ax.errorbar(g_ss.x_Al, g_ss.V, yerr=g_ss.Vstd, fmt="D-", ms=9, capsize=4,
            color="tab:purple", label="Ni(Al)固溶体 fcc-SQS (MLIP)")
for br, g in agg.groupby("branch"):
    g = g.sort_values("x_Al")
    ax.errorbar(g.x_Al, g.V, yerr=g.Vstd, fmt="o-", ms=8, capsize=4,
                color=colors[br], label=labels[br])
ax.plot(mix.x_Al, mix.V_mix, "k--", lw=2.5,
        label=f"半巨視正準 Boltzmann 混合 ({T_ANNEAL_K:.0f} K)")
ax.plot([0.5], [perfect.V_per_atom_A3], "s", ms=12, color="tab:green",
        label="完全B2 (MLIP)")
ax.plot(exp_b2.x_Al, exp_b2.V_bar_A3, "o", ms=11, mfc="none", mec="k", mew=2,
        label="Yamanouchi実験 B2枝")
ax.plot(exp_ss.x_Al, exp_ss.V_bar_A3, "^", ms=11, mfc="none", mec="gray", mew=2,
        label="Yamanouchi実験 Ni(Al)固溶体領域")
ax.set_xlabel(r"Al原子分率 $x_{\mathrm{Al}}$")
ax.set_ylabel(r"平均原子体積 $\bar V$ (Å$^3$/atom)")
ax.set_title(r"Fig. 6(a) 全域: Ni(Al)固溶体とB2不定比枝のMLIP再現")
ax.legend(fontsize=11, loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig_b2_offstoich_vbar_full.png"), dpi=150)
plt.close()

# --- Fig: lattice constant a vs x_Al (anomaly of Ni vacancies) ---------------
fig, ax = plt.subplots(figsize=(10, 7))
for br, g in agg.groupby("branch"):
    g = g.sort_values("x_Al")
    ax.errorbar(g.x_Al, g.a, yerr=g.astd, fmt="o-", ms=9, capsize=4,
                color=colors[br], label=labels[br])
ax.plot(mix.x_Al, mix.a_mix, "k--", lw=2.5,
        label=f"半巨視正準 Boltzmann 混合 ({T_ANNEAL_K:.0f} K)")
ax.plot([0.5], [perfect.a_eff_A], "s", ms=13, color="tab:green", label="完全B2")
ax.plot(exp_b2.x_Al, exp_b2.a_exp_A, "o", ms=11, mfc="none", mec="k", mew=2,
        label="Yamanouchi実験 B2枝 → 格子定数")
ax.set_xlabel(r"Al原子分率 $x_{\mathrm{Al}}$")
ax.set_ylabel(r"B2格子定数 $a$ (Å)")
ax.set_title(r"B2-Ni$_{1-x}$Al$_x$ の格子定数（構造空孔による異常挙動）")
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig_b2_offstoich_a.png"), dpi=150)
plt.close()

# --- Fig: formation energies -------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 7))
for br, g in agg.groupby("branch"):
    g = g.sort_values("x_Al")
    ax.plot(g.x_Al, g.Ef, "o-", ms=9, color=colors[br], label=labels[br])
ax.axhline(0, color="k", lw=1)
ax.set_xlabel(r"Al原子分率 $x_{\mathrm{Al}}$")
ax.set_ylabel(r"生成エネルギー $E_f$ (eV/atom)")
ax.set_title(r"欠陥様式ごとの生成エネルギー（純元素fcc基準、1原子あたり）")
ax.legend(fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig_b2_offstoich_eform.png"), dpi=150)
plt.close()

# --- quantitative comparison -------------------------------------------------
# Compare against the B2 homogeneous range only (0.45 <= x_Al <= 0.60).
# Points beyond 0.60 correspond to intermetallic compounds (Ni2Al3/NiAl3)
# rather than the B2 single-phase branch.
X_B2_MIN = 0.45
X_B2_MAX = 0.60
xs = np.concatenate([mix.x_Al.values, [0.5]])
vs = np.concatenate([mix.V_mix.values, [perfect.V_per_atom_A3]])
order = np.argsort(xs)
xs, vs = xs[order], vs[order]
ve = exp_b2[(exp_b2.x_Al >= X_B2_MIN) & (exp_b2.x_Al <= X_B2_MAX)].copy()
pred = np.interp(ve.x_Al, xs, vs)
resid = pred - ve.V_bar_A3.values
rmse = float(np.sqrt(np.mean(resid ** 2)))
mape = float(np.mean(np.abs(resid) / ve.V_bar_A3.values) * 100)

out = dict(
    n_exp_points_compared=int(len(ve)),
    RMSE_V_A3=round(rmse, 4),
    MAPE_V_pct=round(mape, 3),
    V_B2_perfect=float(perfect.V_per_atom_A3),
    a_B2_perfect=float(perfect.a_eff_A),
    T_boltzmann_K=T_ANNEAL_K,
    weight_method="Helmholtz free energy per occupied atom (semi-grand canonical with analytic configurational entropy)",
    branch_preference={f"{r.x_Al_target:.2f}": dict(preferred=r.preferred,
                                                    w_vacancy_annealT=round(float(r.w_vac), 4),
                                                    dOmega_eV=round(float(r.dOmega_eV), 4) if not pd.isna(r.dOmega_eV) else None)
                       for r in mix.itertuples()},
)
agg.to_csv(os.path.join(AN, "b2_offstoich_branch_means.csv"), index=False)
mix.to_csv(os.path.join(AN, "b2_offstoich_boltzmann_mix.csv"), index=False)
with open(os.path.join(AN, "b2_offstoich_comparison.json"), "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
print("DONE")
