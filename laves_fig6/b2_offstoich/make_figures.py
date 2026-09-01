#!/usr/bin/env python3
"""Post-processing for the B2 off-stoichiometry pipeline (no re-relaxation).

Reads analysis/b2_offstoich_volumes.csv (+ extra CSVs) and the digitized
Fig. 6(a) experimental points, then:

  1. averages over seeds for each composition / defect branch,
  2. computes the total semi-grand potential Ω_i and the Helmholtz free energy
     per occupied atom G_i with analytic configurational entropy,
  3. selects the defect branch with the lower Helmholtz free energy per
     occupied atom G_i at each target composition (fixed lattice composition),
  4. generates volume-, lattice-constant-, and energy-composition figures.

The total semi-grand potential relative to the fcc elements is

    Ω_i = E_i - μ_Ni N_Ni - μ_Al N_Al - k_B T ln g_i

where g_i is the number of ways to place the point defect on its sublattice.
The Helmholtz free energy per occupied atom is

    G_i = Ω_i / N_atom
        = (E_i - μ_Ni N_Ni - μ_Al N_Al) / N_atom - k_B T ln g_i / N_atom

Because vacancy and antisite branches have different numbers of occupied atoms,
Ω_i = N_atom G_i does **not** imply that the lower-G_i branch also has the lower
Ω_i.  At a fixed target lattice composition (fixed x_Al) the relevant potential
for the metastable B2 single phase is the Helmholtz free energy per occupied
atom, G_i = Ω_i / N_atom: it tells which branch is more stable if the alloy is
forced to remain at that composition.  The code therefore selects the branch
with the lower G_i.  A blind total-Ω comparison at fixed x would predict the
Al-rich antisite branch to be lower in Ω at x_Al≈0.60, contradicting the
experimentally observed Ni-vacancy branch and the 1273 K Boltzmann weighting; it
is reported for reference but not used for branch selection.

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


def mark_perfect_b2(ax, x, y, label="完全B2 (MLIP)"):
    """Normal marker at the perfect-B2 composition.

    Uses a single matplotlib marker with a blue face and a red edge to show
    that the vacancy (red) and antisite/MLIP (blue) curves meet the perfect
    B2 state at x_Al=0.50."""
    ax.plot([x], [y], "o", ms=12, mfc="tab:blue", mec="tab:red", mew=2.5,
            zorder=8, label=label)

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

# Pure fcc/bcc end-member references for Vegard comparison
ref = pd.read_csv(os.path.join(AN, "mace_mp_ref_results.csv"))
a2_ref = pd.read_csv(os.path.join(AN, "a2_endmember_energies.csv"))

# fcc (MACE references are single-atom equivalents)
V_fcc_Ni = float(ref[ref.label == "Ni"].volume_per_atom_A3.values[0])
V_fcc_Al = float(ref[ref.label == "Al"].volume_per_atom_A3.values[0])
a_fcc_Ni = (4.0 * V_fcc_Ni) ** (1.0 / 3.0)
a_fcc_Al = (4.0 * V_fcc_Al) ** (1.0 / 3.0)

# bcc A2 end-members
ni_bcc = a2_ref[a2_ref.element == "Ni"].iloc[0]
al_bcc = a2_ref[a2_ref.element == "Al"].iloc[0]
V_bcc_Ni = float(ni_bcc.V_atom_A3)
V_bcc_Al = float(al_bcc.V_atom_A3)
a_bcc_Ni = float(ni_bcc.a_A)
a_bcc_Al = float(al_bcc.a_A)

_x_end = np.array([0.0, 1.0])
V_fcc_veg = (1.0 - _x_end) * V_fcc_Ni + _x_end * V_fcc_Al
V_bcc_veg = (1.0 - _x_end) * V_bcc_Ni + _x_end * V_bcc_Al
a_fcc_veg = (1.0 - _x_end) * a_fcc_Ni + _x_end * a_fcc_Al
a_bcc_veg = (1.0 - _x_end) * a_bcc_Ni + _x_end * a_bcc_Al

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


def G_atom(row):
    """
    Helmholtz free energy per occupied atom in semi-grand form.

    This is the formation energy per atom relative to fcc elements,
    E_f = (E - mu_Ni N_Ni - mu_Al N_Al) / N_atoms, plus the analytic
    configurational entropy -k_B T ln(g) / N_atoms per atom.  The branch
    with the lower value is thermodynamically preferred for a given
    composition at fixed temperature and pressure.
    """
    e = row.energy_eV
    g = n_defect(row)
    n = row.n_atoms
    return (e - MU_NI * row.n_Ni - MU_AL * row.n_Al) / n - KT_EV * ln_comb(NCELL, g) / n


df["n_defect"] = df.apply(n_defect, axis=1)
df["G_atom_eV"] = df.apply(G_atom, axis=1)
df["Omega_total_eV"] = df["G_atom_eV"] * df["n_atoms"]

# aggregate per composition/branch
agg_rows = []
for (x, br), g in df[df.branch != "perfect"].groupby(["x_Al_target", "branch"]):
    agg_rows.append({
        "x_Al_target": x, "branch": br,
        "x_Al": g.x_Al.mean(), "V": g.V_per_atom_A3.mean(),
        "Vstd": g.V_per_atom_A3.std(ddof=1) if len(g) > 1 else 0.0,
        "a": g.a_eff_A.mean(), "astd": g.a_eff_A.std(ddof=1) if len(g) > 1 else 0.0,
        "Ef": g.E_form_eV_atom.mean(),
        "G": g.G_atom_eV.mean(),
        "Gstd": g.G_atom_eV.std(ddof=1) if len(g) > 1 else 0.0,
        "Omega_total_eV": g.Omega_total_eV.mean(),
        "n_atoms": int(round(g.n_atoms.mean())),
        "n": len(g),
    })
agg = pd.DataFrame(agg_rows)

# Select the lower Helmholtz free-energy branch (per-atom G_i) at each target
# composition (fixed x_Al).  For a physical B2 single phase the expected defect is:
#   x_Al < 0.5 -> Ni antisites,   x_Al > 0.5 -> Ni vacancies.
# A target is only admitted if the physically expected branch was sampled; if
# both branches are present, the lower G branch is used.  The perfect
# B2 point at x = 0.5 is always appended.
mix_rows = []
for x, g in agg.groupby("x_Al_target"):
    g = g.set_index("branch")
    avail = set(g.index)
    if x < 0.5 - 1e-6:
        if "antisite" not in avail:
            continue
        if "vacancy" in avail:
            dG = g.loc["vacancy", "G"] - g.loc["antisite", "G"]
            selected = "antisite" if dG > 0.0 else "vacancy"
        else:
            dG = np.nan
            selected = "antisite"
        r = g.loc[selected]
        mix_rows.append(dict(
            x_Al_target=x,
            x_Al=r.x_Al,
            V_mix=r.V,
            a_mix=r.a,
            selected_branch=selected,
            G_atom_eV=r.G,
            Omega_total_eV=r.Omega_total_eV,
            dG_eV=dG,
        ))
    elif x > 0.5 + 1e-6:
        if "vacancy" not in avail:
            continue
        if "antisite" in avail:
            dG = g.loc["vacancy", "G"] - g.loc["antisite", "G"]
            selected = "vacancy" if dG < 0.0 else "antisite"
        else:
            dG = np.nan
            selected = "vacancy"
        r = g.loc[selected]
        mix_rows.append(dict(
            x_Al_target=x,
            x_Al=r.x_Al,
            V_mix=r.V,
            a_mix=r.a,
            selected_branch=selected,
            G_atom_eV=r.G,
            Omega_total_eV=r.Omega_total_eV,
            dG_eV=dG,
        ))
# perfect B2 at x = 0.5
mix_rows.append(dict(
    x_Al_target=perfect.x_Al_target,
    x_Al=perfect.x_Al,
    V_mix=perfect.V_per_atom_A3,
    a_mix=perfect.a_eff_A,
    selected_branch="perfect",
    G_atom_eV=(perfect.energy_eV - MU_NI * perfect.n_Ni - MU_AL * perfect.n_Al) / perfect.n_atoms,
    Omega_total_eV=perfect.energy_eV - MU_NI * perfect.n_Ni - MU_AL * perfect.n_Al,
    dG_eV=0.0,
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
        label=f"$G$ 最安定欠陥モデル選択 ({T_ANNEAL_K:.0f} K, 配置エントロピー込み)")
mark_perfect_b2(ax, 0.5, perfect.V_per_atom_A3, label="完全B2 (MLIP)")
ax.plot(exp_b2.x_Al, exp_b2.V_bar_A3, "o", ms=11, mfc="none", mec="k", mew=2,
        label="Yamanouchi実験 B2 (Fig. 6(a)), 室温")
ax.plot(_x_end, V_fcc_veg, "--", color="tab:green", lw=1.8,
        label="fcc Vegard (Ni–Al)")
ax.plot(_x_end, V_bcc_veg, "--", color="tab:gray", lw=1.8,
        label="bcc A2 Vegard")
ax.plot([0.0, 1.0], [V_fcc_Ni, V_fcc_Al], "o", ms=10, color="tab:green",
        mfc="none", mew=2.0, label="純 fcc Ni/Al (MACE)")
ax.plot([0.0, 1.0], [V_bcc_Ni, V_bcc_Al], "s", ms=10, color="tab:gray",
        mfc="none", mew=2.0, label="純 bcc A2 Ni/Al (MACE)")
ax.set_xlabel(r"$x_{\mathrm{Al}}$")
ax.set_ylabel(r"平均原子体積 $\bar V$ (Å$^3$/atom)")
ax.set_title(r"B2-Ni$_{1-x}$Al$_x$ 平均原子体積と純金属・Vegard 線")
ax.set_xlim(-0.05, 1.05)
ax.legend(fontsize=11)
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
        label=f"$G$ 最安定欠陥モデル選択 ({T_ANNEAL_K:.0f} K)")
mark_perfect_b2(ax, 0.5, perfect.V_per_atom_A3, label="完全B2 (MLIP)")
ax.plot(exp_b2.x_Al, exp_b2.V_bar_A3, "o", ms=11, mfc="none", mec="k", mew=2,
        label="Yamanouchi実験 B2")
ax.plot(exp_ss.x_Al, exp_ss.V_bar_A3, "^", ms=11, mfc="none", mec="gray", mew=2,
        label="Yamanouchi実験 Ni(Al)固溶体領域")
ax.plot(_x_end, V_fcc_veg, "--", color="tab:green", lw=1.8,
        label="fcc Vegard (Ni–Al)")
ax.plot(_x_end, V_bcc_veg, "--", color="tab:gray", lw=1.8,
        label="bcc A2 Vegard")
ax.plot([0.0, 1.0], [V_fcc_Ni, V_fcc_Al], "o", ms=10, color="tab:green",
        mfc="none", mew=2.0, label="純 fcc Ni/Al (MACE)")
ax.plot([0.0, 1.0], [V_bcc_Ni, V_bcc_Al], "s", ms=10, color="tab:gray",
        mfc="none", mew=2.0, label="純 bcc A2 Ni/Al (MACE)")
ax.set_xlabel(r"$x_{\mathrm{Al}}$")
ax.set_ylabel(r"平均原子体積 $\bar V$ (Å$^3$/atom)")
ax.set_title(r"Fig. 6(a) 全域: Ni(Al)固溶体・B2不定比・純金属/Vegard 線")
ax.set_xlim(-0.05, 1.05)
ax.legend(fontsize=10, loc="upper left")
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
        label=f"$G$ 最安定欠陥モデル選択 ({T_ANNEAL_K:.0f} K)")
mark_perfect_b2(ax, 0.5, perfect.a_eff_A, label="完全B2 (MLIP)")
ax.plot(exp_b2.x_Al, exp_b2.a_exp_A, "o", ms=11, mfc="none", mec="k", mew=2,
        label="Yamanouchi実験 B2 → 格子定数")
ax.plot(_x_end, a_bcc_veg, "--", color="tab:gray", lw=1.8,
        label="bcc A2 Vegard")
ax.plot([0.0, 1.0], [a_bcc_Ni, a_bcc_Al], "s", ms=10, color="tab:gray",
        mfc="none", mew=2.0, label="純 bcc A2 Ni/Al (MACE)")
ax.plot(_x_end, a_fcc_veg, "--", color="tab:green", lw=1.8,
        label="fcc Vegard (参考)")
ax.plot([0.0, 1.0], [a_fcc_Ni, a_fcc_Al], "o", ms=10, color="tab:green",
        mfc="none", mew=2.0, label="純 fcc Ni/Al (MACE)")
ax.set_xlabel(r"$x_{\mathrm{Al}}$")
ax.set_ylabel(r"格子定数 $a$ (Å)")
ax.set_title(r"B2-Ni$_{1-x}$Al$_x$ 格子定数と純金属・Vegard 線")
ax.set_xlim(-0.05, 1.05)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig_b2_offstoich_a.png"), dpi=150)
plt.close()

# --- Fig: formation energies -------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 7))
for br, g in agg.groupby("branch"):
    g = g.sort_values("x_Al")
    ax.plot(g.x_Al, g.Ef, "o-", ms=9, color=colors[br], label=labels[br])
ax.axhline(0, color="k", lw=1)
mark_perfect_b2(ax, 0.5, perfect.E_form_eV_atom, label="完全B2")
ax.set_xlabel(r"$x_{\mathrm{Al}}$")
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
# perfect is already included in mix; avoid duplicate
xs = mix.x_Al.values
vs = mix.V_mix.values
order = np.argsort(xs)
xs, vs = xs[order], vs[order]
ve = exp_b2[(exp_b2.x_Al >= X_B2_MIN) & (exp_b2.x_Al <= X_B2_MAX)].copy()

def metrics(sub):
    if sub.empty:
        return dict(n=0, rmse=np.nan, mape=np.nan)
    pred = np.interp(sub.x_Al.values, xs, vs)
    resid = pred - sub.V_bar_A3.values
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mape = float(np.mean(np.abs(resid) / sub.V_bar_A3.values) * 100)
    return dict(n=int(len(sub)), rmse=round(rmse, 4), mape=round(mape, 3))

all_m = metrics(ve)
ni_m = metrics(ve[ve.x_Al < 0.50])
al_m = metrics(ve[ve.x_Al > 0.50])

out = dict(
    n_exp_points_compared=all_m["n"],
    RMSE_V_A3=all_m["rmse"],
    MAPE_V_pct=all_m["mape"],
    RMSE_V_A3_Ni_rich=ni_m["rmse"],
    MAPE_V_pct_Ni_rich=ni_m["mape"],
    n_points_Ni_rich=ni_m["n"],
    RMSE_V_A3_Al_rich=al_m["rmse"],
    MAPE_V_pct_Al_rich=al_m["mape"],
    n_points_Al_rich=al_m["n"],
    V_B2_perfect=float(perfect.V_per_atom_A3),
    a_B2_perfect=float(perfect.a_eff_A),
    T_boltzmann_K=T_ANNEAL_K,
    weight_method="Lower Helmholtz free energy per occupied atom G_i at fixed composition; total Omega_i = N_atom * G_i reported for reference but not used for branch selection because total Omega ordering is not preserved across branches with different N_atom", 
    branch_preference={f"{r.x_Al_target:.2f}": dict(selected_branch=r.selected_branch,
                                                    G_atom_eV=round(float(r.G_atom_eV), 6) if not pd.isna(r.G_atom_eV) else None,
                                                    Omega_total_eV=round(float(r.Omega_total_eV), 6) if not pd.isna(r.Omega_total_eV) else None,
                                                    dG_eV=round(float(r.dG_eV), 6) if not pd.isna(r.dG_eV) else None)
                       for r in mix.itertuples()},
)
agg.to_csv(os.path.join(AN, "b2_offstoich_branch_means.csv"), index=False)
mix.to_csv(os.path.join(AN, "b2_offstoich_boltzmann_mix.csv"), index=False)
with open(os.path.join(AN, "b2_offstoich_comparison.json"), "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
print("DONE")
