#!/usr/bin/env python3
"""Energy vs composition plots for off-stoichiometric B2-NiAl (MACE-MP-0).

Inputs (all pre-computed, nothing is relaxed here):
  analysis/b2_offstoich_branch_means.csv  defect-model means (make_figures.py)
  analysis/b2_offstoich_volumes.csv       perfect B2
  analysis/mace_mp_ref_results.csv        competing ordered phases
  ../niall_ext/analysis/niall_fcc_sqs.csv fcc SQS
  analysis/phonon_fvib.csv                F_vib (x_Al = 0.50-0.58 only)

Outputs:
  figures/fig_energy_composition_0K.png
  figures/fig_energy_composition_0K_hull.png
  figures/fig_energy_composition_finiteT.png
  analysis/energy_composition_table.csv
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from compute_hybrid_c_vac import KB_EV, ln_comb, n_defect_from_row, n_sublattice

BASE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.join(BASE, "analysis")
FIG = os.path.join(BASE, "figures")
T_LIST = [1273.0, 1473.0]
COL = {"antisite": "tab:blue", "vacancy": "tab:red"}
LAB = {"antisite": "反サイトモデル", "vacancy": "空孔モデル"}
PHASE_LABEL = {
    "L12_Ni3Al": r"L1$_2$-Ni$_3$Al",
    "Ni5Al3": r"Ni$_5$Al$_3$",
    "Ni3Al4": r"Ni$_3$Al$_4$",
    "Ni2Al3": r"Ni$_2$Al$_3$",
    "NiAl3": r"NiAl$_3$",
}

plt.rcParams.update({
    "font.size": 20,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.family": ["Noto Sans CJK JP", "sans-serif"],
    "axes.unicode_minus": False,
})


def lower_hull(x, y):
    """Lower convex hull of points (monotone chain)."""
    pts = sorted(set(zip(map(float, x), map(float, y))))
    h = []
    for p in pts:
        while len(h) >= 2 and (
            (h[-1][0] - h[-2][0]) * (p[1] - h[-2][1])
            - (h[-1][1] - h[-2][1]) * (p[0] - h[-2][0])
        ) <= 0:
            h.pop()
        h.append(p)
    return np.array(h)


def load():
    bm = pd.read_csv(os.path.join(AN, "b2_offstoich_branch_means.csv"))
    # At x_Al = 0.5 both defect models are the perfect B2 structure.
    bm = bm[np.abs(bm.x_Al_target - 0.5) > 1e-9].copy()
    bm["n_defect"] = bm.apply(n_defect_from_row, axis=1)
    bm["lng_per_atom"] = bm.apply(
        lambda r: ln_comb(n_sublattice(r), r.n_defect) / r.n_atoms, axis=1)
    vol = pd.read_csv(os.path.join(AN, "b2_offstoich_volumes.csv"))
    ef_b2 = float(vol[vol.branch == "perfect"].E_form_eV_atom.iloc[0])

    fv = pd.read_csv(os.path.join(AN, "phonon_fvib.csv"))
    fv = fv.rename(columns={"x_target": "x_Al_target"})
    bm = bm.merge(fv[["x_Al_target", "branch", "F_vib_1273", "F_vib_1473"]],
                  on=["x_Al_target", "branch"], how="left")
    fv_b2 = fv[fv.branch == "perfect"].iloc[0]
    for T in T_LIST:
        t = f"{T:.0f}"
        bm[f"G_conf_{t}"] = bm.Ef - KB_EV * T * bm.lng_per_atom
        bm[f"G_conf_vib_{t}"] = bm[f"G_conf_{t}"] + bm[f"F_vib_{t}"]
    bm = bm.sort_values(["branch", "x_Al"]).reset_index(drop=True)

    ref = pd.read_csv(os.path.join(AN, "mace_mp_ref_results.csv"))
    ref["x_Al"] = ref.n_Al / ref.n_atoms
    mu_ni = float(ref[ref.label == "Ni"].energy_per_atom_eV.iloc[0])
    mu_al = float(ref[ref.label == "Al"].energy_per_atom_eV.iloc[0])
    sqs = pd.read_csv(os.path.join(BASE, "..", "niall_ext", "analysis", "niall_fcc_sqs.csv"))
    sqs = sqs[sqs.converged].copy()
    sqs["Ef"] = sqs.energy_eV / sqs.n_atoms - ((1 - sqs.x_Al) * mu_ni + sqs.x_Al * mu_al)
    return bm, ef_b2, fv_b2, ref, sqs


def plot_models(ax, bm, ycol, ef_b2, err=True, with_b2=True):
    for br in ("antisite", "vacancy"):
        g = bm[bm.branch == br].dropna(subset=[ycol])
        x, y, e = g.x_Al.values, g[ycol].values, g.Gstd.values
        if with_b2:
            # join the two sides through perfect B2 at x_Al = 0.5
            i = np.searchsorted(x, 0.5)
            x, y, e = (np.insert(x, i, 0.5), np.insert(y, i, ef_b2), np.insert(e, i, 0.0))
        ax.errorbar(x, y, yerr=e if err else None, fmt="o-", ms=7, lw=1.8, capsize=3,
                    color=COL[br], label=LAB[br])
    if with_b2:
        ax.plot([0.5], [ef_b2], "s", ms=14, color="tab:orange", mec="k", zorder=5,
                label="完全 B2-NiAl")


def fig_0K(bm, ef_b2):
    fig, ax = plt.subplots(figsize=(12, 8.5))
    plot_models(ax, bm, "Ef", ef_b2)
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel(r"$x_{\mathrm{Al}}$")
    ax.set_ylabel(r"生成エネルギー $E_f$ (eV/atom)")
    ax.set_title("B2-NiAl 欠陥モデルの 0 K 生成エネルギー\n(MACE-MP-0、純元素 fcc 基準)")
    ax.legend(fontsize=17)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_energy_composition_0K.png"), dpi=150)
    plt.close(fig)


def fig_0K_hull(bm, ef_b2, ref, sqs):
    comp = ref[~ref.label.isin(["Ni", "Al", "B2_NiAl"])]
    hx = np.r_[0.0, 1.0, 0.5, comp.x_Al.values, sqs.x_Al.values]
    hy = np.r_[0.0, 0.0, ef_b2, comp.formation_energy_per_atom_eV.values, sqs.Ef.values]
    hull = lower_hull(hx, hy)

    fig, ax = plt.subplots(figsize=(13, 9))
    ax.plot(hull[:, 0], hull[:, 1], "k--", lw=2,
            label="0 K 凸包 (純元素・規則相・fcc-SQS)")
    plot_models(ax, bm, "Ef", ef_b2)
    ax.plot(sqs.x_Al, sqs.Ef, "^", ms=9, color="tab:purple", label="fcc-SQS")
    ax.plot(comp.x_Al, comp.formation_energy_per_atom_eV, "D", ms=11, color="tab:green",
            mec="k", zorder=6, label="競合する規則相")
    for _, r in comp.iterrows():
        x, y = r.x_Al, r.formation_energy_per_atom_eV
        if r.label == "Ni3Al4":
            ax.annotate(PHASE_LABEL[r.label], (x, y), xytext=(x - 0.005, y - 0.07),
                        ha="left", fontsize=16, color="darkgreen",
                        arrowprops=dict(arrowstyle="-", color="darkgreen"))
            continue
        dx, ha = (0.012, "left") if r.label in ("Ni2Al3", "NiAl3") else (-0.012, "right")
        ax.annotate(PHASE_LABEL[r.label], (x, y), xytext=(x + dx, y - 0.03),
                    ha=ha, fontsize=16, color="darkgreen")
    ax.axhline(0, color="k", lw=1)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.75, 0.05)
    ax.set_xlabel(r"$x_{\mathrm{Al}}$")
    ax.set_ylabel(r"生成エネルギー $E_f$ (eV/atom)")
    ax.set_title("Ni–Al 系の 0 K 生成エネルギーと凸包 (MACE-MP-0)")
    ax.legend(fontsize=15, loc="upper center")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_energy_composition_0K_hull.png"), dpi=150)
    plt.close(fig)
    return hull


def fig_finiteT(bm, ef_b2, fv_b2):
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    for j, T in enumerate(T_LIST):
        t = f"{T:.0f}"
        ax = axes[0, j]
        plot_models(ax, bm, f"G_conf_{t}", ef_b2)
        ax.set_title(rf"{t} K：$E_f - TS_{{\mathrm{{conf}}}}$ (全組成)")
        ax.set_ylabel("自由エネルギー (eV/atom)")
        ax.legend(fontsize=16)

        ax = axes[1, j]
        g_b2 = ef_b2 + float(fv_b2[f"F_vib_{t}"])
        sub = bm.dropna(subset=[f"G_conf_vib_{t}"])
        for br in ("antisite", "vacancy"):
            g = sub[sub.branch == br]
            ax.plot(np.r_[0.5, g.x_Al], np.r_[g_b2, g[f"G_conf_vib_{t}"]], "o-", ms=10,
                    lw=1.8, color=COL[br], label=LAB[br])
        ax.plot([0.5], [g_b2], "s", ms=14, color="tab:orange", mec="k", zorder=5,
                label="完全 B2-NiAl")
        ax.set_title(rf"{t} K：$E_f + F_{{\mathrm{{vib}}}} - TS_{{\mathrm{{conf}}}}$"
                     "\n(フォノン計算済みの組成のみ)")
        ax.set_ylabel("自由エネルギー (eV/atom)")
        ax.legend(fontsize=16)
    for ax in axes.ravel():
        ax.set_xlabel(r"$x_{\mathrm{Al}}$")
    fig.suptitle("B2-NiAl 欠陥モデルの有限温度自由エネルギー (MACE-MP-0)", fontsize=26)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_energy_composition_finiteT.png"), dpi=130)
    plt.close(fig)


def main():
    bm, ef_b2, fv_b2, ref, sqs = load()
    fig_0K(bm, ef_b2)
    hull = fig_0K_hull(bm, ef_b2, ref, sqs)
    fig_finiteT(bm, ef_b2, fv_b2)
    bm["E_above_hull_0K"] = bm.Ef - np.interp(bm.x_Al, hull[:, 0], hull[:, 1])
    cols = ["x_Al_target", "x_Al", "branch", "n", "n_atoms", "n_sites", "n_defect", "Ef",
            "Gstd", "E_above_hull_0K"]
    cols += [f"{p}_{T:.0f}" for T in T_LIST for p in ("G_conf", "F_vib", "G_conf_vib")]
    bm[cols].rename(columns={"Gstd": "Ef_std"}).to_csv(
        os.path.join(AN, "energy_composition_table.csv"), index=False)
    print("hull vertices:", [(round(a, 4), round(b, 4)) for a, b in hull])
    print("perfect B2 Ef:", ef_b2)


if __name__ == "__main__":
    main()
