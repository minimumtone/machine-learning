#!/usr/bin/env python3
"""Binding energies and composition-matched energies of defect pairs / triples in B2-NiAl.

Input : analysis/b2_defect_clusters.csv (run_b2_defect_clusters.py collect)
Output: analysis/b2_defect_cluster_summary.csv
        figures/fig_defect_cluster_binding.png
        figures/fig_defect_cluster_matched.png
        figures/fig_energy_composition_0K_clusters.png

Definitions (E0 = perfect 4x4x4 cell, mu_fu = E0/64 = mu_Ni + mu_Al in B2):
  E_bind  = [E(cluster) - E0] - sum_i [E(isolated_i) - E0]      (negative = bound)
  delta   = dN_Al - dN_Ni                                        (off-stoichiometry)
  E_star  = E - E0 - (dN_Ni + dN_Al)/2 * mu_fu                   (atom-number matched)
  E_star/|delta| compares defects that give the same composition shift.  Its absolute
  value depends on the choice mu_Ni = mu_Al = mu_fu/2, so only differences within one
  side (delta > 0 or delta < 0) are meaningful; E_star for delta = 0 is absolute.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.join(BASE, "analysis")
FIG = os.path.join(BASE, "figures")
KB = 8.617333262e-5
T_LIST = (1273.0, 1473.0)
XMIN = -1.8

plt.rcParams.update({
    "font.family": ["Noto Sans CJK JP", "sans-serif"],
    "font.size": 18, "axes.labelsize": 20, "axes.titlesize": 20,
    "xtick.labelsize": 17, "ytick.labelsize": 17, "legend.fontsize": 15,
    "mathtext.fontset": "dejavusans",
})


def tex(defects):
    out = []
    parts = defects.split("+")
    for p in sorted(set(parts), key=parts.index):
        a, b = p.split("_")
        n = parts.count(p)
        out.append((f"{n}" if n > 1 else "") + rf"{a}$_{{\rm {b}}}$")
    return "+".join(out)


def load():
    df = pd.read_csv(os.path.join(AN, "b2_defect_clusters.csv"))
    e0 = float(df[df.kind == "perfect"].energy_eV.iloc[0])
    mu_fu = float(df.mu_NiAl_fu_eV.iloc[0])
    df["delta"] = df.dN_Al - df.dN_Ni
    df["E_star_eV"] = df.energy_eV - e0 - 0.5 * (df.dN_Ni + df.dN_Al) * mu_fu
    df["E_star_per_delta_eV"] = np.where(df.delta != 0, df.E_star_eV / df.delta.abs().replace(0, np.nan), np.nan)
    return df


def summarize(df):
    rows = []
    iso = df[df.kind == "isolated"]
    for _, r in iso.iterrows():
        rows.append(dict(defects=r.defects, kind="isolated", n_geometries=1, n_transformed=0,
                         best_geometry="", E_bind_min_eV=0.0, delta=r.delta,
                         E_star_min_eV=r.E_star_eV, E_star_per_delta_min_eV=r.E_star_per_delta_eV,
                         E_bind_far_eV=np.nan,
                         **{f"E_bind_boltz_{T:.0f}K_eV": 0.0 for T in T_LIST}))
    for (kind, defects), g in df[df.kind.isin(["pair", "triple"])].groupby(["kind", "defects"], sort=False):
        ok = g[~g.transformed]
        row = dict(defects=defects, kind=kind, n_geometries=len(g),
                   n_transformed=int(g.transformed.sum()), delta=int(g.delta.iloc[0]))
        far = df[(df.kind == "pair_far") & (df.defects == defects)]
        row["E_bind_far_eV"] = float(far.E_bind_eV.iloc[0]) if len(far) else np.nan
        if len(ok):
            b = ok.loc[ok.E_bind_eV.idxmin()]
            row.update(best_geometry=b.geometry, E_bind_min_eV=b.E_bind_eV,
                       E_star_min_eV=b.E_star_eV, E_star_per_delta_min_eV=b.E_star_per_delta_eV)
            for T in T_LIST:
                w = ok.multiplicity * np.exp(-(ok.E_bind_eV - ok.E_bind_eV.min()) / (KB * T))
                row[f"E_bind_boltz_{T:.0f}K_eV"] = float((w * ok.E_bind_eV).sum() / w.sum())
        rows.append(row)
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(AN, "b2_defect_cluster_summary.csv"), index=False)
    return s


def fig_binding(df):
    fig, axes = plt.subplots(1, 2, figsize=(22, 13), gridspec_kw=dict(width_ratios=[1, 1]))
    for ax, kind, title in ((axes[0], "pair", "欠陥対（第1〜第5近接）"),
                            (axes[1], "triple", "欠陥三量体（連結した全形状）")):
        g = df[df.kind == kind]
        order = list(dict.fromkeys(g.defects))
        for y, d in enumerate(order):
            s = g[g.defects == d]
            a = s[~s.transformed]
            b = s[s.transformed]
            ax.scatter(a.E_bind_eV, [y] * len(a), s=150, c="tab:blue", edgecolor="k", zorder=3,
                       label="配置を保持" if y == 0 else None)
            if len(a):
                ax.scatter([a.E_bind_eV.min()], [y], s=330, marker="*", c="tab:red",
                           edgecolor="k", zorder=4, label="最安定配置" if y == 0 else None)
            if kind == "pair":
                f = df[(df.kind == "pair_far") & (df.defects == d)]
                ax.scatter(f.E_bind_eV, [y] * len(f), s=150, marker="x", c="gray", zorder=3,
                           label="最遠配置（セルサイズの目安）" if y == 0 else None)
            if len(b):
                ax.scatter(np.clip(b.E_bind_eV, XMIN + 0.05, None), [y] * len(b), s=150, marker="<",
                           facecolor="none", edgecolor="tab:orange", linewidth=2, zorder=3)
        ax.scatter([], [], s=150, marker="<", facecolor="none", edgecolor="tab:orange",
                   linewidth=2, label="緩和中に別の欠陥へ変化（−1.75 eV 以下は枠端に表示）")
        ax.axvline(0, color="k", lw=1)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([tex(d) for d in order])
        ax.invert_yaxis()
        ax.set_xlim(XMIN, 0.8)
        ax.set_xlabel(r"結合エネルギー $E_{\rm bind}$ (eV)　負＝束縛")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("B2-NiAl の欠陥クラスター結合エネルギー（MACE-MP-0、4×4×4 セル、0 K）", fontsize=24)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(os.path.join(FIG, "fig_defect_cluster_binding.png"), dpi=110)
    plt.close(fig)


def fig_matched(s):
    fig, axes = plt.subplots(1, 3, figsize=(26, 12))
    panels = ((s[s.delta > 0], r"Al 過剰側（$\delta>0$）", "E_star_per_delta_min_eV",
               r"$E^*/|\delta|$：最安定の欠陥との差 (eV)"),
              (s[s.delta < 0], r"Ni 過剰側（$\delta<0$）", "E_star_per_delta_min_eV",
               r"$E^*/|\delta|$：最安定の欠陥との差 (eV)"),
              (s[s.delta == 0], r"組成を変えない欠陥（$\delta=0$）", "E_star_min_eV",
               r"$E^*$ (eV)"))
    col = {"isolated": "tab:green", "pair": "tab:blue", "triple": "tab:purple"}
    lab = {"isolated": "孤立欠陥", "pair": "対", "triple": "三量体"}
    for ax, (g, title, c, xl) in zip(axes, panels):
        g = g.dropna(subset=[c]).sort_values(c)
        if "per_delta" in c:
            g = g.assign(**{c: g[c] - g[c].min()})
        ax.barh(range(len(g)), g[c], color=[col[k] for k in g.kind], edgecolor="k")
        ax.set_yticks(range(len(g)))
        ax.set_yticklabels([tex(d) for d in g.defects])
        ax.invert_yaxis()
        ax.set_xlabel(xl)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)
    axes[2].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=col[k], ec="k") for k in col],
                   labels=[lab[k] for k in col], loc="upper right")
    fig.suptitle("組成変化を揃えたエネルギー比較（各欠陥種の最安定配置、MACE-MP-0、0 K）", fontsize=24)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(FIG, "fig_defect_cluster_matched.png"), dpi=110)
    plt.close(fig)


def seed_points():
    """Individual random-placement results of the existing vacancy / antisite models."""
    import glob
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(
        os.path.join(AN, "b2_offstoich_volumes*.csv")))], ignore_index=True)
    d = d[d.converged.astype(bool) & d.branch.isin(["vacancy", "antisite"])]
    return d.drop_duplicates("structure_id")


def fig_overlay(df):
    bm = pd.read_csv(os.path.join(AN, "energy_composition_table.csv"))
    sp = seed_points()
    p = df[df.kind == "perfect"]
    fig, ax = plt.subplots(figsize=(20, 11))
    for b, c, l in (("antisite", "tab:blue", "反サイトモデル"),
                    ("vacancy", "tab:red", "空孔モデル")):
        g = bm[bm.branch == b]
        g = pd.concat([g, pd.DataFrame(dict(x_Al=[0.5], Ef=[float(p.E_form_eV_atom.iloc[0])],
                                            Ef_std=[0.0]))]).sort_values("x_Al")
        ax.errorbar(g.x_Al, g.Ef, yerr=g.Ef_std, fmt="o-", color=c, ms=10, capsize=4,
                    label=f"{l}（ランダム配置の平均）")
        q = sp[sp.branch == b]
        ax.scatter(q.x_Al, q.E_form_eV_atom, marker="x", s=110, c=c, lw=2, zorder=2,
                   label=f"{l}（各ランダム配置）")
    iso_path = os.path.join(AN, "b2_isolated_vacancies.csv")
    if os.path.exists(iso_path):
        iso = pd.read_csv(iso_path)
        iso = iso[iso.converged.astype(bool)]
        one = df[(df.kind == "isolated") & df.defects.isin(["V_Ni", "V_Al"])]
        for d, c in (("V_Ni", "limegreen"), ("V_Al", "darkgreen")):
            g = iso[iso.defect == d]
            o = one[one.defects == d]
            ax.scatter(np.r_[g.x_Al, o.x_Al], np.r_[g.E_form_eV_atom, o.E_form_eV_atom],
                       marker="s", s=130, c=c, edgecolor="k", zorder=4,
                       label=rf"単空孔 {tex(d)}（互いに離した配置）")
    g = df[(df.kind == "isolated") & df.defects.isin(["Ni_Al", "Al_Ni"])]
    ax.scatter(g.x_Al, g.E_form_eV_atom, marker="s", s=130, c="white", edgecolor="k", zorder=4,
               label="孤立反サイト")
    st = {"pair": ("^", "tab:orange", "対"), "triple": ("D", "tab:purple", "三量体")}
    for k, (m, c, l) in st.items():
        g = df[(df.kind == k) & (~df.transformed)]
        ax.scatter(g.x_Al, g.E_form_eV_atom, marker=m, s=90, c=c, edgecolor="k", alpha=0.8,
                   zorder=3, label=f"{l}（全配置）")
    ax.scatter(p.x_Al, p.E_form_eV_atom, marker="*", s=500, c="gold", edgecolor="k", zorder=5,
               label="完全 B2")
    ax.set_xlim(0.464, 0.536)
    ax.set_ylim(-0.697, -0.625)
    ax.set_xlabel(r"$x_{\rm Al}$")
    ax.set_ylabel(r"生成エネルギー $E_f$ (eV/atom)")
    ax.set_title("0 K 生成エネルギー：単空孔と欠陥クラスターの重ね描き")
    ax.grid(alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_energy_composition_0K_clusters.png"), dpi=120)
    plt.close(fig)


def main():
    df = load()
    df.to_csv(os.path.join(AN, "b2_defect_clusters.csv"), index=False)
    s = summarize(df)
    fig_binding(df)
    fig_matched(s)
    fig_overlay(df)
    pd.set_option("display.width", 250)
    print(s.round(3).to_string())


if __name__ == "__main__":
    main()
