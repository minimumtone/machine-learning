#!/usr/bin/env python3
"""
Vegard則 組成軸プロット — 全元素ペア
======================================

各元素ペア(X,Y)について、組成軸 c_X = 0→1 上に:
  - Vegard則の直線（純元素体積の線形補間）
  - DFT計算値: L1₂ Y₃X (c_X=0.25), B2 XY (c_X=0.50), L1₂ X₃Y (c_X=0.75)
をプロットする。

Author: Satoshi Minamoto (NIMS) / Devin
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "font.family": "sans-serif",
    "font.sans-serif": ["IPAPGothic", "IPAGothic", "DejaVu Sans"],
})

KING_ATOMIC_VOLUMES = {
    "Al":16.602,"Cu":11.810,"Ni":10.941,"Pd":14.716,"Pt":15.095,
    "Au":16.966,"Ag":17.061,"Ir":14.155,"Rh":13.754,
    "Co":11.073,"Ti":17.649,"Zr":23.279,"Hf":22.312,
    "Ru":13.571,"Os":13.977,"Re":14.712,"Mn":12.210,"Zn":15.207,
    "Fe":11.776,"Cr":12.008,"V":13.824,"Nb":17.978,"Mo":15.583,
    "Ta":18.014,"W":15.850,"Si":20.024,"Ge":22.634,"Be":8.111,
    "Mg":23.240,"Y":33.018,"La":37.168,"Ce":34.367,"Sc":24.987,
    "B":7.241,"P":23.000,"Sn":27.053,"Pb":30.321,
    "Er":30.66,"Tb":32.09,"Dy":31.54,"Ca":43.63,
    "Ba":50.0,"Sr":56.0,"Bi":35.39,"Tl":28.59,"In":26.16,
    "Cd":21.58,"Ga":19.58,"As":21.39,"Se":25.81,"Te":34.32,
}

HEA_ELEMENTS = sorted([
    "Al","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Zr","Nb","Mo","Hf","Ta","W","Re","Pd","Pt","Au","Ag",
    "Sc","Y","Mg","Si","Ge","Sn","Ru","Rh","Os","Ir",
])

OUT = Path("vegard_comparison_output")
OUT.mkdir(exist_ok=True)


def volume_to_lattice(v, Z):
    """原子体積から格子定数 a = (V*Z)^{1/3}."""
    return (v * Z) ** (1/3)


def plot_vegard_composition(ax, row, show_ylabel=True, show_xlabel=True):
    """
    1つの元素ペアについて組成軸プロットを描画.
    
    横軸: c_X (element X の組成分率 0→1)
    縦軸: 原子あたり体積 V [Å³/atom]
    
    プロット:
      - Vegard直線: V_Vegard(c_X) = (1-c_X)·V_Y + c_X·V_X
      - DFT点: c_X=0.25 (Y₃X), c_X=0.50 (B2), c_X=0.75 (X₃Y)
    """
    elX = row["elX"]
    elY = row["elY"]

    vX = KING_ATOMIC_VOLUMES.get(elX, np.nan)
    vY = KING_ATOMIC_VOLUMES.get(elY, np.nan)
    if np.isnan(vX) or np.isnan(vY):
        ax.set_visible(False)
        return

    # Vegard line
    c = np.linspace(0, 1, 100)
    v_vegard = (1 - c) * vY + c * vX
    ax.plot(c, v_vegard, "k--", lw=1.2, alpha=0.7, label="Vegard")

    # DFT points
    markers_plotted = False

    # c_X = 0.25: L1₂ Y₃X (Y is majority, X is minority)
    if not pd.isna(row.get("V_Y3X_DFT", np.nan)):
        ax.plot(0.25, row["V_Y3X_DFT"], "o", ms=7, color="blue",
                markeredgecolor="black", markeredgewidth=0.5,
                label=f"L1$_2$ {elY}$_3${elX}", zorder=5)
        markers_plotted = True

    # c_X = 0.50: B2 XY
    if not pd.isna(row.get("V_B2_DFT", np.nan)):
        ax.plot(0.50, row["V_B2_DFT"], "s", ms=7, color="red",
                markeredgecolor="black", markeredgewidth=0.5,
                label=f"B2 {elX}{elY}", zorder=5)
        markers_plotted = True

    # c_X = 0.75: L1₂ X₃Y (X is majority, Y is minority)
    if not pd.isna(row.get("V_X3Y_DFT", np.nan)):
        ax.plot(0.75, row["V_X3Y_DFT"], "^", ms=7, color="green",
                markeredgecolor="black", markeredgewidth=0.5,
                label=f"L1$_2$ {elX}$_3${elY}", zorder=5)
        markers_plotted = True

    # Pure endpoints
    ax.plot(0, vY, "D", ms=5, color="gray", markeredgecolor="black",
            markeredgewidth=0.5, zorder=5)
    ax.plot(1, vX, "D", ms=5, color="gray", markeredgecolor="black",
            markeredgewidth=0.5, zorder=5)

    ax.set_xlim(-0.05, 1.05)
    ax.set_title(f"{elX}–{elY}", fontsize=11, fontweight="bold")
    if show_xlabel:
        ax.set_xlabel(f"$c_{{{elX}}}$")
    if show_ylabel:
        ax.set_ylabel(r"$V$ [Å$^3$/atom]")

    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0\n" + elY, "0.25", "0.5", "0.75", "1\n" + elX], fontsize=8)

    # Compute and annotate Ω_sf values
    annotations = []
    if not pd.isna(row.get("Omega_Y3X", np.nan)):
        annotations.append(f"$\\Omega_{{Y_3X}}$={row['Omega_Y3X']:.3f}")
    if not pd.isna(row.get("Omega_B2", np.nan)):
        annotations.append(f"$\\Omega_{{B2}}$={row['Omega_B2']:.3f}")
    if not pd.isna(row.get("Omega_X3Y", np.nan)):
        annotations.append(f"$\\Omega_{{X_3Y}}$={row['Omega_X3Y']:.3f}")

    if annotations:
        ax.text(0.02, 0.98, "\n".join(annotations), transform=ax.transAxes,
                fontsize=7, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow",
                          edgecolor="gray", alpha=0.8))


def create_all_plots(table, subset="all"):
    """全ペアのVegard組成軸プロットを生成."""

    if subset == "hea":
        df = table[table["elX"].isin(HEA_ELEMENTS) & table["elY"].isin(HEA_ELEMENTS)].copy()
        suffix = "hea"
    else:
        df = table.copy()
        suffix = "all"

    # Sort by elX, then elY
    df = df.sort_values(["elX", "elY"]).reset_index(drop=True)

    n_total = len(df)
    ncols = 6
    nrows = 5
    per_page = ncols * nrows

    n_pages = (n_total + per_page - 1) // per_page
    print(f"  Creating {n_total} plots across {n_pages} pages ({subset})...")

    pdf_path = OUT / f"vegard_composition_{suffix}.pdf"
    with PdfPages(pdf_path) as pdf:
        for page in range(n_pages):
            start = page * per_page
            end = min(start + per_page, n_total)
            n_this = end - start

            fig, axes = plt.subplots(nrows, ncols, figsize=(24, 18))

            for idx in range(per_page):
                r = idx // ncols
                c_idx = idx % ncols
                ax = axes[r, c_idx]

                if idx < n_this:
                    row = df.iloc[start + idx]
                    show_ylabel = (c_idx == 0)
                    show_xlabel = (r == nrows - 1) or (start + idx >= n_total - ncols)
                    plot_vegard_composition(ax, row, show_ylabel=show_ylabel,
                                           show_xlabel=show_xlabel)
                else:
                    ax.set_visible(False)

            fig.suptitle(
                f"Vegard則 vs DFT: 組成軸プロット ({subset.upper()}) — "
                f"Page {page+1}/{n_pages} (pairs {start+1}–{end}/{n_total})",
                fontsize=16, y=0.995)
            fig.tight_layout(rect=[0, 0, 1, 0.98])
            pdf.savefig(fig, dpi=100)
            plt.close(fig)

            if (page + 1) % 10 == 0:
                print(f"    page {page+1}/{n_pages} done")

    print(f"  Saved: {pdf_path}")

    # Also save first page as PNG for preview
    df_first = df.head(per_page)
    fig, axes = plt.subplots(nrows, ncols, figsize=(24, 18))
    for idx in range(per_page):
        r = idx // ncols
        c_idx = idx % ncols
        ax = axes[r, c_idx]
        if idx < len(df_first):
            row = df_first.iloc[idx]
            plot_vegard_composition(ax, row, show_ylabel=(c_idx == 0),
                                   show_xlabel=(r == nrows - 1))
        else:
            ax.set_visible(False)

    fig.suptitle(
        f"Vegard則 vs DFT: 組成軸プロット ({suffix.upper()}) — Page 1",
        fontsize=16, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    png_path = OUT / f"fig10_vegard_composition_{suffix}_p1.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Preview: {png_path}")

    return n_total, n_pages


def create_summary_statistics_plot(table):
    """Vegard偏差の統計サマリー図."""
    both = table.dropna(subset=["V_X3Y_DFT", "V_Y3X_DFT", "V_B2_DFT"]).copy()

    # Compute Vegard deviations
    both["dev_X3Y"] = both["V_X3Y_DFT"] - both["V_X3Y_Vegard"]
    both["dev_Y3X"] = both["V_Y3X_DFT"] - both["V_Y3X_Vegard"]
    both["dev_B2"] = both["V_B2_DFT"] - both["V_B2_Vegard"]

    # Percent deviation
    both["pct_X3Y"] = both["dev_X3Y"] / both["V_X3Y_Vegard"] * 100
    both["pct_Y3X"] = both["dev_Y3X"] / both["V_Y3X_Vegard"] * 100
    both["pct_B2"] = both["dev_B2"] / both["V_B2_Vegard"] * 100

    fig, axes = plt.subplots(2, 3, figsize=(24, 14))

    # (a) Histogram of absolute deviations
    ax = axes[0, 0]
    for col, color, label in [
        ("dev_X3Y", "green", r"L1$_2$ $X_3Y$"),
        ("dev_Y3X", "blue", r"L1$_2$ $Y_3X$"),
        ("dev_B2", "red", "B2"),
    ]:
        ax.hist(both[col], bins=60, alpha=0.5, color=color, label=label,
                edgecolor="black", lw=0.3)
    ax.axvline(0, color="black", lw=2)
    ax.set_xlabel(r"$V_{DFT} - V_{Vegard}$ [Å$^3$/atom]")
    ax.set_ylabel("Count")
    ax.set_title(r"(a) Vegard偏差分布 (絶対値)")
    ax.legend()

    # (b) Histogram of percent deviations
    ax = axes[0, 1]
    for col, color, label in [
        ("pct_X3Y", "green", r"L1$_2$ $X_3Y$"),
        ("pct_Y3X", "blue", r"L1$_2$ $Y_3X$"),
        ("pct_B2", "red", "B2"),
    ]:
        vals = both[col].clip(-50, 50)
        ax.hist(vals, bins=60, alpha=0.5, color=color, label=label,
                edgecolor="black", lw=0.3)
    ax.axvline(0, color="black", lw=2)
    ax.set_xlabel(r"$(V_{DFT} - V_{Vegard})/V_{Vegard}$ [%]")
    ax.set_ylabel("Count")
    ax.set_title("(b) Vegard偏差分布 (%)")
    ax.legend()

    # (c) Scatter: dev_X3Y vs dev_Y3X
    ax = axes[0, 2]
    ax.scatter(both["dev_X3Y"], both["dev_Y3X"], alpha=0.3, s=15, c="purple")
    lim = max(abs(both["dev_X3Y"]).max(), abs(both["dev_Y3X"]).max()) * 0.8
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1.5)
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel(r"$\Delta V$ ($X_3Y$) [Å$^3$]")
    ax.set_ylabel(r"$\Delta V$ ($Y_3X$) [Å$^3$]")
    ax.set_title(r"(c) Vegard偏差: $X_3Y$ vs $Y_3X$")
    corr = np.corrcoef(both["dev_X3Y"], both["dev_Y3X"])[0, 1]
    ax.text(0.05, 0.95, f"$r$ = {corr:.3f}", transform=ax.transAxes, fontsize=13, va="top")

    # (d) |deviation| vs |V_X - V_Y|
    ax = axes[1, 0]
    dv_pure = abs(both["V_X_pure"] - both["V_Y_pure"])
    for col, color, marker, label in [
        ("dev_X3Y", "green", "o", r"$X_3Y$"),
        ("dev_Y3X", "blue", "^", r"$Y_3X$"),
        ("dev_B2", "red", "s", "B2"),
    ]:
        ax.scatter(dv_pure, abs(both[col]), alpha=0.3, s=15, c=color,
                   marker=marker, label=label)
    ax.set_xlabel(r"|$V_X^{pure} - V_Y^{pure}$| [Å$^3$]")
    ax.set_ylabel(r"|$V_{DFT} - V_{Vegard}$| [Å$^3$]")
    ax.set_title("(d) Vegard偏差 vs 純元素体積差")
    ax.legend()

    # (e) Curvature: is deviation positive or negative?
    ax = axes[1, 1]
    # For each pair, classify: positive (V_DFT > V_Vegard), negative, mixed
    pos_x3y = (both["dev_X3Y"] > 0).sum()
    neg_x3y = (both["dev_X3Y"] < 0).sum()
    pos_y3x = (both["dev_Y3X"] > 0).sum()
    neg_y3x = (both["dev_Y3X"] < 0).sum()
    pos_b2 = (both["dev_B2"] > 0).sum()
    neg_b2 = (both["dev_B2"] < 0).sum()

    x = np.arange(3)
    w = 0.35
    ax.bar(x - w/2, [pos_x3y, pos_y3x, pos_b2], w,
           label=r"$V_{DFT} > V_{Vegard}$ (膨張)", color="salmon")
    ax.bar(x + w/2, [neg_x3y, neg_y3x, neg_b2], w,
           label=r"$V_{DFT} < V_{Vegard}$ (収縮)", color="lightblue")
    ax.set_xticks(x)
    ax.set_xticklabels([r"L1$_2$ $X_3Y$", r"L1$_2$ $Y_3X$", "B2"])
    ax.set_ylabel("Count")
    ax.set_title("(e) 膨張/収縮の割合")
    ax.legend()

    # (f) Mean absolute deviation by structure
    ax = axes[1, 2]
    hea_mask = both["elX"].isin(HEA_ELEMENTS) & both["elY"].isin(HEA_ELEMENTS)
    categories = ["All pairs", "HEA pairs"]
    masks = [pd.Series([True]*len(both), index=both.index), hea_mask]
    x = np.arange(3)
    w = 0.35
    for i, (cat, mask) in enumerate(zip(categories, masks)):
        sub = both[mask]
        mad = [sub["dev_X3Y"].abs().mean(), sub["dev_Y3X"].abs().mean(),
               sub["dev_B2"].abs().mean()]
        ax.bar(x + (i - 0.5) * w, mad, w, label=cat,
               color=["steelblue", "darkorange"][i], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([r"L1$_2$ $X_3Y$", r"L1$_2$ $Y_3X$", "B2"])
    ax.set_ylabel(r"MAD [Å$^3$/atom]")
    ax.set_title("(f) 平均絶対Vegard偏差")
    ax.legend()

    fig.tight_layout()
    fig.savefig(OUT / "fig11_vegard_deviation_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  fig11: Vegard deviation summary")


def main():
    print("=" * 60)
    print("Vegard則 組成軸プロット — 全元素ペア")
    print("=" * 60)

    # Load comparison table
    table = pd.read_csv(OUT / "comparison_table.csv")
    print(f"Total pairs: {len(table)}")

    # Filter to pairs with at least some DFT data
    has_data = table.dropna(subset=["a_X3Y", "a_Y3X"], how="all")
    has_all3 = table.dropna(subset=["a_X3Y", "a_Y3X", "a_B2"])
    print(f"Pairs with any L1₂: {len(has_data)}")
    print(f"Pairs with all 3 structures: {len(has_all3)}")

    # Create plots for HEA elements
    n_hea, pages_hea = create_all_plots(has_all3, subset="hea")

    # Create plots for ALL elements
    n_all, pages_all = create_all_plots(has_all3, subset="all")

    # Summary statistics figure
    create_summary_statistics_plot(has_all3)

    print(f"\n完了: HEA {n_hea} pairs ({pages_hea} pages), All {n_all} pairs ({pages_all} pages)")


if __name__ == "__main__":
    main()
