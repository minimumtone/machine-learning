#!/usr/bin/env python3
"""
Pseudo-Vegard's Law Analysis: B2 + L1$_2$ Composition Sweep
=============================================================

For binary A–B systems, three ordered structures provide composition points:

    L1$_2$ A$_3$B  →  75 % A  (CN = 12)
    B2  AB          →  50 % A  (CN = 8)
    L1$_2$ AB$_3$  →  25 % A  (CN = 12)

This script tests whether the lattice parameter (or a derived
structure-independent quantity) varies linearly with composition
across these three structure types, i.e. a *pseudo-Vegard's law*
that ignores the structural change from L1$_2$ (FCC-based) to B2
(BCC-based).

Three representations are compared:
  1. Raw lattice constant  $a$  (structures differ → different scale)
  2. Nearest-neighbour distance  $d_{\\mathrm{nn}}$
     B2:  $d = a\\sqrt{3}/2$,  L1$_2$: $d = a/\\sqrt{2}$
  3. Wigner-Seitz radius  $r_{\\mathrm{WS}} = (V_{\\mathrm{atom}})^{1/3}$
     B2:  $r = (a^3/2)^{1/3}$,  L1$_2$: $r = (a^3/4)^{1/3}$

All figures and report are generated in a single execution pass.
"""

import os
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Presentation-quality font sizes (doubled from default)
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 22,
})


# =====================================================================
# Data loading helpers
# =====================================================================

def normalize_compounds(df: pd.DataFrame, struct_label: str) -> pd.DataFrame:
    """Normalise compound data to canonical (el1 < el2) pair and frac_el1."""
    records = []
    for _, row in df.iterrows():
        a, b = row["element_A"], row["element_B"]
        ca, cb = float(row["count_A"]), float(row["count_B"])
        el1, el2 = sorted([a, b])
        frac_el1 = ca / (ca + cb) if a == el1 else cb / (ca + cb)
        records.append({
            "el1": el1, "el2": el2,
            "frac_el1": round(frac_el1, 2),
            "a_lat": row["lattice_constant"],
            "source": struct_label,
        })
    return pd.DataFrame(records)


def load_all_compounds(data_dir: str) -> pd.DataFrame:
    """Load and normalise L1_2 and B2 compound data from MP + OQMD."""
    frames = []
    for fname, label in [
        ("compounds_MP_L12.csv", "L12_MP"),
        ("compounds_OQMD_L12.csv", "L12_OQMD"),
        ("compounds_MP_B2.csv", "B2_MP"),
        ("compounds_OQMD_B2.csv", "B2_OQMD"),
    ]:
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            frames.append(normalize_compounds(df, label))
    return pd.concat(frames, ignore_index=True)


# =====================================================================
# Structure-independent metrics
# =====================================================================

def d_nn(a: float, struct: str) -> float:
    """Nearest-neighbour distance from lattice constant."""
    if "B2" in struct:
        return a * np.sqrt(3) / 2
    else:  # L12
        return a / np.sqrt(2)


def r_ws(a: float, struct: str) -> float:
    """Wigner-Seitz radius = (V_atom)^{1/3}."""
    if "B2" in struct:
        return (a ** 3 / 2) ** (1.0 / 3)
    else:  # L12
        return (a ** 3 / 4) ** (1.0 / 3)


# =====================================================================
# Find triplet pairs and build analysis table
# =====================================================================

def find_triplet_pairs(df: pd.DataFrame) -> List[Tuple[str, str]]:
    """Find binary pairs that have data at x = 0.25, 0.50, and 0.75."""
    pairs_25 = set(df[df["frac_el1"] == 0.25][["el1", "el2"]].apply(tuple, axis=1))
    pairs_50 = set(df[df["frac_el1"] == 0.50][["el1", "el2"]].apply(tuple, axis=1))
    pairs_75 = set(df[df["frac_el1"] == 0.75][["el1", "el2"]].apply(tuple, axis=1))
    return sorted(pairs_25 & pairs_50 & pairs_75)


def find_all_plottable_pairs(df: pd.DataFrame) -> List[Tuple[str, str]]:
    """Find binary pairs that have data at >= 2 compositions."""
    pairs_25 = set(df[df["frac_el1"] == 0.25][["el1", "el2"]].apply(tuple, axis=1))
    pairs_50 = set(df[df["frac_el1"] == 0.50][["el1", "el2"]].apply(tuple, axis=1))
    pairs_75 = set(df[df["frac_el1"] == 0.75][["el1", "el2"]].apply(tuple, axis=1))
    at_least_two = set()
    for p in pairs_25 | pairs_50 | pairs_75:
        n = sum([p in pairs_25, p in pairs_50, p in pairs_75])
        if n >= 2:
            at_least_two.add(p)
    return sorted(at_least_two)


def build_triplet_table(df: pd.DataFrame,
                        pairs: List[Tuple[str, str]]) -> pd.DataFrame:
    """Build table with one row per pair, columns for 25/50/75% values."""
    rows = []
    for el1, el2 in pairs:
        sub = df[(df["el1"] == el1) & (df["el2"] == el2)]
        row: Dict = {"el1": el1, "el2": el2}

        for frac, label in [(0.25, "25"), (0.50, "50"), (0.75, "75")]:
            pts = sub[sub["frac_el1"] == frac]
            if len(pts) == 0:
                continue
            # Average if multiple sources
            a_avg = pts["a_lat"].mean()
            # Pick one source string for reference
            src = pts.iloc[0]["source"]
            row[f"a_{label}"] = a_avg
            row[f"src_{label}"] = src
            row[f"d_nn_{label}"] = d_nn(a_avg, src)
            row[f"r_ws_{label}"] = r_ws(a_avg, src)

        # Skip if incomplete (need at least 2 compositions)
        n_comp = sum(1 for x in ["25", "50", "75"] if f"a_{x}" in row)
        if n_comp >= 2:
            row["n_compositions"] = n_comp
            rows.append(row)

    return pd.DataFrame(rows)


# =====================================================================
# Linearity analysis
# =====================================================================

def fit_linear(x: np.ndarray, y: np.ndarray) -> Dict:
    """Fit y = mx + b and return slope, intercept, R^2, residuals."""
    coeffs = np.polyfit(x, y, 1)
    m, b = coeffs
    y_pred = m * x + b
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"slope": m, "intercept": b, "R2": r2, "residuals": y - y_pred}


def analyse_linearity(table: pd.DataFrame, metric: str) -> pd.DataFrame:
    """For each pair, fit linear regression on the 3 composition points."""
    results = []
    x = np.array([0.25, 0.50, 0.75])
    for _, row in table.iterrows():
        y = np.array([row[f"{metric}_25"], row[f"{metric}_50"],
                      row[f"{metric}_75"]])
        fit = fit_linear(x, y)
        # Deviation from linearity: max |residual| / range
        y_range = y.max() - y.min()
        max_dev_pct = (np.max(np.abs(fit["residuals"])) / y_range * 100
                       if y_range > 0 else 0.0)
        results.append({
            "el1": row["el1"], "el2": row["el2"],
            f"{metric}_25": y[0], f"{metric}_50": y[1], f"{metric}_75": y[2],
            "slope": fit["slope"], "intercept": fit["intercept"],
            "R2": fit["R2"],
            "max_dev_pct": max_dev_pct,
            "range": y_range,
            "mid_residual": fit["residuals"][1],  # B2 point deviation
        })
    return pd.DataFrame(results)


# =====================================================================
# Plotting
# =====================================================================

def plot_individual_vegard(table: pd.DataFrame, metric: str,
                           ylabel: str, title: str,
                           fig_path: str, n_cols: int = 6):
    """Plot pseudo-Vegard diagrams for all pairs in a grid."""
    n = len(table)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows),
                             squeeze=False)

    x = np.array([0.25, 0.50, 0.75])
    x_fit = np.linspace(0.20, 0.80, 50)

    for idx, (_, row) in enumerate(table.iterrows()):
        ax = axes[idx // n_cols][idx % n_cols]
        y = np.array([row[f"{metric}_25"], row[f"{metric}_50"],
                      row[f"{metric}_75"]])
        fit = fit_linear(x, y)

        # Data points
        ax.scatter([0.25, 0.75], [y[0], y[2]], c="#1f77b4", s=60,
                   zorder=5, marker="o", label="L1$_2$")
        ax.scatter([0.50], [y[1]], c="#d62728", s=60, zorder=5,
                   marker="D", label="B2")
        # Linear fit
        ax.plot(x_fit, fit["slope"] * x_fit + fit["intercept"],
                "k--", alpha=0.5, linewidth=1)

        ax.set_title(f"{row['el1']}–{row['el2']}\n$R^2$={fit['R2']:.4f}",
                     fontsize=11)
        ax.set_xlabel(f"$x_{{{row['el1']}}}$", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.tick_params(labelsize=9)

    # Hide empty axes
    for idx in range(n, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.suptitle(title, fontsize=18, y=1.01)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_triplet_summary_grid(table: pd.DataFrame, metric: str,
                              ylabel: str, fig_path: str,
                              n_cols: int = 6):
    """Summary grid of complete triplet pairs (3 compositions) only.

    Similar to plot_individual_vegard but with richer annotations:
    colour-coded L1_2 / B2 markers, R^2-based subplot border colour,
    and aggregate statistics in a super-title.
    """
    # Filter to rows with all 3 compositions
    triplet = table[table["n_compositions"] == 3].copy()
    if triplet.empty:
        print("  No complete triplet pairs found – skipping summary grid.")
        return

    # Sort by R² descending so best-fit pairs appear first
    x_pts = np.array([0.25, 0.50, 0.75])
    r2_vals = []
    for _, row in triplet.iterrows():
        y = np.array([row[f"{metric}_25"], row[f"{metric}_50"],
                      row[f"{metric}_75"]])
        r2_vals.append(fit_linear(x_pts, y)["R2"])
    triplet = triplet.assign(_r2=r2_vals).sort_values("_r2", ascending=False)

    n = len(triplet)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.2 * n_cols, 3.8 * n_rows),
                             squeeze=False)
    x_fit = np.linspace(0.15, 0.85, 80)

    for idx, (_, row) in enumerate(triplet.iterrows()):
        ax = axes[idx // n_cols][idx % n_cols]
        el1, el2 = row["el1"], row["el2"]
        y = np.array([row[f"{metric}_25"], row[f"{metric}_50"],
                      row[f"{metric}_75"]])
        fit = fit_linear(x_pts, y)

        # Data points: L1_2 (blue circles) and B2 (red diamond)
        ax.scatter([0.25, 0.75], [y[0], y[2]], c="#1f77b4", s=70,
                   zorder=5, marker="o", edgecolors="black",
                   linewidths=0.6, label="L1$_2$")
        ax.scatter([0.50], [y[1]], c="#d62728", s=70, zorder=5,
                   marker="D", edgecolors="black",
                   linewidths=0.6, label="B2")

        # Linear fit
        y_fit = fit["slope"] * x_fit + fit["intercept"]
        ax.plot(x_fit, y_fit, "k--", alpha=0.5, linewidth=1)

        # Colour border by R² quality
        r2 = fit["R2"]
        if r2 >= 0.99:
            border_color = "#2ca02c"   # green
        elif r2 >= 0.95:
            border_color = "#1f77b4"   # blue
        elif r2 >= 0.90:
            border_color = "#ff7f0e"   # orange
        else:
            border_color = "#d62728"   # red
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(2)

        ax.set_title(f"{el1}–{el2}\n$R^2$={r2:.4f}",
                     fontsize=11, color=border_color, fontweight="bold")
        ax.set_xlabel(f"$x_{{{el1}}}$", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlim(0.15, 0.85)
        ax.tick_params(labelsize=9)

    # Hide empty axes
    for idx in range(n, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    # Aggregate stats in suptitle
    r2_arr = np.array(r2_vals)
    n99 = (r2_arr > 0.99).sum()
    n95 = (r2_arr > 0.95).sum()
    fig.suptitle(
        f"Pseudo-Vegard ($r_{{\\mathrm{{WS}}}}$): {n} complete triplet pairs\n"
        f"Mean $R^2$ = {r2_arr.mean():.3f},  Median $R^2$ = {np.median(r2_arr):.3f}  "
        f"($R^2$>0.99: {n99},  $R^2$>0.95: {n95})",
        fontsize=16, y=1.02)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Triplet summary grid ({n} pairs) → {fig_path}")


def plot_individual_vegard_separate(table: pd.DataFrame, metric: str,
                                    ylabel: str, out_dir: str):
    """Save one publication-quality figure per binary pair (2 or 3 points)."""
    os.makedirs(out_dir, exist_ok=True)
    x_fit = np.linspace(0.15, 0.85, 80)
    count = 0

    for _, row in table.iterrows():
        el1, el2 = row["el1"], row["el2"]

        # Collect available data points
        xs, ys, colors, markers, labels_ann = [], [], [], [], []
        for frac, label, clr, mkr, ann in [
            (0.25, "25", "#1f77b4", "o",
             f"L1$_2$ AB$_3$\n(25% {el1})"),
            (0.50, "50", "#d62728", "D",
             f"B2 AB\n(50% {el1})"),
            (0.75, "75", "#1f77b4", "o",
             f"L1$_2$ A$_3$B\n(75% {el1})"),
        ]:
            col = f"{metric}_{label}"
            if col in row and pd.notna(row[col]):
                xs.append(frac)
                ys.append(row[col])
                colors.append(clr)
                markers.append(mkr)
                labels_ann.append(ann)

        if len(xs) < 2:
            continue

        x_arr, y_arr = np.array(xs), np.array(ys)
        fit = fit_linear(x_arr, y_arr)
        n_pts = len(xs)

        fig, ax = plt.subplots(figsize=(7, 5))

        # Plot L1_2 and B2 points separately for legend
        l12_plotted, b2_plotted = False, False
        for xp, yp, clr, mkr in zip(xs, ys, colors, markers):
            lbl = None
            if mkr == "o" and not l12_plotted:
                lbl = "L1$_2$"
                l12_plotted = True
            elif mkr == "D" and not b2_plotted:
                lbl = "B2"
                b2_plotted = True
            ax.scatter([xp], [yp], c=clr, s=120, zorder=5, marker=mkr,
                       edgecolors="black", linewidths=0.8, label=lbl)

        # Linear fit line
        y_fit = fit["slope"] * x_fit + fit["intercept"]
        if n_pts == 3:
            fit_label = f"Linear fit ($R^2$ = {fit['R2']:.4f})"
        else:
            fit_label = "Linear interpolation (2 pts)"
        ax.plot(x_fit, y_fit, "k--", alpha=0.6, linewidth=1.5,
                label=fit_label)

        # Mark missing composition as open marker
        all_fracs = {0.25, 0.50, 0.75}
        missing = all_fracs - set(xs)
        for mf in missing:
            y_pred = fit["slope"] * mf + fit["intercept"]
            mkr_m = "D" if mf == 0.50 else "o"
            ax.scatter([mf], [y_pred], facecolors="none",
                       edgecolors="gray", s=120, zorder=4,
                       marker=mkr_m, linewidths=1.5)
            struct_lbl = "B2" if mf == 0.50 else "L1$_2$"
            ax.annotate(f"{struct_lbl}\n(no data)", (mf, y_pred),
                        textcoords="offset points", xytext=(0, -20),
                        ha="center", fontsize=9, color="gray",
                        fontstyle="italic")

        # Annotations for actual points
        for xp, yp, txt in zip(xs, ys, labels_ann):
            ax.annotate(txt, (xp, yp), textcoords="offset points",
                        xytext=(0, 14), ha="center", fontsize=11,
                        color="#333333")

        ax.set_xlabel(f"$x_{{{el1}}}$ (fraction of {el1})", fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        r2_str = f"$R^2$ = {fit['R2']:.4f}" if n_pts == 3 else "(2 pts)"
        ax.set_title(f"{el1}\u2013{el2}  Pseudo-Vegard ($r_{{\\mathrm{{WS}}}}$)\n"
                     f"{r2_str}", fontsize=18)
        ax.legend(fontsize=13, loc="best")
        ax.set_xlim(0.15, 0.85)
        ax.tick_params(labelsize=13)
        plt.tight_layout()

        fname = f"vegard_{el1}_{el2}_r_ws.png"
        plt.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches="tight")
        plt.close()
        count += 1

    print(f"  Saved {count} individual plots \u2192 {out_dir}/")


def plot_r2_histogram(results: pd.DataFrame, metric_label: str,
                      fig_path: str):
    """Histogram of R^2 values."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(results["R2"], bins=20, edgecolor="black", alpha=0.7,
            color="#2ca02c")
    ax.axvline(results["R2"].mean(), color="red", linestyle="--",
               linewidth=2, label=f"Mean $R^2$ = {results['R2'].mean():.4f}")
    ax.axvline(results["R2"].median(), color="blue", linestyle="--",
               linewidth=2, label=f"Median $R^2$ = {results['R2'].median():.4f}")
    ax.set_xlabel("$R^2$")
    ax.set_ylabel("Count")
    ax.set_title(f"Pseudo-Vegard Linearity: {metric_label}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_mid_deviation(results: pd.DataFrame, metric: str,
                       ylabel: str, fig_path: str):
    """Scatter plot: B2 point deviation from L12-L12 line."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # mid_residual > 0 means B2 above L12-L12 line (expansion)
    # mid_residual < 0 means B2 below (contraction)
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in results["mid_residual"]]
    ax.barh(range(len(results)), results["mid_residual"], color=colors,
            edgecolor="black", alpha=0.7, height=0.7)
    ax.set_yticks(range(len(results)))
    ax.set_yticklabels([f"{r['el1']}-{r['el2']}" for _, r in results.iterrows()],
                       fontsize=9)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel(f"B2 deviation from L1$_2$–L1$_2$ line ({ylabel})")
    ax.set_title(f"B2 Point Deviation from Pseudo-Vegard Line ({ylabel})")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_three_metrics(table: pd.DataFrame, fig_path: str):
    """3-panel comparison: a_lat, d_nn, r_ws for a subset of pairs."""
    # Pick 12 representative pairs spanning different R2 ranges
    for metric in ["d_nn"]:
        res = analyse_linearity(table, metric)
    res_sorted = res.sort_values("R2")
    n_show = min(12, len(res_sorted))
    # pick evenly spaced
    indices = np.linspace(0, len(res_sorted) - 1, n_show, dtype=int)
    sample = res_sorted.iloc[indices]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    metrics = [("a", "$a$ (\\AA)", "Raw Lattice Constant"),
               ("d_nn", "$d_{\\mathrm{nn}}$ (\\AA)", "Nearest-Neighbour Distance"),
               ("r_ws", "$r_{\\mathrm{WS}}$ (\\AA)", "Wigner-Seitz Radius")]

    x = np.array([0.25, 0.50, 0.75])
    cmap = plt.cm.tab20

    for ax, (metric, ylabel, title) in zip(axes, metrics):
        for i, (_, row) in enumerate(sample.iterrows()):
            pair_row = table[(table["el1"] == row["el1"]) &
                             (table["el2"] == row["el2"])].iloc[0]
            y = np.array([pair_row[f"{metric}_{p}"] for p in ["25", "50", "75"]])
            # Normalize to range for comparison
            color = cmap(i / n_show)
            ax.plot(x, y, "o-", color=color, markersize=5, alpha=0.7,
                    label=f"{row['el1']}-{row['el2']}")
        ax.set_xlabel("$x_{\\mathrm{el1}}$ (fraction of element 1)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8, ncol=2, loc="best")

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()


# =====================================================================
# Report generation
# =====================================================================

def generate_report(table: pd.DataFrame,
                    results_a: pd.DataFrame,
                    results_dnn: pd.DataFrame,
                    results_rws: pd.DataFrame,
                    fig_dir: str) -> List[str]:
    """Generate Markdown report."""
    lines: List[str] = []
    lines.append("# 疑似Vegard則解析: B2 + L1$_2$ 組成スイープ\n")
    lines.append("## 概要\n")
    lines.append("二元系A–Bにおいて、L1$_2$ A$_3$B (75%A)、B2 AB (50%A)、")
    lines.append("L1$_2$ AB$_3$ (25%A) の3つの秩序構造から得られる格子定数を用いて、")
    lines.append("組成に対する線形性（疑似Vegard則）を検証する。\n")
    lines.append("構造の違い（B2: BCC基盤 CN=8、L1$_2$: FCC基盤 CN=12）は")
    lines.append("以下の3つの指標で評価する：\n")
    lines.append("1. **生の格子定数 $a$** — 構造の違いを無視")
    lines.append("2. **最近接原子間距離 $d_{\\mathrm{nn}}$** — "
                 "B2: $a\\sqrt{3}/2$、L1$_2$: $a/\\sqrt{2}$")
    lines.append("3. **Wigner-Seitz半径 $r_{\\mathrm{WS}}$** — "
                 "$(V_{\\mathrm{atom}})^{1/3}$\n")
    lines.append(f"解析対象: **{len(table)} 二元系** "
                 f"(MP + OQMD から L1$_2$ A$_3$B, B2 AB, L1$_2$ AB$_3$ "
                 f"すべてが利用可能なペア)\n")

    # Summary statistics
    lines.append("## 1. 線形性の統計\n")
    lines.append("| Metric | Mean $R^2$ | Median $R^2$ | $R^2 > 0.99$ | "
                 "$R^2 > 0.95$ | $R^2 < 0.90$ |")
    lines.append("|:---|:---:|:---:|:---:|:---:|:---:|")

    for label, res in [("$a$ (raw)", results_a),
                       ("$d_{\\mathrm{nn}}$", results_dnn),
                       ("$r_{\\mathrm{WS}}$", results_rws)]:
        n_99 = (res["R2"] > 0.99).sum()
        n_95 = (res["R2"] > 0.95).sum()
        n_lt90 = (res["R2"] < 0.90).sum()
        lines.append(
            f"| {label} | {res['R2'].mean():.4f} | {res['R2'].median():.4f} | "
            f"{n_99} | {n_95} | {n_lt90} |"
        )

    # Detailed table for d_nn (most physically meaningful)
    lines.append("\n## 2. 最近接原子間距離 $d_{\\mathrm{nn}}$ による解析\n")
    lines.append("| Pair | $d_{\\mathrm{nn}}$(25%) | $d_{\\mathrm{nn}}$(50%) | "
                 "$d_{\\mathrm{nn}}$(75%) | $R^2$ | B2 deviation (\\AA) |")
    lines.append("|:---|:---:|:---:|:---:|:---:|:---:|")

    for _, r in results_dnn.sort_values("R2", ascending=False).iterrows():
        lines.append(
            f"| {r['el1']}-{r['el2']} | {r['d_nn_25']:.4f} | "
            f"{r['d_nn_50']:.4f} | {r['d_nn_75']:.4f} | "
            f"{r['R2']:.4f} | {r['mid_residual']:+.4f} |"
        )

    # Raw lattice constant table
    lines.append("\n## 3. 生の格子定数 $a$ による解析\n")
    lines.append("| Pair | $a$(25%) L1$_2$ | $a$(50%) B2 | $a$(75%) L1$_2$ | "
                 "$R^2$ |")
    lines.append("|:---|:---:|:---:|:---:|:---:|")

    for _, r in results_a.sort_values("R2", ascending=False).iterrows():
        lines.append(
            f"| {r['el1']}-{r['el2']} | {r['a_25']:.4f} | "
            f"{r['a_50']:.4f} | {r['a_75']:.4f} | {r['R2']:.4f} |"
        )

    # Figures
    lines.append("\n![Triplet summary grid (r_ws)](vegard_triplet_summary_r_ws.png)\n")
    lines.append("![R2 histogram (r_ws)](vegard_r2_histogram_rws.png)\n")
    lines.append("![B2 deviation](vegard_b2_deviation.png)\n")
    lines.append("![Three metrics comparison](vegard_three_metrics.png)\n")

    # Discussion
    lines.append("## 4. 考察\n")
    lines.append("### 4.1 構造の違いと指標の選択\n")
    lines.append("生の格子定数 $a$ で比較した場合、B2（BCC基盤、$a \\approx 3$–4 \\AA）と")
    lines.append("L1$_2$（FCC基盤、$a \\approx 4$–5 \\AA）のスケール差が大きいため、")
    lines.append("$R^2$ は比較的低くなる傾向がある。\n")
    lines.append("最近接原子間距離 $d_{\\mathrm{nn}}$ またはWigner-Seitz半径 $r_{\\mathrm{WS}}$ に")
    lines.append("変換すると、構造の違いが正規化され、より意味のある比較が可能となる。\n")
    lines.append("### 4.2 B2 点の偏差の意味\n")
    lines.append("L1$_2$–L1$_2$ の2点を結ぶ直線からのB2点の偏差は、")
    lines.append("CN=8（B2）とCN=12（L1$_2$）における原子サイズの違いを反映する。")
    lines.append("B2点が直線より上（正の偏差）の場合、CN=8環境で原子間距離が")
    lines.append("CN=12の外挿値より大きいことを意味し、配位数依存性を示唆する。\n")

    return lines


# =====================================================================
# Main
# =====================================================================

def run_analysis(data_dir: str, fig_dir: str, report_path: str):
    """Run complete pseudo-Vegard analysis."""
    os.makedirs(fig_dir, exist_ok=True)

    print("=" * 70)
    print("Pseudo-Vegard's Law Analysis: B2 + L1_2 Composition Sweep")
    print("=" * 70)

    # Load data
    print("\nLoading compound data …")
    all_data = load_all_compounds(data_dir)
    print(f"  Total compounds: {len(all_data)}")

    # Find triplet pairs (3 compositions) for linearity analysis
    pairs = find_triplet_pairs(all_data)
    print(f"  Triplet pairs (25%/50%/75%): {len(pairs)}")

    # Find all plottable pairs (>= 2 compositions)
    all_pairs = find_all_plottable_pairs(all_data)
    print(f"  Plottable pairs (>= 2 compositions): {len(all_pairs)}")

    # Build analysis tables
    table = build_triplet_table(all_data, pairs)
    table_all = build_triplet_table(all_data, all_pairs)
    print(f"  Valid triplets (3 pts): {len(table)}")
    print(f"  Valid plottable (>= 2 pts): {len(table_all)}")

    # Linearity analysis for each metric
    print("\nAnalysing linearity …")
    results_a = analyse_linearity(table, "a")
    results_dnn = analyse_linearity(table, "d_nn")
    results_rws = analyse_linearity(table, "r_ws")

    for label, res in [("a (raw)", results_a),
                       ("d_nn", results_dnn),
                       ("r_ws", results_rws)]:
        print(f"  {label}: mean R² = {res['R2'].mean():.4f}, "
              f"median R² = {res['R2'].median():.4f}")

    # Save CSV
    table.to_csv(os.path.join(fig_dir, "pseudo_vegard_table.csv"), index=False)
    results_dnn.to_csv(os.path.join(fig_dir, "pseudo_vegard_dnn_results.csv"),
                       index=False)

    # Plots
    print("\nGenerating figures …")

    # 1. R² histogram (r_ws)
    plot_r2_histogram(results_rws, "$r_{\\mathrm{WS}}$",
                      os.path.join(fig_dir, "vegard_r2_histogram_rws.png"))

    # 2. B2 deviation
    results_dnn_sorted = results_dnn.sort_values("mid_residual")
    plot_mid_deviation(results_dnn_sorted, "d_nn", "\\AA",
                       os.path.join(fig_dir, "vegard_b2_deviation.png"))

    # 3. Three-metric comparison
    plot_three_metrics(table, os.path.join(fig_dir, "vegard_three_metrics.png"))

    # 6. Individual per-pair plots for ALL plottable pairs (r_ws)
    print("\nGenerating individual per-pair plots (all plottable pairs) …")
    plot_individual_vegard_separate(
        table_all, "r_ws",
        "$r_{\\mathrm{WS}}$ (\\AA)",
        os.path.join(fig_dir, "individual"))

    # 7. Summary grid of complete triplet pairs only (r_ws)
    print("\nGenerating triplet summary grid (complete 3-pt pairs only) …")
    plot_triplet_summary_grid(
        table_all, "r_ws",
        "$r_{\\mathrm{WS}}$ (\\AA)",
        os.path.join(fig_dir, "vegard_triplet_summary_r_ws.png"))

    # Report
    print("\nGenerating report …")
    report_lines = generate_report(table, results_a, results_dnn,
                                   results_rws, fig_dir)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\nReport saved: {report_path}")
    print("=" * 70)
    print("Analysis complete.")
    print("=" * 70)


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base, "data")
    fig_dir = os.path.join(base, "pseudo_vegard_output")
    report_path = os.path.join(fig_dir, "pseudo_vegard_report.md")

    run_analysis(data_dir, fig_dir, report_path)
