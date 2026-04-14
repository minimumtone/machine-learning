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

        # Skip if incomplete
        if all(f"a_{x}" in row for x in ["25", "50", "75"]):
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
    lines.append("\n![Pseudo-Vegard grid (d_nn)](vegard_grid_d_nn.png)\n")
    lines.append("![R2 histogram](vegard_r2_histogram.png)\n")
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

    # Find triplet pairs
    pairs = find_triplet_pairs(all_data)
    print(f"  Triplet pairs (25%/50%/75%): {len(pairs)}")

    # Build analysis table
    table = build_triplet_table(all_data, pairs)
    print(f"  Valid triplets: {len(table)}")

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

    # 1. Individual pseudo-Vegard diagrams (d_nn)
    plot_individual_vegard(
        table, "d_nn",
        "$d_{\\mathrm{nn}}$ (\\AA)",
        "Pseudo-Vegard: Nearest-Neighbour Distance",
        os.path.join(fig_dir, "vegard_grid_d_nn.png"))

    # 2. Individual pseudo-Vegard diagrams (raw a)
    plot_individual_vegard(
        table, "a",
        "$a$ (\\AA)",
        "Pseudo-Vegard: Raw Lattice Constant",
        os.path.join(fig_dir, "vegard_grid_a_raw.png"))

    # 3. R² histograms
    plot_r2_histogram(results_dnn, "$d_{\\mathrm{nn}}$",
                      os.path.join(fig_dir, "vegard_r2_histogram.png"))

    # 4. B2 deviation
    results_dnn_sorted = results_dnn.sort_values("mid_residual")
    plot_mid_deviation(results_dnn_sorted, "d_nn", "\\AA",
                       os.path.join(fig_dir, "vegard_b2_deviation.png"))

    # 5. Three-metric comparison
    plot_three_metrics(table, os.path.join(fig_dir, "vegard_three_metrics.png"))

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
