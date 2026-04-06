#!/usr/bin/env python3
"""
E_hull filter comparison study: with vs without energy_above_hull < 0.1 eV/atom criterion.

This script runs both filtered and unfiltered calculations in a single pass,
generates comparison figures, and outputs a comprehensive Markdown report.
"""

import os
import sys
import warnings
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

warnings.filterwarnings("ignore")

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

# --- Reuse constants from hea_radius_estimation.py ---
PAULING_RADII = {
    "H": 0.53, "Li": 1.55, "Be": 1.12, "B": 0.98, "C": 0.77, "N": 0.75, "O": 0.73,
    "Na": 1.90, "Mg": 1.60, "Al": 1.43, "Si": 1.17, "P": 1.10, "S": 1.04, "Cl": 0.99,
    "K": 2.35, "Ca": 1.97, "Sc": 1.64, "Ti": 1.47, "V": 1.35, "Cr": 1.29, "Mn": 1.37,
    "Fe": 1.26, "Co": 1.25, "Ni": 1.25, "Cu": 1.28, "Zn": 1.37, "Ga": 1.53, "Ge": 1.22,
    "As": 1.21, "Se": 1.17, "Br": 1.14, "Rb": 2.48, "Sr": 2.15, "Y": 1.82, "Zr": 1.60,
    "Nb": 1.47, "Mo": 1.40, "Tc": 1.35, "Ru": 1.34, "Rh": 1.34, "Pd": 1.37, "Ag": 1.44,
    "Cd": 1.52, "In": 1.67, "Sn": 1.58, "Sb": 1.61, "Te": 1.43, "I": 1.33, "Cs": 2.67,
    "Ba": 2.22, "La": 1.87, "Ce": 1.83, "Pr": 1.82, "Nd": 1.81, "Sm": 1.80, "Eu": 2.04,
    "Gd": 1.80, "Tb": 1.78, "Dy": 1.77, "Ho": 1.76, "Er": 1.75, "Tm": 1.74, "Yb": 1.93,
    "Lu": 1.74, "Hf": 1.59, "Ta": 1.47, "W": 1.41, "Re": 1.37, "Os": 1.35, "Ir": 1.36,
    "Pt": 1.39, "Au": 1.44, "Hg": 1.55, "Tl": 1.71, "Pb": 1.75, "Bi": 1.82, "Th": 1.80,
    "Pa": 1.63, "U": 1.54, "Pu": 1.64,
}


def extract_compounds(api_key: str, energy_above_hull_max: Optional[float] = None) -> pd.DataFrame:
    """Extract B2/L1$_2$ compounds from Materials Project."""
    from mp_api.client import MPRester

    mpr = MPRester(api_key)

    search_kwargs = {
        "spacegroup_number": 221,
        "num_elements": 2,
        "fields": [
            "material_id", "formula_pretty", "structure",
            "energy_per_atom", "energy_above_hull", "composition"
        ]
    }
    if energy_above_hull_max is not None:
        search_kwargs["energy_above_hull"] = (0, energy_above_hull_max)

    label = f"E_hull <= {energy_above_hull_max}" if energy_above_hull_max else "No filter"
    print(f"  [{label}] Querying Materials Project...")
    docs = mpr.materials.summary.search(**search_kwargs)
    print(f"  [{label}] Found {len(docs)} raw documents")

    data = []
    for doc in docs:
        try:
            structure = doc.structure
            lattice = structure.lattice
            a, b, c = lattice.a, lattice.b, lattice.c
            avg = (a + b + c) / 3
            if not (abs(a - avg) / avg < 0.01 and abs(b - avg) / avg < 0.01 and abs(c - avg) / avg < 0.01):
                continue

            composition = doc.composition
            elements = list(composition.elements)
            if len(elements) != 2:
                continue

            element_A = str(elements[0])
            element_B = str(elements[1])
            count_A = composition[elements[0]]
            count_B = composition[elements[1]]
            total = count_A + count_B
            ratio_A = count_A / total
            ratio_B = count_B / total

            tol = 0.05
            if abs(ratio_A - 0.5) < tol and abs(ratio_B - 0.5) < tol:
                stype = "B2"
            elif (abs(ratio_A - 0.75) < tol and abs(ratio_B - 0.25) < tol) or \
                 (abs(ratio_A - 0.25) < tol and abs(ratio_B - 0.75) < tol):
                stype = "L1$_2$"
            else:
                continue

            data.append({
                "material_id": str(doc.material_id),
                "formula": doc.formula_pretty,
                "structure_type": stype,
                "element_A": element_A,
                "element_B": element_B,
                "count_A": count_A,
                "count_B": count_B,
                "lattice_constant": lattice.a,
                "energy_per_atom": doc.energy_per_atom,
                "energy_above_hull": doc.energy_above_hull,
            })
        except Exception:
            continue

    df = pd.DataFrame(data)
    # Remove duplicates: keep lowest energy per composition+structure_type
    if len(df) > 0:
        df = df.sort_values("energy_per_atom").drop_duplicates(
            subset=["element_A", "element_B", "structure_type"], keep="first"
        ).reset_index(drop=True)
    print(f"  [{label}] {len(df)} valid B2/L1$_2$ compounds after dedup")
    return df


def calculate_radii_trf(df: pd.DataFrame, structure_type: Optional[str] = None) -> Tuple[Dict[str, float], Dict]:
    """Calculate effective atomic radii using TRF."""
    if structure_type:
        subset = df[df["structure_type"] == structure_type].copy()
    else:
        subset = df.copy()

    if len(subset) == 0:
        return {}, {"n_compounds": 0, "rmse": float("nan"), "mae": float("nan")}

    elements = sorted(set(subset["element_A"]) | set(subset["element_B"]))
    el2idx = {el: i for i, el in enumerate(elements)}
    x0 = np.array([PAULING_RADII.get(el, 1.4) for el in elements])

    def residuals(radii):
        res = []
        for _, row in subset.iterrows():
            a = row["lattice_constant"]
            iA = el2idx[row["element_A"]]
            iB = el2idx[row["element_B"]]
            if row["structure_type"] == "B2":
                res.append(radii[iA] + radii[iB] - (np.sqrt(3) / 2) * a)
            else:  # L1$_2$
                rA, rB = radii[iA], radii[iB]
                if row["count_A"] > row["count_B"]:
                    r_major, r_minor = rA, rB
                else:
                    r_major, r_minor = rB, rA
                res.append(2 * r_major - a / np.sqrt(2))
                res.append(r_major + r_minor - a / np.sqrt(2))
        return np.array(res)

    result = least_squares(residuals, x0, bounds=(0.5, 3.0), method="trf",
                           ftol=1e-10, xtol=1e-10, gtol=1e-10)
    radii = {el: result.x[el2idx[el]] for el in elements}
    rv = result.fun
    stats = {
        "n_compounds": len(subset),
        "n_elements": len(elements),
        "rmse": np.sqrt(np.mean(rv ** 2)),
        "mae": np.mean(np.abs(rv)),
    }
    return radii, stats


def compare_lattice_constants(df: pd.DataFrame, radii: Dict[str, float]) -> pd.DataFrame:
    """Compare DFT lattice constants vs calculated from radii."""
    rows = []
    for _, row in df.iterrows():
        eA, eB = row["element_A"], row["element_B"]
        if eA not in radii or eB not in radii:
            continue
        rA, rB = radii[eA], radii[eB]
        a_dft = row["lattice_constant"]
        if row["structure_type"] == "B2":
            a_calc = (2 / np.sqrt(3)) * (rA + rB)
        else:
            if row["count_A"] > row["count_B"]:
                r_major, r_minor = rA, rB
            else:
                r_major, r_minor = rB, rA
            a1 = 2 * r_major * np.sqrt(2)
            a2 = (r_major + r_minor) * np.sqrt(2)
            a_calc = (a1 + a2) / 2
        error = a_calc - a_dft
        rel_error = abs(error) / a_dft * 100
        rows.append({
            "formula": row["formula"],
            "structure_type": row["structure_type"],
            "a_dft": a_dft,
            "a_calc": a_calc,
            "error": error,
            "rel_error_pct": rel_error,
            "energy_above_hull": row["energy_above_hull"],
        })
    return pd.DataFrame(rows)


def run_analysis(api_key: str, fig_dir: str):
    """Run the full comparison analysis."""
    os.makedirs(fig_dir, exist_ok=True)

    # =========================================================================
    # Step 1: Extract data - both filtered and unfiltered
    # =========================================================================
    print("=" * 70)
    print("Step 1: Extracting data from Materials Project")
    print("=" * 70)

    df_filtered = extract_compounds(api_key, energy_above_hull_max=0.1)
    df_all = extract_compounds(api_key, energy_above_hull_max=None)

    # Save raw data
    df_filtered.to_csv(os.path.join(fig_dir, "compounds_filtered.csv"), index=False)
    df_all.to_csv(os.path.join(fig_dir, "compounds_all.csv"), index=False)

    n_filtered = len(df_filtered)
    n_all = len(df_all)
    n_unstable_only = n_all - n_filtered
    print(f"\n  Filtered (E_hull < 0.1): {n_filtered} compounds")
    print(f"  All (no filter):         {n_all} compounds")
    print(f"  Unstable-only additions: {n_unstable_only} compounds")

    # =========================================================================
    # Step 2: E_hull distribution
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 2: Analyzing E_hull distribution")
    print("=" * 70)

    ehull_all = df_all["energy_above_hull"].values

    # Stability categories
    n_stable = np.sum(ehull_all == 0)
    n_meta_low = np.sum((ehull_all > 0) & (ehull_all <= 0.05))
    n_meta_mid = np.sum((ehull_all > 0.05) & (ehull_all <= 0.1))
    n_meta_high = np.sum((ehull_all > 0.1) & (ehull_all <= 0.5))
    n_unstable = np.sum(ehull_all > 0.5)

    print(f"  E_hull = 0 (on hull):          {n_stable}")
    print(f"  0 < E_hull <= 0.05 eV/atom:    {n_meta_low}")
    print(f"  0.05 < E_hull <= 0.1 eV/atom:  {n_meta_mid}")
    print(f"  0.1 < E_hull <= 0.5 eV/atom:   {n_meta_high}")
    print(f"  E_hull > 0.5 eV/atom:          {n_unstable}")

    # Figure 1: E_hull distribution histogram
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].hist(ehull_all, bins=50, color="steelblue", edgecolor="black", alpha=0.8)
    axes[0].axvline(0.1, color="red", linestyle="--", linewidth=2, label="0.1 eV/atom threshold")
    axes[0].set_xlabel("$E_{\\mathrm{hull}}$ (eV/atom)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("(a) $E_{\\mathrm{hull}}$ Distribution (All)")
    axes[0].legend()

    # Zoomed view
    ehull_below_1 = ehull_all[ehull_all <= 1.0]
    axes[1].hist(ehull_below_1, bins=50, color="steelblue", edgecolor="black", alpha=0.8)
    axes[1].axvline(0.1, color="red", linestyle="--", linewidth=2, label="0.1 eV/atom threshold")
    axes[1].set_xlabel("$E_{\\mathrm{hull}}$ (eV/atom)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("(b) $E_{\\mathrm{hull}}$ Distribution ($\\leq$ 1.0 eV/atom)")
    axes[1].legend()

    plt.tight_layout()
    fig1_path = os.path.join(fig_dir, "fig1_ehull_distribution.png")
    plt.savefig(fig1_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig1_path}")

    # =========================================================================
    # Step 3: Calculate radii - filtered vs all
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 3: Calculating effective atomic radii")
    print("=" * 70)

    results = {}
    for label, df in [("filtered", df_filtered), ("all", df_all)]:
        print(f"\n  --- {label.upper()} dataset ---")
        r_combined, s_combined = calculate_radii_trf(df)
        r_b2, s_b2 = calculate_radii_trf(df, "B2")
        r_l12, s_l12 = calculate_radii_trf(df, "L1$_2$")

        print(f"  Combined: {s_combined['n_compounds']} cpds, "
              f"RMSE={s_combined['rmse']:.4f} A, MAE={s_combined['mae']:.4f} A")
        print(f"  B2:       {s_b2['n_compounds']} cpds, "
              f"RMSE={s_b2['rmse']:.4f} A, MAE={s_b2['mae']:.4f} A")
        print(f"  L1$_2$:    {s_l12['n_compounds']} cpds, "
              f"RMSE={s_l12['rmse']:.4f} A, MAE={s_l12['mae']:.4f} A")

        comp = compare_lattice_constants(df, r_combined)

        results[label] = {
            "df": df,
            "radii_combined": r_combined,
            "radii_b2": r_b2,
            "radii_l12": r_l12,
            "stats_combined": s_combined,
            "stats_b2": s_b2,
            "stats_l12": s_l12,
            "comparison": comp,
        }

    # =========================================================================
    # Step 4: Comparison figures
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 4: Generating comparison figures")
    print("=" * 70)

    # Figure 2: Parity plots side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, (label, color, title_suffix) in zip(
        axes,
        [("filtered", "steelblue", "$E_{\\mathrm{hull}} < 0.1$ eV/atom"),
         ("all", "darkorange", "No $E_{\\mathrm{hull}}$ filter")]
    ):
        comp = results[label]["comparison"]
        if len(comp) == 0:
            continue
        sc = ax.scatter(comp["a_dft"], comp["a_calc"], c=comp["energy_above_hull"],
                        cmap="RdYlGn_r", edgecolors="black", linewidth=0.5, alpha=0.8, s=50)
        lims = [min(comp["a_dft"].min(), comp["a_calc"].min()) - 0.2,
                max(comp["a_dft"].max(), comp["a_calc"].max()) + 0.2]
        ax.plot(lims, lims, "k--", linewidth=1.5, label="y = x")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("DFT Lattice Constant (A)")
        ax.set_ylabel("Calculated Lattice Constant (A)")
        rmse = np.sqrt(np.mean(comp["error"] ** 2))
        ax.set_title(f"{title_suffix}\nN={len(comp)}, RMSE={rmse:.4f} A")
        ax.legend(loc="upper left")
        plt.colorbar(sc, ax=ax, label="$E_{\\mathrm{hull}}$ (eV/atom)")

    plt.tight_layout()
    fig2_path = os.path.join(fig_dir, "fig2_parity_comparison.png")
    plt.savefig(fig2_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig2_path}")

    # Figure 3: Radius comparison between filtered and all
    common_els = sorted(
        set(results["filtered"]["radii_combined"].keys()) &
        set(results["all"]["radii_combined"].keys())
    )
    if len(common_els) > 0:
        r_filt = np.array([results["filtered"]["radii_combined"][el] for el in common_els])
        r_all_arr = np.array([results["all"]["radii_combined"][el] for el in common_els])
        diff = r_all_arr - r_filt

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Parity
        axes[0].scatter(r_filt, r_all_arr, c="steelblue", edgecolors="black", s=60, alpha=0.8)
        for i, el in enumerate(common_els):
            if abs(diff[i]) > 0.02:
                axes[0].annotate(el, (r_filt[i], r_all_arr[i]), fontsize=10,
                                 xytext=(5, 5), textcoords="offset points")
        lims = [min(r_filt.min(), r_all_arr.min()) - 0.1,
                max(r_filt.max(), r_all_arr.max()) + 0.1]
        axes[0].plot(lims, lims, "k--", linewidth=1.5)
        axes[0].set_xlabel("Radius (filtered) (A)")
        axes[0].set_ylabel("Radius (all) (A)")
        axes[0].set_title(f"(a) Radius: Filtered vs All (N={len(common_els)})")

        # Bar chart of differences
        sorted_idx = np.argsort(np.abs(diff))[::-1]
        top_n = min(25, len(common_els))
        top_idx = sorted_idx[:top_n]
        colors = ["red" if d > 0 else "blue" for d in diff[top_idx]]
        axes[1].barh([common_els[i] for i in top_idx], diff[top_idx], color=colors, alpha=0.7)
        axes[1].set_xlabel("$\\Delta r$ = $r_{\\mathrm{all}}$ - $r_{\\mathrm{filtered}}$ (A)")
        axes[1].set_title(f"(b) Top {top_n} Radius Changes")
        axes[1].axvline(0, color="black", linewidth=0.5)

        plt.tight_layout()
        fig3_path = os.path.join(fig_dir, "fig3_radius_comparison.png")
        plt.savefig(fig3_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fig3_path}")
    else:
        fig3_path = None

    # Figure 4: Error vs E_hull for the "all" dataset
    comp_all = results["all"]["comparison"]
    if len(comp_all) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        axes[0].scatter(comp_all["energy_above_hull"], comp_all["rel_error_pct"],
                        c="darkorange", edgecolors="black", alpha=0.6, s=40)
        axes[0].axvline(0.1, color="red", linestyle="--", linewidth=2, label="0.1 eV/atom")
        axes[0].set_xlabel("$E_{\\mathrm{hull}}$ (eV/atom)")
        axes[0].set_ylabel("Relative Error (%)")
        axes[0].set_title("(a) Lattice Constant Error vs $E_{\\mathrm{hull}}$")
        axes[0].legend()

        # Box plot by E_hull category
        bins = [0, 0.05, 0.1, 0.2, 0.5, comp_all["energy_above_hull"].max() + 0.01]
        labels_box = ["0-0.05", "0.05-0.1", "0.1-0.2", "0.2-0.5", ">0.5"]
        # Trim to only bins that have data
        actual_bins = []
        actual_labels = []
        actual_data = []
        for i in range(len(bins) - 1):
            mask = (comp_all["energy_above_hull"] >= bins[i]) & (comp_all["energy_above_hull"] < bins[i + 1])
            vals = comp_all.loc[mask, "rel_error_pct"].values
            if len(vals) > 0:
                actual_bins.append((bins[i], bins[i + 1]))
                actual_labels.append(labels_box[i])
                actual_data.append(vals)

        if len(actual_data) > 0:
            bp = axes[1].boxplot(actual_data, labels=actual_labels, patch_artist=True)
            colors_box = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#8e44ad"]
            for patch, color in zip(bp["boxes"], colors_box[:len(actual_data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
        axes[1].set_xlabel("$E_{\\mathrm{hull}}$ range (eV/atom)")
        axes[1].set_ylabel("Relative Error (%)")
        axes[1].set_title("(b) Error Distribution by Stability Category")

        plt.tight_layout()
        fig4_path = os.path.join(fig_dir, "fig4_error_vs_ehull.png")
        plt.savefig(fig4_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fig4_path}")
    else:
        fig4_path = None

    # Figure 5: Structure type breakdown
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (label, df) in zip(axes, [("Filtered", df_filtered), ("All", df_all)]):
        counts = df["structure_type"].value_counts()
        ax.bar(counts.index, counts.values, color=["steelblue", "darkorange"], alpha=0.8, edgecolor="black")
        ax.set_ylabel("Count")
        ax.set_title(f"{label}: N={len(df)}")
        for i, (idx, val) in enumerate(counts.items()):
            ax.text(i, val + 0.5, str(val), ha="center", fontsize=14, fontweight="bold")
    plt.suptitle("Structure Type Distribution", fontsize=20)
    plt.tight_layout()
    fig5_path = os.path.join(fig_dir, "fig5_structure_breakdown.png")
    plt.savefig(fig5_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig5_path}")

    # =========================================================================
    # Step 5: Compute summary statistics for report
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 5: Computing summary statistics")
    print("=" * 70)

    summary = {}
    for label in ["filtered", "all"]:
        comp = results[label]["comparison"]
        if len(comp) > 0:
            summary[label] = {
                "n_compounds": len(results[label]["df"]),
                "n_elements": results[label]["stats_combined"]["n_elements"],
                "rmse_radii": results[label]["stats_combined"]["rmse"],
                "mae_radii": results[label]["stats_combined"]["mae"],
                "rmse_lattice": np.sqrt(np.mean(comp["error"] ** 2)),
                "mae_lattice": np.mean(np.abs(comp["error"])),
                "mean_rel_error": comp["rel_error_pct"].mean(),
                "max_rel_error": comp["rel_error_pct"].max(),
                "median_rel_error": comp["rel_error_pct"].median(),
            }
        else:
            summary[label] = {"n_compounds": 0}

    for label in ["filtered", "all"]:
        s = summary[label]
        if s["n_compounds"] > 0:
            print(f"\n  [{label.upper()}]")
            print(f"    Compounds: {s['n_compounds']}, Elements: {s['n_elements']}")
            print(f"    Radii RMSE: {s['rmse_radii']:.4f} A, MAE: {s['mae_radii']:.4f} A")
            print(f"    Lattice RMSE: {s['rmse_lattice']:.4f} A, MAE: {s['mae_lattice']:.4f} A")
            print(f"    Rel Error: mean={s['mean_rel_error']:.2f}%, "
                  f"median={s['median_rel_error']:.2f}%, max={s['max_rel_error']:.2f}%")

    # Elements only in unfiltered
    els_filtered = set(results["filtered"]["radii_combined"].keys())
    els_all = set(results["all"]["radii_combined"].keys())
    new_elements = sorted(els_all - els_filtered)
    print(f"\n  New elements (only in unfiltered): {new_elements}")

    # =========================================================================
    # Step 6: Generate Markdown report
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 6: Generating Markdown report")
    print("=" * 70)

    report_lines = []
    report_lines.append("# $E_{\\mathrm{hull}}$ フィルタ除去の影響分析レポート\n")
    report_lines.append("## 1. 概要\n")
    report_lines.append("B2 / L1$_2$ 構造の二元系化合物について、Materials Project からのデータ抽出時に")
    report_lines.append("適用していた $E_{\\mathrm{hull}} < 0.1$ eV/atom フィルタを除去し、")
    report_lines.append("不安定な化合物も含めた計算結果を比較した。\n")

    report_lines.append("## 2. データセット比較\n")
    report_lines.append("| 項目 | フィルタあり ($E_{\\mathrm{hull}} < 0.1$) | フィルタなし (全データ) |")
    report_lines.append("|:---|:---:|:---:|")
    report_lines.append(f"| 化合物数 | {summary['filtered']['n_compounds']} | {summary['all']['n_compounds']} |")
    if summary["filtered"]["n_compounds"] > 0 and summary["all"]["n_compounds"] > 0:
        report_lines.append(f"| 元素数 | {summary['filtered']['n_elements']} | {summary['all']['n_elements']} |")
        report_lines.append(f"| 追加された化合物 | - | +{n_unstable_only} |")
    report_lines.append("")

    # E_hull distribution
    report_lines.append("### $E_{\\mathrm{hull}}$ 分布（全データ）\n")
    report_lines.append("| カテゴリ | 化合物数 | 割合 |")
    report_lines.append("|:---|:---:|:---:|")
    for cat_label, count in [
        ("$E_{\\mathrm{hull}} = 0$ (凸包上)", n_stable),
        ("$0 < E_{\\mathrm{hull}} \\leq 0.05$", n_meta_low),
        ("$0.05 < E_{\\mathrm{hull}} \\leq 0.1$", n_meta_mid),
        ("$0.1 < E_{\\mathrm{hull}} \\leq 0.5$", n_meta_high),
        ("$E_{\\mathrm{hull}} > 0.5$", n_unstable),
    ]:
        pct = count / n_all * 100 if n_all > 0 else 0
        report_lines.append(f"| {cat_label} | {count} | {pct:.1f}% |")
    report_lines.append("")

    report_lines.append(f"![E_hull分布](figures/fig1_ehull_distribution.png)\n")
    report_lines.append("**図1**: 左: 全データの $E_{\\mathrm{hull}}$ 分布。右: $\\leq$ 1.0 eV/atom の拡大図。赤破線は 0.1 eV/atom 閾値。\n")

    # Radii optimization
    report_lines.append("## 3. 有効原子半径の最適化結果\n")
    report_lines.append("| 指標 | フィルタあり | フィルタなし | 変化 |")
    report_lines.append("|:---|:---:|:---:|:---:|")
    if summary["filtered"]["n_compounds"] > 0 and summary["all"]["n_compounds"] > 0:
        for metric, key, unit in [
            ("半径 RMSE", "rmse_radii", "A"),
            ("半径 MAE", "mae_radii", "A"),
            ("格子定数 RMSE", "rmse_lattice", "A"),
            ("格子定数 MAE", "mae_lattice", "A"),
            ("平均相対誤差", "mean_rel_error", "%"),
            ("中央値相対誤差", "median_rel_error", "%"),
            ("最大相対誤差", "max_rel_error", "%"),
        ]:
            v_f = summary["filtered"][key]
            v_a = summary["all"][key]
            delta = v_a - v_f
            sign = "+" if delta >= 0 else ""
            report_lines.append(f"| {metric} | {v_f:.4f} {unit} | {v_a:.4f} {unit} | {sign}{delta:.4f} {unit} |")
    report_lines.append("")

    report_lines.append(f"![パリティプロット比較](figures/fig2_parity_comparison.png)\n")
    report_lines.append("**図2**: 格子定数のパリティプロット。左: フィルタあり、右: フィルタなし。色は $E_{\\mathrm{hull}}$ を示す。\n")

    if fig3_path:
        report_lines.append(f"![半径比較](figures/fig3_radius_comparison.png)\n")
        report_lines.append("**図3**: (a) フィルタあり/なしでの有効原子半径の比較。(b) 半径変化量の上位元素。\n")

    # Radius changes
    if len(common_els) > 0:
        report_lines.append("### 半径変化の大きい元素（上位10）\n")
        report_lines.append("| 元素 | $r_{\\mathrm{filtered}}$ (A) | $r_{\\mathrm{all}}$ (A) | $\\Delta r$ (A) |")
        report_lines.append("|:---:|:---:|:---:|:---:|")
        diffs_list = [(el, results["filtered"]["radii_combined"][el],
                        results["all"]["radii_combined"][el],
                        results["all"]["radii_combined"][el] - results["filtered"]["radii_combined"][el])
                       for el in common_els]
        diffs_list.sort(key=lambda x: abs(x[3]), reverse=True)
        for el, rf, ra, d in diffs_list[:10]:
            sign = "+" if d >= 0 else ""
            report_lines.append(f"| {el} | {rf:.4f} | {ra:.4f} | {sign}{d:.4f} |")
        report_lines.append("")

    if len(new_elements) > 0:
        report_lines.append(f"### フィルタ除去で新たに含まれた元素\n")
        report_lines.append(f"{', '.join(new_elements)}\n")

    # Error vs stability
    report_lines.append("## 4. 安定性と予測精度の関係\n")
    if fig4_path:
        report_lines.append(f"![誤差 vs E_hull](figures/fig4_error_vs_ehull.png)\n")
        report_lines.append("**図4**: (a) 格子定数の相対誤差 vs $E_{\\mathrm{hull}}$。(b) 安定性カテゴリ別の誤差分布。\n")

    # Error by stability category
    if len(comp_all) > 0:
        report_lines.append("| $E_{\\mathrm{hull}}$ 範囲 | 化合物数 | 平均誤差 (%) | RMSE (A) |")
        report_lines.append("|:---|:---:|:---:|:---:|")
        for low, high, cat_label in [
            (0, 0.05, "$0 - 0.05$"),
            (0.05, 0.1, "$0.05 - 0.1$"),
            (0.1, 0.2, "$0.1 - 0.2$"),
            (0.2, 0.5, "$0.2 - 0.5$"),
            (0.5, 999, "$> 0.5$"),
        ]:
            mask = (comp_all["energy_above_hull"] >= low) & (comp_all["energy_above_hull"] < high)
            subset = comp_all[mask]
            if len(subset) > 0:
                mean_err = subset["rel_error_pct"].mean()
                rmse_lat = np.sqrt(np.mean(subset["error"] ** 2))
                report_lines.append(f"| {cat_label} | {len(subset)} | {mean_err:.2f} | {rmse_lat:.4f} |")
        report_lines.append("")

    report_lines.append(f"![構造タイプ](figures/fig5_structure_breakdown.png)\n")
    report_lines.append("**図5**: 構造タイプ（B2 / L1$_2$）の分布比較。\n")

    # Conclusion
    report_lines.append("## 5. 結論\n")
    if summary["filtered"]["n_compounds"] > 0 and summary["all"]["n_compounds"] > 0:
        delta_rmse = summary["all"]["rmse_lattice"] - summary["filtered"]["rmse_lattice"]
        delta_mean = summary["all"]["mean_rel_error"] - summary["filtered"]["mean_rel_error"]
        if delta_rmse > 0:
            report_lines.append(f"- フィルタ除去により格子定数 RMSE は {delta_rmse:.4f} A 増加した。")
            report_lines.append(f"  これは不安定な化合物の格子定数が硬球モデルからより大きく逸脱するためと考えられる。")
        else:
            report_lines.append(f"- フィルタ除去により格子定数 RMSE は {abs(delta_rmse):.4f} A 改善した。")
            report_lines.append(f"  データ数の増加による統計的ロバスト性の向上が寄与していると考えられる。")

        report_lines.append(f"- 平均相対誤差は {summary['filtered']['mean_rel_error']:.2f}% → "
                          f"{summary['all']['mean_rel_error']:.2f}% に変化（{'+' if delta_mean >= 0 else ''}{delta_mean:.2f} pp）。")
        report_lines.append(f"- 化合物数は {n_filtered} → {n_all}（+{n_unstable_only}）に増加。")
        if len(new_elements) > 0:
            report_lines.append(f"- フィルタ除去により新たに {len(new_elements)} 元素のデータが利用可能になった。")
        report_lines.append("")
        report_lines.append("### 推奨事項\n")
        report_lines.append("1. **初期スクリーニング**: $E_{\\mathrm{hull}} < 0.1$ eV/atom は依然として妥当な初期フィルタだが、")
        report_lines.append("   化学系に応じた閾値調整が望ましい。")
        report_lines.append("2. **不安定相の活用**: 不安定な化合物のデータを含めることで、")
        report_lines.append("   元素被覆率（element coverage）が向上し、HEA設計に有用な情報が得られる。")
        report_lines.append("3. **品質管理**: $E_{\\mathrm{hull}} > 0.5$ eV/atom の化合物は予測精度が低下する傾向があるため、")
        report_lines.append("   重み付けや外れ値処理の導入を検討すべき。")

    report_text = "\n".join(report_lines)
    report_path = os.path.join(fig_dir, "ehull_comparison_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"  Report saved to: {report_path}")

    # Save radii tables
    for label in ["filtered", "all"]:
        radii = results[label]["radii_combined"]
        if radii:
            rows = []
            for el in sorted(radii.keys()):
                rows.append({
                    "element": el,
                    "radius_combined": radii[el],
                    "pauling_radius": PAULING_RADII.get(el, float("nan")),
                })
            pd.DataFrame(rows).to_csv(
                os.path.join(fig_dir, f"radii_{label}.csv"), index=False
            )

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)

    return results, summary


if __name__ == "__main__":
    api_key = os.environ.get("MP_API_KEY", "")
    if not api_key:
        if len(sys.argv) > 1:
            api_key = sys.argv[1]
        else:
            print("Please provide MP_API_KEY via environment variable or argument")
            sys.exit(1)

    fig_dir = sys.argv[2] if len(sys.argv) > 2 else "ehull_comparison_output/figures"
    run_analysis(api_key, fig_dir)
