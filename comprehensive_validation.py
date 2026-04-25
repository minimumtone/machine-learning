#!/usr/bin/env python3
"""
Comprehensive Validation: r_WS vs Contact Radii, Bootstrap Uncertainty, DFT Comparison
=======================================================================================

This script performs all validation analyses in a single execution pass:

1. r_WS bootstrap uncertainty estimation (per-element)
2. DFT lattice constant → r_WS vs optimised contact radii comparison
3. Periodic table: r_WS vs contact radii comparison figures
4. Structure dependence: B2 vs L1₂ r_WS difference (with corrected formula)
5. Parity plots: r_WS(DFT) vs r_contact for all 4 cases
6. Element-group analysis of r_WS vs contact radii discrepancy

r_WS is defined as the proper Wigner-Seitz radius:
    r_WS = (3 V_atom / 4π)^{1/3}
where V_atom = a^3 / Z  (Z = atoms per unit cell: B2→2, L1₂→4).
"""

import os
import sys
from typing import Dict, List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Presentation-quality font sizes
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
# Constants and helpers
# =====================================================================

DATA_DIR = "data"
FIG_DIR = "validation_output"

# Element group classification
ELEMENT_GROUPS = {
    "3d": ["Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"],
    "4d": ["Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd"],
    "5d": ["Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg"],
    "Lanthanide": ["La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
                   "Tb", "Dy", "Ho", "Er", "Tm", "Yb"],
    "Actinide": ["Ac", "Th", "Pa", "U", "Np", "Pu", "Am"],
    "p-block": ["Al", "Ga", "Ge", "In", "Sn", "Tl", "Pb", "Bi", "Si", "As", "Sb", "Te"],
    "s-block": ["Be", "Mg", "Ca", "Sr", "Ba"],
}

def element_group(el: str) -> str:
    for grp, members in ELEMENT_GROUPS.items():
        if el in members:
            return grp
    return "Other"


def r_ws_from_V(V_atom: float) -> float:
    """Wigner-Seitz radius from atomic volume."""
    return (3 * V_atom / (4 * np.pi)) ** (1.0 / 3)


def V_atom_from_a(a: float, struct: str) -> float:
    """Atomic volume from lattice constant."""
    if "B2" in struct:
        return a ** 3 / 2
    else:  # L12
        return a ** 3 / 4


def r_ws_from_a(a: float, struct: str) -> float:
    """Wigner-Seitz radius from lattice constant."""
    return r_ws_from_V(V_atom_from_a(a, struct))


def r_ws_from_radii_B2(rA: float, rB: float) -> float:
    """r_WS for B2 compound from optimised contact radii via max-contact model."""
    a_pred = max(2 * rA, 2 * rB, (2 / np.sqrt(3)) * (rA + rB))
    return r_ws_from_a(a_pred, "B2")


def r_ws_from_radii_L12(r_major: float, r_minor: float) -> float:
    """r_WS for L1₂ compound from optimised contact radii via max-contact model."""
    a_pred = max(2 * np.sqrt(2) * r_major,
                 np.sqrt(2) * (r_major + r_minor),
                 2 * r_minor)
    return r_ws_from_a(a_pred, "L12")


# =====================================================================
# Data loading
# =====================================================================

def load_radii(path: str) -> Dict[str, float]:
    """Load element → radius mapping from CSV."""
    df = pd.read_csv(path)
    return dict(zip(df["element"], df["radius"]))


def load_compounds(path: str) -> pd.DataFrame:
    """Load compound data with lattice constants."""
    return pd.read_csv(path)


# =====================================================================
# 1. Per-element r_WS from DFT data
# =====================================================================

def compute_element_rws(compounds_df: pd.DataFrame, struct: str) -> Dict[str, List[float]]:
    """For each element, collect r_WS values from all compounds containing it.
    
    For a compound AB with lattice constant a:
      V_atom = a^3 / Z
      r_WS = (3 V_atom / 4π)^{1/3}
    
    This r_WS is the *compound-averaged* atomic volume radius, not
    element-specific. We assign it to both elements in the compound.
    """
    element_rws: Dict[str, List[float]] = {}
    for _, row in compounds_df.iterrows():
        a = row["lattice_constant"]
        rws = r_ws_from_a(a, struct)
        for el_col in ["element_A", "element_B"]:
            el = row[el_col]
            if el not in element_rws:
                element_rws[el] = []
            element_rws[el].append(rws)
    return element_rws


def compute_element_rws_weighted(compounds_df: pd.DataFrame, struct: str) -> Dict[str, List[float]]:
    """For each element, collect r_WS weighted by its fraction in the compound.
    
    For B2 (AB): each element has 50% → r_WS is equal for both.
    For L1₂ (A₃B): majority element gets r_WS at 75%, minority at 25%.
    Since V_atom is per-atom average, we just assign r_WS to each element.
    """
    return compute_element_rws(compounds_df, struct)


def element_mean_rws(element_rws: Dict[str, List[float]]) -> Dict[str, float]:
    """Mean r_WS per element."""
    return {el: float(np.mean(vals)) for el, vals in element_rws.items()}


def element_std_rws(element_rws: Dict[str, List[float]]) -> Dict[str, float]:
    """Std of r_WS per element."""
    return {el: float(np.std(vals)) if len(vals) > 1 else 0.0
            for el, vals in element_rws.items()}


# =====================================================================
# 2. Bootstrap uncertainty for r_WS
# =====================================================================

def bootstrap_rws(compounds_df: pd.DataFrame, struct: str,
                  n_bootstrap: int = 1000, seed: int = 42) -> Dict:
    """Bootstrap uncertainty estimation for per-element mean r_WS.
    
    Resamples compounds with replacement, computes per-element mean r_WS
    for each resample, and returns mean, std, and 95% CI per element.
    """
    rng = np.random.RandomState(seed)
    n = len(compounds_df)
    
    # Collect all elements
    all_elements = sorted(set(compounds_df["element_A"]) | set(compounds_df["element_B"]))
    
    boot_means: Dict[str, List[float]] = {el: [] for el in all_elements}
    
    for _ in range(n_bootstrap):
        # Resample compounds with replacement
        idx = rng.randint(0, n, size=n)
        sample = compounds_df.iloc[idx]
        
        # Compute per-element r_WS for this sample
        el_rws = compute_element_rws(sample, struct)
        for el in all_elements:
            if el in el_rws and len(el_rws[el]) > 0:
                boot_means[el].append(np.mean(el_rws[el]))
    
    result = {
        "mean": {}, "std": {}, "ci_lower": {}, "ci_upper": {},
        "n_compounds": {}, "n_success": n_bootstrap
    }
    for el in all_elements:
        vals = np.array(boot_means[el])
        if len(vals) > 0:
            result["mean"][el] = float(np.mean(vals))
            result["std"][el] = float(np.std(vals))
            result["ci_lower"][el] = float(np.percentile(vals, 2.5))
            result["ci_upper"][el] = float(np.percentile(vals, 97.5))
            result["n_compounds"][el] = len([1 for _, r in compounds_df.iterrows()
                                             if r["element_A"] == el or r["element_B"] == el])
    return result


# =====================================================================
# 3. r_WS from contact radii (predicted lattice constant)
# =====================================================================

def compute_rws_from_contact_radii(compounds_df: pd.DataFrame,
                                    radii: Dict[str, float],
                                    struct: str) -> Dict[str, List[float]]:
    """Compute r_WS using predicted lattice constant from contact radii."""
    element_rws: Dict[str, List[float]] = {}
    for _, row in compounds_df.iterrows():
        eA, eB = row["element_A"], row["element_B"]
        if eA not in radii or eB not in radii:
            continue
        rA, rB = radii[eA], radii[eB]
        
        if "B2" in struct:
            a_pred = max(2 * rA, 2 * rB, (2 / np.sqrt(3)) * (rA + rB))
        else:  # L12
            cA, cB = float(row["count_A"]), float(row["count_B"])
            if cA > cB:
                r_major, r_minor = rA, rB
            else:
                r_major, r_minor = rB, rA
            a_pred = max(2 * np.sqrt(2) * r_major,
                         np.sqrt(2) * (r_major + r_minor),
                         2 * r_minor)
        
        rws = r_ws_from_a(a_pred, struct)
        for el in [eA, eB]:
            if el not in element_rws:
                element_rws[el] = []
            element_rws[el].append(rws)
    return element_rws


# =====================================================================
# 4. Compound-level parity: r_WS(DFT) vs r_WS(predicted)
# =====================================================================

def compound_rws_parity(compounds_df: pd.DataFrame,
                        radii: Dict[str, float],
                        struct: str) -> pd.DataFrame:
    """For each compound, compute r_WS from DFT a and from predicted a."""
    rows = []
    for _, row in compounds_df.iterrows():
        eA, eB = row["element_A"], row["element_B"]
        if eA not in radii or eB not in radii:
            continue
        rA, rB = radii[eA], radii[eB]
        
        a_dft = row["lattice_constant"]
        if "B2" in struct:
            a_pred = max(2 * rA, 2 * rB, (2 / np.sqrt(3)) * (rA + rB))
        else:
            cA, cB = float(row["count_A"]), float(row["count_B"])
            if cA > cB:
                r_major, r_minor = rA, rB
            else:
                r_major, r_minor = rB, rA
            a_pred = max(2 * np.sqrt(2) * r_major,
                         np.sqrt(2) * (r_major + r_minor),
                         2 * r_minor)
        
        rws_dft = r_ws_from_a(a_dft, struct)
        rws_pred = r_ws_from_a(a_pred, struct)
        rows.append({
            "formula": row.get("formula", f"{eA}{eB}"),
            "element_A": eA, "element_B": eB,
            "a_dft": a_dft, "a_pred": a_pred,
            "rws_dft": rws_dft, "rws_pred": rws_pred,
            "rws_diff": rws_pred - rws_dft,
            "rws_rel_err": (rws_pred - rws_dft) / rws_dft * 100,
            "a_rel_err": (a_pred - a_dft) / a_dft * 100,
        })
    return pd.DataFrame(rows)


# =====================================================================
# 5. Plotting functions
# =====================================================================

def plot_rws_vs_contact_parity(mean_rws_dft: Dict[str, float],
                                radii: Dict[str, float],
                                title: str, outpath: str):
    """Scatter plot: element-average r_WS(DFT) vs contact radius.
    Elements deviating >2σ from the mean ratio are highlighted."""
    common = sorted(set(mean_rws_dft.keys()) & set(radii.keys()))
    if not common:
        return
    
    rws_vals = [mean_rws_dft[el] for el in common]
    contact_vals = [radii[el] for el in common]
    ratios = [rc / rws for rc, rws in zip(contact_vals, rws_vals)]
    
    mean_ratio = np.mean(ratios)
    std_ratio = np.std(ratios)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Colour by element group
    colors = {"3d": "tab:red", "4d": "tab:blue", "5d": "tab:green",
              "Lanthanide": "tab:orange", "Actinide": "tab:purple",
              "p-block": "tab:brown", "s-block": "tab:pink", "Other": "gray"}
    
    for el, rws, rc, ratio in zip(common, rws_vals, contact_vals, ratios):
        grp = element_group(el)
        is_outlier = abs(ratio - mean_ratio) > 2 * std_ratio
        marker_size = 80 if is_outlier else 50
        edgecolor = "black" if is_outlier else "none"
        lw = 1.5 if is_outlier else 0
        ax.scatter(rws, rc, c=colors.get(grp, "gray"), s=marker_size,
                   zorder=3 + (1 if is_outlier else 0), alpha=0.8,
                   edgecolors=edgecolor, linewidths=lw)
        # Label outliers in bold, others in normal
        if is_outlier:
            ax.annotate(el, (rws, rc), fontsize=10, ha="left", va="bottom",
                        fontweight="bold", color="red",
                        xytext=(4, 4), textcoords="offset points")
        else:
            ax.annotate(el, (rws, rc), fontsize=7, ha="left", va="bottom",
                        xytext=(2, 2), textcoords="offset points")
    
    # Diagonal
    all_vals = rws_vals + contact_vals
    vmin, vmax = min(all_vals) * 0.95, max(all_vals) * 1.05
    ax.plot([vmin, vmax], [vmin, vmax], "k--", lw=1, alpha=0.5,
            label="$r_\\mathrm{WS} = r_\\mathrm{contact}$")
    
    # Statistics
    rws_arr = np.array(rws_vals)
    rc_arr = np.array(contact_vals)
    rmse = np.sqrt(np.mean((rws_arr - rc_arr) ** 2))
    r2 = 1 - np.sum((rc_arr - rws_arr) ** 2) / np.sum((rc_arr - np.mean(rc_arr)) ** 2)
    
    ax.set_xlabel("$r_\\mathrm{WS}$ from DFT (Å)")
    ax.set_ylabel("Optimised contact radius (Å)")
    ax.set_title(title)
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_aspect("equal")
    
    # Legend for groups
    for grp, c in colors.items():
        ax.scatter([], [], c=c, s=50, label=grp)
    ax.legend(fontsize=9, loc="upper left")
    
    n_outliers = sum(1 for r in ratios if abs(r - mean_ratio) > 2 * std_ratio)
    ax.text(0.98, 0.05,
            f"RMSE = {rmse:.3f} Å\n$R^2$ = {r2:.3f}\n"
            f"$\\langle r_c / r_{{WS}} \\rangle$ = {mean_ratio:.3f} ± {std_ratio:.3f}\n"
            f"Outliers (>2σ): {n_outliers}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=12,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")
    return {"rmse": rmse, "r2": r2, "mean_ratio": mean_ratio,
            "std_ratio": std_ratio, "n": len(common), "n_outliers": n_outliers}


def plot_compound_rws_parity(parity_df: pd.DataFrame, title: str, outpath: str):
    """Parity plot: r_WS(DFT) vs r_WS(predicted) at compound level.
    Outliers (|rel_err| > 2*std) are labelled with compound formula."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Identify outliers: |rws_rel_err| > 2*std
    rel_err = parity_df["rws_rel_err"].abs()
    threshold = rel_err.mean() + 2 * rel_err.std()
    is_outlier = rel_err > threshold
    
    # Plot normal points
    normal = parity_df[~is_outlier]
    ax.scatter(normal["rws_dft"], normal["rws_pred"],
               s=10, alpha=0.3, c="steelblue")
    
    # Plot outliers with labels
    outliers = parity_df[is_outlier]
    ax.scatter(outliers["rws_dft"], outliers["rws_pred"],
               s=30, alpha=0.8, c="red", zorder=5, edgecolors="black", linewidths=0.5)
    for _, row in outliers.iterrows():
        ax.annotate(row["formula"], (row["rws_dft"], row["rws_pred"]),
                    fontsize=7, ha="left", va="bottom", color="red",
                    xytext=(3, 3), textcoords="offset points")
    
    vmin = min(parity_df["rws_dft"].min(), parity_df["rws_pred"].min()) * 0.95
    vmax = max(parity_df["rws_dft"].max(), parity_df["rws_pred"].max()) * 1.05
    ax.plot([vmin, vmax], [vmin, vmax], "k--", lw=1, alpha=0.5)
    
    rmse = np.sqrt(np.mean(parity_df["rws_diff"] ** 2))
    mae = np.mean(np.abs(parity_df["rws_diff"]))
    r2 = 1 - np.sum(parity_df["rws_diff"] ** 2) / np.sum(
        (parity_df["rws_dft"] - parity_df["rws_dft"].mean()) ** 2)
    
    ax.set_xlabel("$r_\\mathrm{WS}$ from DFT lattice constant (Å)")
    ax.set_ylabel("$r_\\mathrm{WS}$ from optimised radii (Å)")
    ax.set_title(title)
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_aspect("equal")
    
    ax.text(0.98, 0.05,
            f"N = {len(parity_df)}\nRMSE = {rmse:.4f} Å\nMAE = {mae:.4f} Å\n$R^2$ = {r2:.4f}\n"
            f"Outliers (red): {len(outliers)} (>{threshold:.1f}% rel.err.)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=12,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")
    return {"rmse": rmse, "mae": mae, "r2": r2, "n": len(parity_df),
            "n_outliers": len(outliers), "outlier_threshold": threshold}


def plot_bootstrap_uncertainty(boot_result: Dict, title: str, outpath: str):
    """Bar chart of per-element r_WS with bootstrap 95% CI."""
    elements = sorted(boot_result["mean"].keys(),
                      key=lambda e: boot_result["mean"][e])
    
    means = [boot_result["mean"][el] for el in elements]
    ci_lo = [boot_result["ci_lower"][el] for el in elements]
    ci_hi = [boot_result["ci_upper"][el] for el in elements]
    stds = [boot_result["std"][el] for el in elements]
    
    err_lo = [max(0, m - lo) for m, lo in zip(means, ci_lo)]
    err_hi = [max(0, hi - m) for m, hi in zip(means, ci_hi)]
    
    fig, ax = plt.subplots(figsize=(20, 8))
    
    colors = []
    color_map = {"3d": "tab:red", "4d": "tab:blue", "5d": "tab:green",
                 "Lanthanide": "tab:orange", "Actinide": "tab:purple",
                 "p-block": "tab:brown", "s-block": "tab:pink", "Other": "gray"}
    for el in elements:
        colors.append(color_map.get(element_group(el), "gray"))
    
    ax.bar(range(len(elements)), means, color=colors, alpha=0.7)
    ax.errorbar(range(len(elements)), means, yerr=[err_lo, err_hi],
                fmt="none", ecolor="black", capsize=2, lw=1)
    
    ax.set_xticks(range(len(elements)))
    ax.set_xticklabels(elements, rotation=90, fontsize=10)
    ax.set_ylabel("$r_\\mathrm{WS}$ (Å)")
    ax.set_title(title)
    
    # Legend
    for grp, c in color_map.items():
        ax.bar([], [], color=c, alpha=0.7, label=grp)
    ax.legend(fontsize=10, loc="upper left", ncol=2)
    
    # Stats annotation
    med_std = np.median(stds)
    max_std = max(stds)
    max_std_el = elements[stds.index(max_std)]
    ci_widths = [hi - lo for lo, hi in zip(ci_lo, ci_hi)]
    avg_ci = np.mean(ci_widths)
    
    ax.text(0.98, 0.95,
            f"Median std = {med_std:.4f} Å\nMax std = {max_std:.4f} Å ({max_std_el})\nAvg 95% CI width = {avg_ci:.4f} Å",
            transform=ax.transAxes, ha="right", va="top", fontsize=12,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")
    return {"med_std": med_std, "max_std": max_std, "max_std_el": max_std_el,
            "avg_ci_width": avg_ci}


def plot_rws_contact_ratio_periodic(mean_rws: Dict[str, float],
                                     radii: Dict[str, float],
                                     title: str, outpath: str):
    """Periodic-table-style heatmap of r_contact / r_WS ratio."""
    # Periodic table layout (row, col) for common elements
    PT_POS = {
        "H": (0, 0), "He": (0, 17),
        "Li": (1, 0), "Be": (1, 1), "B": (1, 12), "C": (1, 13), "N": (1, 14), "O": (1, 15), "F": (1, 16), "Ne": (1, 17),
        "Na": (2, 0), "Mg": (2, 1), "Al": (2, 12), "Si": (2, 13), "P": (2, 14), "S": (2, 15), "Cl": (2, 16), "Ar": (2, 17),
        "K": (3, 0), "Ca": (3, 1), "Sc": (3, 2), "Ti": (3, 3), "V": (3, 4), "Cr": (3, 5), "Mn": (3, 6),
        "Fe": (3, 7), "Co": (3, 8), "Ni": (3, 9), "Cu": (3, 10), "Zn": (3, 11),
        "Ga": (3, 12), "Ge": (3, 13), "As": (3, 14), "Se": (3, 15), "Br": (3, 16), "Kr": (3, 17),
        "Rb": (4, 0), "Sr": (4, 1), "Y": (4, 2), "Zr": (4, 3), "Nb": (4, 4), "Mo": (4, 5), "Tc": (4, 6),
        "Ru": (4, 7), "Rh": (4, 8), "Pd": (4, 9), "Ag": (4, 10), "Cd": (4, 11),
        "In": (4, 12), "Sn": (4, 13), "Sb": (4, 14), "Te": (4, 15), "I": (4, 16), "Xe": (4, 17),
        "Cs": (5, 0), "Ba": (5, 1), "La": (8, 2), "Hf": (5, 3), "Ta": (5, 4), "W": (5, 5), "Re": (5, 6),
        "Os": (5, 7), "Ir": (5, 8), "Pt": (5, 9), "Au": (5, 10), "Hg": (5, 11),
        "Tl": (5, 12), "Pb": (5, 13), "Bi": (5, 14),
        "Fr": (6, 0), "Ra": (6, 1), "Ac": (9, 2),
        # Lanthanides (row 8)
        "Ce": (8, 3), "Pr": (8, 4), "Nd": (8, 5), "Pm": (8, 6), "Sm": (8, 7),
        "Eu": (8, 8), "Gd": (8, 9), "Tb": (8, 10), "Dy": (8, 11), "Ho": (8, 12),
        "Er": (8, 13), "Tm": (8, 14), "Yb": (8, 15), "Lu": (5, 2),
        # Actinides (row 9)
        "Th": (9, 3), "Pa": (9, 4), "U": (9, 5), "Np": (9, 6), "Pu": (9, 7),
        "Am": (9, 8),
    }
    
    common = sorted(set(mean_rws.keys()) & set(radii.keys()))
    if not common:
        return
    
    ratios = {el: radii[el] / mean_rws[el] for el in common}
    
    fig, ax = plt.subplots(figsize=(20, 10))
    
    vmin_r = min(ratios.values())
    vmax_r = max(ratios.values())
    cmap = plt.cm.RdYlBu_r
    norm = plt.Normalize(vmin=vmin_r, vmax=vmax_r)
    
    for el in common:
        if el not in PT_POS:
            continue
        row, col = PT_POS[el]
        ratio = ratios[el]
        color = cmap(norm(ratio))
        
        rect = plt.Rectangle((col, -row), 0.9, 0.9, facecolor=color,
                              edgecolor="black", linewidth=0.5)
        ax.add_patch(rect)
        ax.text(col + 0.45, -row + 0.6, el, ha="center", va="center",
                fontsize=10, fontweight="bold")
        ax.text(col + 0.45, -row + 0.25, f"{ratio:.3f}", ha="center",
                va="center", fontsize=7)
    
    # Grey cells for elements with no data
    for el, (row, col) in PT_POS.items():
        if el not in common:
            rect = plt.Rectangle((col, -row), 0.9, 0.9, facecolor="lightgray",
                                  edgecolor="black", linewidth=0.5, alpha=0.3)
            ax.add_patch(rect)
            ax.text(col + 0.45, -row + 0.45, el, ha="center", va="center",
                    fontsize=8, color="gray")
    
    ax.set_xlim(-0.5, 18.5)
    ax.set_ylim(-10.5, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=18, pad=20)
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=20, pad=0.02)
    cbar.set_label("$r_\\mathrm{contact} / r_\\mathrm{WS}$", fontsize=14)
    
    # Stats
    ratio_arr = np.array(list(ratios.values()))
    ax.text(0.02, 0.02,
            f"Mean ratio = {np.mean(ratio_arr):.3f} ± {np.std(ratio_arr):.3f}\n"
            f"Range = {vmin_r:.3f} – {vmax_r:.3f}\n"
            f"N = {len(common)} elements",
            transform=ax.transAxes, fontsize=12, va="bottom",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")
    return {"mean_ratio": float(np.mean(ratio_arr)),
            "std_ratio": float(np.std(ratio_arr)),
            "min_ratio": float(vmin_r), "max_ratio": float(vmax_r)}


def plot_rws_B2_vs_L12(rws_B2: Dict[str, float], rws_L12: Dict[str, float],
                        title: str, outpath: str):
    """Scatter: r_WS(B2) vs r_WS(L1₂) per element."""
    common = sorted(set(rws_B2.keys()) & set(rws_L12.keys()))
    if not common:
        return
    
    b2_vals = [rws_B2[el] for el in common]
    l12_vals = [rws_L12[el] for el in common]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    color_map = {"3d": "tab:red", "4d": "tab:blue", "5d": "tab:green",
                 "Lanthanide": "tab:orange", "Actinide": "tab:purple",
                 "p-block": "tab:brown", "s-block": "tab:pink", "Other": "gray"}
    
    # Detect outliers: |diff_pct| > mean + 2*std
    diff_pct_all = [(b - l) / l * 100 for b, l in zip(b2_vals, l12_vals)]
    mean_dp = np.mean(np.abs(diff_pct_all))
    std_dp = np.std(np.abs(diff_pct_all))
    outlier_thresh = mean_dp + 2 * std_dp
    
    for el, b2, l12 in zip(common, b2_vals, l12_vals):
        grp = element_group(el)
        dp = abs((b2 - l12) / l12 * 100)
        is_outlier = dp > outlier_thresh
        sz = 120 if is_outlier else 50
        ec = "black" if is_outlier else "none"
        lw = 2 if is_outlier else 0.5
        ax.scatter(l12, b2, c=color_map.get(grp, "gray"), s=sz, zorder=3,
                   alpha=0.8, edgecolors=ec, linewidths=lw)
        if is_outlier:
            ax.annotate(el, (l12, b2), fontsize=10, fontweight="bold",
                        color="red", ha="left", va="bottom")
        else:
            ax.annotate(el, (l12, b2), fontsize=8, ha="left", va="bottom")
    
    all_vals = b2_vals + l12_vals
    vmin, vmax = min(all_vals) * 0.95, max(all_vals) * 1.05
    ax.plot([vmin, vmax], [vmin, vmax], "k--", lw=1, alpha=0.5)
    
    diff_pct = [(b - l) / l * 100 for b, l in zip(b2_vals, l12_vals)]
    mean_diff = np.mean(diff_pct)
    
    ax.set_xlabel("$r_\\mathrm{WS}$ from L1$_2$ compounds (Å)")
    ax.set_ylabel("$r_\\mathrm{WS}$ from B2 compounds (Å)")
    ax.set_title(title)
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_aspect("equal")
    
    for grp, c in color_map.items():
        ax.scatter([], [], c=c, s=50, label=grp)
    ax.legend(fontsize=9, loc="upper left")
    
    rmse = np.sqrt(np.mean(np.array([(b - l) for b, l in zip(b2_vals, l12_vals)]) ** 2))
    ax.text(0.98, 0.05,
            f"N = {len(common)}\nMean diff = {mean_diff:+.2f}%\nRMSE = {rmse:.4f} Å",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=12,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")
    return {"mean_diff_pct": mean_diff, "rmse": rmse, "n": len(common)}


def plot_bootstrap_comparison(boot_contact: Dict, boot_rws: Dict,
                               case_label: str, outpath: str):
    """Side-by-side comparison of contact radii bootstrap vs r_WS bootstrap."""
    # Get common elements
    common = sorted(set(boot_contact.get("mean", {}).keys()) &
                    set(boot_rws.get("mean", {}).keys()))
    if not common:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    for ax, boot, ylabel, title_suffix in [
        (axes[0], boot_contact, "Contact radius (Å)", "Contact Radii"),
        (axes[1], boot_rws, "$r_\\mathrm{WS}$ (Å)", "Wigner-Seitz Radii"),
    ]:
        elements = sorted(boot["mean"].keys(), key=lambda e: boot["mean"].get(e, 0))
        means = [boot["mean"][el] for el in elements]
        ci_lo = [boot["ci_lower"][el] for el in elements]
        ci_hi = [boot["ci_upper"][el] for el in elements]
        err_lo = [m - lo for m, lo in zip(means, ci_lo)]
        err_hi = [hi - m for m, hi in zip(means, ci_hi)]
        stds = [boot["std"][el] for el in elements]
        
        color_map = {"3d": "tab:red", "4d": "tab:blue", "5d": "tab:green",
                     "Lanthanide": "tab:orange", "Actinide": "tab:purple",
                     "p-block": "tab:brown", "s-block": "tab:pink", "Other": "gray"}
        colors = [color_map.get(element_group(el), "gray") for el in elements]
        
        ax.bar(range(len(elements)), means, color=colors, alpha=0.7)
        ax.errorbar(range(len(elements)), means, yerr=[err_lo, err_hi],
                    fmt="none", ecolor="black", capsize=1, lw=0.5)
        ax.set_xticks(range(len(elements)))
        ax.set_xticklabels(elements, rotation=90, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{case_label}: {title_suffix}")
        
        med_std = np.median(stds)
        max_std = max(stds)
        max_el = elements[stds.index(max_std)]
        ax.text(0.98, 0.95,
                f"Med std = {med_std:.4f} Å\nMax std = {max_std:.4f} Å ({max_el})",
                transform=ax.transAxes, ha="right", va="top", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_element_group_analysis(mean_rws: Dict[str, float],
                                 radii: Dict[str, float],
                                 title: str, outpath: str):
    """Box plot of r_contact/r_WS ratio by element group."""
    common = sorted(set(mean_rws.keys()) & set(radii.keys()))
    
    group_ratios: Dict[str, List[float]] = {}
    for el in common:
        grp = element_group(el)
        ratio = radii[el] / mean_rws[el]
        if grp not in group_ratios:
            group_ratios[grp] = []
        group_ratios[grp].append(ratio)
    
    groups = sorted(group_ratios.keys(), key=lambda g: np.median(group_ratios[g]))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data = [group_ratios[g] for g in groups]
    labels = [f"{g}\n(n={len(group_ratios[g])})" for g in groups]
    
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    
    color_map = {"3d": "tab:red", "4d": "tab:blue", "5d": "tab:green",
                 "Lanthanide": "tab:orange", "Actinide": "tab:purple",
                 "p-block": "tab:brown", "s-block": "tab:pink", "Other": "gray"}
    for patch, g in zip(bp["boxes"], groups):
        patch.set_facecolor(color_map.get(g, "gray"))
        patch.set_alpha(0.6)
    
    ax.axhline(y=1.0, color="black", linestyle="--", lw=1, alpha=0.5)
    ax.set_ylabel("$r_\\mathrm{contact} / r_\\mathrm{WS}$")
    ax.set_title(title)
    
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


# =====================================================================
# Main execution
# =====================================================================

def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    
    report_lines = ["# Comprehensive Validation Report\n"]
    report_lines.append(f"r_WS definition: $r_{{WS}} = (3 V_{{atom}} / 4\\pi)^{{1/3}}$\n\n")
    
    # Load all data
    print("Loading data...")
    cases = {
        "MP-B2":   {"compounds": "compounds_MP_B2.csv",   "radii": "radii_MP_B2.csv",   "struct": "B2"},
        "MP-L12":  {"compounds": "compounds_MP_L12.csv",  "radii": "radii_MP_L12.csv",  "struct": "L12"},
        "OQMD-B2": {"compounds": "compounds_OQMD_B2.csv", "radii": "radii_OQMD_B2.csv", "struct": "B2"},
        "OQMD-L12":{"compounds": "compounds_OQMD_L12.csv","radii": "radii_OQMD_L12.csv","struct": "L12"},
    }
    
    all_data = {}
    for case_name, info in cases.items():
        comp_path = os.path.join(DATA_DIR, info["compounds"])
        rad_path = os.path.join(DATA_DIR, info["radii"])
        if not os.path.exists(comp_path) or not os.path.exists(rad_path):
            print(f"  Skipping {case_name}: missing data")
            continue
        compounds = load_compounds(comp_path)
        radii = load_radii(rad_path)
        all_data[case_name] = {
            "compounds": compounds,
            "radii": radii,
            "struct": info["struct"],
        }
        print(f"  {case_name}: {len(compounds)} compounds, {len(radii)} elements")
    
    # =========================================================
    # Section 1: Bootstrap uncertainty for r_WS
    # =========================================================
    print("\n=== Bootstrap Uncertainty for r_WS ===")
    report_lines.append("## 1. Bootstrap Uncertainty for r_WS\n\n")
    
    boot_rws_results = {}
    for case_name, data in all_data.items():
        print(f"  Bootstrap {case_name} (1000 resamples)...")
        boot = bootstrap_rws(data["compounds"], data["struct"],
                              n_bootstrap=1000, seed=42)
        boot_rws_results[case_name] = boot
        
        stats = plot_bootstrap_uncertainty(
            boot, f"r_WS Bootstrap Uncertainty: {case_name}",
            os.path.join(FIG_DIR, f"bootstrap_rws_{case_name.replace('-', '_')}.png"))
        
        report_lines.append(f"### {case_name}\n")
        report_lines.append(f"- Median std: {stats['med_std']:.4f} Å\n")
        report_lines.append(f"- Max std: {stats['max_std']:.4f} Å ({stats['max_std_el']})\n")
        report_lines.append(f"- Avg 95% CI width: {stats['avg_ci_width']:.4f} Å\n\n")
    
    # =========================================================
    # Section 2: r_WS(DFT) vs contact radii comparison
    # =========================================================
    print("\n=== r_WS vs Contact Radii ===")
    report_lines.append("## 2. Element-Average r_WS(DFT) vs Contact Radii\n\n")
    
    rws_means = {}
    for case_name, data in all_data.items():
        el_rws = compute_element_rws(data["compounds"], data["struct"])
        mean_rws = element_mean_rws(el_rws)
        rws_means[case_name] = mean_rws
        
        stats = plot_rws_vs_contact_parity(
            mean_rws, data["radii"],
            f"r_WS(DFT) vs Contact Radius: {case_name}",
            os.path.join(FIG_DIR, f"rws_vs_contact_{case_name.replace('-', '_')}.png"))
        
        if stats:
            report_lines.append(f"### {case_name}\n")
            report_lines.append(f"- RMSE: {stats['rmse']:.4f} Å\n")
            report_lines.append(f"- R²: {stats['r2']:.4f}\n")
            report_lines.append(f"- Mean r_contact/r_WS: {stats['mean_ratio']:.4f}\n")
            report_lines.append(f"- N elements: {stats['n']}\n\n")
    
    # =========================================================
    # Section 3: Compound-level r_WS parity
    # =========================================================
    print("\n=== Compound-Level r_WS Parity ===")
    report_lines.append("## 3. Compound-Level r_WS Parity (DFT vs Predicted)\n\n")
    
    for case_name, data in all_data.items():
        parity_df = compound_rws_parity(data["compounds"], data["radii"], data["struct"])
        
        stats = plot_compound_rws_parity(
            parity_df, f"Compound r_WS Parity: {case_name}",
            os.path.join(FIG_DIR, f"compound_rws_parity_{case_name.replace('-', '_')}.png"))
        
        report_lines.append(f"### {case_name}\n")
        report_lines.append(f"- N compounds: {stats['n']}\n")
        report_lines.append(f"- RMSE: {stats['rmse']:.4f} Å\n")
        report_lines.append(f"- MAE: {stats['mae']:.4f} Å\n")
        report_lines.append(f"- R²: {stats['r2']:.4f}\n\n")
        
        # Save CSV
        csv_path = os.path.join(FIG_DIR, f"compound_rws_parity_{case_name.replace('-', '_')}.csv")
        parity_df.to_csv(csv_path, index=False)
    
    # =========================================================
    # Section 4: Periodic table r_contact/r_WS ratio
    # =========================================================
    print("\n=== Periodic Table: r_contact / r_WS ===")
    report_lines.append("## 4. Periodic Table: r_contact / r_WS Ratio\n\n")
    
    for case_name, data in all_data.items():
        mean_rws = rws_means[case_name]
        stats = plot_rws_contact_ratio_periodic(
            mean_rws, data["radii"],
            f"$r_{{contact}} / r_{{WS}}$: {case_name}",
            os.path.join(FIG_DIR, f"periodic_ratio_{case_name.replace('-', '_')}.png"))
        
        if stats:
            report_lines.append(f"### {case_name}\n")
            report_lines.append(f"- Mean ratio: {stats['mean_ratio']:.4f} ± {stats['std_ratio']:.4f}\n")
            report_lines.append(f"- Range: {stats['min_ratio']:.4f} – {stats['max_ratio']:.4f}\n\n")
    
    # =========================================================
    # Section 5: B2 vs L1₂ r_WS comparison
    # =========================================================
    print("\n=== B2 vs L1₂ r_WS ===")
    report_lines.append("## 5. B2 vs L1₂ r_WS Structure Dependence\n\n")
    
    for db in ["MP", "OQMD"]:
        b2_key = f"{db}-B2"
        l12_key = f"{db}-L12"
        if b2_key in rws_means and l12_key in rws_means:
            stats = plot_rws_B2_vs_L12(
                rws_means[b2_key], rws_means[l12_key],
                f"r_WS: B2 vs L1$_2$ ({db})",
                os.path.join(FIG_DIR, f"rws_B2_vs_L12_{db}.png"))
            
            if stats:
                report_lines.append(f"### {db}\n")
                report_lines.append(f"- Mean diff (B2−L1₂): {stats['mean_diff_pct']:+.2f}%\n")
                report_lines.append(f"- RMSE: {stats['rmse']:.4f} Å\n")
                report_lines.append(f"- N elements: {stats['n']}\n\n")
    
    # =========================================================
    # Section 6: Element group analysis
    # =========================================================
    print("\n=== Element Group Analysis ===")
    report_lines.append("## 6. Element Group Analysis (r_contact / r_WS)\n\n")
    
    for case_name, data in all_data.items():
        mean_rws = rws_means[case_name]
        plot_element_group_analysis(
            mean_rws, data["radii"],
            f"r_contact/r_WS by Element Group: {case_name}",
            os.path.join(FIG_DIR, f"group_ratio_{case_name.replace('-', '_')}.png"))
        
        # Compute group statistics
        common = sorted(set(mean_rws.keys()) & set(data["radii"].keys()))
        group_stats: Dict[str, Dict] = {}
        for el in common:
            grp = element_group(el)
            ratio = data["radii"][el] / mean_rws[el]
            if grp not in group_stats:
                group_stats[grp] = {"ratios": [], "elements": []}
            group_stats[grp]["ratios"].append(ratio)
            group_stats[grp]["elements"].append(el)
        
        report_lines.append(f"### {case_name}\n")
        report_lines.append("| Group | N | Mean ratio | Std | Min | Max |\n")
        report_lines.append("|-------|---|-----------|-----|-----|-----|\n")
        for grp in sorted(group_stats.keys()):
            r = np.array(group_stats[grp]["ratios"])
            report_lines.append(
                f"| {grp} | {len(r)} | {np.mean(r):.4f} | {np.std(r):.4f} | "
                f"{np.min(r):.4f} | {np.max(r):.4f} |\n")
        report_lines.append("\n")
    
    # =========================================================
    # Section 7: Summary table — all 4 cases
    # =========================================================
    print("\n=== Summary Statistics ===")
    report_lines.append("## 7. Summary: r_WS vs Contact Radii Across All Cases\n\n")
    report_lines.append("| Case | N_compounds | N_elements | rWS med_std | rWS avg_CI | r_c/r_WS mean | r_c/r_WS std |\n")
    report_lines.append("|------|------------|-----------|------------|-----------|--------------|-------------|\n")
    
    for case_name, data in all_data.items():
        n_comp = len(data["compounds"])
        n_el = len(data["radii"])
        boot = boot_rws_results.get(case_name, {})
        med_std = np.median(list(boot.get("std", {}).values())) if boot.get("std") else 0
        ci_widths = [(boot["ci_upper"].get(el, 0) - boot["ci_lower"].get(el, 0))
                     for el in boot.get("mean", {}).keys()]
        avg_ci = np.mean(ci_widths) if ci_widths else 0
        
        mean_rws = rws_means.get(case_name, {})
        common = sorted(set(mean_rws.keys()) & set(data["radii"].keys()))
        if common:
            ratios = [data["radii"][el] / mean_rws[el] for el in common]
            r_mean = np.mean(ratios)
            r_std = np.std(ratios)
        else:
            r_mean = r_std = 0
        
        report_lines.append(
            f"| {case_name} | {n_comp} | {n_el} | {med_std:.4f} | {avg_ci:.4f} | "
            f"{r_mean:.4f} | {r_std:.4f} |\n")
    
    report_lines.append("\n")
    
    # =========================================================
    # Section 8: Per-element detailed table (CSV)
    # =========================================================
    print("\n=== Per-Element Detailed Table ===")
    
    # Build a master table with all radii types per element
    all_elements = set()
    for case_name, data in all_data.items():
        all_elements.update(data["radii"].keys())
        all_elements.update(rws_means.get(case_name, {}).keys())
    
    rows = []
    for el in sorted(all_elements):
        row_data = {"element": el, "group": element_group(el)}
        for case_name in all_data.keys():
            data = all_data[case_name]
            rws_m = rws_means.get(case_name, {})
            r_contact = data["radii"].get(el, np.nan)
            r_ws_val = rws_m.get(el, np.nan)
            boot = boot_rws_results.get(case_name, {})
            rws_std = boot.get("std", {}).get(el, np.nan)
            rws_ci_lo = boot.get("ci_lower", {}).get(el, np.nan)
            rws_ci_hi = boot.get("ci_upper", {}).get(el, np.nan)
            n_comp = boot.get("n_compounds", {}).get(el, 0)
            
            prefix = case_name.replace("-", "_")
            row_data[f"{prefix}_r_contact"] = r_contact
            row_data[f"{prefix}_r_ws"] = r_ws_val
            row_data[f"{prefix}_rws_std"] = rws_std
            row_data[f"{prefix}_rws_ci_lo"] = rws_ci_lo
            row_data[f"{prefix}_rws_ci_hi"] = rws_ci_hi
            row_data[f"{prefix}_n_compounds"] = n_comp
            if not np.isnan(r_contact) and not np.isnan(r_ws_val) and r_ws_val > 0:
                row_data[f"{prefix}_ratio"] = r_contact / r_ws_val
            else:
                row_data[f"{prefix}_ratio"] = np.nan
        rows.append(row_data)
    
    master_df = pd.DataFrame(rows)
    csv_path = os.path.join(FIG_DIR, "element_radii_comparison.csv")
    master_df.to_csv(csv_path, index=False, float_format="%.5f")
    print(f"  Saved: {csv_path}")
    
    # =========================================================
    # Write report
    # =========================================================
    report_path = os.path.join(FIG_DIR, "comprehensive_validation_report.md")
    with open(report_path, "w") as f:
        f.writelines(report_lines)
    print(f"\nReport saved: {report_path}")
    print("All done.")


if __name__ == "__main__":
    main()
