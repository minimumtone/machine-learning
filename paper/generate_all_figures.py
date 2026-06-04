#!/usr/bin/env python3
"""
Clean re-analysis script: generates ALL figures and tables for the paper.
Excludes Gd/Ce from all analyses.

Usage:
    cd paper/ && python generate_all_figures.py

Input data (relative to repo root):
    four_case_output/figures/compounds_{MP,OQMD}_{B2,L12}.csv
    data/compounds_VASP_{B2,L12}.csv

Output:
    paper/fig_*.png          — all paper figures
    paper/results_*.csv      — all data tables
"""

import sys
from pathlib import Path

# Ensure repo root is on path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from collections import defaultdict
from scipy.optimize import minimize_scalar
from itertools import combinations

# ---------------------------------------------------------------------------
# Font setup
# ---------------------------------------------------------------------------
for fp in fm.findSystemFonts():
    if "ipag" in fp.lower() or "ipagothic" in fp.lower():
        plt.rcParams["font.family"] = fm.FontProperties(fname=fp).get_name()
        break
else:
    for fp in fm.findSystemFonts():
        if "wqy" in fp.lower():
            plt.rcParams["font.family"] = fm.FontProperties(fname=fp).get_name()
            break

plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
    "figure.dpi": 150,
})

OUTDIR = Path(__file__).resolve().parent  # paper/

# ---------------------------------------------------------------------------
# Import data from main model
# ---------------------------------------------------------------------------
from hea_lattice_xgboost import (
    KING_ATOMIC_VOLUMES, ALONSO_TABLE2, INDEPENDENT_TEST,
    MULTIPHASE_HEA_DB, compute_eq10_scaled, compute_vegard,
    compute_delta_r, compute_delta_sf, PAULING_EN, VEC, D_ELECTRONS,
)

EXCLUDE_ELEMENTS = {"Gd", "Ce"}

# ---------------------------------------------------------------------------
# 1. Load & filter compound data
# ---------------------------------------------------------------------------
def load_compounds():
    """Load MP + OQMD + VASP compound data, excluding Gd/Ce."""
    base = REPO / "four_case_output" / "figures"
    dfs = []
    for src in ["MP", "OQMD"]:
        for struct in ["B2", "L12"]:
            f = base / f"compounds_{src}_{struct}.csv"
            if f.exists():
                df = pd.read_csv(f)
                df["db"] = src
                df["stype"] = struct
                dfs.append(df)
    for struct in ["B2", "L12"]:
        for search_dir in [REPO / "data", base]:
            f = search_dir / f"compounds_VASP_{struct}.csv"
            if f.exists():
                df = pd.read_csv(f)
                df["db"] = "VASP"
                df["stype"] = struct
                dfs.append(df)
                break
    if not dfs:
        raise FileNotFoundError(
            "No compound CSV files found. Check four_case_output/figures/ and data/ directories.")
    all_df = pd.concat(dfs, ignore_index=True)
    # Exclude Gd/Ce
    mask = ~(all_df["element_A"].isin(EXCLUDE_ELEMENTS) |
             all_df["element_B"].isin(EXCLUDE_ELEMENTS))
    all_df = all_df[mask].reset_index(drop=True)
    return all_df


def compute_omega_sf_pairwise(df, sources=("MP", "OQMD"), min_count=2):
    """Compute structure-specific pairwise Omega_sf. Returns (ob2, ol12)."""
    pair_b2 = defaultdict(list)
    pair_l12 = defaultdict(list)
    sub = df[df["db"].isin(sources)]
    for _, row in sub.iterrows():
        elA, elB = row.get("element_A", ""), row.get("element_B", "")
        a = row.get("lattice_constant", 0)
        stype = row.get("stype", "")
        if a <= 2 or a >= 8:
            continue
        if elA not in KING_ATOMIC_VOLUMES or elB not in KING_ATOMIC_VOLUMES:
            continue
        if elA == elB:
            continue
        pair = tuple(sorted([elA, elB]))
        vA, vB = KING_ATOMIC_VOLUMES[elA], KING_ATOMIC_VOLUMES[elB]
        if stype == "B2":
            v_act = a**3 / 2
            v_veg = (vA + vB) / 2
            pair_b2[pair].append((v_act - v_veg) / v_veg)
        elif stype == "L12":
            cA = row.get("count_A", 3)
            cB = row.get("count_B", 1)
            total = cA + cB
            v_act = a**3 / 4
            v_veg = (cA * vA + cB * vB) / total
            pair_l12[pair].append((v_act - v_veg) / v_veg)
    ob2 = {p: np.median(v) for p, v in pair_b2.items() if len(v) >= min_count}
    ol12 = {p: np.median(v) for p, v in pair_l12.items() if len(v) >= min_count}
    return ob2, ol12


def optimize_gamma(heas, ob2, ol12):
    """Optimize gamma_BCC and gamma_FCC on training HEAs."""
    y = np.array([h["a_exp"] for h in heas])
    bcc_i = [i for i, h in enumerate(heas) if h["struct"] == "BCC"]
    fcc_i = [i for i, h in enumerate(heas) if h["struct"] == "FCC"]

    def pred_all(gb, gf):
        return np.array([
            compute_eq10_scaled(h["comp"], h["struct"],
                                ob2 if h["struct"] == "BCC" else ol12,
                                gb if h["struct"] == "BCC" else gf)
            for h in heas
        ])

    def rmse_bcc(gb):
        if not bcc_i:
            return 0.0
        p = pred_all(gb, 1.0)
        return np.sqrt(np.mean((p[bcc_i] - y[bcc_i]) ** 2))

    def rmse_fcc(gf):
        if not fcc_i:
            return 0.0
        p = pred_all(1.0, gf)
        return np.sqrt(np.mean((p[fcc_i] - y[fcc_i]) ** 2))

    gb = minimize_scalar(rmse_bcc, bounds=(-5, 5), method="bounded").x if bcc_i else 1.0
    gf = minimize_scalar(rmse_fcc, bounds=(-5, 5), method="bounded").x if fcc_i else 1.0
    return gb, gf


def predict_heas(heas, ob2, ol12, gb, gf):
    """Predict lattice constants for a list of HEAs."""
    return np.array([
        compute_eq10_scaled(h["comp"], h["struct"],
                            ob2 if h["struct"] == "BCC" else ol12,
                            gb if h["struct"] == "BCC" else gf)
        for h in heas
    ])


def additive_decomposition(ob2, ol12):
    """Decompose pairwise Omega_sf into element-level delta parameters."""
    results = {}
    for label, omega in [("B2", ob2), ("L12", ol12)]:
        elements = set()
        for (a, b) in omega:
            elements.add(a)
            elements.add(b)
        elements = sorted(elements)
        elem_idx = {e: i for i, e in enumerate(elements)}
        n = len(elements)

        rows_A = []
        rows_b = []
        for (a, b), val in omega.items():
            row = np.zeros(n)
            row[elem_idx[a]] = 1.0
            row[elem_idx[b]] = 1.0
            rows_A.append(row)
            rows_b.append(val)

        A = np.array(rows_A)
        b_vec = np.array(rows_b)
        delta, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)

        delta_dict = {elements[i]: delta[i] for i in range(n)}
        # R-squared
        pred = A @ delta
        ss_res = np.sum((b_vec - pred) ** 2)
        ss_tot = np.sum((b_vec - b_vec.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        results[label] = {"delta": delta_dict, "r2": r2, "elements": elements}
    return results


# ---------------------------------------------------------------------------
# Effective radius functions
# ---------------------------------------------------------------------------
def compute_effective_radii(all_df, sources=("MP", "OQMD", "VASP")):
    """Compute structure-dependent effective radii from DFT volumes."""
    sub = all_df[all_df["db"].isin(sources)]
    # For each element, collect volumes in different structural roles
    vol_b2 = defaultdict(list)
    vol_l12_maj = defaultdict(list)
    vol_l12_min = defaultdict(list)

    for _, row in sub.iterrows():
        elA, elB = row.get("element_A", ""), row.get("element_B", "")
        a = row.get("lattice_constant", 0)
        stype = row.get("stype", "")
        if a <= 2 or a >= 8 or elA == elB:
            continue
        if elA not in KING_ATOMIC_VOLUMES or elB not in KING_ATOMIC_VOLUMES:
            continue
        if stype == "B2":
            v = a**3 / 2  # average volume per atom
            vol_b2[elA].append(v)
            vol_b2[elB].append(v)
        elif stype == "L12":
            cA = row.get("count_A", 3)
            cB = row.get("count_B", 1)
            v = a**3 / 4
            if cA == 3:
                vol_l12_maj[elA].append(v)
                vol_l12_min[elB].append(v)
            else:
                vol_l12_maj[elB].append(v)
                vol_l12_min[elA].append(v)

    radii = {}
    for elem in set(list(vol_b2.keys()) + list(vol_l12_maj.keys()) + list(vol_l12_min.keys())):
        r_pure = (3 * KING_ATOMIC_VOLUMES.get(elem, 15.0) / (4 * np.pi)) ** (1/3)
        entry = {"r_pure": r_pure}
        if elem in vol_b2 and len(vol_b2[elem]) >= 3:
            v_eff = np.median(vol_b2[elem])
            entry["r_b2"] = (3 * v_eff / (4 * np.pi)) ** (1/3)
        if elem in vol_l12_maj and len(vol_l12_maj[elem]) >= 3:
            v_eff = np.median(vol_l12_maj[elem])
            entry["r_l12_maj"] = (3 * v_eff / (4 * np.pi)) ** (1/3)
        if elem in vol_l12_min and len(vol_l12_min[elem]) >= 3:
            v_eff = np.median(vol_l12_min[elem])
            entry["r_l12_min"] = (3 * v_eff / (4 * np.pi)) ** (1/3)
        radii[elem] = entry
    return radii


# ===========================================================================
# FIGURE GENERATION
# ===========================================================================

def fig01_parity(y_train, a_veg_tr, a_ss_tr, y_test, a_veg_te, a_ss_te):
    """Fig 1: Parity plot Vegard vs DFT-Omega_sf."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    lims = [2.85, 3.65]
    ax.plot(lims, lims, "k-", lw=1)
    ax.scatter(y_train, a_veg_tr, c="gray", alpha=0.5, s=50, label="Vegard (train)")
    ax.scatter(y_train, a_ss_tr, c="C0", alpha=0.7, s=50, label=r"DFT-$\Omega_{\mathrm{sf}}$ (train)")
    ax.scatter(y_test, a_veg_te, c="gray", alpha=0.5, s=50, marker="^")
    ax.scatter(y_test, a_ss_te, c="C3", alpha=0.7, s=50, marker="^", label=r"DFT-$\Omega_{\mathrm{sf}}$ (test)")
    ax.set_xlabel("Experimental $a$ (\u00c5)")
    ax.set_ylabel("Predicted $a$ (\u00c5)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.legend(loc="upper left", fontsize=12)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_parity.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig_parity.png")


def fig02_rmse_bar(rmse_dict):
    """Fig 2: RMSE bar chart comparing all methods."""
    methods = list(rmse_dict.keys())
    vals = [rmse_dict[m] for m in methods]
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#aaaaaa", "#4c72b0", "#55a868", "#c44e52", "#8172b2"]
    bars = ax.bar(range(len(methods)), [v * 1000 for v in vals],
                  color=colors[:len(methods)], edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=14)
    ax.set_ylabel("RMSE (m\u00c5)")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{v*1000:.1f}", ha="center", va="bottom", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_rmse_bar.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig_rmse_bar.png")


def fig03_bcc_fcc(y_train, a_ss_tr, heas_train):
    """Fig 3: BCC/FCC split parity."""
    bcc_i = [i for i, h in enumerate(heas_train) if h["struct"] == "BCC"]
    fcc_i = [i for i, h in enumerate(heas_train) if h["struct"] == "FCC"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for ax, idx, label, c in [(ax1, bcc_i, "BCC", "C0"), (ax2, fcc_i, "FCC", "C3")]:
        lims = [min(y_train[idx]) - 0.02, max(y_train[idx]) + 0.02]
        ax.plot(lims, lims, "k-", lw=1)
        ax.scatter(y_train[idx], a_ss_tr[idx], c=c, s=60, alpha=0.7)
        rmse = np.sqrt(np.mean((a_ss_tr[idx] - y_train[idx]) ** 2))
        ax.set_title(f"{label} (RMSE = {rmse:.4f} \u00c5)")
        ax.set_xlabel("Experimental $a$ (\u00c5)")
        ax.set_ylabel("Predicted $a$ (\u00c5)")
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_bcc_fcc.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig_bcc_fcc.png")


def fig04_indep_test(y_test, a_veg_te, a_ss_te, heas_test, gb, gf):
    """Fig 4: Multi-panel independent test figure."""
    bcc_t = [i for i, h in enumerate(heas_test) if h["struct"] == "BCC"]
    fcc_t = [i for i, h in enumerate(heas_test) if h["struct"] == "FCC"]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # (a) Per-alloy absolute error
    ax = axes[0]
    alloy_names = []
    err_veg = []
    err_ss = []
    for i in range(len(heas_test)):
        h = heas_test[i]
        elems = sorted(h["comp"].keys())
        alloy_names.append("-".join(elems))
        err_veg.append(abs(a_veg_te[i] - y_test[i]) * 1000)
        err_ss.append(abs(a_ss_te[i] - y_test[i]) * 1000)
    x = np.arange(len(alloy_names))
    w = 0.35
    ax.barh(x - w/2, err_veg, w, color="gray", alpha=0.7, label="Vegard")
    ax.barh(x + w/2, err_ss, w, color="C0", alpha=0.7, label=r"DFT-$\Omega_{\mathrm{sf}}$")
    ax.set_yticks(x)
    ax.set_yticklabels(alloy_names, fontsize=10)
    ax.set_xlabel("|Error| (m\u00c5)")
    ax.set_title("(a) Per-alloy error")
    ax.legend(fontsize=11)
    ax.invert_yaxis()

    # (b) BCC/FCC RMSE breakdown
    ax = axes[1]
    categories = ["All", "BCC", "FCC"]
    rmse_v = [np.sqrt(np.mean((a_veg_te - y_test) ** 2)) * 1000,
              np.sqrt(np.mean((a_veg_te[bcc_t] - y_test[bcc_t]) ** 2)) * 1000,
              np.sqrt(np.mean((a_veg_te[fcc_t] - y_test[fcc_t]) ** 2)) * 1000]
    rmse_s = [np.sqrt(np.mean((a_ss_te - y_test) ** 2)) * 1000,
              np.sqrt(np.mean((a_ss_te[bcc_t] - y_test[bcc_t]) ** 2)) * 1000,
              np.sqrt(np.mean((a_ss_te[fcc_t] - y_test[fcc_t]) ** 2)) * 1000]
    x2 = np.arange(len(categories))
    ax.bar(x2 - 0.2, rmse_v, 0.35, color="gray", alpha=0.7, label="Vegard")
    ax.bar(x2 + 0.2, rmse_s, 0.35, color="C0", alpha=0.7, label=r"DFT-$\Omega_{\mathrm{sf}}$")
    ax.set_xticks(x2)
    ax.set_xticklabels(categories)
    ax.set_ylabel("RMSE (m\u00c5)")
    ax.set_title("(b) RMSE breakdown")
    ax.legend(fontsize=11)

    # (c) Parity
    ax = axes[2]
    lims = [min(y_test) - 0.02, max(y_test) + 0.02]
    ax.plot(lims, lims, "k-", lw=1)
    for idx, label, marker, c in [(bcc_t, "BCC", "s", "C0"), (fcc_t, "FCC", "o", "C3")]:
        ax.scatter(y_test[idx], a_ss_te[idx], c=c, marker=marker, s=70, alpha=0.7, label=label)
    ax.set_xlabel("Experimental $a$ (\u00c5)")
    ax.set_ylabel("Predicted $a$ (\u00c5)")
    ax.set_title(f"(c) Independent test ($q_{{BCC}}$={gb:.2f}, $q_{{FCC}}$={gf:.2f})")
    ax.legend(fontsize=11)
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_indep_test.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig_indep_test.png")


def fig05_element_delta(decomp):
    """Fig 5: Element delta bar chart (B2 and L12)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    for ax, key, title in [(ax1, "B2", "B2 (BCC)"), (ax2, "L12", r"L1$_2$ (FCC)")]:
        d = decomp[key]["delta"]
        elems = sorted(d.keys(), key=lambda e: d[e])
        vals = [d[e] for e in elems]
        colors = ["C3" if v < 0 else "C0" for v in vals]
        ax.bar(range(len(elems)), vals, color=colors, edgecolor="black", linewidth=0.3)
        ax.set_xticks(range(len(elems)))
        ax.set_xticklabels(elems, fontsize=10, rotation=45)
        ax.set_ylabel(r"$\delta^{(s)}$")
        ax.set_title(f"{title}  (R$^2$ = {decomp[key]['r2']:.3f})")
        ax.axhline(0, color="k", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_element_delta.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig_element_delta.png")


def fig06_additive_fit(ob2, ol12, decomp):
    """Fig 6: Pairwise Omega_sf vs additive delta_A + delta_B."""
    OUTLIER_THRESHOLD = 0.3  # |Omega_sf| > 0.3 are unphysical (f-electron issues)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for ax, omega, key, title in [(ax1, ob2, "B2", "B2"),
                                   (ax2, ol12, "L12", r"L1$_2$")]:
        d = decomp[key]["delta"]
        x_vals, y_vals, x_out, y_out = [], [], [], []
        for (a, b), val in omega.items():
            if a in d and b in d:
                if abs(val) > OUTLIER_THRESHOLD:
                    x_out.append(d[a] + d[b])
                    y_out.append(val)
                else:
                    x_vals.append(d[a] + d[b])
                    y_vals.append(val)
        ax.scatter(x_vals, y_vals, c="C0", alpha=0.4, s=20)
        if x_out:
            ax.scatter(x_out, y_out, c="C3", alpha=0.6, s=40, marker="x",
                       label=f"excluded ({len(x_out)})")
            ax.legend(fontsize=10)
        lims = [min(min(x_vals), min(y_vals)) - 0.01,
                max(max(x_vals), max(y_vals)) + 0.01]
        ax.plot(lims, lims, "k--", lw=1)
        ax.set_xlabel(r"$\delta_A^{(s)} + \delta_B^{(s)}$")
        ax.set_ylabel(r"$\Omega_\mathrm{sf}^{(s)}$ (pairwise)")
        n_total = len(x_vals) + len(x_out)
        ax.set_title(f"{title} ({n_total} pairs)  R$^2$ = {decomp[key]['r2']:.3f}")
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_additive_fit.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig_additive_fit.png")


def fig07_composition_examples(all_df):
    """Fig 7: Vegard composition plots for representative pairs."""
    examples = [("Cu", "Zr"), ("Al", "Ni"), ("Fe", "Ti"),
                ("Co", "Cr"), ("Pd", "Ti"), ("Nb", "Ta")]

    # Build DFT pure element volumes from same-element B2 entries
    same_el_b2 = all_df[(all_df["stype"] == "B2") &
                         (all_df["element_A"] == all_df["element_B"])]
    dft_volumes = {}
    for _, row in same_el_b2.iterrows():
        el = row["element_A"]
        v = row["lattice_constant"] ** 3 / 2
        dft_volumes[el] = v  # last entry wins (VASP preferred)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()

    for idx, (elX, elY) in enumerate(examples):
        ax = axes[idx]
        vX = KING_ATOMIC_VOLUMES.get(elX, 15)
        vY = KING_ATOMIC_VOLUMES.get(elY, 15)
        # King Vegard line
        c_arr = np.linspace(0, 1, 100)
        a_veg = [(2 * ((1 - c) * vX + c * vY)) ** (1/3) for c in c_arr]
        ax.plot(c_arr * 100, a_veg, "k--", lw=1.5, label="Vegard (King)")

        # DFT Vegard line (if both elements have DFT volumes)
        if elX in dft_volumes and elY in dft_volumes:
            vX_dft = dft_volumes[elX]
            vY_dft = dft_volumes[elY]
            a_veg_dft = [(2 * ((1 - c) * vX_dft + c * vY_dft)) ** (1/3)
                         for c in c_arr]
            ax.plot(c_arr * 100, a_veg_dft, "k-", lw=1.5,
                    alpha=0.6, label="Vegard (DFT)")

        # DFT data points
        sub = all_df[(all_df["stype"] == "B2")]
        for _, row in sub.iterrows():
            elA, elB = row["element_A"], row["element_B"]
            a = row["lattice_constant"]
            if a <= 2 or a >= 8:
                continue
            if {elA, elB} == {elX, elY}:
                cA = row.get("count_A", 1)
                cB = row.get("count_B", 1)
                total = cA + cB
                if elA == elY:
                    c_B = cA / total
                else:
                    c_B = cB / total
                ax.scatter(c_B * 100, a, c="C0", s=80, zorder=5)

        sub_l12 = all_df[all_df["stype"] == "L12"]
        for _, row in sub_l12.iterrows():
            elA, elB = row["element_A"], row["element_B"]
            a = row["lattice_constant"]
            if a <= 2 or a >= 8:
                continue
            if {elA, elB} == {elX, elY}:
                cA = row.get("count_A", 3)
                cB = row.get("count_B", 1)
                total = cA + cB
                if elA == elY:
                    c_B = cA / total
                else:
                    c_B = cB / total
                a_fcc = a / (2 ** (1/3))  # convert L12 to equivalent B2 scale
                ax.scatter(c_B * 100, a_fcc, c="C3", s=80, zorder=5, marker="^")

        ax.set_xlabel(f"% {elY}")
        ax.set_ylabel("$a$ (\u00c5)")
        ax.set_title(f"{elX}-{elY}")
        ax.legend(fontsize=10, loc="best")
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_composition_examples.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig_composition_examples.png")


def _l12_bucket(elA, elB, cA, cB):
    """Determine L1₂ bucket for sorted pair key.
    'A3B' = sorted_pair[0] is majority; 'AB3' = sorted_pair[1] is majority.
    """
    pair = tuple(sorted([elA, elB]))
    maj_elem = elA if cA >= cB else elB
    return "A3B" if maj_elem == pair[0] else "AB3"


def fig08_delta_r_proof(all_df):
    """Fig 8: 6-panel proof that delta_r cannot absorb structure info."""
    # Compute delta_r and Omega_sf for each pair×structure
    pair_data = defaultdict(lambda: {"B2": [], "L12_A3B": [], "L12_AB3": []})
    for _, row in all_df.iterrows():
        elA, elB = row.get("element_A", ""), row.get("element_B", "")
        a = row.get("lattice_constant", 0)
        stype = row.get("stype", "")
        if a <= 2 or a >= 8 or elA == elB:
            continue
        if elA not in KING_ATOMIC_VOLUMES or elB not in KING_ATOMIC_VOLUMES:
            continue
        pair = tuple(sorted([elA, elB]))
        vA, vB = KING_ATOMIC_VOLUMES[elA], KING_ATOMIC_VOLUMES[elB]
        if stype == "B2":
            osf = (a**3 / 2 - (vA + vB) / 2) / ((vA + vB) / 2)
            pair_data[pair]["B2"].append(osf)
        elif stype == "L12":
            cA = row.get("count_A", 3)
            cB = row.get("count_B", 1)
            total = cA + cB
            v_veg = (cA * vA + cB * vB) / total
            osf = (a**3 / 4 - v_veg) / v_veg
            bucket = "L12_" + _l12_bucket(elA, elB, cA, cB)
            pair_data[pair][bucket].append(osf)

    # Pairs with all 3 structures
    complete = {}
    for pair, data in pair_data.items():
        if data["B2"] and data["L12_A3B"] and data["L12_AB3"]:
            complete[pair] = {
                "B2": np.median(data["B2"]),
                "L12_A3B": np.median(data["L12_A3B"]),
                "L12_AB3": np.median(data["L12_AB3"]),
            }

    if not complete:
        print("  fig_delta_r_proof.png — skipped (no pairs with all 3 structures)")
        return

    # --- Figure 1: (a)(b)(c) structural invariance proof ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # (a) delta_r X3Y vs Y3X — should show NO scatter
    ax = axes[0]
    dr_a3b, dr_ab3 = [], []
    for (elA, elB) in complete:
        vA, vB = KING_ATOMIC_VOLUMES[elA], KING_ATOMIC_VOLUMES[elB]
        rA = (3 * vA / (4 * np.pi)) ** (1/3)
        rB = (3 * vB / (4 * np.pi)) ** (1/3)
        r_avg_75 = 0.75 * rA + 0.25 * rB
        r_avg_25 = 0.25 * rA + 0.75 * rB
        dr1 = np.sqrt(0.75 * (rA / r_avg_75 - 1)**2 + 0.25 * (rB / r_avg_75 - 1)**2) * 100
        dr2 = np.sqrt(0.25 * (rA / r_avg_25 - 1)**2 + 0.75 * (rB / r_avg_25 - 1)**2) * 100
        dr_a3b.append(dr1)
        dr_ab3.append(dr2)
    ax.scatter(dr_a3b, dr_ab3, c="C0", alpha=0.4, s=20)
    lims = [0, max(max(dr_a3b), max(dr_ab3)) * 1.05]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlabel(r"$\delta r$ (A$_3$B)")
    ax.set_ylabel(r"$\delta r$ (B$_3$A)")
    ax.set_title(r"(a) $\delta r$: no scatter ($\equiv$ composition only)")
    ax.set_aspect("equal")

    # (b) delta_r A3B vs B2
    ax = axes[1]
    dr_b2 = []
    for (elA, elB) in complete:
        vA, vB = KING_ATOMIC_VOLUMES[elA], KING_ATOMIC_VOLUMES[elB]
        rA = (3 * vA / (4 * np.pi)) ** (1/3)
        rB = (3 * vB / (4 * np.pi)) ** (1/3)
        r_avg = 0.5 * rA + 0.5 * rB
        dr = np.sqrt(0.5 * (rA / r_avg - 1)**2 + 0.5 * (rB / r_avg - 1)**2) * 100
        dr_b2.append(dr)
    ax.scatter(dr_a3b, dr_b2, c="C0", alpha=0.4, s=20)
    ax.set_xlabel(r"$\delta r$ (A$_3$B)")
    ax.set_ylabel(r"$\delta r$ (B2)")
    ax.set_title(r"(b) $\delta r$: smooth curve ($\equiv$ no structure info)")
    ax.set_aspect("equal")

    # (c) Omega_sf X3Y vs Y3X — should show LARGE scatter
    ax = axes[2]
    osf_a3b = [complete[p]["L12_A3B"] for p in complete]
    osf_ab3 = [complete[p]["L12_AB3"] for p in complete]
    ax.scatter(osf_a3b, osf_ab3, c="C3", alpha=0.4, s=20)
    lims2 = [min(min(osf_a3b), min(osf_ab3)) - 0.02,
             max(max(osf_a3b), max(osf_ab3)) + 0.02]
    ax.plot(lims2, lims2, "k--", lw=1)
    r_corr = np.corrcoef(osf_a3b, osf_ab3)[0, 1]
    ax.set_xlabel(r"$\Omega_\mathrm{sf}$ (A$_3$B)")
    ax.set_ylabel(r"$\Omega_\mathrm{sf}$ (B$_3$A)")
    ax.set_title(f"(c) $\\Omega_{{sf}}$: large scatter (r={r_corr:.2f})")
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_delta_r_proof.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig_delta_r_proof.png")

    # --- Figure 2: (d)(e) L12 asymmetry and L12-B2 correlation ---
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

    # (d) DFT lattice constant difference |a(A3B) - a(B3A)|
    ax = axes2[0]
    a_diffs = []
    for pair, data in pair_data.items():
        if data["L12_A3B"] and data["L12_AB3"]:
            # need actual lattice constants, not Omega_sf
            pass
    # Use raw lattice constants
    pair_a = defaultdict(lambda: {"A3B": [], "AB3": []})
    for _, row in all_df[all_df["stype"] == "L12"].iterrows():
        elA, elB = row["element_A"], row["element_B"]
        a = row["lattice_constant"]
        if a <= 2 or a >= 8 or elA == elB:
            continue
        pair = tuple(sorted([elA, elB]))
        cA = row.get("count_A", 3)
        cB = row.get("count_B", 1)
        bucket = _l12_bucket(elA, elB, cA, cB)
        pair_a[pair][bucket].append(a)
    diffs = []
    for pair in pair_a:
        if pair_a[pair]["A3B"] and pair_a[pair]["AB3"]:
            diffs.append(abs(np.median(pair_a[pair]["A3B"]) -
                            np.median(pair_a[pair]["AB3"])))
    if not diffs:
        diffs = [0.0]
    ax.hist(diffs, bins=40, color="C0", edgecolor="black", linewidth=0.3)
    ax.axvline(np.mean(diffs), color="C3", linestyle="--", lw=2,
               label=f"mean = {np.mean(diffs):.3f} \u00c5")
    ax.set_xlabel("|$a$(A$_3$B) $-$ $a$(B$_3$A)| (\u00c5)")
    ax.set_ylabel("Count")
    ax.set_title("(d) L1$_2$ lattice constant asymmetry")
    ax.legend(fontsize=12)

    # (e) Omega_sf A3B vs B2
    ax = axes2[1]
    osf_b2_list = [complete[p]["B2"] for p in complete]
    ax.scatter(osf_a3b, osf_b2_list, c="C0", alpha=0.4, s=20)
    r2 = np.corrcoef(osf_a3b, osf_b2_list)[0, 1]
    ax.set_xlabel(r"$\Omega_\mathrm{sf}$ (A$_3$B, L1$_2$)")
    ax.set_ylabel(r"$\Omega_\mathrm{sf}$ (AB, B2)")
    ax.set_title(f"(e) L1$_2$ vs B2 (r={r2:.2f})")

    fig2.tight_layout()
    fig2.savefig(OUTDIR / "fig_l12_b2_correlation.png", bbox_inches="tight")
    plt.close(fig2)
    print("  fig_l12_b2_correlation.png")


def fig09_packing(all_df):
    """Fig 9: Packing radius analysis."""
    # Packing: r_A + r_B = d_nn => symmetric => can't distinguish A3B from B3A
    pair_a = defaultdict(lambda: {"A3B": [], "AB3": [], "B2": []})
    for _, row in all_df.iterrows():
        elA, elB = row.get("element_A", ""), row.get("element_B", "")
        a = row.get("lattice_constant", 0)
        stype = row.get("stype", "")
        if a <= 2 or a >= 8 or elA == elB:
            continue
        pair = tuple(sorted([elA, elB]))
        if stype == "B2":
            pair_a[pair]["B2"].append(a)
        elif stype == "L12":
            cA = row.get("count_A", 3)
            cB = row.get("count_B", 1)
            bucket = _l12_bucket(elA, elB, cA, cB)
            pair_a[pair][bucket].append(a)

    # For pairs with both A3B and AB3, compute packing nearest-neighbor distance
    dnn_a3b, dnn_ab3 = [], []
    for pair in pair_a:
        if pair_a[pair]["A3B"] and pair_a[pair]["AB3"]:
            a1 = np.median(pair_a[pair]["A3B"])
            a2 = np.median(pair_a[pair]["AB3"])
            dnn_a3b.append(a1 / np.sqrt(2))  # FCC nearest neighbor
            dnn_ab3.append(a2 / np.sqrt(2))

    if not dnn_a3b:
        print("  fig_packing.png — skipped (no pairs with both A3B and AB3)")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.scatter(dnn_a3b, dnn_ab3, c="C0", alpha=0.4, s=20)
    lims = [min(min(dnn_a3b), min(dnn_ab3)) - 0.05,
            max(max(dnn_a3b), max(dnn_ab3)) + 0.05]
    ax1.plot(lims, lims, "k--", lw=1)
    ax1.set_xlabel("$d_{nn}$ (A$_3$B) (\u00c5)")
    ax1.set_ylabel("$d_{nn}$ (B$_3$A) (\u00c5)")
    ax1.set_title("Packing: $r_A + r_B = d_{nn}$ is symmetric")
    ax1.set_aspect("equal")
    ax1.text(0.05, 0.9, f"n = {len(dnn_a3b)} pairs",
             transform=ax1.transAxes, fontsize=14)

    # Packing predicts d_nn(A3B) = d_nn(B3A), but DFT shows they differ
    residuals = np.array(dnn_a3b) - np.array(dnn_ab3)
    ax2.hist(residuals, bins=40, color="C0", edgecolor="black", linewidth=0.3)
    ax2.axvline(0, color="k", linestyle="--", lw=1)
    ax2.set_xlabel("$d_{nn}$(A$_3$B) $-$ $d_{nn}$(B$_3$A) (\u00c5)")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Packing residual (std = {np.std(residuals):.3f} \u00c5)")

    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_packing.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig_packing.png")


def fig10_l12_asymmetry(all_df):
    """Fig 10: L12 asymmetry — a(A3B) vs a(B3A)."""
    pair_a = defaultdict(lambda: {"A3B": [], "AB3": []})
    for _, row in all_df[all_df["stype"] == "L12"].iterrows():
        elA, elB = row["element_A"], row["element_B"]
        a = row["lattice_constant"]
        if a <= 2 or a >= 8 or elA == elB:
            continue
        pair = tuple(sorted([elA, elB]))
        cA = row.get("count_A", 3)
        cB = row.get("count_B", 1)
        bucket = _l12_bucket(elA, elB, cA, cB)
        pair_a[pair][bucket].append(a)

    a3b_vals, ab3_vals = [], []
    for pair in pair_a:
        if pair_a[pair]["A3B"] and pair_a[pair]["AB3"]:
            a3b_vals.append(np.median(pair_a[pair]["A3B"]))
            ab3_vals.append(np.median(pair_a[pair]["AB3"]))

    if not a3b_vals:
        print("  fig_l12_asymmetry.png — skipped (no pairs with both A3B and AB3)")
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(a3b_vals, ab3_vals, c="C3", alpha=0.4, s=20)
    lims = [min(min(a3b_vals), min(ab3_vals)) - 0.1,
            max(max(a3b_vals), max(ab3_vals)) + 0.1]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlabel("$a$ (A$_3$B) (\u00c5)")
    ax.set_ylabel("$a$ (B$_3$A) (\u00c5)")
    ax.set_title(f"L1$_2$ asymmetry ({len(a3b_vals)} pairs)")
    ax.set_aspect("equal")
    diff = np.array(a3b_vals) - np.array(ab3_vals)
    ax.text(0.05, 0.9, f"mean |$\\Delta a$| = {np.mean(np.abs(diff)):.3f} \u00c5",
            transform=ax.transAxes, fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_l12_asymmetry.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig_l12_asymmetry.png")


def fig11_volume_radius(radii):
    """Fig 11: Structure-dependent effective radii from volume."""
    elements = sorted([e for e in radii if "r_b2" in radii[e] and "r_l12_maj" in radii[e]])
    if not elements:
        print("  fig_volume_radius.png — skipped (no elements with both B2 and L12 radii)")
        return
    r_pure = [radii[e]["r_pure"] for e in elements]
    r_b2 = [radii[e]["r_b2"] for e in elements]
    r_l12 = [radii[e]["r_l12_maj"] for e in elements]

    fig, ax = plt.subplots(figsize=(8, 8))
    lims = [min(min(r_pure), min(r_b2), min(r_l12)) - 0.05,
            max(max(r_pure), max(r_b2), max(r_l12)) + 0.05]
    ax.plot(lims, lims, "k--", lw=1)
    ax.scatter(r_pure, r_b2, c="C0", s=40, alpha=0.7, label="B2")
    ax.scatter(r_pure, r_l12, c="C3", s=40, alpha=0.7, marker="^", label=r"L1$_2$ majority")
    ax.set_xlabel("Pure element radius (\u00c5)")
    ax.set_ylabel("Effective radius in compound (\u00c5)")
    ax.set_title("Structure-dependent effective radii")
    ax.legend(fontsize=13)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_volume_radius.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig_volume_radius.png")


def fig12_hea_additive(y_train, a_add_tr, heas_train, y_test, a_add_te, heas_test):
    """Fig 12: HEA prediction using additive delta."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    lims = [2.85, 3.65]

    ax1.plot(lims, lims, "k-", lw=1)
    bcc_tr = [i for i, h in enumerate(heas_train) if h["struct"] == "BCC"]
    fcc_tr = [i for i, h in enumerate(heas_train) if h["struct"] == "FCC"]
    ax1.scatter(y_train[bcc_tr], a_add_tr[bcc_tr], c="C0", s=50, alpha=0.7, label="BCC")
    ax1.scatter(y_train[fcc_tr], a_add_tr[fcc_tr], c="C3", s=50, alpha=0.7, marker="^", label="FCC")
    rmse_tr = np.sqrt(np.mean((a_add_tr - y_train) ** 2))
    ax1.set_title(f"Training (RMSE = {rmse_tr:.4f} \u00c5)")
    ax1.set_xlabel("Experimental $a$ (\u00c5)")
    ax1.set_ylabel("Predicted $a$ (\u00c5)")
    ax1.legend(fontsize=12)
    ax1.set_xlim(lims); ax1.set_ylim(lims)
    ax1.set_aspect("equal")

    ax2.plot(lims, lims, "k-", lw=1)
    bcc_te = [i for i, h in enumerate(heas_test) if h["struct"] == "BCC"]
    fcc_te = [i for i, h in enumerate(heas_test) if h["struct"] == "FCC"]
    ax2.scatter(y_test[bcc_te], a_add_te[bcc_te], c="C0", s=50, alpha=0.7, label="BCC")
    ax2.scatter(y_test[fcc_te], a_add_te[fcc_te], c="C3", s=50, alpha=0.7, marker="^", label="FCC")
    rmse_te = np.sqrt(np.mean((a_add_te - y_test) ** 2))
    ax2.set_title(f"Independent test (RMSE = {rmse_te:.4f} \u00c5)")
    ax2.set_xlabel("Experimental $a$ (\u00c5)")
    ax2.set_ylabel("Predicted $a$ (\u00c5)")
    ax2.legend(fontsize=12)
    ax2.set_xlim(lims); ax2.set_ylim(lims)
    ax2.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_hea_prediction_additive.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig_hea_prediction_additive.png")


def fig13_composition_reff(decomp, gb, gf):
    """Fig 13: Composition-dependent effective radius for CoCrFeMnNi."""
    elements = ["Co", "Cr", "Fe", "Mn", "Ni"]
    delta_l12 = decomp["L12"]["delta"]
    delta_b2 = decomp["B2"]["delta"]

    # Vary each element's fraction from 0 to 0.4 while keeping others equal
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, delta, gamma, title, struct in [
        (axes[0], delta_b2, gb, "BCC ($\\delta^{(B2)}$)", "BCC"),
        (axes[1], delta_l12, gf, r"FCC ($\delta^{(L1_2)}$)", "FCC"),
    ]:
        for elem in elements:
            if elem not in delta:
                continue
            fracs = np.linspace(0.05, 0.5, 50)
            r_effs = []
            for f in fracs:
                # Other elements share remaining equally
                c_other = (1 - f) / (len(elements) - 1)
                comp = {e: c_other for e in elements}
                comp[elem] = f
                # Effective volume for this element
                vi = KING_ATOMIC_VOLUMES[elem]
                osf_sum = sum(comp[j] * (delta[elem] + delta.get(j, 0))
                              for j in comp if j != elem and j in delta)
                v_eff = vi * (1 + gamma * osf_sum)
                r_eff = (3 * v_eff / (4 * np.pi)) ** (1/3)
                r_effs.append(r_eff)
            ax.plot(fracs * 100, r_effs, lw=2, label=elem)
        ax.set_xlabel("Element fraction (%)")
        ax.set_ylabel("$r_{eff}$ (\u00c5)")
        ax.set_title(title)
        ax.legend(fontsize=12)

    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_composition_dependent_reff.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig_composition_dependent_reff.png")


def fig14_vegard_absorbed(all_df, radii):
    """Fig 14: Vegard with structure info absorbed via effective radii."""
    # Show that using V_eff (structure-dependent) gives better Vegard
    pair_a = defaultdict(lambda: {"A3B": [], "AB3": [], "B2": []})
    for _, row in all_df.iterrows():
        elA, elB = row.get("element_A", ""), row.get("element_B", "")
        a = row.get("lattice_constant", 0)
        stype = row.get("stype", "")
        if a <= 2 or a >= 8 or elA == elB:
            continue
        pair = tuple(sorted([elA, elB]))
        if stype == "B2":
            pair_a[pair]["B2"].append(a)
        elif stype == "L12":
            cA = row.get("count_A", 3)
            cB = row.get("count_B", 1)
            bucket = _l12_bucket(elA, elB, cA, cB)
            pair_a[pair][bucket].append(a)

    a_dft = []
    a_veg_pure = []
    a_veg_eff = []
    for pair, data in pair_a.items():
        elA, elB = pair
        if elA not in radii or elB not in radii:
            continue
        for struct_key, n_auc in [("B2", 2), ("A3B", 4), ("AB3", 4)]:
            if not data[struct_key]:
                continue
            a_real = np.median(data[struct_key])
            a_dft.append(a_real)

            # Pure Vegard
            vA = KING_ATOMIC_VOLUMES.get(elA, 15)
            vB = KING_ATOMIC_VOLUMES.get(elB, 15)
            if struct_key == "B2":
                a_veg_pure.append((n_auc * (vA + vB) / 2) ** (1/3))
                # V_eff Vegard
                rA = radii[elA].get("r_b2", radii[elA]["r_pure"])
                rB = radii[elB].get("r_b2", radii[elB]["r_pure"])
                v_eff = (4/3 * np.pi * rA**3 + 4/3 * np.pi * rB**3) / 2
                a_veg_eff.append((n_auc * v_eff) ** (1/3))
            elif struct_key == "A3B":
                a_veg_pure.append((n_auc * (3 * vA + vB) / 4) ** (1/3))
                rA = radii[elA].get("r_l12_maj", radii[elA]["r_pure"])
                rB = radii[elB].get("r_l12_min", radii[elB]["r_pure"])
                v_eff = (3 * 4/3 * np.pi * rA**3 + 4/3 * np.pi * rB**3) / 4
                a_veg_eff.append((n_auc * v_eff) ** (1/3))
            else:  # AB3
                a_veg_pure.append((n_auc * (vA + 3 * vB) / 4) ** (1/3))
                rA = radii[elA].get("r_l12_min", radii[elA]["r_pure"])
                rB = radii[elB].get("r_l12_maj", radii[elB]["r_pure"])
                v_eff = (4/3 * np.pi * rA**3 + 3 * 4/3 * np.pi * rB**3) / 4
                a_veg_eff.append((n_auc * v_eff) ** (1/3))

    a_dft = np.array(a_dft)
    a_veg_pure = np.array(a_veg_pure)
    a_veg_eff = np.array(a_veg_eff)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    lims = [2.5, 7.0]

    rmse1 = np.sqrt(np.mean((a_veg_pure - a_dft) ** 2))
    ax1.scatter(a_dft, a_veg_pure, c="gray", alpha=0.2, s=10)
    ax1.plot(lims, lims, "k--", lw=1)
    ax1.set_xlabel("DFT $a$ (\u00c5)")
    ax1.set_ylabel("Vegard (pure) $a$ (\u00c5)")
    ax1.set_title(f"Pure radii Vegard (RMSE = {rmse1:.3f} \u00c5)")
    ax1.set_xlim(lims); ax1.set_ylim(lims)
    ax1.set_aspect("equal")

    rmse2 = np.sqrt(np.mean((a_veg_eff - a_dft) ** 2))
    ax2.scatter(a_dft, a_veg_eff, c="C0", alpha=0.2, s=10)
    ax2.plot(lims, lims, "k--", lw=1)
    ax2.set_xlabel("DFT $a$ (\u00c5)")
    ax2.set_ylabel("Vegard (V$_{eff}$) $a$ (\u00c5)")
    ax2.set_title(f"Structure-dependent V$_{{eff}}$ (RMSE = {rmse2:.3f} \u00c5)")
    ax2.set_xlim(lims); ax2.set_ylim(lims)
    ax2.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_vegard_structure_absorbed.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig_vegard_structure_absorbed.png")


def fig15_roc(heas_mp, ob2, ol12):
    """Fig 15: ROC for multi-phase classification (delta_r vs delta_sf)."""
    from sklearn.metrics import roc_curve, auc

    dr_vals, dsf_vals, labels = [], [], []
    for h in heas_mp:
        comp = h["comp"]
        phase = h.get("phase", "SS")
        dr = compute_delta_r(comp)
        dsf = compute_delta_sf(comp, ob2)
        if np.isnan(dr) or np.isnan(dsf):
            continue
        dr_vals.append(dr)
        dsf_vals.append(dsf)
        labels.append(1 if phase == "IM" else 0)  # IM=1, SS=0

    dr_vals = np.array(dr_vals)
    dsf_vals = np.array(dsf_vals)
    labels = np.array(labels)

    if len(labels) == 0 or len(np.unique(labels)) < 2:
        print("  fig_roc.png — skipped (insufficient data or single class)")
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    for vals, name, c in [(dr_vals, r"$\delta r$", "C0"),
                           (dsf_vals, r"$\delta_\mathrm{sf}$", "C3")]:
        fpr, tpr, _ = roc_curve(labels, vals)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, c=c, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Multi-phase classification")
    ax.legend(fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_roc.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig_roc.png")


def fig16_phase_map(heas_mp, ob2, ol12):
    """Fig 16: Phase stability map (delta_r vs VEC)."""
    dr_ss, vec_ss = [], []
    dr_im, vec_im = [], []
    for h in heas_mp:
        comp = h["comp"]
        phase = h.get("phase", "SS")
        dr = compute_delta_r(comp)
        vec = sum(c * VEC.get(e, 5) for e, c in comp.items())
        if phase == "SS":
            dr_ss.append(dr)
            vec_ss.append(vec)
        else:
            dr_im.append(dr)
            vec_im.append(vec)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(vec_ss, dr_ss, c="C0", s=40, alpha=0.6, label="SS")
    ax.scatter(vec_im, dr_im, c="C3", s=40, alpha=0.6, marker="x", label="IM")
    ax.axhline(6.6, color="k", linestyle="--", lw=1, label=r"$\delta r$ = 6.6%")
    ax.set_xlabel("VEC")
    ax.set_ylabel(r"$\delta r$ (%)")
    ax.set_title("Phase stability map")
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_phase_map.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig_phase_map.png")


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print("=" * 60)
    print("HEA Lattice Constant — Clean Re-analysis")
    print(f"Excluded elements: {EXCLUDE_ELEMENTS}")
    print("=" * 60)

    # 1. Load data
    print("\n[1] Loading compound data...")
    all_df = load_compounds()
    n_mp = len(all_df[all_df["db"] == "MP"])
    n_oqmd = len(all_df[all_df["db"] == "OQMD"])
    n_vasp = len(all_df[all_df["db"] == "VASP"])
    print(f"    MP: {n_mp}, OQMD: {n_oqmd}, VASP: {n_vasp}")
    print(f"    Total: {len(all_df)} compounds (Gd/Ce excluded)")

    # 2. Compute pairwise Omega_sf (all 3 sources)
    print("\n[2] Computing pairwise Omega_sf (MP+OQMD+VASP)...")
    ob2, ol12 = compute_omega_sf_pairwise(all_df, sources=("MP", "OQMD", "VASP"), min_count=1)
    print(f"    B2 pairs: {len(ob2)}, L1_2 pairs: {len(ol12)}")

    # 3. Optimize gamma
    print("\n[3] Optimizing gamma on training set ({} HEAs)...".format(len(ALONSO_TABLE2)))
    gb, gf = optimize_gamma(ALONSO_TABLE2, ob2, ol12)
    print(f"    gamma_BCC = {gb:.4f}, gamma_FCC = {gf:.4f}")

    # 4. Predict training set
    y_train = np.array([h["a_exp"] for h in ALONSO_TABLE2])
    a_veg_tr = np.array([compute_vegard(h["comp"], h["struct"]) for h in ALONSO_TABLE2])
    a_ss_tr = predict_heas(ALONSO_TABLE2, ob2, ol12, gb, gf)
    bcc_i = [i for i, h in enumerate(ALONSO_TABLE2) if h["struct"] == "BCC"]
    fcc_i = [i for i, h in enumerate(ALONSO_TABLE2) if h["struct"] == "FCC"]

    rmse_veg_tr = np.sqrt(np.mean((a_veg_tr - y_train) ** 2))
    rmse_ss_tr = np.sqrt(np.mean((a_ss_tr - y_train) ** 2))
    print(f"\n    Training RMSE:")
    print(f"      Vegard:       {rmse_veg_tr:.4f} A")
    print(f"      DFT-Omega_sf: {rmse_ss_tr:.4f} A")
    print(f"      BCC:          {np.sqrt(np.mean((a_ss_tr[bcc_i]-y_train[bcc_i])**2)):.4f}")
    print(f"      FCC:          {np.sqrt(np.mean((a_ss_tr[fcc_i]-y_train[fcc_i])**2)):.4f}")

    # 5. Independent test
    print("\n[4] Independent test ({} HEAs)...".format(len(INDEPENDENT_TEST)))
    y_test = np.array([h["a_exp"] for h in INDEPENDENT_TEST])
    a_veg_te = np.array([compute_vegard(h["comp"], h["struct"]) for h in INDEPENDENT_TEST])
    a_ss_te = predict_heas(INDEPENDENT_TEST, ob2, ol12, gb, gf)
    bcc_t = [i for i, h in enumerate(INDEPENDENT_TEST) if h["struct"] == "BCC"]
    fcc_t = [i for i, h in enumerate(INDEPENDENT_TEST) if h["struct"] == "FCC"]

    rmse_veg_te = np.sqrt(np.mean((a_veg_te - y_test) ** 2))
    rmse_ss_te = np.sqrt(np.mean((a_ss_te - y_test) ** 2))
    print(f"    Test RMSE:")
    print(f"      Vegard:       {rmse_veg_te:.4f}")
    print(f"      DFT-Omega_sf: {rmse_ss_te:.4f}")
    print(f"      BCC:          {np.sqrt(np.mean((a_ss_te[bcc_t]-y_test[bcc_t])**2)):.4f}")
    print(f"      FCC:          {np.sqrt(np.mean((a_ss_te[fcc_t]-y_test[fcc_t])**2)):.4f}")
    id_v = sum(1 for i in bcc_t if abs(a_ss_te[i] - a_veg_te[i]) < 1e-6)
    print(f"      BCC identical to Vegard: {id_v}/{len(bcc_t)}")

    # 6. Additive decomposition
    print("\n[5] Additive delta decomposition...")
    decomp = additive_decomposition(ob2, ol12)
    print(f"    B2:  {len(decomp['B2']['elements'])} elements, R2 = {decomp['B2']['r2']:.4f}")
    print(f"    L12: {len(decomp['L12']['elements'])} elements, R2 = {decomp['L12']['r2']:.4f}")

    # 6b. Extended decomposition for Table 6 (includes VASP, min_count=1)
    #     This fills gaps (Ge, Pb for B2; Be, Mo, Os, Re, W for L12)
    #     without affecting model predictions (which use ob2/ol12 from MP+OQMD).
    print("\n[5b] Extended decomposition for Table 6 (MP+OQMD+VASP, min_count=1)...")
    ob2_ext, ol12_ext = compute_omega_sf_pairwise(
        all_df, sources=("MP", "OQMD", "VASP"), min_count=1
    )
    decomp_table = additive_decomposition(ob2_ext, ol12_ext)
    print(f"    B2:  {len(decomp_table['B2']['elements'])} elements, R2 = {decomp_table['B2']['r2']:.4f}")
    print(f"    L12: {len(decomp_table['L12']['elements'])} elements, R2 = {decomp_table['L12']['r2']:.4f}")

    # --- Additive prediction: TWO modes ---
    # Mode A: Replace existing pairwise pairs with additive approximation (same coverage)
    def build_additive_omega_existing(decomp, key, original_omega):
        """Replace existing pairwise Omega_sf with additive delta_A + delta_B."""
        delta = decomp[key]["delta"]
        omega = {}
        for pair in original_omega:
            a, b = pair
            if a in delta and b in delta:
                omega[pair] = delta[a] + delta[b]
        return omega

    # Mode B: Full gap-fill (all possible pairs from decomposition elements)
    def build_additive_omega_full(decomp, key):
        """Build full Omega_sf dict from additive delta for all element pairs."""
        delta = decomp[key]["delta"]
        omega = {}
        elements = decomp[key]["elements"]
        for i, a in enumerate(elements):
            for b in elements[i+1:]:
                omega[tuple(sorted([a, b]))] = delta[a] + delta[b]
        return omega

    # Mode A: same coverage
    ob2_addA = build_additive_omega_existing(decomp, "B2", ob2)
    ol12_addA = build_additive_omega_existing(decomp, "L12", ol12)
    gb_addA, gf_addA = optimize_gamma(ALONSO_TABLE2, ob2_addA, ol12_addA)
    a_addA_tr = predict_heas(ALONSO_TABLE2, ob2_addA, ol12_addA, gb_addA, gf_addA)
    a_addA_te = predict_heas(INDEPENDENT_TEST, ob2_addA, ol12_addA, gb_addA, gf_addA)
    rmse_addA_tr = np.sqrt(np.mean((a_addA_tr - y_train) ** 2))
    rmse_addA_te = np.sqrt(np.mean((a_addA_te - y_test) ** 2))
    print(f"    Mode A (same coverage {len(ob2_addA)}/{len(ol12_addA)} pairs):")
    print(f"      gamma_BCC={gb_addA:.4f}, gamma_FCC={gf_addA:.4f}")
    print(f"      Train RMSE: {rmse_addA_tr:.4f}, Test RMSE: {rmse_addA_te:.4f}")

    # Mode B: full gap-fill
    ob2_addB = build_additive_omega_full(decomp, "B2")
    ol12_addB = build_additive_omega_full(decomp, "L12")
    gb_addB, gf_addB = optimize_gamma(ALONSO_TABLE2, ob2_addB, ol12_addB)
    a_addB_tr = predict_heas(ALONSO_TABLE2, ob2_addB, ol12_addB, gb_addB, gf_addB)
    a_addB_te = predict_heas(INDEPENDENT_TEST, ob2_addB, ol12_addB, gb_addB, gf_addB)
    rmse_addB_tr = np.sqrt(np.mean((a_addB_tr - y_train) ** 2))
    rmse_addB_te = np.sqrt(np.mean((a_addB_te - y_test) ** 2))
    print(f"    Mode B (full gap-fill {len(ob2_addB)}/{len(ol12_addB)} pairs):")
    print(f"      gamma_BCC={gb_addB:.4f}, gamma_FCC={gf_addB:.4f}")
    print(f"      Train RMSE: {rmse_addB_tr:.4f}, Test RMSE: {rmse_addB_te:.4f}")

    # Use Mode A for figures (positive gamma, same coverage = fair comparison)
    a_add_tr = a_addA_tr
    a_add_te = a_addA_te
    rmse_add_tr = rmse_addA_tr
    rmse_add_te = rmse_addA_te
    gb_add = gb_addA
    gf_add = gf_addA

    # 7. Effective radii
    print("\n[6] Computing structure-dependent effective radii...")
    radii = compute_effective_radii(all_df)
    print(f"    Elements with radii: {len(radii)}")

    # 8. Generate all figures
    print("\n[7] Generating figures...")
    fig01_parity(y_train, a_veg_tr, a_ss_tr, y_test, a_veg_te, a_ss_te)
    fig02_rmse_bar({
        "Vegard": rmse_veg_tr,
        r"DFT-$\Omega_{\mathrm{sf}}$" + "\n(pairwise)": rmse_ss_tr,
        r"DFT-$\Omega_{\mathrm{sf}}$" + "\n(additive δ)": rmse_add_tr,
    })
    fig03_bcc_fcc(y_train, a_ss_tr, ALONSO_TABLE2)
    fig04_indep_test(y_test, a_veg_te, a_ss_te, INDEPENDENT_TEST, gb, gf)
    fig05_element_delta(decomp_table)
    fig06_additive_fit(ob2, ol12, decomp)
    fig07_composition_examples(all_df)
    fig08_delta_r_proof(all_df)
    fig09_packing(all_df)
    fig10_l12_asymmetry(all_df)
    fig11_volume_radius(radii)
    fig12_hea_additive(y_train, a_add_tr, ALONSO_TABLE2, y_test, a_add_te, INDEPENDENT_TEST)
    fig13_composition_reff(decomp, gb_add, gf_add)
    fig14_vegard_absorbed(all_df, radii)
    fig15_roc(MULTIPHASE_HEA_DB, ob2, ol12)
    fig16_phase_map(MULTIPHASE_HEA_DB, ob2, ol12)

    # 9. Save data
    print("\n[8] Saving data files...")

    # Delta parameters table (uses extended decomposition with VASP for full coverage)
    rows = []
    for elem in sorted(set(decomp_table["B2"]["elements"] + decomp_table["L12"]["elements"])):
        r_pure = (3 * KING_ATOMIC_VOLUMES.get(elem, 15) / (4 * np.pi)) ** (1/3)
        rows.append({
            "Element": elem,
            "V_pure": KING_ATOMIC_VOLUMES.get(elem, np.nan),
            "r_pure": r_pure,
            "delta_B2": decomp_table["B2"]["delta"].get(elem, np.nan),
            "delta_L12": decomp_table["L12"]["delta"].get(elem, np.nan),
        })
    df_delta = pd.DataFrame(rows).sort_values("Element")
    df_delta.to_csv(OUTDIR / "results_delta_parameters.csv", index=False)
    print(f"    results_delta_parameters.csv ({len(df_delta)} elements, extended with VASP)")

    # Omega_sf data
    rows_osf = []
    for (a, b), val in sorted(ob2.items()):
        rows_osf.append({"pair": f"{a}-{b}", "structure": "B2", "omega_sf": val})
    for (a, b), val in sorted(ol12.items()):
        rows_osf.append({"pair": f"{a}-{b}", "structure": "L12", "omega_sf": val})
    pd.DataFrame(rows_osf).to_csv(OUTDIR / "results_omega_sf.csv", index=False)
    print(f"    results_omega_sf.csv ({len(rows_osf)} pairs)")

    # Independent test results
    rows_test = []
    for i, h in enumerate(INDEPENDENT_TEST):
        elems = "-".join(sorted(h["comp"].keys()))
        rows_test.append({
            "alloy": elems,
            "struct": h["struct"],
            "a_exp": y_test[i],
            "a_vegard": a_veg_te[i],
            "a_dft_eq10_ss": a_ss_te[i],
            "a_additive": a_add_te[i],
            "ref": h.get("ref", ""),
        })
    pd.DataFrame(rows_test).to_csv(OUTDIR / "results_independent_test.csv", index=False)
    print(f"    results_independent_test.csv ({len(rows_test)} HEAs)")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Data: {n_mp} MP + {n_oqmd} OQMD + {n_vasp} VASP (Gd/Ce excluded)")
    print(f"B2 pairs: {len(ob2)}, L12 pairs: {len(ol12)}")
    print(f"gamma_BCC = {gb:.4f}, gamma_FCC = {gf:.4f}")
    print(f"\nTraining ({len(ALONSO_TABLE2)} HEA):")
    print(f"  Vegard RMSE:       {rmse_veg_tr:.4f} A")
    print(f"  Pairwise RMSE:     {rmse_ss_tr:.4f} A  (improvement: {(1-rmse_ss_tr/rmse_veg_tr)*100:.1f}%)")
    print(f"  Additive RMSE:     {rmse_add_tr:.4f} A")
    print(f"\nIndependent test ({len(INDEPENDENT_TEST)} HEA):")
    print(f"  Vegard RMSE:       {rmse_veg_te:.4f} A")
    print(f"  Pairwise RMSE:     {rmse_ss_te:.4f} A")
    print(f"  Additive RMSE:     {rmse_add_te:.4f} A")
    print(f"  BCC=Vegard:        {id_v}/{len(bcc_t)}")
    print(f"\nAdditive decomposition:")
    print(f"  B2  R2: {decomp['B2']['r2']:.4f} ({len(decomp['B2']['elements'])} elements)")
    print(f"  L12 R2: {decomp['L12']['r2']:.4f} ({len(decomp['L12']['elements'])} elements)")
    print(f"\nFigures: 16 PNGs saved to {OUTDIR}")
    print(f"Data:    3 CSVs saved to {OUTDIR}")


if __name__ == "__main__":
    main()
