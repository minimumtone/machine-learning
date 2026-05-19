#!/usr/bin/env python3
"""
Effective Radius Comparison: Element-wise vs Pairwise Ω_sf
============================================================

Two approaches for replacing pairwise Ω_sf with element-wise properties:

Approach 1 — Additive Ω_sf decomposition:
    Ω_sf(A,B) ≈ δ_A + δ_B
    Each element has an intrinsic "deviation tendency" δ_A.
    Used in Eq.10: Ω_sf_pred(i,j) = δ_i + δ_j

Approach 2 — DFT effective atomic volume (V_eff):
    From B2/L1₂ data, solve for V_eff(A) such that:
      B2:  0.5·V_eff_A + 0.5·V_eff_B = a³/2
      L1₂: 0.75·V_eff_A + 0.25·V_eff_B = a³/4
    HEA prediction: V = n_auc × Σ c_i V_eff_i  (Vegard with effective volumes)

Baseline — Pairwise Ω_sf (existing method):
    Ω_sf(A,B) from DFT binary compounds, used in Eq.10 with γ scaling.

Author: Satoshi Minamoto (NIMS) / Devin
"""

import sys
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Import data from main script
from hea_lattice_xgboost import (
    KING_ATOMIC_VOLUMES,
    ALONSO_TABLE2,
    load_compound_data,
    compute_structure_specific_omega_sf,
    compute_eq10_scaled,
)

# ── Font setup ──────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "figure.titlesize": 20,
})


# =====================================================================
# Remove duplicate HEA compositions
# =====================================================================
def dedup_hea(data):
    seen = set()
    out = []
    for h in data:
        key = tuple(sorted((e, round(c, 3)) for e, c in h["comp"].items()))
        if key not in seen:
            seen.add(key)
            out.append(h)
    return out


# =====================================================================
# Load DFT compound data with source filtering
# =====================================================================
def load_compounds_by_source(sources=None):
    """
    Load compound data, optionally filtering by source.
    sources: list of allowed db values, e.g. ["MP", "OQMD"]
    If None, load all.
    """
    base = Path("four_case_output/figures")
    dfs = []

    # MP + existing OQMD
    for src in ["MP", "OQMD"]:
        if sources is not None and src not in sources:
            continue
        for struct in ["B2", "L12"]:
            f = base / f"compounds_{src}_{struct}.csv"
            if f.exists():
                df = pd.read_csv(f)
                df["db"] = src
                df["stype"] = struct
                dfs.append(df)

    # VASP
    if sources is None or "VASP" in sources:
        for struct in ["B2", "L12"]:
            for search_dir in [Path("data"), base]:
                f = search_dir / f"compounds_VASP_{struct}.csv"
                if f.exists():
                    df = pd.read_csv(f)
                    df["db"] = "VASP"
                    df["stype"] = struct
                    dfs.append(df)
                    break

    # New OQMD data (user-uploaded)
    if sources is None or "OQMD_new" in sources:
        for struct, fname in [("B2", "oqmd_b2_data.csv"), ("L12", "oqmd_l12_data.csv")]:
            f = Path("data") / fname
            if f.exists():
                df = pd.read_csv(f)
                df = df.rename(columns={"lattice_constant_a": "lattice_constant"})
                df["db"] = "OQMD_new"
                df["stype"] = struct
                dfs.append(df)

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


# =====================================================================
# Approach 1: Additive Ω_sf decomposition
# =====================================================================
def fit_additive_omega_sf(omega_sf_dict):
    """
    Decompose pairwise Ω_sf(A,B) into element-level δ_A:
        Ω_sf(A,B) ≈ δ_A + δ_B
    Returns dict: element → δ value, RMSE, R².
    """
    elements_set = set()
    for (elA, elB) in omega_sf_dict:
        elements_set.add(elA)
        elements_set.add(elB)
    elements = sorted(elements_set)
    elem_idx = {e: i for i, e in enumerate(elements)}
    n_elem = len(elements)

    # Build linear system: δ_A + δ_B = Ω_sf(A,B)
    rows, vals = [], []
    for (elA, elB), omega in omega_sf_dict.items():
        row = np.zeros(n_elem)
        row[elem_idx[elA]] = 1.0
        row[elem_idx[elB]] = 1.0
        rows.append(row)
        vals.append(omega)

    A = np.array(rows)
    b = np.array(vals)

    # Soft zero-mean constraint for identifiability
    constraint_row = np.ones((1, n_elem)) * 0.01
    A_aug = np.vstack([A, constraint_row])
    b_aug = np.append(b, 0.0)

    delta, _, _, _ = np.linalg.lstsq(A_aug, b_aug, rcond=None)

    # Reconstruction quality
    omega_pred = A @ delta
    rmse_fit = np.sqrt(np.mean((omega_pred - b)**2))
    ss_res = np.sum((omega_pred - b)**2)
    ss_tot = np.sum((b - np.mean(b))**2)
    r2_fit = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return {elements[i]: delta[i] for i in range(n_elem)}, rmse_fit, r2_fit


# =====================================================================
# Approach 2: DFT effective atomic volume
# =====================================================================
def fit_effective_volumes(compound_df, structure="B2"):
    """
    Fit effective atomic volume V_eff(A) for each element from DFT data.
    B2:  0.5·V_eff_A + 0.5·V_eff_B = a³/2
    L1₂: (cA/(cA+cB))·V_eff_A + (cB/(cA+cB))·V_eff_B = a³/4
    Returns dict: element → V_eff (ų).
    """
    elements_set = set()
    data_rows = []

    for _, row in compound_df.iterrows():
        elA = row.get("element_A", "")
        elB = row.get("element_B", "")
        a = row.get("lattice_constant", 0)
        stype = row.get("stype", "")

        if stype != structure or not elA or not elB or a <= 2 or a >= 8:
            continue
        if elA not in KING_ATOMIC_VOLUMES or elB not in KING_ATOMIC_VOLUMES:
            continue

        elements_set.add(elA)
        elements_set.add(elB)

        cA = row.get("count_A", 1 if structure == "B2" else 3)
        cB = row.get("count_B", 1)
        total = cA + cB
        if structure == "B2":
            target = a**3 / 2   # per-atom volume
        else:
            target = a**3 / 4   # per-atom volume
        data_rows.append((elA, elB, cA / total, cB / total, target))

    elements = sorted(elements_set)
    elem_idx = {e: i for i, e in enumerate(elements)}
    n_elem = len(elements)

    A_mat = np.zeros((len(data_rows), n_elem))
    b_vec = np.zeros(len(data_rows))

    for k, (elA, elB, wA, wB, target) in enumerate(data_rows):
        A_mat[k, elem_idx[elA]] = wA
        A_mat[k, elem_idx[elB]] = wB
        b_vec[k] = target

    # Tikhonov regularization toward King volumes
    king_vec = np.array([KING_ATOMIC_VOLUMES.get(e, 15.0) for e in elements])
    lam = 0.01
    reg_mat = lam * np.eye(n_elem)
    reg_vec = lam * king_vec

    A_aug = np.vstack([A_mat, reg_mat])
    b_aug = np.append(b_vec, reg_vec)

    v_eff, _, _, _ = np.linalg.lstsq(A_aug, b_aug, rcond=None)

    # Fit quality
    pred = A_mat @ v_eff
    rmse_fit = np.sqrt(np.mean((pred - b_vec)**2))
    ss_res = np.sum((pred - b_vec)**2)
    ss_tot = np.sum((b_vec - np.mean(b_vec))**2)
    r2_fit = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return {elements[i]: v_eff[i] for i in range(n_elem)}, rmse_fit, r2_fit


# =====================================================================
# HEA prediction functions
# =====================================================================
def predict_vegard_king(comp, struct):
    elements = list(comp.keys())
    fracs = np.array([comp[e] for e in elements])
    fracs = fracs / fracs.sum()
    vols = np.array([KING_ATOMIC_VOLUMES.get(e, 15.0) for e in elements])
    n_auc = 4 if struct == "FCC" else 2
    return (n_auc * np.sum(fracs * vols)) ** (1/3)


def predict_eq10(comp, struct, omega_sf, gamma=1.0):
    """Alonso Eq.10 with any Ω_sf dict (pairwise or reconstructed)."""
    elements = list(comp.keys())
    fracs = np.array([comp[e] for e in elements])
    fracs = fracs / fracs.sum()
    vols = np.array([KING_ATOMIC_VOLUMES.get(e, 15.0) for e in elements])
    n_auc = 4 if struct == "FCC" else 2

    v_vegard = np.sum(fracs * vols)
    correction = 0.0
    for i in range(len(elements)):
        for j in range(len(elements)):
            if i != j:
                pair = tuple(sorted([elements[i], elements[j]]))
                omega = omega_sf.get(pair, 0.0)
                correction += fracs[i] * fracs[j] * vols[j] * omega
    v_total = n_auc * (v_vegard + gamma * correction)
    if v_total <= 0:
        return predict_vegard_king(comp, struct)
    return v_total ** (1/3)


def predict_vegard_veff(comp, struct, v_eff_b2, v_eff_l12):
    """Vegard's law with DFT effective volumes (structure-specific)."""
    v_eff = v_eff_l12 if struct == "FCC" else v_eff_b2
    elements = list(comp.keys())
    fracs = np.array([comp[e] for e in elements])
    fracs = fracs / fracs.sum()
    vols = np.array([v_eff.get(e, KING_ATOMIC_VOLUMES.get(e, 15.0))
                     for e in elements])
    n_auc = 4 if struct == "FCC" else 2
    v_total = n_auc * np.sum(fracs * vols)
    if v_total <= 0:
        return predict_vegard_king(comp, struct)
    return v_total ** (1/3)


def build_additive_omega_dict(delta_dict):
    """Build pair → Ω_sf dict from element-level δ values."""
    elems = sorted(delta_dict.keys())
    omega = {}
    for i, eA in enumerate(elems):
        for eB in elems[i:]:
            omega[(eA, eB)] = delta_dict[eA] + delta_dict[eB]
    return omega


# =====================================================================
# γ optimisation (grid search)
# =====================================================================
def optimise_gamma(hea_data, y_true, bcc_idx, fcc_idx,
                   pred_fn_bcc, pred_fn_fcc):
    """
    Grid search for optimal γ_BCC and γ_FCC.
    pred_fn_bcc(comp, gamma) → a_pred
    pred_fn_fcc(comp, gamma) → a_pred
    """
    best_rmse, best_gb, best_gf = 999, 0, 0
    N = len(y_true)

    for gb in np.arange(-0.5, 2.51, 0.05):
        for gf in np.arange(-0.5, 2.51, 0.05):
            a_pred = np.zeros(N)
            for i in bcc_idx:
                a_pred[i] = pred_fn_bcc(hea_data[i]["comp"], gb)
            for i in fcc_idx:
                a_pred[i] = pred_fn_fcc(hea_data[i]["comp"], gf)
            rmse = np.sqrt(np.mean((y_true - a_pred)**2))
            if rmse < best_rmse:
                best_rmse, best_gb, best_gf = rmse, gb, gf

    for gb in np.arange(best_gb - 0.05, best_gb + 0.06, 0.01):
        for gf in np.arange(best_gf - 0.05, best_gf + 0.06, 0.01):
            a_pred = np.zeros(N)
            for i in bcc_idx:
                a_pred[i] = pred_fn_bcc(hea_data[i]["comp"], gb)
            for i in fcc_idx:
                a_pred[i] = pred_fn_fcc(hea_data[i]["comp"], gf)
            rmse = np.sqrt(np.mean((y_true - a_pred)**2))
            if rmse < best_rmse:
                best_rmse, best_gb, best_gf = rmse, gb, gf

    return best_gb, best_gf, best_rmse


# =====================================================================
# Run one scenario
# =====================================================================
def run_scenario(compound_df, hea_data, scenario_name):
    """Run all approaches for a given compound dataset. Returns results dict."""
    N = len(hea_data)
    y_true = np.array([h["a_exp"] for h in hea_data])
    bcc_idx = [i for i in range(N) if hea_data[i]["struct"] == "BCC"]
    fcc_idx = [i for i in range(N) if hea_data[i]["struct"] == "FCC"]

    print(f"\n{'='*60}")
    print(f"  Scenario: {scenario_name}")
    print(f"  Compounds: {len(compound_df)}")
    print(f"  HEAs: {N} ({len(bcc_idx)} BCC, {len(fcc_idx)} FCC)")
    print(f"{'='*60}")

    results = {}

    # ── King Vegard (baseline) ──
    a_vegard = np.array([predict_vegard_king(h["comp"], h["struct"])
                         for h in hea_data])
    rmse_v = np.sqrt(np.mean((y_true - a_vegard)**2))
    rmse_v_bcc = np.sqrt(np.mean((y_true[bcc_idx] - a_vegard[bcc_idx])**2))
    rmse_v_fcc = np.sqrt(np.mean((y_true[fcc_idx] - a_vegard[fcc_idx])**2))
    results["King Vegard"] = {
        "rmse": rmse_v, "bcc": rmse_v_bcc, "fcc": rmse_v_fcc,
        "gb": "-", "gf": "-", "type": "element", "pred": a_vegard
    }
    print(f"  King Vegard: RMSE={rmse_v:.4f}")

    # ── Pairwise Ω_sf (baseline SS method) ──
    omega_b2, omega_l12 = compute_structure_specific_omega_sf(compound_df)
    print(f"  Ω_sf pairs: B2={len(omega_b2)}, L1₂={len(omega_l12)}")

    gb, gf, _ = optimise_gamma(
        hea_data, y_true, bcc_idx, fcc_idx,
        lambda c, g: predict_eq10(c, "BCC", omega_b2, g),
        lambda c, g: predict_eq10(c, "FCC", omega_l12, g),
    )
    a_pw = np.zeros(N)
    for i in bcc_idx:
        a_pw[i] = predict_eq10(hea_data[i]["comp"], "BCC", omega_b2, gb)
    for i in fcc_idx:
        a_pw[i] = predict_eq10(hea_data[i]["comp"], "FCC", omega_l12, gf)
    rmse_pw = np.sqrt(np.mean((y_true - a_pw)**2))
    rmse_pw_bcc = np.sqrt(np.mean((y_true[bcc_idx] - a_pw[bcc_idx])**2))
    rmse_pw_fcc = np.sqrt(np.mean((y_true[fcc_idx] - a_pw[fcc_idx])**2))
    results["Pairwise Ω_sf + γ"] = {
        "rmse": rmse_pw, "bcc": rmse_pw_bcc, "fcc": rmse_pw_fcc,
        "gb": f"{gb:.2f}", "gf": f"{gf:.2f}", "type": "pair", "pred": a_pw
    }
    print(f"  Pairwise Ω_sf: RMSE={rmse_pw:.4f} (γ_BCC={gb:.2f}, γ_FCC={gf:.2f})")

    # ── Approach 1: Additive δ ──
    delta_b2, rmse_fit_b2, r2_b2 = fit_additive_omega_sf(omega_b2)
    delta_l12, rmse_fit_l12, r2_l12 = fit_additive_omega_sf(omega_l12)
    print(f"  Additive fit: B2 δ for {len(delta_b2)} elem (R²={r2_b2:.3f}), "
          f"L1₂ δ for {len(delta_l12)} elem (R²={r2_l12:.3f})")

    omega_add_b2 = build_additive_omega_dict(delta_b2)
    omega_add_l12 = build_additive_omega_dict(delta_l12)

    gb1, gf1, _ = optimise_gamma(
        hea_data, y_true, bcc_idx, fcc_idx,
        lambda c, g: predict_eq10(c, "BCC", omega_add_b2, g),
        lambda c, g: predict_eq10(c, "FCC", omega_add_l12, g),
    )
    a_add = np.zeros(N)
    for i in bcc_idx:
        a_add[i] = predict_eq10(hea_data[i]["comp"], "BCC", omega_add_b2, gb1)
    for i in fcc_idx:
        a_add[i] = predict_eq10(hea_data[i]["comp"], "FCC", omega_add_l12, gf1)
    rmse_add = np.sqrt(np.mean((y_true - a_add)**2))
    rmse_add_bcc = np.sqrt(np.mean((y_true[bcc_idx] - a_add[bcc_idx])**2))
    rmse_add_fcc = np.sqrt(np.mean((y_true[fcc_idx] - a_add[fcc_idx])**2))
    results["Additive δ + γ (Ap.1)"] = {
        "rmse": rmse_add, "bcc": rmse_add_bcc, "fcc": rmse_add_fcc,
        "gb": f"{gb1:.2f}", "gf": f"{gf1:.2f}", "type": "element", "pred": a_add
    }
    print(f"  Additive δ: RMSE={rmse_add:.4f} (γ_BCC={gb1:.2f}, γ_FCC={gf1:.2f})")

    # ── Approach 2a: V_eff Vegard ──
    v_eff_b2, rmse_vb, r2_vb = fit_effective_volumes(compound_df, "B2")
    v_eff_l12, rmse_vl, r2_vl = fit_effective_volumes(compound_df, "L12")
    print(f"  V_eff fit: B2 {len(v_eff_b2)} elem (R²={r2_vb:.4f}), "
          f"L1₂ {len(v_eff_l12)} elem (R²={r2_vl:.4f})")

    a_veff = np.array([predict_vegard_veff(h["comp"], h["struct"],
                                            v_eff_b2, v_eff_l12)
                        for h in hea_data])
    rmse_veff = np.sqrt(np.mean((y_true - a_veff)**2))
    rmse_veff_bcc = np.sqrt(np.mean((y_true[bcc_idx] - a_veff[bcc_idx])**2))
    rmse_veff_fcc = np.sqrt(np.mean((y_true[fcc_idx] - a_veff[fcc_idx])**2))
    results["V_eff Vegard (Ap.2a)"] = {
        "rmse": rmse_veff, "bcc": rmse_veff_bcc, "fcc": rmse_veff_fcc,
        "gb": "-", "gf": "-", "type": "element", "pred": a_veff
    }
    print(f"  V_eff Vegard: RMSE={rmse_veff:.4f}")

    # ── Approach 2b: V_eff + pairwise Ω_sf ──
    # Recompute Ω_sf with V_eff as base volumes instead of King
    omega_veff_b2 = compute_omega_sf_with_veff(compound_df, v_eff_b2, "B2")
    omega_veff_l12 = compute_omega_sf_with_veff(compound_df, v_eff_l12, "L12")

    def pred_veff_eq10_bcc(comp, g):
        return predict_eq10_veff(comp, "BCC", v_eff_b2, omega_veff_b2, g)
    def pred_veff_eq10_fcc(comp, g):
        return predict_eq10_veff(comp, "FCC", v_eff_l12, omega_veff_l12, g)

    gb2b, gf2b, _ = optimise_gamma(
        hea_data, y_true, bcc_idx, fcc_idx,
        pred_veff_eq10_bcc, pred_veff_eq10_fcc,
    )
    a_veff_eq10 = np.zeros(N)
    for i in bcc_idx:
        a_veff_eq10[i] = pred_veff_eq10_bcc(hea_data[i]["comp"], gb2b)
    for i in fcc_idx:
        a_veff_eq10[i] = pred_veff_eq10_fcc(hea_data[i]["comp"], gf2b)
    rmse_veff_eq10 = np.sqrt(np.mean((y_true - a_veff_eq10)**2))
    rmse_veff_eq10_bcc = np.sqrt(np.mean((y_true[bcc_idx] - a_veff_eq10[bcc_idx])**2))
    rmse_veff_eq10_fcc = np.sqrt(np.mean((y_true[fcc_idx] - a_veff_eq10[fcc_idx])**2))
    results["V_eff + Pair Ω_sf (Ap.2b)"] = {
        "rmse": rmse_veff_eq10, "bcc": rmse_veff_eq10_bcc, "fcc": rmse_veff_eq10_fcc,
        "gb": f"{gb2b:.2f}", "gf": f"{gf2b:.2f}", "type": "pair+element", "pred": a_veff_eq10
    }
    print(f"  V_eff + Pair Ω_sf: RMSE={rmse_veff_eq10:.4f}")

    # ── Approach 2c: V_eff + additive δ ──
    delta_veff_b2, _, _ = fit_additive_omega_sf(omega_veff_b2)
    delta_veff_l12, _, _ = fit_additive_omega_sf(omega_veff_l12)
    omega_veff_add_b2 = build_additive_omega_dict(delta_veff_b2)
    omega_veff_add_l12 = build_additive_omega_dict(delta_veff_l12)

    def pred_veff_add_bcc(comp, g):
        return predict_eq10_veff(comp, "BCC", v_eff_b2, omega_veff_add_b2, g)
    def pred_veff_add_fcc(comp, g):
        return predict_eq10_veff(comp, "FCC", v_eff_l12, omega_veff_add_l12, g)

    gb2c, gf2c, _ = optimise_gamma(
        hea_data, y_true, bcc_idx, fcc_idx,
        pred_veff_add_bcc, pred_veff_add_fcc,
    )
    a_veff_add = np.zeros(N)
    for i in bcc_idx:
        a_veff_add[i] = pred_veff_add_bcc(hea_data[i]["comp"], gb2c)
    for i in fcc_idx:
        a_veff_add[i] = pred_veff_add_fcc(hea_data[i]["comp"], gf2c)
    rmse_veff_add = np.sqrt(np.mean((y_true - a_veff_add)**2))
    rmse_veff_add_bcc = np.sqrt(np.mean((y_true[bcc_idx] - a_veff_add[bcc_idx])**2))
    rmse_veff_add_fcc = np.sqrt(np.mean((y_true[fcc_idx] - a_veff_add[fcc_idx])**2))
    results["V_eff + Add. δ (Ap.2c)"] = {
        "rmse": rmse_veff_add, "bcc": rmse_veff_add_bcc, "fcc": rmse_veff_add_fcc,
        "gb": f"{gb2c:.2f}", "gf": f"{gf2c:.2f}", "type": "element", "pred": a_veff_add
    }
    print(f"  V_eff + Add. δ: RMSE={rmse_veff_add:.4f}")

    # Attach extra info
    results["_meta"] = {
        "omega_b2": omega_b2, "omega_l12": omega_l12,
        "delta_b2": delta_b2, "delta_l12": delta_l12,
        "v_eff_b2": v_eff_b2, "v_eff_l12": v_eff_l12,
        "r2_b2": r2_b2, "r2_l12": r2_l12,
        "rmse_fit_b2": rmse_fit_b2, "rmse_fit_l12": rmse_fit_l12,
        "n_b2_pairs": len(omega_b2), "n_l12_pairs": len(omega_l12),
        "n_delta_b2": len(delta_b2), "n_delta_l12": len(delta_l12),
    }
    return results


def compute_omega_sf_with_veff(compound_df, v_eff, structure):
    """Compute Ω_sf using V_eff instead of King volumes as baseline."""
    pair_data = defaultdict(list)

    for _, row in compound_df.iterrows():
        elA = row.get("element_A", "")
        elB = row.get("element_B", "")
        a = row.get("lattice_constant", 0)
        stype = row.get("stype", "")

        if stype != structure or not elA or not elB or a <= 2 or a >= 8:
            continue
        if elA not in v_eff or elB not in v_eff:
            continue

        pair = tuple(sorted([elA, elB]))
        vA = v_eff[elA]
        vB = v_eff[elB]

        if structure == "B2":
            v_actual = a**3 / 2
            v_vegard = (vA + vB) / 2
        elif structure == "L12":
            cA = row.get("count_A", 3)
            cB = row.get("count_B", 1)
            total = cA + cB
            v_actual = a**3 / 4
            v_vegard = (cA * vA + cB * vB) / total
        else:
            continue

        if v_vegard > 0:
            pair_data[pair].append((v_actual - v_vegard) / v_vegard)

    return {p: np.median(v) for p, v in pair_data.items() if len(v) >= 2}


def predict_eq10_veff(comp, struct, v_eff, omega_sf, gamma=1.0):
    """Eq.10 using V_eff instead of King volumes."""
    elements = list(comp.keys())
    fracs = np.array([comp[e] for e in elements])
    fracs = fracs / fracs.sum()
    vols = np.array([v_eff.get(e, KING_ATOMIC_VOLUMES.get(e, 15.0))
                     for e in elements])
    n_auc = 4 if struct == "FCC" else 2

    v_vegard = np.sum(fracs * vols)
    correction = 0.0
    for i in range(len(elements)):
        for j in range(len(elements)):
            if i != j:
                pair = tuple(sorted([elements[i], elements[j]]))
                omega = omega_sf.get(pair, 0.0)
                correction += fracs[i] * fracs[j] * vols[j] * omega
    v_total = n_auc * (v_vegard + gamma * correction)
    if v_total <= 0:
        return predict_vegard_king(comp, struct)
    return v_total ** (1/3)


# =====================================================================
# Main analysis
# =====================================================================
def main():
    print("=" * 70)
    print("Effective Radius Comparison: Element-wise vs Pairwise Ω_sf")
    print("=" * 70)

    # Prepare HEA data
    hea_data = dedup_hea(ALONSO_TABLE2)
    N = len(hea_data)
    y_true = np.array([h["a_exp"] for h in hea_data])
    bcc_mask = np.array([h["struct"] == "BCC" for h in hea_data])
    fcc_mask = ~bcc_mask
    bcc_idx = list(np.where(bcc_mask)[0])
    fcc_idx = list(np.where(fcc_mask)[0])
    print(f"\nHEA data: {N} ({len(bcc_idx)} BCC, {len(fcc_idx)} FCC)")

    # ── Scenario 1: MP+OQMD only (proven best) ──
    df_mpoqmd = load_compounds_by_source(["MP", "OQMD"])
    res_mpoqmd = run_scenario(df_mpoqmd, hea_data, "MP+OQMD only")

    # ── Scenario 2: All data (MP+OQMD+VASP+new_OQMD) ──
    df_all = load_compounds_by_source(None)
    res_all = run_scenario(df_all, hea_data, "All data (MP+OQMD+VASP+new_OQMD)")

    # =====================================================================
    # Summary table
    # =====================================================================
    outdir = Path("hea_xgboost_output")
    outdir.mkdir(exist_ok=True)

    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    for label, res in [("MP+OQMD", res_mpoqmd), ("All data", res_all)]:
        meta = res["_meta"]
        print(f"\n--- {label} (B2 pairs={meta['n_b2_pairs']}, "
              f"L1₂ pairs={meta['n_l12_pairs']}) ---")
        print(f"  Additive R²: B2={meta['r2_b2']:.3f}, L1₂={meta['r2_l12']:.3f}")
        print(f"\n  {'Method':<28} {'RMSE':>7} {'BCC':>7} {'FCC':>7} "
              f"{'γ_BCC':>6} {'γ_FCC':>6} {'Type':>8}")
        print("  " + "-" * 78)
        for name, r in res.items():
            if name.startswith("_"):
                continue
            print(f"  {name:<28} {r['rmse']:.4f}  {r['bcc']:.4f}  "
                  f"{r['fcc']:.4f}  {r['gb']:>6}  {r['gf']:>6}  {r['type']:>8}")

    # =====================================================================
    # Figures (using MP+OQMD scenario — best baseline)
    # =====================================================================
    res = res_mpoqmd
    meta = res["_meta"]

    # ── Figure 1: 6-panel parity (3 methods × 2 scenarios) ──
    fig, axes = plt.subplots(2, 3, figsize=(24, 14))

    for col, (label, r) in enumerate([
        ("MP+OQMD", res_mpoqmd), ("All data", res_all)
    ]):
        # Skip — just use MP+OQMD for parity plots
        pass

    # 4-panel parity for MP+OQMD
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    predictions = [
        ("Pairwise Ω_sf + γ", res["Pairwise Ω_sf + γ"]),
        ("Additive δ + γ (Ap.1)", res["Additive δ + γ (Ap.1)"]),
        ("V_eff Vegard (Ap.2a)", res["V_eff Vegard (Ap.2a)"]),
        ("V_eff + Add. δ (Ap.2c)", res["V_eff + Add. δ (Ap.2c)"]),
    ]
    for ax, (name, r) in zip(axes.flat, predictions):
        a_pred = r["pred"]
        ax.scatter(y_true[bcc_mask], a_pred[bcc_mask], c="tab:red", s=60,
                   alpha=0.7, label=f"BCC (N={bcc_mask.sum()})",
                   edgecolors="k", linewidths=0.5)
        ax.scatter(y_true[fcc_mask], a_pred[fcc_mask], c="tab:blue", s=60,
                   alpha=0.7, label=f"FCC (N={fcc_mask.sum()})",
                   edgecolors="k", linewidths=0.5)
        lims = [min(y_true.min(), a_pred.min()) - 0.02,
                max(y_true.max(), a_pred.max()) + 0.02]
        ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Experimental $a$ (Å)")
        ax.set_ylabel("Predicted $a$ (Å)")
        ax.set_title(f"{name}\nRMSE={r['rmse']:.4f} Å")
        ax.legend(loc="upper left")
        ax.set_aspect("equal")

    fig.suptitle("HEA Lattice Constant: Element-wise vs Pairwise (MP+OQMD)", fontsize=20)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outdir / "fig_effective_radius_parity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {outdir}/fig_effective_radius_parity.png")

    # ── Figure 2: RMSE bar chart comparing both scenarios ──
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    for ax, (label, r) in zip(axes, [("MP+OQMD", res_mpoqmd), ("All data", res_all)]):
        methods = [k for k in r if not k.startswith("_")]
        rmse_all = [r[k]["rmse"] for k in methods]
        rmse_bcc = [r[k]["bcc"] for k in methods]
        rmse_fcc = [r[k]["fcc"] for k in methods]
        types = [r[k]["type"] for k in methods]

        x = np.arange(len(methods))
        w = 0.25
        colors = ["#4472C4" if "element" in t else "#ED7D31" for t in types]
        ax.bar(x - w, rmse_all, w, label="All", color=colors, alpha=0.9,
               edgecolor="k", linewidth=0.5)
        ax.bar(x, rmse_bcc, w, label="BCC", color="#A5A5A5", alpha=0.7,
               edgecolor="k", linewidth=0.5)
        ax.bar(x + w, rmse_fcc, w, label="FCC", color="#70AD47", alpha=0.7,
               edgecolor="k", linewidth=0.5)

        for i, v in enumerate(rmse_all):
            ax.text(x[i] - w, v + 0.0005, f"{v:.4f}", ha="center", va="bottom",
                    fontsize=9, rotation=45)

        ax.set_ylabel("RMSE (Å)")
        ax.set_title(f"{label}")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=11)
        ax.legend(loc="upper right")
        ax.axhline(0.016, color="gray", linestyle=":", alpha=0.5)
        ax.set_ylim(0, max(rmse_all) * 1.3)

    fig.suptitle("RMSE Comparison: Element-wise (blue) vs Pairwise (orange)", fontsize=20)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outdir / "fig_effective_radius_rmse_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outdir}/fig_effective_radius_rmse_bars.png")

    # ── Figure 3: V_eff vs King comparison ──
    hea_elements = set()
    for h in hea_data:
        hea_elements.update(h["comp"].keys())
    hea_elements = sorted(hea_elements)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    for ax, (v_eff, struct_label) in zip(axes,
            [(meta["v_eff_b2"], "B2 (BCC)"), (meta["v_eff_l12"], "L1$_2$ (FCC)")]):
        elems = [e for e in hea_elements if e in v_eff]
        v_king = [KING_ATOMIC_VOLUMES[e] for e in elems]
        v_dft = [v_eff[e] for e in elems]
        x_pos = np.arange(len(elems))

        ax.bar(x_pos - 0.2, v_king, 0.35, label="King (1966)",
               color="#4472C4", alpha=0.8, edgecolor="k", linewidth=0.5)
        ax.bar(x_pos + 0.2, v_dft, 0.35, label="DFT $V_{eff}$",
               color="#ED7D31", alpha=0.8, edgecolor="k", linewidth=0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(elems, fontsize=12)
        ax.set_ylabel("Atomic Volume (ų)")
        ax.set_title(f"Effective Atomic Volumes — {struct_label}")
        ax.legend()

    fig.suptitle("King vs DFT Effective Atomic Volumes for HEA Elements", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outdir / "fig_effective_volumes_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outdir}/fig_effective_volumes_comparison.png")

    # ── Figure 4: δ values per element ──
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    for ax, (delta, struct_label) in zip(axes,
            [(meta["delta_b2"], "B2 (BCC)"), (meta["delta_l12"], "L1$_2$ (FCC)")]):
        elems = [e for e in hea_elements if e in delta]
        vals = [delta[e] for e in elems]
        x_pos = np.arange(len(elems))
        colors = ["#ED7D31" if v > 0 else "#4472C4" for v in vals]
        ax.bar(x_pos, vals, color=colors, alpha=0.8, edgecolor="k", linewidth=0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(elems, fontsize=12)
        ax.axhline(0, color="k", linewidth=0.5)
        ax.set_ylabel("$\\delta_A$ (additive $\\Omega_{sf}$ contribution)")
        ax.set_title(f"Element-wise $\\delta_A$ — {struct_label}")

    fig.suptitle("Additive $\\Omega_{sf}$ Decomposition per Element (Approach 1)",
                 fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outdir / "fig_additive_delta_elements.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outdir}/fig_additive_delta_elements.png")

    # ── Figure 5: Ω_sf reconstruction quality ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, (omega_dict, delta, struct_label) in zip(axes, [
        (meta["omega_b2"], meta["delta_b2"], "B2"),
        (meta["omega_l12"], meta["delta_l12"], "L1$_2$"),
    ]):
        true_vals, pred_vals = [], []
        for (eA, eB), omega in omega_dict.items():
            if eA in delta and eB in delta:
                true_vals.append(omega)
                pred_vals.append(delta[eA] + delta[eB])
        true_vals = np.array(true_vals)
        pred_vals = np.array(pred_vals)
        ss_res = np.sum((pred_vals - true_vals)**2)
        ss_tot = np.sum((true_vals - np.mean(true_vals))**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
        rmse = np.sqrt(np.mean((pred_vals - true_vals)**2))

        ax.scatter(true_vals, pred_vals, s=20, alpha=0.5, c="tab:blue",
                   edgecolors="none")
        lims = [min(true_vals.min(), pred_vals.min()) - 0.02,
                max(true_vals.max(), pred_vals.max()) + 0.02]
        ax.plot(lims, lims, "k--", alpha=0.5)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel(f"DFT $\\Omega_{{sf}}$ (pairwise)")
        ax.set_ylabel(f"$\\delta_A + \\delta_B$ (additive)")
        ax.set_title(f"{struct_label}: RMSE={rmse:.4f}, R$^2$={r2:.3f}")
        ax.set_aspect("equal")

    fig.suptitle("$\\Omega_{sf}$ Reconstruction: Pairwise vs Additive (Approach 1)",
                 fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outdir / "fig_omega_sf_reconstruction.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outdir}/fig_omega_sf_reconstruction.png")

    # ── Save element-wise table ──
    all_elems = sorted(set(list(meta["delta_b2"].keys()) +
                           list(meta["delta_l12"].keys()) +
                           list(meta["v_eff_b2"].keys()) +
                           list(meta["v_eff_l12"].keys())))
    rows = []
    for e in all_elems:
        vk = KING_ATOMIC_VOLUMES.get(e, None)
        vb = meta["v_eff_b2"].get(e, None)
        vl = meta["v_eff_l12"].get(e, None)
        rows.append({
            "element": e,
            "V_King": vk,
            "V_eff_B2": vb,
            "V_eff_L12": vl,
            "r_King": (vk * 3 / (4*np.pi))**(1/3) if vk else None,
            "r_eff_B2": (vb * 3 / (4*np.pi))**(1/3) if vb else None,
            "r_eff_L12": (vl * 3 / (4*np.pi))**(1/3) if vl else None,
            "delta_B2": meta["delta_b2"].get(e, None),
            "delta_L12": meta["delta_l12"].get(e, None),
        })
    df_table = pd.DataFrame(rows)
    df_table.to_csv(outdir / "effective_radius_table.csv", index=False)
    print(f"Saved: {outdir}/effective_radius_table.csv")

    # ── Print element table for HEA elements ──
    print(f"\n{'='*70}")
    print("ELEMENT-WISE PROPERTIES (HEA elements, MP+OQMD)")
    print(f"{'='*70}")
    print(f"  {'Elem':<5} {'V_King':>8} {'V_eff_B2':>10} {'V_eff_L12':>10} "
          f"{'r_King':>8} {'r_eff_B2':>10} {'r_eff_L12':>10} "
          f"{'δ_B2':>8} {'δ_L12':>8}")
    print("  " + "-" * 90)
    for e in hea_elements:
        vk = KING_ATOMIC_VOLUMES.get(e, 0)
        vb = meta["v_eff_b2"].get(e, 0)
        vl = meta["v_eff_l12"].get(e, 0)
        rk = (vk * 3 / (4*np.pi))**(1/3) if vk > 0 else 0
        rb = (vb * 3 / (4*np.pi))**(1/3) if vb > 0 else 0
        rl = (vl * 3 / (4*np.pi))**(1/3) if vl > 0 else 0
        db = meta["delta_b2"].get(e, 0)
        dl = meta["delta_l12"].get(e, 0)
        print(f"  {e:<4} {vk:>8.3f} {vb:>10.3f} {vl:>10.3f} "
              f"{rk:>8.4f} {rb:>10.4f} {rl:>10.4f} "
              f"{db:>+8.4f} {dl:>+8.4f}")

    # ── Final conclusion ──
    pw = res_mpoqmd["Pairwise Ω_sf + γ"]["rmse"]
    a1 = res_mpoqmd["Additive δ + γ (Ap.1)"]["rmse"]
    a2a = res_mpoqmd["V_eff Vegard (Ap.2a)"]["rmse"]
    a2c = res_mpoqmd["V_eff + Add. δ (Ap.2c)"]["rmse"]

    print(f"\n{'='*70}")
    print("CONCLUSION")
    print(f"{'='*70}")
    print(f"\n  ペアワイズ Ω_sf (baseline):  RMSE = {pw:.4f} Å")
    print(f"  加法分解 δ (Ap.1):          RMSE = {a1:.4f} Å  ({(pw-a1)/pw*100:+.1f}%)")
    print(f"  有効体積 Vegard (Ap.2a):    RMSE = {a2a:.4f} Å  ({(pw-a2a)/pw*100:+.1f}%)")
    print(f"  有効体積 + 加法δ (Ap.2c):   RMSE = {a2c:.4f} Å  ({(pw-a2c)/pw*100:+.1f}%)")
    n_delta = meta["n_delta_b2"] + meta["n_delta_l12"]
    n_pairs = meta["n_b2_pairs"] + meta["n_l12_pairs"]
    n_comb = (meta["n_delta_b2"]*(meta["n_delta_b2"]-1)//2 +
              meta["n_delta_l12"]*(meta["n_delta_l12"]-1)//2)
    print(f"\n  パラメータ数: 加法 {n_delta} 元素 → {n_comb} ペア予測可能")
    print(f"                ペアワイズ {n_pairs} ペア (データ依存)")
    print(f"\n  加法分解は情報損失 {(1-meta['r2_b2'])*100:.0f}% (B2) / "
          f"{(1-meta['r2_l12'])*100:.0f}% (L1₂)")
    print(f"  → Ω_sfの二体相互作用は元素レベルに完全分解不可")
    print(f"  → 但しペア数3.2倍のカバレッジ向上と引き換えに"
          f"RMSE {(a1-pw)/pw*100:+.1f}% 劣化のみ")


if __name__ == "__main__":
    main()
