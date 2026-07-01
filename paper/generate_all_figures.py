#!/usr/bin/env python3
"""
Clean re-analysis script: generates ALL figures and tables for the paper.
Excludes 4f rare earths (Gd, Ce, La, Pr, Nd, Sm, Eu, Tb, Dy, Ho, Er, Tm, Yb, Lu) and Y from all analyses.

Usage:
    cd paper/ && python generate_all_figures.py

Input data (relative to repo root):
    data/compounds_{MP,OQMD,VASP}_{B2,L12}.csv

Output:
    paper/fig_*.png          — all paper figures
    paper/results_*.csv      — all data tables
"""

import sys
import csv
import re
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
from scipy.stats import pearsonr
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

EXCLUDE_ELEMENTS = {
    "Gd", "Ce",  # 4f instability (already excluded)
    "La", "Pr", "Nd", "Sm", "Eu", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",  # 4f RE
    "Y",  # similar RE behavior
}

# ---------------------------------------------------------------------------
# 1. Load & filter compound data
# ---------------------------------------------------------------------------
def load_compounds():
    """Load MP + OQMD + VASP compound data, excluding 4f RE + Y."""
    data_dir = REPO / "data"
    dfs = []
    for src in ["MP", "OQMD", "VASP"]:
        for struct in ["B2", "L12"]:
            f = data_dir / f"compounds_{src}_{struct}.csv"
            if f.exists():
                df = pd.read_csv(f)
                df["db"] = src
                df["stype"] = struct
                dfs.append(df)
    if not dfs:
        raise FileNotFoundError(
            "No compound CSV files found in data/ directory.")
    all_df = pd.concat(dfs, ignore_index=True)
    # Exclude Gd/Ce
    mask = ~(all_df["element_A"].isin(EXCLUDE_ELEMENTS) |
             all_df["element_B"].isin(EXCLUDE_ELEMENTS))
    all_df = all_df[mask].reset_index(drop=True)
    return all_df


def compute_omega_sf_pairwise(df, sources=("MP", "OQMD", "VASP"), min_count=1):
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


def additive_decomposition(ob2, ol12, outlier_threshold=0.3):
    """Decompose pairwise Omega_sf into element-level delta parameters.

    Pairs with |Omega_sf| > outlier_threshold are excluded from the fit
    (consistent with fig06_additive_fit visual labeling).
    """
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
        n_excluded = 0
        for (a, b), val in omega.items():
            if abs(val) > outlier_threshold:
                n_excluded += 1
                continue
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
        rmse = float(np.sqrt(np.mean((b_vec - pred) ** 2)))
        results[label] = {
            "delta": delta_dict, "r2": r2, "rmse": rmse,
            "elements": elements, "n_excluded": n_excluded,
        }
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


# ---------------------------------------------------------------------------
# SQS analysis: DFT-consistent Vegard reference
# ---------------------------------------------------------------------------
SQS_FILE = REPO / "data" / "sqs_results.csv"


def load_sqs_data():
    """Load BCC SQS 1:1 results and compute Omega_sf with DFT Vegard reference.

    Returns dict with keys:
        omega_dft: {pair: Omega_sf} using SQS pure element volumes as Vegard ref
        omega_king: {pair: Omega_sf} using King volumes as Vegard ref
        pure_vol: {element: volume_per_atom} from SQS A8A8
        pure_a: {element: lattice_constant} from SQS A8A8
        n_converged: total converged BCC SQS 1:1 rows
        n_pairs: number of unique pairs (after 4f+Y exclusion)
    """
    if not SQS_FILE.exists():
        return None

    with open(SQS_FILE) as f:
        rows = list(csv.DictReader(f))

    # Load MP DFT BCC pure element volumes for reliability check
    mp_bcc_file = REPO / "data" / "mp_pure_elements_bcc.csv"
    mp_bcc_vol = {}
    if mp_bcc_file.exists():
        mp_df = pd.read_csv(mp_bcc_file)
        mp_bcc_vol = dict(zip(mp_df["element"], mp_df["volume_per_atom"]))

    # Extract BCC SQS pure element data (A8A8 same element)
    pure_vol_raw = {}
    pure_a = {}
    for r in rows:
        if r["status"] != "OK" or r.get("relax_converged", "") != "yes":
            continue
        if r["lattice_type"] != "bcc":
            continue
        pairs = re.findall(r"([A-Z][a-z]?)(\d+)", r["dir"])
        if len(pairs) != 2:
            continue
        el1, n1 = pairs[0][0], int(pairs[0][1])
        el2, n2 = pairs[1][0], int(pairs[1][1])
        if el1 == el2 and n1 == 8 and n2 == 8:
            try:
                a = float(r["a_bcc_A"])
            except (ValueError, KeyError):
                continue
            if 2.0 < a < 8.0:
                pure_a[el1] = a
                pure_vol_raw[el1] = a**3 / 2.0

    # Reliability check: flag SQS pure volumes >3% from King/MP
    pure_vol = {}
    sqs_override_log = []
    for el, v_sqs in pure_vol_raw.items():
        v_king = KING_ATOMIC_VOLUMES.get(el)
        v_mp = mp_bcc_vol.get(el)
        use_sqs = True
        if v_king is not None:
            pct = abs(v_sqs - v_king) / v_king * 100
            if pct > 3.0:
                # SQS unreliable; prefer MP if close to King, else King
                if v_mp is not None and abs(v_mp - v_king) / v_king * 100 <= 3.0:
                    pure_vol[el] = v_mp
                    sqs_override_log.append(f"  {el}: SQS {v_sqs:.3f} -> MP {v_mp:.3f} (SQS-King={pct:+.1f}%)")
                else:
                    pure_vol[el] = v_king
                    sqs_override_log.append(f"  {el}: SQS {v_sqs:.3f} -> King {v_king:.3f} (SQS-King={pct:+.1f}%)")
                use_sqs = False
        if use_sqs:
            pure_vol[el] = v_sqs
    if sqs_override_log:
        print(f"  SQS BCC pure volume overrides ({len(sqs_override_log)} elements):")
        for line in sqs_override_log:
            print(line)

    # BCC SQS 1:1 (A8B8) pairs
    omega_dft = {}
    omega_king = {}
    sqs_a = {}
    n_converged = 0

    for r in rows:
        if r["status"] != "OK" or r.get("relax_converged", "") != "yes":
            continue
        if r["lattice_type"] != "bcc":
            continue
        pairs = re.findall(r"([A-Z][a-z]?)(\d+)", r["dir"])
        if len(pairs) != 2:
            continue
        el1, n1 = pairs[0][0], int(pairs[0][1])
        el2, n2 = pairs[1][0], int(pairs[1][1])
        if n1 != 8 or n2 != 8 or el1 == el2:
            continue
        n_converged += 1
        if el1 in EXCLUDE_ELEMENTS or el2 in EXCLUDE_ELEMENTS:
            continue
        if el1 not in KING_ATOMIC_VOLUMES or el2 not in KING_ATOMIC_VOLUMES:
            continue
        try:
            a = float(r["a_bcc_A"])
        except (ValueError, KeyError):
            continue
        if a < 2.0 or a > 8.0:
            continue

        pair = tuple(sorted([el1, el2]))
        v_actual = a**3 / 2.0
        sqs_a[pair] = a

        # King Vegard reference
        v_king = 0.5 * KING_ATOMIC_VOLUMES[pair[0]] + 0.5 * KING_ATOMIC_VOLUMES[pair[1]]
        omega_king[pair] = (v_actual - v_king) / v_king

        # DFT Vegard reference (requires both pure elements)
        if pair[0] in pure_vol and pair[1] in pure_vol:
            v_dft = 0.5 * pure_vol[pair[0]] + 0.5 * pure_vol[pair[1]]
            omega_dft[pair] = (v_actual - v_dft) / v_dft

    # FCC SQS pure element data (A16A16 same element)
    mp_fcc_file = REPO / "data" / "mp_pure_elements_fcc.csv"
    mp_fcc_vol = {}
    if mp_fcc_file.exists():
        mp_fcc_df = pd.read_csv(mp_fcc_file)
        mp_fcc_vol = dict(zip(mp_fcc_df["element"], mp_fcc_df["volume_per_atom"]))

    fcc_pure_vol_raw = {}
    for r in rows:
        if r["status"] != "OK" or r.get("relax_converged", "") != "yes":
            continue
        if r["lattice_type"] != "fcc":
            continue
        pairs = re.findall(r"([A-Z][a-z]?)(\d+)", r["dir"])
        if len(pairs) != 2:
            continue
        el1, n1 = pairs[0][0], int(pairs[0][1])
        el2, n2 = pairs[1][0], int(pairs[1][1])
        if el1 == el2 and n1 == 16 and n2 == 16:
            try:
                vol = float(r.get("volume_A3", "0"))
                natoms = int(r.get("natoms", "32"))
            except (ValueError, KeyError):
                continue
            if vol > 0:
                fcc_pure_vol_raw[el1] = vol / natoms

    # Reliability check for FCC pure volumes
    fcc_pure_vol = {}
    fcc_override_log = []
    for el, v_sqs in fcc_pure_vol_raw.items():
        v_king = KING_ATOMIC_VOLUMES.get(el)
        v_mp = mp_fcc_vol.get(el)
        use_sqs = True
        if v_king is not None:
            pct = abs(v_sqs - v_king) / v_king * 100
            if pct > 3.0:
                if v_mp is not None and abs(v_mp - v_king) / v_king * 100 <= 3.0:
                    fcc_pure_vol[el] = v_mp
                    fcc_override_log.append(f"  {el}: SQS {v_sqs:.3f} -> MP {v_mp:.3f} (SQS-King={pct:+.1f}%)")
                else:
                    fcc_pure_vol[el] = v_king
                    fcc_override_log.append(f"  {el}: SQS {v_sqs:.3f} -> King {v_king:.3f} (SQS-King={pct:+.1f}%)")
                use_sqs = False
        if use_sqs:
            fcc_pure_vol[el] = v_sqs
    if fcc_override_log:
        print(f"  SQS FCC pure volume overrides ({len(fcc_override_log)} elements):")
        for line in fcc_override_log:
            print(line)

    # FCC SQS 1:1 (A16B16) pairs
    fcc_omega_king = {}
    fcc_omega_dft = {}
    fcc_n_converged = 0
    for r in rows:
        if r["status"] != "OK" or r.get("relax_converged", "") != "yes":
            continue
        if r["lattice_type"] != "fcc":
            continue
        pairs = re.findall(r"([A-Z][a-z]?)(\d+)", r["dir"])
        if len(pairs) != 2:
            continue
        el1, n1 = pairs[0][0], int(pairs[0][1])
        el2, n2 = pairs[1][0], int(pairs[1][1])
        if n1 != 16 or n2 != 16 or el1 == el2:
            continue
        fcc_n_converged += 1
        if el1 in EXCLUDE_ELEMENTS or el2 in EXCLUDE_ELEMENTS:
            continue
        try:
            vol = float(r.get("volume_A3", "0"))
            natoms = int(r.get("natoms", "32"))
        except (ValueError, KeyError):
            continue
        if vol <= 0:
            continue
        pair = tuple(sorted([el1, el2]))
        v_actual = vol / natoms

        # King Vegard reference
        if pair[0] in KING_ATOMIC_VOLUMES and pair[1] in KING_ATOMIC_VOLUMES:
            v_king = 0.5 * KING_ATOMIC_VOLUMES[pair[0]] + 0.5 * KING_ATOMIC_VOLUMES[pair[1]]
            fcc_omega_king[pair] = (v_actual - v_king) / v_king

        # DFT Vegard reference
        if pair[0] in fcc_pure_vol and pair[1] in fcc_pure_vol:
            v_dft = 0.5 * fcc_pure_vol[pair[0]] + 0.5 * fcc_pure_vol[pair[1]]
            fcc_omega_dft[pair] = (v_actual - v_dft) / v_dft

    # Count non-excluded pure elements
    n_pure_bcc = len({el for el in pure_vol if el not in EXCLUDE_ELEMENTS})
    n_pure_fcc = len({el for el in fcc_pure_vol if el not in EXCLUDE_ELEMENTS})

    return {
        "omega_dft": omega_dft,
        "omega_king": omega_king,
        "pure_vol": pure_vol,
        "pure_vol_raw": pure_vol_raw,
        "pure_a": pure_a,
        "sqs_a": sqs_a,
        "n_converged": n_converged,
        "n_pairs_king": len(omega_king),
        "n_pairs_dft": len(omega_dft),
        "n_pure_elements": n_pure_bcc,
        "fcc_pure_vol": fcc_pure_vol,
        "fcc_pure_vol_raw": fcc_pure_vol_raw,
        "fcc_omega_king": fcc_omega_king,
        "fcc_omega_dft": fcc_omega_dft,
        "fcc_n_converged": fcc_n_converged,
        "fcc_n_pairs": len(fcc_omega_king),
        "fcc_n_pairs_dft": len(fcc_omega_dft),
        "n_pure_fcc_elements": n_pure_fcc,
    }


def analyze_sqs(sqs_data, ob2, ol12, heas_train, heas_test):
    """Run SQS + DFT Vegard analysis and return metrics dict.

    Args:
        sqs_data: output of load_sqs_data()
        ob2: B2 Omega_sf dict (for comparison)
        ol12: L12 Omega_sf dict (for FCC SQS comparison)
        heas_train: training BCC HEAs
        heas_test: test BCC HEAs
    Returns:
        dict of metrics for paper_metrics.json
    """
    if sqs_data is None:
        return {}

    omega_dft = sqs_data["omega_dft"]
    omega_king = sqs_data["omega_king"]

    def _rmse(omega_dict, q, heas):
        y = np.array([h["a_exp"] for h in heas])
        p = np.array([
            compute_eq10_scaled(h["comp"], h["struct"], omega_dict, q)
            for h in heas
        ])
        return float(np.sqrt(np.mean((p - y) ** 2)))

    def _optimize_q(omega_dict, heas):
        y = np.array([h["a_exp"] for h in heas])

        def obj(q):
            p = np.array([
                compute_eq10_scaled(h["comp"], h["struct"], omega_dict, q)
                for h in heas
            ])
            return float(np.sqrt(np.mean((p - y) ** 2)))

        res = minimize_scalar(obj, bounds=(-5, 5), method="bounded")
        return res.x, res.fun

    # Vegard baseline
    rmse_veg_train = _rmse({}, 0, heas_train)
    rmse_veg_test = _rmse({}, 0, heas_test)

    # B2 reference
    q_b2, rmse_b2_train = _optimize_q(ob2, heas_train)
    rmse_b2_test = _rmse(ob2, q_b2, heas_test)

    # SQS + King Vegard
    q_king, rmse_king_train = _optimize_q(omega_king, heas_train)
    rmse_king_test = _rmse(omega_king, q_king, heas_test)
    rmse_king_test_q1 = _rmse(omega_king, 1.0, heas_test)

    # SQS + DFT Vegard
    q_dft, rmse_dft_train = _optimize_q(omega_dft, heas_train)
    rmse_dft_test = _rmse(omega_dft, q_dft, heas_test)
    rmse_dft_test_q1 = _rmse(omega_dft, 1.0, heas_test)
    rmse_dft_train_q1 = _rmse(omega_dft, 1.0, heas_train)

    # Correlation: B2 vs SQS (DFT Vegard)
    common_b2_dft = set(ob2.keys()) & set(omega_dft.keys())
    if len(common_b2_dft) > 2:
        x_b2 = [ob2[p] for p in common_b2_dft]
        y_dft = [omega_dft[p] for p in common_b2_dft]
        r_b2_dft, _ = pearsonr(x_b2, y_dft)
        slope_b2_dft = float(np.polyfit(x_b2, y_dft, 1)[0])
    else:
        r_b2_dft, slope_b2_dft = 0.0, 0.0

    # Correlation: B2 vs SQS (King Vegard)
    common_b2_king = set(ob2.keys()) & set(omega_king.keys())
    if len(common_b2_king) > 2:
        x_b2k = [ob2[p] for p in common_b2_king]
        y_king = [omega_king[p] for p in common_b2_king]
        r_b2_king, _ = pearsonr(x_b2k, y_king)
        slope_b2_king = float(np.polyfit(x_b2k, y_king, 1)[0])
    else:
        r_b2_king, slope_b2_king = 0.0, 0.0

    # Omega_sf distribution for SQS (DFT Vegard)
    vals_dft = list(omega_dft.values())
    n_positive = sum(1 for v in vals_dft if v > 0)
    n_negative = sum(1 for v in vals_dft if v <= 0)

    # q sensitivity: test RMSE at q=0.6, 0.8, 1.0, 1.2, 1.4
    q_scan = {}
    for q_val in [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 2.0]:
        q_scan[f"q{q_val:.1f}"] = round(_rmse(omega_dft, q_val, heas_test), 4)

    # FCC: L12 vs FCC SQS 1:1 correlation
    fcc_omega_king = sqs_data.get("fcc_omega_king", {})
    common_l12_fcc = set(ol12.keys()) & set(fcc_omega_king.keys())
    if len(common_l12_fcc) > 2:
        x_l12 = [ol12[p] for p in common_l12_fcc]
        y_fcc = [fcc_omega_king[p] for p in common_l12_fcc]
        r_l12_fcc, _ = pearsonr(x_l12, y_fcc)
        slope_l12_fcc = float(np.polyfit(x_l12, y_fcc, 1)[0])
    else:
        r_l12_fcc, slope_l12_fcc = 0.0, 0.0

    metrics = {
        "n_sqs_pure_elements": sqs_data["n_pure_elements"],
        "n_sqs_pairs_king": sqs_data["n_pairs_king"],
        "n_sqs_pairs_dft": sqs_data["n_pairs_dft"],
        "n_sqs_converged_11": sqs_data["n_converged"],
        "q_BCC_b2": round(q_b2, 4),
        "q_BCC_sqs_king": round(q_king, 6),
        "q_BCC_sqs_dft_opt": round(q_dft, 3),
        "q_BCC_sqs_dft_adopted": 1.0,
        "RMSE_vegard_BCC_train": round(rmse_veg_train, 4),
        "RMSE_vegard_BCC_test": round(rmse_veg_test, 4),
        "RMSE_b2_BCC_train": round(rmse_b2_train, 4),
        "RMSE_b2_BCC_test": round(rmse_b2_test, 4),
        "RMSE_sqs_king_train": round(rmse_king_train, 4),
        "RMSE_sqs_king_test": round(rmse_king_test, 4),
        "RMSE_sqs_king_test_q1": round(rmse_king_test_q1, 4),
        "RMSE_sqs_dft_train_qopt": round(rmse_dft_train, 4),
        "RMSE_sqs_dft_test_qopt": round(rmse_dft_test, 4),
        "RMSE_sqs_dft_train_q1": round(rmse_dft_train_q1, 4),
        "RMSE_sqs_dft_test_q1": round(rmse_dft_test_q1, 4),
        "improvement_sqs_dft_q1_vs_vegard_pct": round(
            (1 - rmse_dft_test_q1 / rmse_veg_test) * 100, 1
        ),
        "improvement_b2_vs_vegard_pct": round(
            (1 - rmse_b2_test / rmse_veg_test) * 100, 1
        ),
        "correlation_b2_vs_sqs_dft_n": len(common_b2_dft),
        "correlation_b2_vs_sqs_dft_r": round(r_b2_dft, 3),
        "correlation_b2_vs_sqs_dft_slope": round(slope_b2_dft, 3),
        "correlation_b2_vs_sqs_king_n": len(common_b2_king),
        "correlation_b2_vs_sqs_king_r": round(r_b2_king, 3),
        "correlation_b2_vs_sqs_king_slope": round(slope_b2_king, 3),
        "omega_dft_positive_pct": round(n_positive / len(vals_dft) * 100, 0),
        "omega_dft_negative_pct": round(n_negative / len(vals_dft) * 100, 0),
        "omega_dft_mean": round(float(np.mean(vals_dft)), 4),
        "omega_dft_range_min": round(float(np.min(vals_dft)), 4),
        "omega_dft_range_max": round(float(np.max(vals_dft)), 4),
        "q_sensitivity_test": q_scan,
        "fcc_sqs_n_pairs": sqs_data.get("fcc_n_pairs", 0),
        "fcc_sqs_n_converged": sqs_data.get("fcc_n_converged", 0),
        "correlation_l12_vs_fcc_sqs_n": len(common_l12_fcc),
        "correlation_l12_vs_fcc_sqs_r": round(r_l12_fcc, 3),
        "correlation_l12_vs_fcc_sqs_slope": round(slope_l12_fcc, 3),
    }
    return metrics


def analyze_ml_residual(y_train, a_ss_tr, heas_train, ob2, ol12):
    """ML residual correction analysis (Section 4.12).

    Tests whether ML models can improve upon the physics model residuals.
    Uses LOO-CV to avoid overfitting on the small (64 HEA) dataset.

    Returns dict of metrics for paper_metrics.json.
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import LeaveOneOut

    residuals = y_train - a_ss_tr  # physics model residual
    n = len(heas_train)

    # Features: composition-based descriptors
    X = np.zeros((n, 6))  # VEC, deltaR, deltaSf, mean_EN, nElements, VEC_std
    for i, h in enumerate(heas_train):
        comp = h["comp"]
        elements = list(comp.keys())
        fracs = np.array([comp[e] for e in elements])
        vecs = np.array([VEC.get(e, 0) for e in elements])
        ens = np.array([PAULING_EN.get(e, 0) for e in elements])
        omega = ob2 if h["struct"] == "BCC" else ol12
        X[i, 0] = np.dot(fracs, vecs)  # mean VEC
        X[i, 1] = compute_delta_r(comp)
        X[i, 2] = compute_delta_sf(comp, omega)
        X[i, 3] = np.dot(fracs, ens)  # mean EN
        X[i, 4] = len(elements)
        X[i, 5] = np.std(vecs)  # VEC std

    # LOO-CV with Ridge regression (residual correction)
    loo = LeaveOneOut()
    residual_pred = np.zeros(n)
    for train_idx, test_idx in loo.split(X):
        model = Ridge(alpha=1.0)
        model.fit(X[train_idx], residuals[train_idx])
        residual_pred[test_idx] = model.predict(X[test_idx])

    corrected = a_ss_tr + residual_pred
    rmse_ridge_loo = float(np.sqrt(np.mean((corrected - y_train) ** 2)))

    # Direct XGBoost (overfitting test)
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        gb = GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                        random_state=42)
        gb.fit(X, y_train)
        pred_gb_train = gb.predict(X)
        rmse_xgb_train = float(np.sqrt(np.mean((pred_gb_train - y_train) ** 2)))

        # LOO for XGBoost
        gb_loo_pred = np.zeros(n)
        for train_idx, test_idx in loo.split(X):
            gb_loo = GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                                random_state=42)
            gb_loo.fit(X[train_idx], y_train[train_idx])
            gb_loo_pred[test_idx] = gb_loo.predict(X[test_idx])
        rmse_xgb_loo = float(np.sqrt(np.mean((gb_loo_pred - y_train) ** 2)))
    except ImportError:
        rmse_xgb_train = 0.0
        rmse_xgb_loo = 0.0

    # Physics model RMSE for comparison
    rmse_physics = float(np.sqrt(np.mean((a_ss_tr - y_train) ** 2)))

    return {
        "RMSE_physics_train": round(rmse_physics, 4),
        "RMSE_ridge_loo": round(rmse_ridge_loo, 4),
        "RMSE_xgb_train": round(rmse_xgb_train, 4),
        "RMSE_xgb_loo": round(rmse_xgb_loo, 4),
        "improvement_ridge_loo_pct": round(
            (1 - rmse_ridge_loo / rmse_physics) * 100, 1
        ),
    }


def analyze_dft_self_consistent(all_df, ob2, ol12, heas_train, heas_test):
    """Recompute Omega_sf using VASP homonuclear compound volumes as Vegard reference.

    Pure element volumes are extracted from VASP homonuclear rows in the compound
    CSVs (B2: a^3/2, L12: a^3/4). These are NOT the same as MP pure-element DFT
    volumes (data/mp_pure_elements_{bcc,fcc}.csv), which use different cell setups.
    For the "all-source" omega (ob2_dft_all, ol12_dft_all), MP/OQMD compound
    pair volumes are referenced against these VASP homonuclear endpoints.

    Returns dict of metrics for paper_metrics.json.
    """
    # Extract VASP pure element volumes from homonuclear compound data
    vasp_b2 = all_df[(all_df["db"] == "VASP") & (all_df["stype"] == "B2")]
    vasp_l12 = all_df[(all_df["db"] == "VASP") & (all_df["stype"] == "L12")]

    # B2 pure elements (element_A == element_B)
    pure_b2 = vasp_b2[vasp_b2["element_A"] == vasp_b2["element_B"]]
    dft_vol_b2 = {}
    for _, row in pure_b2.iterrows():
        el = row["element_A"]
        a = row["lattice_constant"]
        if 2.0 < a < 8.0:
            dft_vol_b2[el] = a**3 / 2.0

    # L12 pure elements
    pure_l12 = vasp_l12[vasp_l12["element_A"] == vasp_l12["element_B"]]
    dft_vol_l12 = {}
    for _, row in pure_l12.iterrows():
        el = row["element_A"]
        a = row["lattice_constant"]
        if 2.0 < a < 8.0:
            dft_vol_l12[el] = a**3 / 4.0

    # Recompute B2 Omega_sf with DFT Vegard reference
    ob2_dft = {}
    vasp_b2_pairs = vasp_b2[vasp_b2["element_A"] != vasp_b2["element_B"]]
    for _, row in vasp_b2_pairs.iterrows():
        elA, elB = row["element_A"], row["element_B"]
        a = row["lattice_constant"]
        if a <= 2 or a >= 8:
            continue
        if elA in EXCLUDE_ELEMENTS or elB in EXCLUDE_ELEMENTS:
            continue
        pair = tuple(sorted([elA, elB]))
        if pair[0] not in dft_vol_b2 or pair[1] not in dft_vol_b2:
            continue
        v_act = a**3 / 2.0
        v_veg_dft = 0.5 * dft_vol_b2[pair[0]] + 0.5 * dft_vol_b2[pair[1]]
        ob2_dft[pair] = (v_act - v_veg_dft) / v_veg_dft

    # Recompute L12 Omega_sf with DFT Vegard reference
    ol12_dft = {}
    vasp_l12_pairs = vasp_l12[vasp_l12["element_A"] != vasp_l12["element_B"]]
    for _, row in vasp_l12_pairs.iterrows():
        elA, elB = row["element_A"], row["element_B"]
        a = row["lattice_constant"]
        if a <= 2 or a >= 8:
            continue
        if elA in EXCLUDE_ELEMENTS or elB in EXCLUDE_ELEMENTS:
            continue
        pair = tuple(sorted([elA, elB]))
        cA = row.get("count_A", 3)
        cB = row.get("count_B", 1)
        total = cA + cB
        if pair[0] not in dft_vol_l12 or pair[1] not in dft_vol_l12:
            continue
        v_act = a**3 / 4.0
        # For L12 A3B: V_Vegard = (3*V_A + V_B)/4
        fA = cA / total
        fB = cB / total
        # Determine which element is A (majority) and B (minority)
        if elA == pair[0]:
            v_veg_dft = fA * dft_vol_l12[pair[0]] + fB * dft_vol_l12[pair[1]]
        else:
            v_veg_dft = fA * dft_vol_l12[pair[1]] + fB * dft_vol_l12[pair[0]]
        ol12_dft[pair] = (v_act - v_veg_dft) / v_veg_dft

    # Also use MP/OQMD B2/L12 data with DFT Vegard reference
    # For the "all sources" approach, recompute using all compound data
    ob2_dft_all = {}
    ol12_dft_all = {}
    sources = ("MP", "OQMD", "VASP")
    sub = all_df[all_df["db"].isin(sources)]
    for _, row in sub.iterrows():
        elA = row.get("element_A", "")
        elB = row.get("element_B", "")
        a = row.get("lattice_constant", 0)
        stype = row.get("stype", "")
        if a <= 2 or a >= 8 or elA == elB:
            continue
        if elA in EXCLUDE_ELEMENTS or elB in EXCLUDE_ELEMENTS:
            continue
        pair = tuple(sorted([elA, elB]))
        if stype == "B2":
            if pair[0] not in dft_vol_b2 or pair[1] not in dft_vol_b2:
                continue
            v_act = a**3 / 2.0
            v_veg = 0.5 * dft_vol_b2[pair[0]] + 0.5 * dft_vol_b2[pair[1]]
            ob2_dft_all.setdefault(pair, []).append((v_act - v_veg) / v_veg)
        elif stype == "L12":
            if pair[0] not in dft_vol_l12 or pair[1] not in dft_vol_l12:
                continue
            cA = row.get("count_A", 3)
            cB = row.get("count_B", 1)
            total = cA + cB
            v_act = a**3 / 4.0
            fA = cA / total
            fB = cB / total
            if elA == pair[0]:
                v_veg = fA * dft_vol_l12[pair[0]] + fB * dft_vol_l12[pair[1]]
            else:
                v_veg = fA * dft_vol_l12[pair[1]] + fB * dft_vol_l12[pair[0]]
            ol12_dft_all.setdefault(pair, []).append((v_act - v_veg) / v_veg)

    ob2_dft_med = {p: np.median(v) for p, v in ob2_dft_all.items() if len(v) >= 2}
    ol12_dft_med = {p: np.median(v) for p, v in ol12_dft_all.items() if len(v) >= 2}

    # Optimize q with DFT Vegard reference on training 64 HEAs
    y_tr = np.array([h["a_exp"] for h in heas_train])
    bcc_tr = [i for i, h in enumerate(heas_train) if h["struct"] == "BCC"]
    fcc_tr = [i for i, h in enumerate(heas_train) if h["struct"] == "FCC"]

    def _pred(heas, ob, ol, qb, qf):
        return np.array([
            compute_eq10_scaled(h["comp"], h["struct"],
                                ob if h["struct"] == "BCC" else ol,
                                qb if h["struct"] == "BCC" else qf)
            for h in heas
        ])

    def _rmse_bcc(q):
        p = _pred(heas_train, ob2_dft_med, ol12_dft_med, q, 1.0)
        return float(np.sqrt(np.mean((p[bcc_tr] - y_tr[bcc_tr]) ** 2)))

    def _rmse_fcc(q):
        p = _pred(heas_train, ob2_dft_med, ol12_dft_med, 1.0, q)
        return float(np.sqrt(np.mean((p[fcc_tr] - y_tr[fcc_tr]) ** 2)))

    q_bcc_dft = minimize_scalar(_rmse_bcc, bounds=(0, 5), method="bounded").x
    q_fcc_dft = minimize_scalar(_rmse_fcc, bounds=(0, 5), method="bounded").x

    # --- Training 64 HEA evaluation ---
    p_opt_tr = _pred(heas_train, ob2_dft_med, ol12_dft_med, q_bcc_dft, q_fcc_dft)
    rmse_opt_tr = float(np.sqrt(np.mean((p_opt_tr - y_tr) ** 2)))

    p_q1_tr = _pred(heas_train, ob2_dft_med, ol12_dft_med, 1.0, 1.0)
    rmse_q1_tr = float(np.sqrt(np.mean((p_q1_tr - y_tr) ** 2)))

    p_veg_tr = _pred(heas_train, ob2_dft_med, ol12_dft_med, 0.0, 0.0)
    rmse_veg_tr = float(np.sqrt(np.mean((p_veg_tr - y_tr) ** 2)))

    # King reference (use actual optimized q from main model, not hard-coded)
    gb_king, gf_king = optimize_gamma(heas_train, ob2, ol12)
    p_king_opt_tr = _pred(heas_train, ob2, ol12, gb_king, gf_king)
    rmse_king_opt_tr = float(np.sqrt(np.mean((p_king_opt_tr - y_tr) ** 2)))

    # --- Independent test evaluation ---
    y_te = np.array([h["a_exp"] for h in heas_test])
    bcc_te = [i for i, h in enumerate(heas_test) if h["struct"] == "BCC"]
    fcc_te = [i for i, h in enumerate(heas_test) if h["struct"] == "FCC"]

    p_opt_te = _pred(heas_test, ob2_dft_med, ol12_dft_med, q_bcc_dft, q_fcc_dft)
    rmse_opt_te = float(np.sqrt(np.mean((p_opt_te - y_te) ** 2)))
    rmse_opt_te_bcc = float(np.sqrt(np.mean((p_opt_te[bcc_te] - y_te[bcc_te]) ** 2)))
    rmse_opt_te_fcc = float(np.sqrt(np.mean((p_opt_te[fcc_te] - y_te[fcc_te]) ** 2)))

    p_veg_te = _pred(heas_test, ob2_dft_med, ol12_dft_med, 0.0, 0.0)
    rmse_veg_te = float(np.sqrt(np.mean((p_veg_te - y_te) ** 2)))
    rmse_veg_te_bcc = float(np.sqrt(np.mean((p_veg_te[bcc_te] - y_te[bcc_te]) ** 2)))

    return {
        "n_pure_b2": len(dft_vol_b2),
        "n_pure_l12": len(dft_vol_l12),
        "n_pairs_b2_dft": len(ob2_dft_med),
        "n_pairs_l12_dft": len(ol12_dft_med),
        "q_BCC_dft_ref": round(q_bcc_dft, 4),
        "q_FCC_dft_ref": round(q_fcc_dft, 4),
        # Training 64
        "RMSE_dft_ref_opt_q_train": round(rmse_opt_tr, 4),
        "RMSE_dft_ref_q1_train": round(rmse_q1_tr, 4),
        "RMSE_vegard_train": round(rmse_veg_tr, 4),
        "RMSE_king_opt_train": round(rmse_king_opt_tr, 4),
        "improvement_dft_ref_opt_train_pct": round(
            (1 - rmse_opt_tr / rmse_veg_tr) * 100, 1),
        # Test 28
        "RMSE_dft_ref_opt_q_test": round(rmse_opt_te, 4),
        "RMSE_dft_ref_opt_q_test_BCC": round(rmse_opt_te_bcc, 4),
        "RMSE_dft_ref_opt_q_test_FCC": round(rmse_opt_te_fcc, 4),
        "RMSE_vegard_test": round(rmse_veg_te, 4),
        "improvement_dft_ref_opt_test_pct": round(
            (1 - rmse_opt_te / rmse_veg_te) * 100, 1),
        "improvement_dft_ref_opt_test_BCC_pct": round(
            (1 - rmse_opt_te_bcc / rmse_veg_te_bcc) * 100, 1),
    }


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
    """Fig 3: Combined BCC/FCC parity (single panel)."""
    bcc_i = [i for i, h in enumerate(heas_train) if h["struct"] == "BCC"]
    fcc_i = [i for i, h in enumerate(heas_train) if h["struct"] == "FCC"]

    fig, ax = plt.subplots(figsize=(8, 8))
    lims = [min(y_train) - 0.05, max(y_train) + 0.05]
    ax.plot(lims, lims, "k-", lw=1)

    rmse_bcc = np.sqrt(np.mean((a_ss_tr[bcc_i] - y_train[bcc_i]) ** 2))
    rmse_fcc = np.sqrt(np.mean((a_ss_tr[fcc_i] - y_train[fcc_i]) ** 2))
    rmse_all = np.sqrt(np.mean((a_ss_tr - y_train) ** 2))

    ax.scatter(y_train[bcc_i], a_ss_tr[bcc_i], c="C0", marker="s",
               s=70, alpha=0.7, label=f"BCC ({len(bcc_i)}, RMSE={rmse_bcc:.4f} \u00c5)")
    ax.scatter(y_train[fcc_i], a_ss_tr[fcc_i], c="C3", marker="o",
               s=70, alpha=0.7, label=f"FCC ({len(fcc_i)}, RMSE={rmse_fcc:.4f} \u00c5)")

    ax.set_xlabel("Experimental $a$ (\u00c5)", fontsize=14)
    ax.set_ylabel("Predicted $a$ (\u00c5)", fontsize=14)
    ax.set_title(f"Training 64 HEA (RMSE = {rmse_all:.4f} \u00c5)", fontsize=16)
    ax.legend(fontsize=13)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_bcc_fcc.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig_bcc_fcc.png")


def fig04_indep_test(y_test, a_veg_te, a_ss_te, heas_test, gb, gf):
    """Fig 4: Multi-panel independent test figure.

    Layout: top row = (a) per-alloy error bars (full width)
            bottom row = (b) RMSE breakdown + (c) parity plot side by side
    Improved readability: larger figure, sorted bars, bigger fonts.
    """
    bcc_t = [i for i, h in enumerate(heas_test) if h["struct"] == "BCC"]
    fcc_t = [i for i, h in enumerate(heas_test) if h["struct"] == "FCC"]

    fig = plt.figure(figsize=(20, 16))
    # Top: per-alloy error (full width)
    ax_a = fig.add_axes([0.06, 0.45, 0.90, 0.52])
    # Bottom-left: RMSE breakdown
    ax_b = fig.add_axes([0.06, 0.06, 0.38, 0.32])
    # Bottom-right: parity
    ax_c = fig.add_axes([0.55, 0.06, 0.38, 0.32])

    # (a) Per-alloy absolute error — grouped by BCC/FCC, sorted within each
    alloy_names, err_veg, err_ss, structs = [], [], [], []
    for i in range(len(heas_test)):
        h = heas_test[i]
        elems = sorted(h["comp"].keys())
        alloy_names.append("-".join(elems))
        err_veg.append(abs(a_veg_te[i] - y_test[i]) * 1000)
        err_ss.append(abs(a_ss_te[i] - y_test[i]) * 1000)
        structs.append(h["struct"])

    # Group by structure, sort within each group by DFT error
    bcc_idx = [i for i, s in enumerate(structs) if s == "BCC"]
    fcc_idx = [i for i, s in enumerate(structs) if s == "FCC"]
    bcc_idx = sorted(bcc_idx, key=lambda i: err_ss[i])
    fcc_idx = sorted(fcc_idx, key=lambda i: err_ss[i])
    ordered = bcc_idx + fcc_idx  # BCC on top, FCC on bottom

    ordered_names = [alloy_names[i] for i in ordered]
    ordered_veg = [err_veg[i] for i in ordered]
    ordered_ss = [err_ss[i] for i in ordered]
    ordered_structs = [structs[i] for i in ordered]

    n_bcc = len(bcc_idx)
    n_fcc = len(fcc_idx)
    n_total = n_bcc + n_fcc

    x = np.arange(n_total)
    w = 0.38
    # Color bars by structure
    for k in range(n_total):
        bar_color_veg = "#888888"
        bar_color_ss = "C0" if k < n_bcc else "C3"
        ax_a.barh(x[k] - w/2, ordered_veg[k], w, color=bar_color_veg, alpha=0.7)
        ax_a.barh(x[k] + w/2, ordered_ss[k], w, color=bar_color_ss, alpha=0.7)

    # Legend handles
    from matplotlib.patches import Patch
    ax_a.legend(handles=[
        Patch(facecolor="#888888", alpha=0.7, label="Vegard"),
        Patch(facecolor="C0", alpha=0.7, label=r"DFT-$\Omega_{\mathrm{sf}}$ (BCC)"),
        Patch(facecolor="C3", alpha=0.7, label=r"DFT-$\Omega_{\mathrm{sf}}$ (FCC)"),
    ], fontsize=13, loc="lower right")

    ax_a.set_yticks(x)
    ax_a.set_yticklabels(ordered_names, fontsize=13, fontfamily="monospace")
    ax_a.set_xlabel("|Error| (m\u00c5)", fontsize=16)
    ax_a.set_title("(a) Per-alloy absolute error", fontsize=18)
    ax_a.invert_yaxis()

    # Separator line between BCC and FCC
    if n_bcc > 0 and n_fcc > 0:
        sep_y = n_bcc - 0.5
        ax_a.axhline(sep_y, color="black", linewidth=1.5, linestyle="--")
        ax_a.text(ax_a.get_xlim()[1] * 0.92, n_bcc / 2 - 0.5,
                  f"BCC ({n_bcc})", ha="center", va="center",
                  fontsize=14, fontweight="bold", color="C0")
        ax_a.text(ax_a.get_xlim()[1] * 0.92, n_bcc + n_fcc / 2 - 0.5,
                  f"FCC ({n_fcc})", ha="center", va="center",
                  fontsize=14, fontweight="bold", color="C3")

    # (b) BCC/FCC RMSE breakdown
    categories = ["All", "BCC", "FCC"]
    rmse_v = [np.sqrt(np.mean((a_veg_te - y_test) ** 2)) * 1000,
              np.sqrt(np.mean((a_veg_te[bcc_t] - y_test[bcc_t]) ** 2)) * 1000,
              np.sqrt(np.mean((a_veg_te[fcc_t] - y_test[fcc_t]) ** 2)) * 1000]
    rmse_s = [np.sqrt(np.mean((a_ss_te - y_test) ** 2)) * 1000,
              np.sqrt(np.mean((a_ss_te[bcc_t] - y_test[bcc_t]) ** 2)) * 1000,
              np.sqrt(np.mean((a_ss_te[fcc_t] - y_test[fcc_t]) ** 2)) * 1000]
    x2 = np.arange(len(categories))
    bars_v = ax_b.bar(x2 - 0.2, rmse_v, 0.35, color="gray", alpha=0.7, label="Vegard")
    bars_s = ax_b.bar(x2 + 0.2, rmse_s, 0.35, color="C0", alpha=0.7, label=r"DFT-$\Omega_{\mathrm{sf}}$")
    # Add value labels on bars
    for bar in list(bars_v) + list(bars_s):
        h = bar.get_height()
        ax_b.text(bar.get_x() + bar.get_width()/2, h + 0.3, f"{h:.1f}",
                  ha="center", va="bottom", fontsize=12)
    ax_b.set_xticks(x2)
    ax_b.set_xticklabels(categories, fontsize=14)
    ax_b.set_ylabel("RMSE (m\u00c5)", fontsize=14)
    ax_b.set_title("(b) RMSE breakdown", fontsize=18)
    ax_b.legend(fontsize=12)

    # (c) Parity
    lims = [min(y_test) - 0.02, max(y_test) + 0.02]
    ax_c.plot(lims, lims, "k-", lw=1)
    for idx, label, marker, c in [(bcc_t, "BCC", "s", "C0"), (fcc_t, "FCC", "o", "C3")]:
        ax_c.scatter(y_test[idx], a_ss_te[idx], c=c, marker=marker, s=90, alpha=0.7, label=label)
    ax_c.set_xlabel("Experimental $a$ (\u00c5)", fontsize=14)
    ax_c.set_ylabel("Predicted $a$ (\u00c5)", fontsize=14)
    ax_c.set_title(f"(c) Independent test ($q_{{BCC}}$={gb:.2f}, $q_{{FCC}}$={gf:.2f})", fontsize=18)
    ax_c.legend(fontsize=13)
    ax_c.set_aspect("equal")

    fig.savefig(OUTDIR / "fig_indep_test.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig_indep_test.png")


def fig05_element_delta(decomp):
    """Fig 5: Element delta — B2 and L12 grouped bar chart (single panel)."""
    d_b2 = decomp["B2"]["delta"]
    d_l12 = decomp["L12"]["delta"]
    # Union of elements, sorted by B2 delta
    all_elems = sorted(set(d_b2.keys()) | set(d_l12.keys()),
                       key=lambda e: d_b2.get(e, 0))
    n = len(all_elems)
    x = np.arange(n)
    w = 0.38

    fig, ax = plt.subplots(figsize=(max(16, n * 0.6), 7))
    vals_b2 = [d_b2.get(e, 0) for e in all_elems]
    vals_l12 = [d_l12.get(e, 0) for e in all_elems]
    ax.bar(x - w/2, vals_b2, w, color="C0", alpha=0.8, edgecolor="black",
           linewidth=0.3, label=f"B2 (R$^2$={decomp['B2']['r2']:.3f})")
    ax.bar(x + w/2, vals_l12, w, color="C3", alpha=0.8, edgecolor="black",
           linewidth=0.3, label=r"L1$_2$" + f" (R$^2$={decomp['L12']['r2']:.3f})")
    ax.set_xticks(x)
    ax.set_xticklabels(all_elems, fontsize=14, rotation=45, ha="right")
    ax.set_ylabel(r"$\delta^{(s)}$", fontsize=16)
    ax.set_title(f"Additive element parameters ({n} elements)", fontsize=18)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.legend(fontsize=14, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_element_delta.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig_element_delta.png")


def fig06_additive_fit(ob2, ol12, decomp):
    """Fig 6: Pairwise Omega_sf vs additive delta_A + delta_B."""
    OUTLIER_THRESHOLD = 0.3  # |Omega_sf| > 0.3 are unphysical (f-electron issues)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Collect all data to determine unified axis limits
    all_vals = []
    panel_data = []
    for omega, key, title in [(ob2, "B2", "B2"),
                               (ol12, "L12", r"L1$_2$")]:
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
        all_vals.extend(x_vals + y_vals)
        panel_data.append((x_vals, y_vals, x_out, y_out, key, title))

    # Unified axis range
    v_min = min(all_vals) - 0.02
    v_max = max(all_vals) + 0.02
    unified_lims = [v_min, v_max]

    from matplotlib.ticker import MultipleLocator
    for ax, (x_vals, y_vals, x_out, y_out, key, title) in zip(
            [ax1, ax2], panel_data):
        ax.scatter(x_vals, y_vals, c="C0", alpha=0.4, s=20)
        if x_out:
            ax.scatter(x_out, y_out, c="C3", alpha=0.6, s=40, marker="x",
                       label=f"excluded ({len(x_out)}, "
                             r"$|\Omega_{\mathrm{sf}}|>0.3$)")
            ax.legend(fontsize=10)
        ax.plot(unified_lims, unified_lims, "k--", lw=1)
        ax.set_xlim(unified_lims)
        ax.set_ylim(unified_lims)
        ax.xaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.05))
        ax.yaxis.set_minor_locator(MultipleLocator(0.05))
        ax.set_xlabel(r"$\delta_A^{(s)} + \delta_B^{(s)}$", fontsize=13)
        ax.set_ylabel(r"$\Omega_\mathrm{sf}^{(s)}$ (pairwise)", fontsize=13)
        n_total = len(x_vals) + len(x_out)
        ax.set_title(f"{title} ({n_total} pairs)  R$^2$ = {decomp[key]['r2']:.3f}",
                     fontsize=14)
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_additive_fit.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig_additive_fit.png")


def fig07_composition_examples(all_df):
    """Fig 7a/7b: Structure-matched Vegard composition plots.

    B2 figure: King, MP-BCC, and VASP-BCC (magnetic) Vegard lines.
    L12 figure: MP FCC volumes as FCC endpoints.
    Each figure uses structure-consistent DFT endpoints so that the
    deviation from Vegard is the pure Omega_sf, free of structure-
    mismatch artifacts.
    """
    examples = [("Cu", "Zr"), ("Al", "Ni"), ("Fe", "Ti"),
                ("Co", "Cr"), ("Pd", "Ti"), ("Nb", "Ti")]

    # Build DFT pure element volumes from Materials Project
    dft_vol_b2 = {}   # V_X^BCC from MP Im-3m
    dft_vol_l12 = {}  # V_X^FCC from MP Fm-3m
    mp_bcc_file = REPO / "data" / "mp_pure_elements_bcc.csv"
    mp_fcc_file = REPO / "data" / "mp_pure_elements_fcc.csv"
    if mp_bcc_file.exists():
        mp_bcc = pd.read_csv(mp_bcc_file)
        for _, row in mp_bcc.iterrows():
            dft_vol_b2[row["element"]] = row["volume_per_atom"]
    if mp_fcc_file.exists():
        mp_fcc = pd.read_csv(mp_fcc_file)
        for _, row in mp_fcc.iterrows():
            dft_vol_l12[row["element"]] = row["volume_per_atom"]
    # Fallback to VASP homonuclear if MP data missing
    if not dft_vol_b2 or not dft_vol_l12:
        for stype, vol_dict, n_auc in [("B2", dft_vol_b2, 2),
                                        ("L12", dft_vol_l12, 4)]:
            homo = all_df[(all_df["stype"] == stype) &
                          (all_df["element_A"] == all_df["element_B"])]
            for _, row in homo.iterrows():
                el = row["element_A"]
                a = row["lattice_constant"]
                if a <= 2 or a >= 8:
                    continue
                vol_dict.setdefault(el, a ** 3 / n_auc)

    # Build VASP-BCC (magnetic) pure element volumes
    vasp_vol_b2 = {}  # V_X^BCC from VASP ISPIN=2 magnetic recalc
    mag_b2_file = REPO / "data" / "magnetic_b2_results.csv"
    mag_b2_df = None
    if mag_b2_file.exists():
        mag_b2_df = pd.read_csv(mag_b2_file)
        # Use pure_ entries (explicit MAGMOM) for magnetic elements
        for _, row in mag_b2_df[mag_b2_df["type"] == "pure"].iterrows():
            el = row["element_A"]
            a = row["lattice_constant_A"]
            if 2 < a < 8:
                vasp_vol_b2[el] = a ** 3 / 2
        # Fill remaining from B2 self-pairs (X1X1 where A==B)
        self_pairs = mag_b2_df[
            (mag_b2_df["type"] == "B2") &
            (mag_b2_df["element_A"] == mag_b2_df["element_B"])
        ]
        for _, row in self_pairs.iterrows():
            el = row["element_A"]
            a = row["lattice_constant_A"]
            if 2 < a < 8:
                vasp_vol_b2.setdefault(el, a ** 3 / 2)

    # --- Fig 7a: B2 (BCC-like) ---
    fig_b2, axes_b2 = plt.subplots(2, 3, figsize=(18, 11))
    axes_b2 = axes_b2.flatten()
    c_arr = np.linspace(0, 1, 100)

    for idx, (elX, elY) in enumerate(examples):
        ax = axes_b2[idx]
        # King Vegard line (dashed reference)
        vX_k = KING_ATOMIC_VOLUMES.get(elX, 15)
        vY_k = KING_ATOMIC_VOLUMES.get(elY, 15)
        a_king = [(2 * ((1 - c) * vX_k + c * vY_k)) ** (1/3) for c in c_arr]
        ax.plot(c_arr * 100, a_king, "k--", lw=1.5, alpha=0.5,
                label="Vegard (King)")

        # MP-BCC Vegard line (dotted)
        if elX in dft_vol_b2 and elY in dft_vol_b2:
            vX_d = dft_vol_b2[elX]
            vY_d = dft_vol_b2[elY]
            a_dft = [(2 * ((1 - c) * vX_d + c * vY_d)) ** (1/3) for c in c_arr]
            ax.plot(c_arr * 100, a_dft, "C0:", lw=1.5, alpha=0.6,
                    label="Vegard (MP-BCC)")

        # VASP-BCC (magnetic) Vegard line (solid)
        if elX in vasp_vol_b2 and elY in vasp_vol_b2:
            vX_v = vasp_vol_b2[elX]
            vY_v = vasp_vol_b2[elY]
            a_vasp = [(2 * ((1 - c) * vX_v + c * vY_v)) ** (1/3)
                      for c in c_arr]
            ax.plot(c_arr * 100, a_vasp, "C0-", lw=2,
                    label="Vegard (VASP-BCC)")

        # B2 DFT data points — existing (gray, if magnetic data overlays)
        has_mag = (mag_b2_df is not None and
                   any(e in {"Fe", "Co", "Ni", "Mn", "Cr"}
                       for e in [elX, elY]))

        sub = all_df[all_df["stype"] == "B2"]
        ispin1_labeled = False
        for _, row in sub.iterrows():
            elA, elB = row["element_A"], row["element_B"]
            a = row["lattice_constant"]
            if a <= 2 or a >= 8:
                continue
            if {elA, elB} == {elX, elY} or \
               (elA == elB and elA in {elX, elY}):
                cA = row.get("count_A", 1)
                cB = row.get("count_B", 1)
                total = cA + cB
                if elA == elB:
                    c_B = 0.0 if elA == elX else 1.0
                elif elA == elY:
                    c_B = cA / total
                else:
                    c_B = cB / total
                color = "gray" if has_mag else "C0"
                alpha = 0.4 if has_mag else 1.0
                label = None
                if has_mag and not ispin1_labeled:
                    label = "MAGMOM default"
                    ispin1_labeled = True
                ax.scatter(c_B * 100, a, c=color, s=60, zorder=4,
                           alpha=alpha, label=label)

        # Magnetic B2 data points (orange, overlaid)
        if has_mag and mag_b2_df is not None:
            mag_sub = mag_b2_df[mag_b2_df["type"] == "B2"]
            plotted_mag = set()
            first_label = True
            for _, row in mag_sub.iterrows():
                elA, elB = row["element_A"], row["element_B"]
                a = row["lattice_constant_A"]
                if a <= 2 or a >= 8:
                    continue
                if {elA, elB} == {elX, elY} or \
                   (elA == elB and elA in {elX, elY}):
                    if elA == elB:
                        pure_row = mag_b2_df[
                            mag_b2_df["directory"] == f"pure_{elA}"]
                        if len(pure_row) > 0:
                            a = pure_row.iloc[0]["lattice_constant_A"]
                        c_B = 0.0 if elA == elX else 1.0
                        key = (elA, c_B)
                    else:
                        c_B = 0.5
                        key = (frozenset([elA, elB]), c_B)
                    if key in plotted_mag:
                        continue
                    plotted_mag.add(key)
                    label = "MAGMOM explicit" if first_label else None
                    ax.scatter(c_B * 100, a, c="C1", s=100, zorder=5,
                               marker="D", edgecolors="k", linewidths=0.5,
                               label=label)
                    first_label = False

        ax.set_xlabel(f"% {elY}")
        ax.set_ylabel("$a$ (\u00c5)")
        ax.set_title(f"B2: {elX}\u2013{elY}")
        ax.legend(fontsize=9, loc="best")

    fig_b2.tight_layout()
    fig_b2.savefig(OUTDIR / "fig_composition_b2.png", bbox_inches="tight")
    plt.close(fig_b2)
    print("  fig_composition_b2.png")

    # --- Fig 7b: L12 (FCC-like) ---
    fig_l12, axes_l12 = plt.subplots(2, 3, figsize=(18, 11))
    axes_l12 = axes_l12.flatten()

    for idx, (elX, elY) in enumerate(examples):
        ax = axes_l12[idx]
        # King Vegard line (dashed, in FCC scale: n_auc=4)
        vX_k = KING_ATOMIC_VOLUMES.get(elX, 15)
        vY_k = KING_ATOMIC_VOLUMES.get(elY, 15)
        a_king = [(4 * ((1 - c) * vX_k + c * vY_k)) ** (1/3) for c in c_arr]
        ax.plot(c_arr * 100, a_king, "k--", lw=1.5, alpha=0.5,
                label="Vegard (King)")

        # Structure-matched DFT Vegard line (solid)
        if elX in dft_vol_l12 and elY in dft_vol_l12:
            vX_d = dft_vol_l12[elX]
            vY_d = dft_vol_l12[elY]
            a_dft = [(4 * ((1 - c) * vX_d + c * vY_d)) ** (1/3) for c in c_arr]
            ax.plot(c_arr * 100, a_dft, "C3-", lw=2,
                    label=r"Vegard (MP-FCC)")

        # L12 DFT data points (including homonuclear endpoints)
        sub = all_df[all_df["stype"] == "L12"]
        for _, row in sub.iterrows():
            elA, elB = row["element_A"], row["element_B"]
            a = row["lattice_constant"]
            if a <= 2 or a >= 8:
                continue
            if {elA, elB} == {elX, elY} or \
               (elA == elB and elA in {elX, elY}):
                cA = row.get("count_A", 3)
                cB = row.get("count_B", 1)
                total = cA + cB
                if elA == elB:
                    c_B = 0.0 if elA == elX else 1.0
                elif elA == elY:
                    c_B = cA / total
                else:
                    c_B = cB / total
                ax.scatter(c_B * 100, a, c="C3", s=80, zorder=5, marker="^")

        ax.set_xlabel(f"% {elY}")
        ax.set_ylabel("$a$ (\u00c5)")
        ax.set_title(r"L1$_2$: " + f"{elX}\u2013{elY}")
        ax.legend(fontsize=10, loc="best")

    fig_l12.tight_layout()
    fig_l12.savefig(OUTDIR / "fig_composition_l12.png", bbox_inches="tight")
    plt.close(fig_l12)
    print("  fig_composition_l12.png")


def _l12_bucket(elA, elB, cA, cB):
    """Determine L1₂ bucket for sorted pair key.
    'A3B' = sorted_pair[0] is majority; 'AB3' = sorted_pair[1] is majority.
    """
    pair = tuple(sorted([elA, elB]))
    maj_elem = elA if cA >= cB else elB
    return "A3B" if maj_elem == pair[0] else "AB3"


def fig07b_vegard_heatmap(all_df):
    """Fig 7c/7d: Heatmap of Vegard deviation (Omega_sf) for all element pairs.

    Uses structure-matched DFT homonuclear endpoints so the deviation
    is pure Omega_sf without structure-mismatch artifacts.
    Separate heatmaps for B2 (BCC) and L12 (FCC).
    Returns dict of stats for paper_metrics.
    """
    stats = {}
    # Build DFT homonuclear volumes
    dft_vol = {}
    for stype, n_auc in [("B2", 2), ("L12", 4)]:
        homo = all_df[(all_df["stype"] == stype) &
                      (all_df["element_A"] == all_df["element_B"])]
        for _, row in homo.iterrows():
            el = row["element_A"]
            a = row["lattice_constant"]
            if 2 < a < 8:
                dft_vol[(el, stype)] = a ** 3 / n_auc

    for stype, n_auc, cmap_name, fig_suffix in [
        ("B2", 2, "RdBu_r", "b2"),
        ("L12", 4, "RdBu_r", "l12"),
    ]:
        sub = all_df[(all_df["stype"] == stype) &
                     (all_df["element_A"] != all_df["element_B"])]

        # Compute Omega_sf with structure-matched DFT endpoints
        pair_omega = defaultdict(list)
        for _, row in sub.iterrows():
            elA, elB = row["element_A"], row["element_B"]
            a = row["lattice_constant"]
            if a <= 2 or a >= 8:
                continue
            vA_key = (elA, stype)
            vB_key = (elB, stype)
            if vA_key not in dft_vol or vB_key not in dft_vol:
                continue
            vA, vB = dft_vol[vA_key], dft_vol[vB_key]
            cA = row.get("count_A", 1 if stype == "B2" else 3)
            cB = row.get("count_B", 1)
            total = cA + cB
            v_act = a ** 3 / n_auc
            v_veg = (cA * vA + cB * vB) / total
            if v_veg > 0:
                omega = (v_act - v_veg) / v_veg
                pair = tuple(sorted([elA, elB]))
                pair_omega[pair].append(omega)

        omega_median = {p: np.median(v) for p, v in pair_omega.items() if v}

        # Collect elements that appear in pairs
        elements = sorted({el for pair in omega_median for el in pair})
        n_el = len(elements)
        el_idx = {el: i for i, el in enumerate(elements)}

        # Build matrix
        mat = np.full((n_el, n_el), np.nan)
        for (a, b), val in omega_median.items():
            i, j = el_idx[a], el_idx[b]
            mat[i, j] = val
            mat[j, i] = val

        # Plot
        fig, ax = plt.subplots(figsize=(16, 14))
        vmax = np.nanpercentile(np.abs(mat[~np.isnan(mat)]), 95)
        im = ax.imshow(mat, cmap=cmap_name, vmin=-vmax, vmax=vmax,
                       aspect="equal", interpolation="nearest")
        ax.set_xticks(range(n_el))
        ax.set_xticklabels(elements, fontsize=12, rotation=90)
        ax.set_yticks(range(n_el))
        ax.set_yticklabels(elements, fontsize=12)
        struct_label = "B2 (BCC)" if stype == "B2" else r"L1$_2$ (FCC)"
        ax.set_title(f"Vegard deviation $\\Omega_{{\\mathrm{{sf}}}}$ — {struct_label}"
                     f"  ({len(omega_median)} pairs, DFT endpoints)",
                     fontsize=18)
        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label(r"$\Omega_{\mathrm{sf}}$ (structure-matched DFT)",
                       fontsize=14)
        cbar.ax.tick_params(labelsize=12)
        fig.tight_layout()
        fname = f"fig_vegard_heatmap_{fig_suffix}.png"
        fig.savefig(OUTDIR / fname, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  {fname} ({n_el} elements, {len(omega_median)} pairs)")
        stats[f"heatmap_{fig_suffix}_n_pairs"] = len(omega_median)
        stats[f"heatmap_{fig_suffix}_n_elements"] = n_el
    return stats


def fig07c_vegard_parity(all_df):
    """Vegard parity plots: V_DFT vs V_Vegard for all pairs.
    Returns dict of stats for paper_metrics.
    """
    stats = {}
    # Build DFT homonuclear volumes
    dft_vol = {}
    for stype, n_auc in [("B2", 2), ("L12", 4)]:
        homo = all_df[(all_df["stype"] == stype) &
                      (all_df["element_A"] == all_df["element_B"])]
        for _, row in homo.iterrows():
            el = row["element_A"]
            a = row["lattice_constant"]
            if 2 < a < 8:
                dft_vol[(el, stype)] = a ** 3 / n_auc

    for stype, n_auc, color, marker, fig_suffix in [
        ("B2", 2, "C0", "o", "b2"),
        ("L12", 4, "C3", "^", "l12"),
    ]:
        sub = all_df[(all_df["stype"] == stype) &
                     (all_df["element_A"] != all_df["element_B"])]
        v_dft_list, v_veg_list = [], []
        for _, row in sub.iterrows():
            elA, elB = row["element_A"], row["element_B"]
            a = row["lattice_constant"]
            if a <= 2 or a >= 8:
                continue
            vA_key, vB_key = (elA, stype), (elB, stype)
            if vA_key not in dft_vol or vB_key not in dft_vol:
                continue
            vA, vB = dft_vol[vA_key], dft_vol[vB_key]
            cA = row.get("count_A", 1 if stype == "B2" else 3)
            cB = row.get("count_B", 1)
            total = cA + cB
            v_act = a ** 3 / n_auc
            v_veg = (cA * vA + cB * vB) / total
            if v_veg > 0:
                v_dft_list.append(v_act)
                v_veg_list.append(v_veg)

        v_dft_arr = np.array(v_dft_list)
        v_veg_arr = np.array(v_veg_list)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(v_veg_arr, v_dft_arr, c=color, marker=marker,
                   s=15, alpha=0.4, edgecolors="none")
        vmin = min(v_veg_arr.min(), v_dft_arr.min()) * 0.95
        vmax = max(v_veg_arr.max(), v_dft_arr.max()) * 1.05
        ax.plot([vmin, vmax], [vmin, vmax], "k--", lw=1.5, alpha=0.6,
                label="Vegard (y = x)")
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$V_{\mathrm{Vegard}}$ (Å³/atom, DFT endpoints)", fontsize=13)
        ax.set_ylabel(r"$V_{\mathrm{DFT}}$ (Å³/atom)", fontsize=13)
        struct_label = "B2 (BCC)" if stype == "B2" else r"L1$_2$ (FCC)"
        # R² and RMSE
        ss_res = np.sum((v_dft_arr - v_veg_arr) ** 2)
        ss_tot = np.sum((v_dft_arr - v_dft_arr.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        rmse_v = np.sqrt(np.mean((v_dft_arr - v_veg_arr) ** 2))
        ax.set_title(f"Vegard parity — {struct_label}  "
                     f"({len(v_dft_arr)} points, R²={r2:.4f}, "
                     f"RMSE={rmse_v:.3f} Å³/atom)", fontsize=12)
        ax.legend(fontsize=11, loc="upper left")
        fig.tight_layout()
        fname = f"fig_vegard_parity_{fig_suffix}.png"
        fig.savefig(OUTDIR / fname, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  {fname} ({len(v_dft_arr)} points, R²={r2:.4f})")
        stats[f"parity_{fig_suffix}_n_points"] = len(v_dft_arr)
        stats[f"parity_{fig_suffix}_R2"] = round(r2, 4)
        stats[f"parity_{fig_suffix}_RMSE"] = round(float(rmse_v), 3)
    return stats


def fig07d_vegard_vs_radius(all_df):
    """Omega_sf vs atomic radius difference for all pairs.
    Returns dict of stats for paper_metrics.
    """
    stats = {}
    # Build DFT homonuclear volumes
    dft_vol = {}
    for stype, n_auc in [("B2", 2), ("L12", 4)]:
        homo = all_df[(all_df["stype"] == stype) &
                      (all_df["element_A"] == all_df["element_B"])]
        for _, row in homo.iterrows():
            el = row["element_A"]
            a = row["lattice_constant"]
            if 2 < a < 8:
                dft_vol[(el, stype)] = a ** 3 / n_auc

    for stype, n_auc, color, marker, fig_suffix in [
        ("B2", 2, "C0", "o", "b2"),
        ("L12", 4, "C3", "^", "l12"),
    ]:
        sub = all_df[(all_df["stype"] == stype) &
                     (all_df["element_A"] != all_df["element_B"])]

        # Collect per-pair data
        pair_data = defaultdict(lambda: {"omega": [], "rA": 0, "rB": 0})
        for _, row in sub.iterrows():
            elA, elB = row["element_A"], row["element_B"]
            a = row["lattice_constant"]
            if a <= 2 or a >= 8:
                continue
            vA_key, vB_key = (elA, stype), (elB, stype)
            if vA_key not in dft_vol or vB_key not in dft_vol:
                continue
            vA, vB = dft_vol[vA_key], dft_vol[vB_key]
            cA = row.get("count_A", 1 if stype == "B2" else 3)
            cB = row.get("count_B", 1)
            total = cA + cB
            v_act = a ** 3 / n_auc
            v_veg = (cA * vA + cB * vB) / total
            if v_veg > 0:
                omega = (v_act - v_veg) / v_veg
                pair = tuple(sorted([elA, elB]))
                pair_data[pair]["omega"].append(omega)
                # Pure element radii from DFT volume: r = (3V/4π)^(1/3)
                rA = (3 * vA / (4 * np.pi)) ** (1/3)
                rB = (3 * vB / (4 * np.pi)) ** (1/3)
                pair_data[pair]["rA"] = rA
                pair_data[pair]["rB"] = rB

        dr_list, omega_list = [], []
        for pair, d in pair_data.items():
            if d["omega"]:
                dr = abs(d["rA"] - d["rB"])
                omega_med = np.median(d["omega"])
                dr_list.append(dr)
                omega_list.append(omega_med)

        dr_arr = np.array(dr_list)
        omega_arr = np.array(omega_list)

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.scatter(dr_arr, omega_arr, c=color, marker=marker,
                   s=20, alpha=0.4, edgecolors="none")
        ax.axhline(0, color="k", ls="--", lw=1, alpha=0.5)
        ax.set_xlabel(r"$|\Delta r_{\mathrm{DFT}}|$ (Å)", fontsize=13)
        ax.set_ylabel(r"$\Omega_{\mathrm{sf}}$ (structure-matched DFT)", fontsize=13)
        struct_label = "B2 (BCC)" if stype == "B2" else r"L1$_2$ (FCC)"
        corr = np.corrcoef(dr_arr, omega_arr)[0, 1] if len(dr_arr) > 2 else 0
        ax.set_title(f"Vegard deviation vs radius difference — {struct_label}  "
                     f"({len(dr_arr)} pairs, r = {corr:.3f})", fontsize=12)
        fig.tight_layout()
        fname = f"fig_vegard_vs_radius_{fig_suffix}.png"
        fig.savefig(OUTDIR / fname, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  {fname} ({len(dr_arr)} pairs, r={corr:.3f})")
        stats[f"radius_{fig_suffix}_n_pairs"] = len(dr_arr)
        stats[f"radius_{fig_suffix}_corr"] = round(float(corr), 3)
    return stats


def fig08_delta_r_proof(all_df):
    """Fig 8: 6-panel proof that delta_r cannot absorb structure info.
    Returns dict of stats for paper_metrics.
    """
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
        return {}

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

    # (c) Omega_sf X3Y vs Y3X — should show scatter (structure-dependent)
    ax = axes[2]
    osf_a3b, osf_ab3 = [], []
    for p in complete:
        osf_a3b.append(complete[p]["L12_A3B"])
        osf_ab3.append(complete[p]["L12_AB3"])
    ax.scatter(osf_a3b, osf_ab3, c="C0", alpha=0.4, s=20)
    lims2 = [min(min(osf_a3b), min(osf_ab3)) - 0.02,
             max(max(osf_a3b), max(osf_ab3)) + 0.02]
    ax.plot(lims2, lims2, "k--", lw=1)
    r_all = np.corrcoef(osf_a3b, osf_ab3)[0, 1] if len(osf_a3b) > 2 else 0
    ax.set_xlabel(r"$\Omega_\mathrm{sf}$ (A$_3$B)")
    ax.set_ylabel(r"$\Omega_\mathrm{sf}$ (B$_3$A)")
    ax.set_title(f"(c) $\\Omega_{{sf}}$: r={r_all:.2f} ({len(osf_a3b)} pairs)")
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_delta_r_proof.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig_delta_r_proof.png")

    # --- Figure 2: (d)(e) L12 asymmetry and L12-B2 correlation ---
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

    # (d) DFT lattice constant difference |a(A3B) - a(B3A)|
    ax = axes2[0]
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
    osf_a3b_all = [complete[p]["L12_A3B"] for p in complete]
    osf_b2_list = [complete[p]["B2"] for p in complete]
    ax.scatter(osf_a3b_all, osf_b2_list, c="C0", alpha=0.4, s=20)
    r2 = np.corrcoef(osf_a3b_all, osf_b2_list)[0, 1]
    ax.set_xlabel(r"$\Omega_\mathrm{sf}$ (A$_3$B, L1$_2$)")
    ax.set_ylabel(r"$\Omega_\mathrm{sf}$ (AB, B2)")
    ax.set_title(f"(e) L1$_2$ vs B2 (r={r2:.2f})")

    fig2.tight_layout()
    fig2.savefig(OUTDIR / "fig_l12_b2_correlation.png", bbox_inches="tight")
    plt.close(fig2)
    print("  fig_l12_b2_correlation.png")
    return {
        "delta_r_proof_n_pairs": len(complete),
        "delta_r_proof_panel_c_r": round(float(r_all), 2),
        "l12_asymmetry_n_pairs": len(diffs),
        "l12_asymmetry_mean_diff": round(float(np.mean(diffs)), 3),
        "l12_b2_corr_r": round(float(r2), 2),
    }


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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
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
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
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
        labels.append(0 if phase == "SS" else 1)  # non-SS (IM+AM)=1

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

    # 2. Compute pairwise Omega_sf (all 3 sources, min_count=1)
    print("\n[2] Computing pairwise Omega_sf (MP+OQMD+VASP)...")
    ob2, ol12 = compute_omega_sf_pairwise(all_df)
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
    print("\n    Training RMSE:")
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
    print("    Test RMSE:")
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

    # 6b. Extended decomposition for Table 6
    #     Now identical to main ob2/ol12 (defaults changed to all sources, min_count=1).
    print("\n[5b] Extended decomposition for Table 6...")
    ob2_ext, ol12_ext = ob2, ol12
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
    heatmap_stats = fig07b_vegard_heatmap(all_df)
    parity_stats = fig07c_vegard_parity(all_df)
    radius_stats = fig07d_vegard_vs_radius(all_df)
    proof_stats = fig08_delta_r_proof(all_df)
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
    print("\nAdditive decomposition:")
    print(f"  B2  R2: {decomp['B2']['r2']:.4f} ({len(decomp['B2']['elements'])} elements)")
    print(f"  L12 R2: {decomp['L12']['r2']:.4f} ({len(decomp['L12']['elements'])} elements)")
    print(f"\nFigures: 16 PNGs saved to {OUTDIR}")
    print(f"Data:    3 CSVs saved to {OUTDIR}")

    # -----------------------------------------------------------------------
    # 9b. SQS analysis (BCC only, DFT-consistent Vegard reference)
    # -----------------------------------------------------------------------
    print("\n[9] SQS + DFT Vegard analysis...")
    sqs_data = load_sqs_data()
    if sqs_data is not None:
        bcc_train = [h for h in ALONSO_TABLE2 if h["struct"] == "BCC"]
        bcc_test = [h for h in INDEPENDENT_TEST if h["struct"] == "BCC"]
        sqs_metrics = analyze_sqs(sqs_data, ob2, ol12, bcc_train, bcc_test)
        print(f"    SQS pure elements: {sqs_data['n_pure_elements']}")
        print(f"    SQS pairs (King): {sqs_data['n_pairs_king']}")
        print(f"    SQS pairs (DFT Vegard): {sqs_data['n_pairs_dft']}")
        print(f"    q_BCC (SQS+DFT, optimal): {sqs_metrics['q_BCC_sqs_dft_opt']}")
        print(f"    q_BCC (SQS+DFT, adopted): 1.0")
        print(f"    BCC test RMSE (B2, q={sqs_metrics['q_BCC_b2']:.3f}): "
              f"{sqs_metrics['RMSE_b2_BCC_test']}")
        print(f"    BCC test RMSE (SQS+DFT, q=1): "
              f"{sqs_metrics['RMSE_sqs_dft_test_q1']}")
        print(f"    Improvement vs Vegard: "
              f"{sqs_metrics['improvement_sqs_dft_q1_vs_vegard_pct']}%")
        print(f"    B2 vs SQS(DFT) correlation: r={sqs_metrics['correlation_b2_vs_sqs_dft_r']}, "
              f"slope={sqs_metrics['correlation_b2_vs_sqs_dft_slope']}")

        # --- SQS: ALL test evaluation (BCC: SQS omega, FCC: L12 omega) ---
        omega_dft_sqs = sqs_data["omega_dft"]
        omega_king_sqs = sqs_data["omega_king"]
        q_sqs_king = sqs_metrics["q_BCC_sqs_king"]
        q_sqs_dft_opt = sqs_metrics["q_BCC_sqs_dft_opt"]

        y_te_all = np.array([h["a_exp"] for h in INDEPENDENT_TEST])
        bcc_te_idx = [i for i, h in enumerate(INDEPENDENT_TEST) if h["struct"] == "BCC"]
        fcc_te_idx = [i for i, h in enumerate(INDEPENDENT_TEST) if h["struct"] == "FCC"]

        def _eval28_sqs(omega_bcc, omega_fcc, qb, qf):
            p = np.array([
                compute_eq10_scaled(h["comp"], h["struct"],
                    omega_bcc if h["struct"] == "BCC" else omega_fcc,
                    qb if h["struct"] == "BCC" else qf)
                for h in INDEPENDENT_TEST])
            return {
                "ALL": round(float(np.sqrt(np.mean((y_te_all - p) ** 2))), 4),
                "BCC": round(float(np.sqrt(np.mean((y_te_all[bcc_te_idx] - p[bcc_te_idx]) ** 2))), 4),
                "FCC": round(float(np.sqrt(np.mean((y_te_all[fcc_te_idx] - p[fcc_te_idx]) ** 2))), 4),
            }

        vegard_te = _eval28_sqs({}, ol12, 0.0, 0.0)
        sqs_king_te = _eval28_sqs(omega_king_sqs, ol12, q_sqs_king, gf)
        sqs_dft_q1_te = _eval28_sqs(omega_dft_sqs, ol12, 1.0, gf)
        sqs_dft_qopt_te = _eval28_sqs(omega_dft_sqs, ol12, q_sqs_dft_opt, gf)

        n_te_sqs = len(INDEPENDENT_TEST)
        sqs_metrics["test_vegard"] = vegard_te
        sqs_metrics["test_sqs_king"] = sqs_king_te
        sqs_metrics["test_sqs_dft_q1"] = sqs_dft_q1_te
        sqs_metrics["test_sqs_dft_qopt"] = sqs_dft_qopt_te

        print(f"    --- ALL {n_te_sqs} test (BCC: SQS, FCC: L12) ---")
        print(f"    Vegard:          ALL={vegard_te['ALL']}, BCC={vegard_te['BCC']}, FCC={vegard_te['FCC']}")
        print(f"    SQS+King:        ALL={sqs_king_te['ALL']}, BCC={sqs_king_te['BCC']}, FCC={sqs_king_te['FCC']}")
        print(f"    SQS+DFT (q=1):   ALL={sqs_dft_q1_te['ALL']}, BCC={sqs_dft_q1_te['BCC']}, FCC={sqs_dft_q1_te['FCC']}")
        print(f"    SQS+DFT (q_opt): ALL={sqs_dft_qopt_te['ALL']}, BCC={sqs_dft_qopt_te['BCC']}, FCC={sqs_dft_qopt_te['FCC']}")

        # --- SQS additive decomposition ---
        print("\n[9a] SQS additive decomposition...")
        omega_dft = sqs_data["omega_dft"]
        omega_king = sqs_data["omega_king"]

        # Decompose SQS Omega_sf into element-level delta
        for label, omega_src in [("SQS_DFT", omega_dft), ("SQS_King", omega_king)]:
            elements = set()
            for (a, b) in omega_src:
                elements.add(a)
                elements.add(b)
            elements = sorted(elements)
            elem_idx = {e: i for i, e in enumerate(elements)}
            n_elem = len(elements)

            rows_A = []
            rows_b = []
            for (a, b), val in omega_src.items():
                row = np.zeros(n_elem)
                row[elem_idx[a]] = 1.0
                row[elem_idx[b]] = 1.0
                rows_A.append(row)
                rows_b.append(val)

            A_mat = np.array(rows_A)
            b_vec = np.array(rows_b)
            delta, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)

            delta_dict = {elements[i]: delta[i] for i in range(n_elem)}
            pred = A_mat @ delta
            ss_res = np.sum((b_vec - pred) ** 2)
            ss_tot = np.sum((b_vec - b_vec.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            rmse_decomp = float(np.sqrt(np.mean((b_vec - pred) ** 2)))

            # Build additive omega for BCC HEA prediction
            omega_add = {}
            for pair in omega_src:
                a, b = pair
                if a in delta_dict and b in delta_dict:
                    omega_add[pair] = delta_dict[a] + delta_dict[b]

            # Predict BCC HEAs with additive SQS delta (q=1 for DFT, q_opt for King)
            if label == "SQS_DFT":
                q_use = 1.0
                rmse_train_add = float(np.sqrt(np.mean(np.array([
                    (compute_eq10_scaled(h["comp"], h["struct"], omega_add, q_use)
                     - h["a_exp"]) ** 2
                    for h in bcc_train
                ]))))
                rmse_test_add = float(np.sqrt(np.mean(np.array([
                    (compute_eq10_scaled(h["comp"], h["struct"], omega_add, q_use)
                     - h["a_exp"]) ** 2
                    for h in bcc_test
                ]))))
                sqs_metrics["sqs_additive_dft_n_elements"] = n_elem
                sqs_metrics["sqs_additive_dft_n_pairs"] = len(omega_src)
                sqs_metrics["sqs_additive_dft_R2"] = round(r2, 4)
                sqs_metrics["sqs_additive_dft_RMSE_omega"] = round(rmse_decomp, 4)
                sqs_metrics["sqs_additive_dft_RMSE_BCC_train_q1"] = round(rmse_train_add, 4)
                sqs_metrics["sqs_additive_dft_RMSE_BCC_test_q1"] = round(rmse_test_add, 4)
                sqs_metrics["sqs_additive_dft_improvement_vs_vegard_pct"] = round(
                    (1 - rmse_test_add / sqs_metrics["RMSE_vegard_BCC_test"]) * 100, 1
                )
                print(f"    {label}: {n_elem} elements, {len(omega_src)} pairs, "
                      f"R2={r2:.4f}, RMSE(Ω)={rmse_decomp:.4f}")
                print(f"      BCC test RMSE (q=1, additive): {rmse_test_add:.4f}")
                print(f"      vs pairwise (q=1): {sqs_metrics['RMSE_sqs_dft_test_q1']}")
            else:
                sqs_metrics["sqs_additive_king_n_elements"] = n_elem
                sqs_metrics["sqs_additive_king_n_pairs"] = len(omega_src)
                sqs_metrics["sqs_additive_king_R2"] = round(r2, 4)
                sqs_metrics["sqs_additive_king_RMSE_omega"] = round(rmse_decomp, 4)
                print(f"    {label}: {n_elem} elements, {len(omega_src)} pairs, "
                      f"R2={r2:.4f}, RMSE(Ω)={rmse_decomp:.4f}")

        # --- Omega_sf distribution histogram: B2 vs SQS ---
        print("\n[9b] Omega_sf distribution histogram (B2 vs SQS)...")
        omega_dft = sqs_data["omega_dft"]
        omega_king_sqs = sqs_data["omega_king"]

        b2_vals = np.array(list(ob2.values()))
        sqs_king_vals = np.array(list(omega_king_sqs.values()))
        sqs_dft_vals = np.array(list(omega_dft.values()))

        # Common pairs for scatter
        common_pairs = sorted(set(ob2.keys()) & set(omega_dft.keys()))
        b2_common = np.array([ob2[p] for p in common_pairs])
        sqs_dft_common = np.array([omega_dft[p] for p in common_pairs])

        fig_hist, axes = plt.subplots(1, 3, figsize=(18, 6))
        bins = np.linspace(-0.35, 0.20, 56)

        ax = axes[0]
        ax.hist(b2_vals, bins=bins, alpha=0.6,
                label=f'B2 (King Vegard)\n{len(ob2)} pairs',
                color='steelblue', edgecolor='white')
        ax.hist(sqs_king_vals, bins=bins, alpha=0.6,
                label=f'SQS 1:1 (King Vegard)\n{len(omega_king_sqs)} pairs',
                color='coral', edgecolor='white')
        ax.set_xlabel(r'$\Omega_{\mathrm{sf}}$', fontsize=14)
        ax.set_ylabel('Count', fontsize=14)
        ax.set_title('B2 vs SQS (King Vegard ref.)', fontsize=14)
        ax.legend(fontsize=11)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)

        ax = axes[1]
        ax.hist(b2_vals, bins=bins, alpha=0.6,
                label=f'B2 (King Vegard)\n{len(ob2)} pairs',
                color='steelblue', edgecolor='white')
        ax.hist(sqs_dft_vals, bins=bins, alpha=0.6,
                label=f'SQS 1:1 (DFT Vegard)\n{len(omega_dft)} pairs',
                color='forestgreen', edgecolor='white')
        ax.set_xlabel(r'$\Omega_{\mathrm{sf}}$', fontsize=14)
        ax.set_ylabel('Count', fontsize=14)
        ax.set_title('B2 vs SQS (DFT Vegard ref.)', fontsize=14)
        ax.legend(fontsize=11)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)

        ax = axes[2]
        ax.scatter(b2_common, sqs_dft_common, alpha=0.4, s=20, color='purple')
        r_scatter = float(np.corrcoef(b2_common, sqs_dft_common)[0, 1])
        slope_scatter = float(np.polyfit(b2_common, sqs_dft_common, 1)[0])
        ax.plot([-0.35, 0.20], [-0.35, 0.20], 'k--', alpha=0.3, label='1:1')
        ax.set_xlabel(r'$\Omega_{\mathrm{sf}}^{\mathrm{B2}}$ (King)', fontsize=14)
        ax.set_ylabel(r'$\Omega_{\mathrm{sf}}^{\mathrm{SQS}}$ (DFT Vegard)', fontsize=14)
        ax.set_title(f'B2 vs SQS correlation\n'
                     f'r = {r_scatter:.3f}, slope = {slope_scatter:.3f}, '
                     f'{len(common_pairs)} pairs', fontsize=13)
        ax.set_aspect('equal')
        ax.axhline(0, color='gray', linestyle=':', alpha=0.3)
        ax.axvline(0, color='gray', linestyle=':', alpha=0.3)

        fig_hist.tight_layout()
        fig_hist.savefig(OUTDIR / "fig_omega_hist_b2_sqs.png",
                         bbox_inches="tight", dpi=150)
        plt.close(fig_hist)
        print("  fig_omega_hist_b2_sqs.png")

        # Add distribution stats to metrics
        sqs_metrics["omega_b2_mean"] = round(float(b2_vals.mean()), 4)
        sqs_metrics["omega_b2_std"] = round(float(b2_vals.std()), 4)
        sqs_metrics["omega_sqs_king_mean"] = round(float(sqs_king_vals.mean()), 4)
        sqs_metrics["omega_sqs_king_std"] = round(float(sqs_king_vals.std()), 4)
        sqs_metrics["omega_sqs_dft_mean"] = round(float(sqs_dft_vals.mean()), 4)
        sqs_metrics["omega_sqs_dft_std"] = round(float(sqs_dft_vals.std()), 4)
        sqs_metrics["omega_b2_negative_pct"] = round(
            float(np.sum(b2_vals < 0) / len(b2_vals) * 100), 1)

    else:
        sqs_metrics = {}
        print("    SQS data not found, skipping.")

    # -----------------------------------------------------------------------
    # 9c. DFT self-consistent Omega_sf (B2/L12 with DFT Vegard reference)
    # -----------------------------------------------------------------------
    print("\n[9c] DFT self-consistent Omega_sf analysis...")
    dft_sc_metrics = analyze_dft_self_consistent(
        all_df, ob2, ol12, ALONSO_TABLE2, INDEPENDENT_TEST
    )
    print(f"    q_BCC (DFT ref): {dft_sc_metrics['q_BCC_dft_ref']}")
    print(f"    q_FCC (DFT ref): {dft_sc_metrics['q_FCC_dft_ref']}")
    print(f"    Train RMSE (opt q): {dft_sc_metrics['RMSE_dft_ref_opt_q_train']}")
    print(f"    Test  RMSE (opt q): {dft_sc_metrics['RMSE_dft_ref_opt_q_test']}")
    print(f"    Test  RMSE BCC:     {dft_sc_metrics['RMSE_dft_ref_opt_q_test_BCC']}")
    print(f"    Test  RMSE FCC:     {dft_sc_metrics['RMSE_dft_ref_opt_q_test_FCC']}")
    print(f"    Improvement (test vs Vegard): "
          f"{dft_sc_metrics['improvement_dft_ref_opt_test_pct']}%")
    print(f"    Improvement (test BCC vs Vegard): "
          f"{dft_sc_metrics['improvement_dft_ref_opt_test_BCC_pct']}%")

    # -----------------------------------------------------------------------
    # 9d. ML residual correction analysis
    # -----------------------------------------------------------------------
    print("\n[9d] ML residual correction analysis...")
    ml_metrics = analyze_ml_residual(y_train, a_ss_tr, ALONSO_TABLE2, ob2, ol12)
    print(f"    Physics model RMSE: {ml_metrics['RMSE_physics_train']}")
    print(f"    Ridge LOO-CV RMSE: {ml_metrics['RMSE_ridge_loo']}")
    print(f"    XGBoost train RMSE: {ml_metrics['RMSE_xgb_train']}")
    print(f"    XGBoost LOO RMSE: {ml_metrics['RMSE_xgb_loo']}")

    # -----------------------------------------------------------------------
    # 10. Paper metrics JSON — ALL numerical values used in the manuscript
    # -----------------------------------------------------------------------
    import json

    rmse_veg_te_bcc = float(np.sqrt(np.mean((a_veg_te[bcc_t] - y_test[bcc_t]) ** 2)))
    rmse_veg_te_fcc = float(np.sqrt(np.mean((a_veg_te[fcc_t] - y_test[fcc_t]) ** 2)))
    rmse_ss_te_bcc = float(np.sqrt(np.mean((a_ss_te[bcc_t] - y_test[bcc_t]) ** 2)))
    rmse_ss_te_fcc = float(np.sqrt(np.mean((a_ss_te[fcc_t] - y_test[fcc_t]) ** 2)))
    rmse_ss_tr_bcc = float(np.sqrt(np.mean((a_ss_tr[bcc_i] - y_train[bcc_i]) ** 2)))
    rmse_ss_tr_fcc = float(np.sqrt(np.mean((a_ss_tr[fcc_i] - y_train[fcc_i]) ** 2)))

    # Binary system Vegard RMSE (for Fig14 discussion)
    # Compute from all B2+L12 pairs using King volumes
    binary_b2_rmse_veg = []
    binary_b2_rmse_veff = []
    for (a, b), osf in ob2.items():
        vA, vB = KING_ATOMIC_VOLUMES.get(a, 0), KING_ATOMIC_VOLUMES.get(b, 0)
        if vA == 0 or vB == 0:
            continue
        for c in [0.25, 0.5, 0.75]:
            v_veg = c * vA + (1 - c) * vB
            a_veg = (2 * v_veg) ** (1/3)
            v_eff = v_veg * (1 + gb * ((1 - c) * osf))
            a_eff = (2 * v_eff) ** (1/3)
            # DFT value: reconstruct from omega_sf
            v_dft = v_veg * (1 + osf) if c == 0.5 else v_veg  # approx
            a_dft = (2 * v_dft) ** (1/3)
            if c == 0.5:
                binary_b2_rmse_veg.append((a_veg - a_dft) ** 2)
                binary_b2_rmse_veff.append((a_eff - a_dft) ** 2)

    # Phase classification metrics (from MULTIPHASE_HEA_DB)
    # Convention: non-SS (IM+AM)=1, SS=0. Higher dr/dsf predicts non-SS.
    # Aligned with fig15_roc labeling and standard HEA literature.
    from sklearn.metrics import roc_auc_score, f1_score
    dr_vals, dsf_vals, labels = [], [], []
    for h in MULTIPHASE_HEA_DB:
        dr = compute_delta_r(h["comp"])
        dsf = compute_delta_sf(h["comp"], ob2)
        if np.isnan(dr) or np.isnan(dsf):
            continue
        dr_vals.append(dr)
        dsf_vals.append(dsf)
        labels.append(0 if h.get("phase", "SS") == "SS" else 1)
    dr_vals_arr = np.array(dr_vals)
    dsf_vals_arr = np.array(dsf_vals)
    labels_arr = np.array(labels)
    auc_dr = float(roc_auc_score(labels, dr_vals))
    auc_dsf = float(roc_auc_score(labels, dsf_vals))

    # Optimal thresholds (maximize F1)
    def find_best_threshold(vals, labs):
        best_f1, best_thr = 0, 0
        for thr in np.linspace(min(vals), max(vals), 1000):
            preds = (np.array(vals) >= thr).astype(int)
            f1 = float(f1_score(labs, preds, zero_division=0))
            if f1 > best_f1:
                best_f1, best_thr = f1, thr
        return best_thr, best_f1

    best_thr_dr, f1_dr = find_best_threshold(dr_vals, labels)
    best_thr_dsf, f1_dsf = find_best_threshold(dsf_vals, labels)

    # --- Bootstrap CI for ΔRMSE (Vegard − Ω_sf) ---
    print("\n[Bootstrap] ΔRMSE confidence intervals (10,000 iterations)...")
    n_boot = 10_000
    rng = np.random.RandomState(42)
    n_te = len(y_test)

    err_veg = a_veg_te - y_test       # Vegard errors
    err_ss = a_ss_te - y_test          # DFT-Ω_sf errors

    def _boot_delta_rmse(idx_subset, n_iter=n_boot):
        """Bootstrap ΔRMSE = RMSE_Vegard − RMSE_Ωsf for a subset."""
        ev = err_veg[idx_subset]
        es = err_ss[idx_subset]
        n = len(ev)
        deltas = np.empty(n_iter)
        for b in range(n_iter):
            ix = rng.randint(0, n, size=n)
            deltas[b] = np.sqrt(np.mean(ev[ix] ** 2)) - np.sqrt(np.mean(es[ix] ** 2))
        delta_obs = np.sqrt(np.mean(ev ** 2)) - np.sqrt(np.mean(es ** 2))
        ci_lo = float(np.percentile(deltas, 2.5))
        ci_hi = float(np.percentile(deltas, 97.5))
        return round(delta_obs, 4), round(ci_lo, 4), round(ci_hi, 4)

    all_idx = np.arange(n_te)
    bcc_idx = np.array(bcc_t)
    fcc_idx = np.array(fcc_t)

    boot_all = _boot_delta_rmse(all_idx)
    boot_bcc = _boot_delta_rmse(bcc_idx)
    boot_fcc = _boot_delta_rmse(fcc_idx)

    print(f"    ALL {n_te}: ΔRMSE={boot_all[0]:+.4f}, 95% CI=[{boot_all[1]:+.4f}, {boot_all[2]:+.4f}]")
    print(f"    BCC {len(bcc_idx)}: ΔRMSE={boot_bcc[0]:+.4f}, 95% CI=[{boot_bcc[1]:+.4f}, {boot_bcc[2]:+.4f}]")
    print(f"    FCC {len(fcc_idx)}: ΔRMSE={boot_fcc[0]:+.4f}, 95% CI=[{boot_fcc[1]:+.4f}, {boot_fcc[2]:+.4f}]")

    bootstrap_ci = {
        "n_iterations": n_boot,
        "seed": 42,
        "ALL": {
            "N": int(n_te),
            "delta_RMSE": boot_all[0],
            "CI_95_lower": boot_all[1],
            "CI_95_upper": boot_all[2],
        },
        "BCC": {
            "N": int(len(bcc_idx)),
            "delta_RMSE": boot_bcc[0],
            "CI_95_lower": boot_bcc[1],
            "CI_95_upper": boot_bcc[2],
        },
        "FCC": {
            "N": int(len(fcc_idx)),
            "delta_RMSE": boot_fcc[0],
            "CI_95_lower": boot_fcc[1],
            "CI_95_upper": boot_fcc[2],
        },
    }

    # --- LOO-DCV (Leave-One-Out Double Cross Validation) ---
    # Outer loop: leave 1 alloy out; inner loop: optimize q on remaining N-1
    print("\n[LOO-DCV] Leave-One-Out Double Cross Validation...")
    all_heas = list(ALONSO_TABLE2) + list(INDEPENDENT_TEST)
    N_all = len(all_heas)
    loo_preds = np.empty(N_all)
    loo_vegard = np.empty(N_all)
    loo_y = np.array([h["a_exp"] for h in all_heas])
    loo_structs = [h["struct"] for h in all_heas]

    for i in range(N_all):
        remaining = all_heas[:i] + all_heas[i + 1:]
        gb_i, gf_i = optimize_gamma(remaining, ob2, ol12)
        h = all_heas[i]
        loo_preds[i] = compute_eq10_scaled(
            h["comp"], h["struct"],
            ob2 if h["struct"] == "BCC" else ol12,
            gb_i if h["struct"] == "BCC" else gf_i,
        )
        loo_vegard[i] = compute_vegard(h["comp"], h["struct"])

    loo_bcc = np.array([i for i, s in enumerate(loo_structs) if s == "BCC"])
    loo_fcc = np.array([i for i, s in enumerate(loo_structs) if s == "FCC"])

    loo_rmse_all = float(np.sqrt(np.mean((loo_preds - loo_y) ** 2)))
    loo_rmse_bcc = float(np.sqrt(np.mean((loo_preds[loo_bcc] - loo_y[loo_bcc]) ** 2)))
    loo_rmse_fcc = float(np.sqrt(np.mean((loo_preds[loo_fcc] - loo_y[loo_fcc]) ** 2)))
    loo_veg_all = float(np.sqrt(np.mean((loo_vegard - loo_y) ** 2)))
    loo_veg_bcc = float(np.sqrt(np.mean((loo_vegard[loo_bcc] - loo_y[loo_bcc]) ** 2)))
    loo_veg_fcc = float(np.sqrt(np.mean((loo_vegard[loo_fcc] - loo_y[loo_fcc]) ** 2)))

    print(f"  LOO-DCV ({N_all} alloys):")
    print(f"    ALL: Vegard={loo_veg_all:.4f}, Ω_sf={loo_rmse_all:.4f} "
          f"({(1 - loo_rmse_all / loo_veg_all) * 100:.1f}% improvement)")
    print(f"    BCC ({len(loo_bcc)}): Vegard={loo_veg_bcc:.4f}, Ω_sf={loo_rmse_bcc:.4f} "
          f"({(1 - loo_rmse_bcc / loo_veg_bcc) * 100:.1f}% improvement)")
    print(f"    FCC ({len(loo_fcc)}): Vegard={loo_veg_fcc:.4f}, Ω_sf={loo_rmse_fcc:.4f} "
          f"({(1 - loo_rmse_fcc / loo_veg_fcc) * 100:.1f}% improvement)")

    loo_dcv_results = {
        "N": N_all,
        "N_BCC": int(len(loo_bcc)),
        "N_FCC": int(len(loo_fcc)),
        "ALL": {
            "RMSE_Vegard": round(loo_veg_all, 4),
            "RMSE_Omega_sf": round(loo_rmse_all, 4),
            "improvement_pct": round((1 - loo_rmse_all / loo_veg_all) * 100, 1),
        },
        "BCC": {
            "RMSE_Vegard": round(loo_veg_bcc, 4),
            "RMSE_Omega_sf": round(loo_rmse_bcc, 4),
            "improvement_pct": round((1 - loo_rmse_bcc / loo_veg_bcc) * 100, 1),
        },
        "FCC": {
            "RMSE_Vegard": round(loo_veg_fcc, 4),
            "RMSE_Omega_sf": round(loo_rmse_fcc, 4),
            "improvement_pct": round((1 - loo_rmse_fcc / loo_veg_fcc) * 100, 1),
        },
    }

    # --- Bootstrap with q re-optimization (10,000 iterations) ---
    print("\n[Bootstrap-DCV] ΔRMSE CI with q re-optimization (10,000 iterations)...")
    n_boot_dcv = 10_000
    rng_dcv = np.random.RandomState(42)
    boot_deltas_all = np.empty(n_boot_dcv)
    boot_deltas_bcc = np.empty(n_boot_dcv)
    boot_deltas_fcc = np.empty(n_boot_dcv)

    for b in range(n_boot_dcv):
        if b % 1000 == 0:
            print(f"    iteration {b}/{n_boot_dcv}...")
        ix = rng_dcv.randint(0, N_all, size=N_all)
        train_b = [all_heas[j] for j in ix]
        oob_mask = np.ones(N_all, dtype=bool)
        oob_mask[ix] = False
        oob_idx = np.where(oob_mask)[0]
        if len(oob_idx) < 2:
            boot_deltas_all[b] = 0.0
            boot_deltas_bcc[b] = 0.0
            boot_deltas_fcc[b] = 0.0
            continue
        gb_b, gf_b = optimize_gamma(train_b, ob2, ol12)
        oob_heas = [all_heas[j] for j in oob_idx]
        oob_y = np.array([h["a_exp"] for h in oob_heas])
        oob_pred = predict_heas(oob_heas, ob2, ol12, gb_b, gf_b)
        oob_veg = np.array([compute_vegard(h["comp"], h["struct"]) for h in oob_heas])
        rmse_v = np.sqrt(np.mean((oob_veg - oob_y) ** 2))
        rmse_o = np.sqrt(np.mean((oob_pred - oob_y) ** 2))
        boot_deltas_all[b] = rmse_v - rmse_o

        oob_bcc = [j for j, h in enumerate(oob_heas) if h["struct"] == "BCC"]
        oob_fcc = [j for j, h in enumerate(oob_heas) if h["struct"] == "FCC"]
        if len(oob_bcc) >= 2:
            boot_deltas_bcc[b] = (
                np.sqrt(np.mean((oob_veg[oob_bcc] - oob_y[oob_bcc]) ** 2)) -
                np.sqrt(np.mean((oob_pred[oob_bcc] - oob_y[oob_bcc]) ** 2))
            )
        else:
            boot_deltas_bcc[b] = np.nan
        if len(oob_fcc) >= 2:
            boot_deltas_fcc[b] = (
                np.sqrt(np.mean((oob_veg[oob_fcc] - oob_y[oob_fcc]) ** 2)) -
                np.sqrt(np.mean((oob_pred[oob_fcc] - oob_y[oob_fcc]) ** 2))
            )
        else:
            boot_deltas_fcc[b] = np.nan

    def _boot_summary(deltas, label):
        valid = deltas[~np.isnan(deltas)]
        n_valid = len(valid)
        mean_d = float(np.mean(valid))
        ci_lo = float(np.percentile(valid, 2.5))
        ci_hi = float(np.percentile(valid, 97.5))
        print(f"    {label}: ΔRMSE={mean_d:+.4f}, 95% CI=[{ci_lo:+.4f}, {ci_hi:+.4f}] "
              f"(n_valid={n_valid})")
        return {
            "mean_delta_RMSE": round(mean_d, 4),
            "CI_95_lower": round(ci_lo, 4),
            "CI_95_upper": round(ci_hi, 4),
            "n_valid": n_valid,
        }

    boot_dcv_all = _boot_summary(boot_deltas_all, f"ALL {N_all}")
    boot_dcv_bcc = _boot_summary(boot_deltas_bcc, f"BCC {len(loo_bcc)}")
    boot_dcv_fcc = _boot_summary(boot_deltas_fcc, f"FCC {len(loo_fcc)}")

    bootstrap_dcv_results = {
        "n_iterations": n_boot_dcv,
        "seed": 42,
        "method": "OOB with q re-optimization per bootstrap sample",
        "ALL": boot_dcv_all,
        "BCC": boot_dcv_bcc,
        "FCC": boot_dcv_fcc,
    }

    metrics = {
        "_description": "All numerical values used in the paper. Generated by generate_all_figures.py.",
        "data": {
            "n_compounds_total": int(len(all_df)),
            "n_MP": int(n_mp),
            "n_OQMD": int(n_oqmd),
            "n_VASP": int(n_vasp),
            "n_B2_pairs": int(len(ob2)),
            "n_L12_pairs": int(len(ol12)),
            "n_training_HEA": int(len(ALONSO_TABLE2)),
            "n_test_HEA": int(len(INDEPENDENT_TEST)),
            "n_BCC_train": int(len(bcc_i)),
            "n_FCC_train": int(len(fcc_i)),
            "n_BCC_test": int(len(bcc_t)),
            "n_FCC_test": int(len(fcc_t)),
        },
        "calibration": {
            "q_BCC": round(float(gb), 4),
            "q_FCC": round(float(gf), 4),
        },
        "training_64HEA": {
            "RMSE_Vegard": round(float(rmse_veg_tr), 4),
            "RMSE_DFT_Omega_sf": round(float(rmse_ss_tr), 4),
            "RMSE_additive": round(float(rmse_add_tr), 4),
            "improvement_vs_Vegard_pct": round((1 - rmse_ss_tr / rmse_veg_tr) * 100, 1),
            "RMSE_BCC": round(float(rmse_ss_tr_bcc), 4),
            "RMSE_FCC": round(float(rmse_ss_tr_fcc), 4),
        },
        "independent_test": {
            "RMSE_Vegard": round(float(rmse_veg_te), 4),
            "RMSE_DFT_Omega_sf": round(float(rmse_ss_te), 4),
            "RMSE_additive": round(float(rmse_add_te), 4),
            "improvement_vs_Vegard_pct": round((1 - rmse_ss_te / rmse_veg_te) * 100, 1),
            "RMSE_Vegard_BCC": round(float(rmse_veg_te_bcc), 4),
            "RMSE_Vegard_FCC": round(float(rmse_veg_te_fcc), 4),
            "RMSE_BCC": round(float(rmse_ss_te_bcc), 4),
            "RMSE_FCC": round(float(rmse_ss_te_fcc), 4),
            "improvement_BCC_pct": round((1 - rmse_ss_te_bcc / rmse_veg_te_bcc) * 100, 1),
            "improvement_FCC_pct": round((1 - rmse_ss_te_fcc / rmse_veg_te_fcc) * 100, 1),
            "BCC_identical_to_Vegard": f"{id_v}/{len(bcc_t)}",
        },
        "additive_decomposition": {
            "B2_R2": round(float(decomp["B2"]["r2"]), 4),
            "L12_R2": round(float(decomp["L12"]["r2"]), 4),
            "B2_RMSE_Omega_sf": round(float(decomp["B2"]["rmse"]), 4),
            "L12_RMSE_Omega_sf": round(float(decomp["L12"]["rmse"]), 4),
            "B2_n_elements": int(len(decomp["B2"]["elements"])),
            "L12_n_elements": int(len(decomp["L12"]["elements"])),
            "B2_extended_R2": round(float(decomp_table["B2"]["r2"]), 4),
            "L12_extended_R2": round(float(decomp_table["L12"]["r2"]), 4),
            "B2_extended_n_elements": int(len(decomp_table["B2"]["elements"])),
            "L12_extended_n_elements": int(len(decomp_table["L12"]["elements"])),
            "additive_mode_A_q_BCC": round(float(gb_addA), 4),
            "additive_mode_A_q_FCC": round(float(gf_addA), 4),
            "additive_mode_A_RMSE_train": round(float(rmse_addA_tr), 4),
            "additive_mode_A_RMSE_test": round(float(rmse_addA_te), 4),
        },
        "phase_classification": {
            "n_HEA": int(len(MULTIPHASE_HEA_DB)),
            "AUC_delta_r": round(float(auc_dr), 3),
            "AUC_delta_sf": round(float(auc_dsf), 3),
            "accuracy_delta_r": round(float(np.mean(
                [(1 if d >= best_thr_dr else 0) == la for d, la in zip(dr_vals, labels)])), 3),
            "accuracy_delta_sf": round(float(np.mean(
                [(1 if d >= best_thr_dsf else 0) == la for d, la in zip(dsf_vals, labels)])), 3),
            "F1_delta_r": round(float(f1_dr), 3),
            "F1_delta_sf": round(float(f1_dsf), 3),
            "threshold_delta_r_pct": round(float(best_thr_dr), 2),
            "threshold_delta_sf": round(float(best_thr_dsf), 4),
            "convention": "non-SS (IM+AM) = positive class",
        },
        "noise_floor": {
            "sigma_approx": 0.016,
            "comment": "Experimental reproducibility limit from literature variance",
        },
        "bootstrap_ci": bootstrap_ci,
        "loo_dcv": loo_dcv_results,
        "bootstrap_dcv": bootstrap_dcv_results,
    }

    # --- Compute delta_r_proof, l12_asymmetry, effective_radius metrics ---
    F_ELEMENTS = {"La","Ce","Pr","Nd","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu"}

    # delta_r_proof: pairs with all 3 structures (B2, L12_A3B, L12_AB3)
    pair_data_m = defaultdict(lambda: {"B2": [], "L12_A3B": [], "L12_AB3": []})
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
            pair_data_m[pair]["B2"].append(osf)
        elif stype == "L12":
            cA = row.get("count_A", 3)
            cB = row.get("count_B", 1)
            total = cA + cB
            v_veg = (cA * vA + cB * vB) / total
            osf = (a**3 / 4 - v_veg) / v_veg
            bucket = "L12_" + _l12_bucket(elA, elB, cA, cB)
            pair_data_m[pair][bucket].append(osf)

    complete_m = {}
    for pair, data in pair_data_m.items():
        if data["B2"] and data["L12_A3B"] and data["L12_AB3"]:
            complete_m[pair] = {
                "B2": np.median(data["B2"]),
                "L12_A3B": np.median(data["L12_A3B"]),
                "L12_AB3": np.median(data["L12_AB3"]),
            }

    if complete_m:
        all_a3b_m = [complete_m[p]["L12_A3B"] for p in complete_m]
        all_ab3_m = [complete_m[p]["L12_AB3"] for p in complete_m]
        all_b2_m = [complete_m[p]["B2"] for p in complete_m]
        r_all_m = round(float(np.corrcoef(all_a3b_m, all_ab3_m)[0, 1]), 2) if len(all_a3b_m) > 2 else 0
        r_l12_b2_m = round(float(np.corrcoef(all_a3b_m, all_b2_m)[0, 1]), 2) if len(all_a3b_m) > 2 else 0
        metrics["delta_r_proof"] = {
            "delta_r_proof_n_complete": len(complete_m),
            "delta_r_proof_r_all": r_all_m,
            "l12_vs_b2_corr_r": r_l12_b2_m,
        }

    # l12_asymmetry: |a(A3B) - a(B3A)| distribution
    # No King filter — matches the figure which uses all DFT pairs
    pair_a_m = defaultdict(lambda: {"A3B": [], "AB3": []})
    for _, row in all_df[all_df["stype"] == "L12"].iterrows():
        elA, elB = row["element_A"], row["element_B"]
        a = row["lattice_constant"]
        if a <= 2 or a >= 8 or elA == elB:
            continue
        pair = tuple(sorted([elA, elB]))
        cA = row.get("count_A", 3)
        cB = row.get("count_B", 1)
        bucket = _l12_bucket(elA, elB, cA, cB)
        pair_a_m[pair][bucket].append(a)
    diffs_m = []
    for pair in pair_a_m:
        if pair_a_m[pair]["A3B"] and pair_a_m[pair]["AB3"]:
            diffs_m.append(abs(np.median(pair_a_m[pair]["A3B"]) -
                              np.median(pair_a_m[pair]["AB3"])))
    if diffs_m:
        metrics["l12_asymmetry"] = {
            "l12_asymmetry_n_pairs": len(diffs_m),
            "l12_asymmetry_std": round(float(np.std(diffs_m)), 2),
            "l12_asymmetry_mean_abs_diff": round(float(np.mean(diffs_m)), 3),
            "l12_asymmetry_max_abs_diff": round(float(np.max(diffs_m)), 2),
        }

    # effective_radius: minority/majority deviations from L12 packing
    if radii:
        min_devs, maj_devs = [], []
        for elem, r_data in radii.items():
            if "r_l12_min" in r_data and "r_l12_maj" in r_data:
                r_king = (3 * KING_ATOMIC_VOLUMES.get(elem, 15) / (4 * np.pi)) ** (1/3)
                min_devs.append(abs(r_data["r_l12_min"] - r_king))
                maj_devs.append(abs(r_data["r_l12_maj"] - r_king))
        if min_devs and maj_devs:
            min_mean = round(float(np.mean(min_devs)), 3)
            maj_mean = round(float(np.mean(maj_devs)), 3)
            ratio = round(min_mean / maj_mean * 100, 0) if maj_mean > 0 else 0
            metrics["effective_radius"] = {
                "eff_radius_min_dev": min_mean,
                "eff_radius_maj_dev": maj_mean,
                "eff_radius_min_over_maj_pct": ratio,
            }

    # Add SQS metrics
    if sqs_metrics:
        metrics["sqs_analysis"] = sqs_metrics

    # Add DFT self-consistent metrics
    metrics["dft_self_consistent"] = dft_sc_metrics

    # Add ML residual metrics
    metrics["ml_residual"] = ml_metrics

    # Add figure-specific stats (ensures figure annotations match paper text)
    fig_stats = {}
    fig_stats.update(heatmap_stats or {})
    fig_stats.update(parity_stats or {})
    fig_stats.update(radius_stats or {})
    fig_stats.update(proof_stats or {})
    metrics["figure_stats"] = fig_stats

    with open(OUTDIR / "paper_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print("\n    paper_metrics.json saved (all numerical values for manuscript)")


if __name__ == "__main__":
    main()
