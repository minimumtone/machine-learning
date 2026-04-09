#!/usr/bin/env python3
"""
Four-case comparison study: MP-B2, MP-L1_2, OQMD-B2, OQMD-L1_2.

Extracts binary cubic (Pm-3m) compounds from both Materials Project and OQMD,
classifies them as B2 or L1_2, calculates effective atomic radii via TRF
optimisation, and generates a comprehensive comparison report with figures.
"""

import multiprocessing
import os
import re
import sys
import time
import warnings
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy.optimize import least_squares, minimize
from threadpoolctl import threadpool_limits

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

# Lattice constant range for valid B2/L1_2 unit cells (Angstrom).
# Values outside this range likely indicate superstructures (e.g. Rh17S15 ~10 A).
LATTICE_CONST_MIN = 2.0
LATTICE_CONST_MAX = 8.0

# Non-metal elements to exclude from the analysis.
# Hard-sphere model is not suitable for compounds with strong covalent/ionic bonding.
NON_METAL_ELEMENTS = frozenset({
    "H", "He",
    "C", "N", "O", "F", "Ne",
    "P", "S", "Cl", "Ar",
    "Se", "Br", "Kr",
    "Te", "I", "Xe",
    "At", "Rn",
})

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
    "Pa": 1.63, "U": 1.54, "Pu": 1.64, "Np": 1.55, "Am": 1.73,
}


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------

def _parse_oqmd_sites(sites_list: List[str]) -> Dict[str, int]:
    """Parse OQMD site strings like 'Fe @ 0 0 0' to element counts."""
    counts: Dict[str, int] = {}
    for s in sites_list:
        el = s.split("@")[0].strip()
        # strip trailing digits for elements like "Fe1"
        el = re.match(r"[A-Z][a-z]?", el).group()
        counts[el] = counts.get(el, 0) + 1
    return counts


def extract_mp_all(api_key: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extract B2 and L1_2 compounds from Materials Project in a single API call."""
    from mp_api.client import MPRester

    mpr = MPRester(api_key)
    docs = mpr.materials.summary.search(
        spacegroup_number=221,
        num_elements=2,
        fields=[
            "material_id", "formula_pretty", "structure",
            "energy_per_atom", "energy_above_hull", "composition"
        ]
    )
    print(f"    MP raw documents: {len(docs)}")

    tol = 0.05
    b2_data, l12_data = [], []
    for doc in docs:
        try:
            structure = doc.structure
            lattice = structure.lattice
            a, b, c = lattice.a, lattice.b, lattice.c
            avg = (a + b + c) / 3
            if not (abs(a - avg) / avg < 0.01
                    and abs(b - avg) / avg < 0.01
                    and abs(c - avg) / avg < 0.01):
                continue

            composition = doc.composition
            elements = list(composition.elements)
            if len(elements) != 2:
                continue

            element_A = str(elements[0])
            element_B = str(elements[1])
            if element_A in NON_METAL_ELEMENTS or element_B in NON_METAL_ELEMENTS:
                continue
            count_A = composition[elements[0]]
            count_B = composition[elements[1]]
            total = count_A + count_B
            ratio_A = count_A / total
            ratio_B = count_B / total

            row = {
                "material_id": str(doc.material_id),
                "formula": doc.formula_pretty,
                "element_A": element_A,
                "element_B": element_B,
                "count_A": count_A,
                "count_B": count_B,
                "lattice_constant": lattice.a,
                "energy_per_atom": doc.energy_per_atom,
                "energy_above_hull": doc.energy_above_hull,
                "source": "MP",
            }

            if not (LATTICE_CONST_MIN <= lattice.a <= LATTICE_CONST_MAX):
                continue

            if abs(ratio_A - 0.5) < tol and abs(ratio_B - 0.5) < tol:
                row["structure_type"] = "B2"
                b2_data.append(row)
            elif ((abs(ratio_A - 0.75) < tol and abs(ratio_B - 0.25) < tol) or
                  (abs(ratio_A - 0.25) < tol and abs(ratio_B - 0.75) < tol)):
                row["structure_type"] = "L1$_2$"
                l12_data.append(row)
        except Exception:
            continue

    dfs = {}
    for label, raw in [("B2", b2_data), ("L1$_2$", l12_data)]:
        df = pd.DataFrame(raw)
        if len(df) > 0:
            df = df.sort_values("energy_per_atom").drop_duplicates(
                subset=["element_A", "element_B"], keep="first"
            ).reset_index(drop=True)
        dfs[label] = df

    print(f"    MP B2:   {len(dfs['B2'])} compounds")
    print(f"    MP L1$_2$: {len(dfs['L1$_2$'])} compounds")
    return dfs["B2"], dfs["L1$_2$"]


def _fetch_oqmd_paginated(prototype: str, page_size: int = 500,
                           timeout: int = 180) -> List[dict]:
    """Fetch all OQMD entries for a given prototype with pagination."""
    base = "http://oqmd.org/oqmdapi/formationenergy"
    all_entries: List[dict] = []
    offset = 0
    max_retries = 5
    while True:
        url = (f"{base}?filter=prototype={prototype}"
               f"&limit={page_size}&offset={offset}&format=json")
        success = False
        for attempt in range(max_retries):
            try:
                resp = requests.get(url, timeout=timeout)
                resp.raise_for_status()
                success = True
                break
            except Exception as e:
                wait_sec = 10 * (attempt + 1)
                print(f"    OQMD page offset={offset} attempt {attempt+1}/{max_retries} "
                      f"failed: {e}  (retry in {wait_sec}s)")
                time.sleep(wait_sec)
        if not success:
            print(f"    OQMD page offset={offset} failed after {max_retries} attempts, "
                  f"returning {len(all_entries)} entries so far")
            return all_entries
        data = resp.json()
        entries = data.get("data", [])
        all_entries.extend(entries)
        total_avail = data["meta"]["data_available"]
        print(f"    OQMD fetched {len(all_entries)}/{total_avail} "
              f"(offset={offset})")
        if len(entries) == 0 or len(all_entries) >= total_avail:
            break
        offset += page_size
    return all_entries


def extract_oqmd_compounds(structure_type: str) -> pd.DataFrame:
    """Extract B2 or L1_2 compounds from OQMD (no E_hull filter)."""
    proto = "B2_CsCl" if structure_type == "B2" else "AuCu3"
    entries = _fetch_oqmd_paginated(proto)
    print(f"    OQMD raw entries ({proto}): {len(entries)}")

    data = []
    for entry in entries:
        try:
            uc = entry.get("unit_cell")
            if uc is None:
                continue
            # Unit cell is a 3x3 list; for cubic, a = length of first vector
            a = np.linalg.norm(uc[0])
            b = np.linalg.norm(uc[1])
            c = np.linalg.norm(uc[2])
            avg = (a + b + c) / 3
            if not (abs(a - avg) / avg < 0.02
                    and abs(b - avg) / avg < 0.02
                    and abs(c - avg) / avg < 0.02):
                continue

            if not (LATTICE_CONST_MIN <= a <= LATTICE_CONST_MAX):
                continue

            sites = entry.get("sites", [])
            el_counts = _parse_oqmd_sites(sites)
            if len(el_counts) != 2:
                continue

            elements = sorted(el_counts.keys())
            element_A, element_B = elements[0], elements[1]
            if element_A in NON_METAL_ELEMENTS or element_B in NON_METAL_ELEMENTS:
                continue
            count_A, count_B = el_counts[element_A], el_counts[element_B]

            stability = entry.get("stability")
            if stability is None:
                stability = float("nan")

            data.append({
                "material_id": f"oqmd-{entry.get('entry_id', 'unknown')}",
                "formula": entry.get("name", f"{element_A}{element_B}"),
                "structure_type": structure_type,
                "element_A": element_A,
                "element_B": element_B,
                "count_A": count_A,
                "count_B": count_B,
                "lattice_constant": a,
                "energy_per_atom": entry.get("delta_e", float("nan")),
                "energy_above_hull": stability,
                "source": "OQMD",
            })
        except Exception:
            continue

    df = pd.DataFrame(data)
    if len(df) > 0:
        df = df.sort_values("energy_per_atom").drop_duplicates(
            subset=["element_A", "element_B"], keep="first"
        ).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Radius calculation (same TRF approach as existing code)
# ---------------------------------------------------------------------------

def _determine_active_contacts_vec(radii_arr: np.ndarray,
                                    iA_arr: np.ndarray, iB_arr: np.ndarray,
                                    im_arr: np.ndarray, in_arr: np.ndarray,
                                    stype: str) -> np.ndarray:
    """Vectorised active-contact determination.  Returns array of 0/1/2
    encoding AA/BB/AB respectively."""
    _sqrt2 = np.sqrt(2)
    _2s3 = 2.0 / np.sqrt(3)
    if stype == "B2":
        a_AA = 2.0 * radii_arr[iA_arr]
        a_BB = 2.0 * radii_arr[iB_arr]
        a_AB = _2s3 * (radii_arr[iA_arr] + radii_arr[iB_arr])
        stacked = np.column_stack([a_AA, a_BB, a_AB])
    else:  # L1_2
        rm = radii_arr[im_arr]
        rn = radii_arr[in_arr]
        a_AA = 2.0 * _sqrt2 * rm
        a_AB = _sqrt2 * (rm + rn)
        a_BB = 2.0 * rn
        stacked = np.column_stack([a_AA, a_AB, a_BB])
    return stacked.argmax(axis=1)


# contact-type integer codes used throughout the vectorised path
_CT_AA, _CT_BB, _CT_AB = 0, 1, 2
# For L1_2 the column order is AA, AB, BB → codes 0, 1, 2 but
# _CT_AB=2 would collide with BB.  Keep separate mappings:
_B2_CODES  = np.array([0, 1, 2])   # AA, BB, AB columns
_L12_CODES = np.array([0, 1, 2])   # AA, AB, BB columns


def calculate_radii_trf(df: pd.DataFrame,
                        max_iter: int = 50,
                        initial_guess: Optional[Dict[str, float]] = None,
                        ) -> Tuple[Dict[str, float], Dict]:
    """Calculate effective atomic radii using iterative constrained
    optimisation (SLSQP) with fully vectorised inner functions.

    At each iteration the dominant contact condition (max) is determined
    for every compound from the current radii.  A constrained optimisation
    sub-problem is then solved: the objective minimises residuals for the
    active contacts while inequality constraints enforce that every
    non-dominant contact satisfies a_contact <= a_DFT.  The loop repeats
    until no compound switches its active contact (self-consistent radii).
    """
    if len(df) == 0:
        return {}, {"n_compounds": 0, "n_elements": 0,
                     "rmse": float("nan"), "mae": float("nan")}

    elements = sorted(set(df["element_A"]) | set(df["element_B"]))
    el2idx = {el: i for i, el in enumerate(elements)}
    n_el = len(elements)

    if initial_guess:
        x0 = np.array([initial_guess.get(el, 1.4) for el in elements])
    else:
        x0 = np.array([PAULING_RADII.get(el, 1.4) for el in elements])

    stype = df["structure_type"].iloc[0]

    _sqrt2 = np.sqrt(2)
    _2s3 = 2.0 / np.sqrt(3)

    # Pre-extract numpy arrays (done once)
    n_comp = len(df)
    a_vals = df["lattice_constant"].values.astype(np.float64)
    iA_arr = np.array([el2idx[e] for e in df["element_A"]], dtype=np.intp)
    iB_arr = np.array([el2idx[e] for e in df["element_B"]], dtype=np.intp)
    cA_arr = df["count_A"].values
    cB_arr = df["count_B"].values

    # For L1_2: pre-compute major / minor element indices
    is_A_major = cA_arr > cB_arr
    im_arr = np.where(is_A_major, iA_arr, iB_arr)  # major
    in_arr = np.where(is_A_major, iB_arr, iA_arr)  # minor

    # --- iterative loop ---
    current_radii = x0.copy()
    prev_ct = np.full(n_comp, -1, dtype=np.intp)
    n_iterations = 0
    _bounds = [(0.5, 3.5)] * n_el

    for iteration in range(max_iter):
        ct = _determine_active_contacts_vec(current_radii,
                                            iA_arr, iB_arr,
                                            im_arr, in_arr, stype)
        if np.array_equal(ct, prev_ct):
            break
        prev_ct = ct.copy()
        n_iterations = iteration + 1

        # --- build masks for current contact assignment ---
        if stype == "B2":
            mAA = ct == 0; mBB = ct == 1; mAB = ct == 2  # noqa: E702
        else:
            mAA = ct == 0; mAB = ct == 1; mBB = ct == 2  # noqa: E702

        # ---- vectorised objective (0.5 * SSR) ----
        if stype == "B2":
            def _obj(radii, _mAA=mAA, _mBB=mBB, _mAB=mAB):
                rA = radii[iA_arr]; rB = radii[iB_arr]
                d = np.empty(n_comp)
                d[_mAA] = 2.0 * rA[_mAA] - a_vals[_mAA]
                d[_mBB] = 2.0 * rB[_mBB] - a_vals[_mBB]
                d[_mAB] = _2s3 * (rA[_mAB] + rB[_mAB]) - a_vals[_mAB]
                return 0.5 * np.dot(d, d)

            def _grad(radii, _mAA=mAA, _mBB=mBB, _mAB=mAB):
                rA = radii[iA_arr]; rB = radii[iB_arr]
                d = np.empty(n_comp)
                d[_mAA] = 2.0 * rA[_mAA] - a_vals[_mAA]
                d[_mBB] = 2.0 * rB[_mBB] - a_vals[_mBB]
                d[_mAB] = _2s3 * (rA[_mAB] + rB[_mAB]) - a_vals[_mAB]
                g = np.zeros(n_el)
                np.add.at(g, iA_arr[_mAA], 2.0 * d[_mAA])
                np.add.at(g, iB_arr[_mBB], 2.0 * d[_mBB])
                np.add.at(g, iA_arr[_mAB], _2s3 * d[_mAB])
                np.add.at(g, iB_arr[_mAB], _2s3 * d[_mAB])
                return g
        else:  # L1_2
            def _obj(radii, _mAA=mAA, _mAB=mAB, _mBB=mBB):
                rm = radii[im_arr]; rn = radii[in_arr]
                d = np.empty(n_comp)
                d[_mAA] = 2.0 * _sqrt2 * rm[_mAA] - a_vals[_mAA]
                d[_mAB] = _sqrt2 * (rm[_mAB] + rn[_mAB]) - a_vals[_mAB]
                d[_mBB] = 2.0 * rn[_mBB] - a_vals[_mBB]
                return 0.5 * np.dot(d, d)

            def _grad(radii, _mAA=mAA, _mAB=mAB, _mBB=mBB):
                rm = radii[im_arr]; rn = radii[in_arr]
                d = np.empty(n_comp)
                d[_mAA] = 2.0 * _sqrt2 * rm[_mAA] - a_vals[_mAA]
                d[_mAB] = _sqrt2 * (rm[_mAB] + rn[_mAB]) - a_vals[_mAB]
                d[_mBB] = 2.0 * rn[_mBB] - a_vals[_mBB]
                g = np.zeros(n_el)
                np.add.at(g, im_arr[_mAA], 2.0 * _sqrt2 * d[_mAA])
                np.add.at(g, im_arr[_mAB], _sqrt2 * d[_mAB])
                np.add.at(g, in_arr[_mAB], _sqrt2 * d[_mAB])
                np.add.at(g, in_arr[_mBB], 2.0 * d[_mBB])
                return g

        # ---- vectorised inequality constraints ----
        # Build constraint index arrays & pre-compute constant Jacobian
        # (constraints are linear in radii → Jacobian is constant)
        c_idx_list: List[int] = []      # compound index
        c_type_list: List[int] = []     # 0=AA, 1=BB, 2=AB for B2 / 0=AA, 1=AB, 2=BB for L12
        if stype == "B2":
            for i in range(n_comp):
                c = ct[i]
                if c != 0:  # not AA
                    c_idx_list.append(i); c_type_list.append(0)
                if c != 1:  # not BB
                    c_idx_list.append(i); c_type_list.append(1)
                if c != 2:  # not AB
                    c_idx_list.append(i); c_type_list.append(2)
        else:
            for i in range(n_comp):
                c = ct[i]
                if c != 0:  # not AA
                    c_idx_list.append(i); c_type_list.append(0)
                if c != 1:  # not AB
                    c_idx_list.append(i); c_type_list.append(1)
                if c != 2:  # not BB
                    c_idx_list.append(i); c_type_list.append(2)

        n_ineq = len(c_idx_list)
        if n_ineq > 0:
            ci = np.array(c_idx_list, dtype=np.intp)
            ctp = np.array(c_type_list, dtype=np.intp)
            c_a = a_vals[ci]
            cmAA = ctp == 0; cmBB_or_AB1 = ctp == 1; cmAB_or_BB2 = ctp == 2

            if stype == "B2":
                c_iA = iA_arr[ci]; c_iB = iB_arr[ci]

                def _ineq(radii, _cmAA=cmAA, _cmBB=cmBB_or_AB1,
                          _cmAB=cmAB_or_BB2):
                    rA = radii[c_iA]; rB = radii[c_iB]
                    v = np.empty(n_ineq)
                    v[_cmAA] = c_a[_cmAA] - 2.0 * rA[_cmAA]
                    v[_cmBB] = c_a[_cmBB] - 2.0 * rB[_cmBB]
                    v[_cmAB] = c_a[_cmAB] - _2s3 * (rA[_cmAB] + rB[_cmAB])
                    return v

                # constant Jacobian
                jac = np.zeros((n_ineq, n_el))
                aa_rows = np.where(cmAA)[0]
                jac[aa_rows, c_iA[aa_rows]] += -2.0
                bb_rows = np.where(cmBB_or_AB1)[0]
                jac[bb_rows, c_iB[bb_rows]] += -2.0
                ab_rows = np.where(cmAB_or_BB2)[0]
                jac[ab_rows, c_iA[ab_rows]] += -_2s3
                jac[ab_rows, c_iB[ab_rows]] += -_2s3
            else:  # L1_2
                c_im = im_arr[ci]; c_in = in_arr[ci]

                def _ineq(radii, _cmAA=cmAA, _cmAB=cmBB_or_AB1,
                          _cmBB=cmAB_or_BB2):
                    rm = radii[c_im]; rn = radii[c_in]
                    v = np.empty(n_ineq)
                    v[_cmAA] = c_a[_cmAA] - 2.0 * _sqrt2 * rm[_cmAA]
                    v[_cmAB] = c_a[_cmAB] - _sqrt2 * (rm[_cmAB] + rn[_cmAB])
                    v[_cmBB] = c_a[_cmBB] - 2.0 * rn[_cmBB]
                    return v

                jac = np.zeros((n_ineq, n_el))
                aa_rows = np.where(cmAA)[0]
                jac[aa_rows, c_im[aa_rows]] += -2.0 * _sqrt2
                ab_rows = np.where(cmBB_or_AB1)[0]
                jac[ab_rows, c_im[ab_rows]] += -_sqrt2
                jac[ab_rows, c_in[ab_rows]] += -_sqrt2
                bb_rows = np.where(cmAB_or_BB2)[0]
                jac[bb_rows, c_in[bb_rows]] += -2.0

            _const_jac = jac

            def _ineq_jac(radii):
                return _const_jac
        else:
            def _ineq(radii):
                return np.array([])

            def _ineq_jac(radii):
                return np.empty((0, n_el))

        result = minimize(
            _obj, current_radii, jac=_grad,
            method="SLSQP", bounds=_bounds,
            constraints={"type": "ineq", "fun": _ineq, "jac": _ineq_jac},
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        current_radii = result.x

    radii = {el: current_radii[el2idx[el]] for el in elements}

    # Final stats using max (the true model) — vectorised
    rA_f = np.array([radii[e] for e in df["element_A"]])
    rB_f = np.array([radii[e] for e in df["element_B"]])
    if stype == "B2":
        a_calc = np.maximum(np.maximum(2 * rA_f, 2 * rB_f),
                            _2s3 * (rA_f + rB_f))
    else:
        rm_f = np.where(is_A_major, rA_f, rB_f)
        rn_f = np.where(is_A_major, rB_f, rA_f)
        a_calc = np.maximum(np.maximum(2 * _sqrt2 * rm_f,
                                       _sqrt2 * (rm_f + rn_f)),
                            2 * rn_f)
    final_res = a_calc - a_vals

    # Count constraint violations — vectorised
    if stype == "B2":
        all_contacts = np.column_stack([2 * rA_f, 2 * rB_f,
                                        _2s3 * (rA_f + rB_f)])
    else:
        all_contacts = np.column_stack([2 * _sqrt2 * rm_f,
                                        _sqrt2 * (rm_f + rn_f),
                                        2 * rn_f])
    n_violated = int(np.sum(all_contacts > a_vals[:, None] + 1e-6))

    return radii, {
        "n_compounds": len(df),
        "n_elements": len(elements),
        "rmse": float(np.sqrt(np.mean(final_res ** 2))),
        "mae": float(np.mean(np.abs(final_res))),
        "contact_iterations": n_iterations,
        "n_constraint_violations": n_violated,
    }


# ---------------------------------------------------------------------------
# Module-level state for multiprocessing bootstrap workers
# ---------------------------------------------------------------------------
_BOOT_DF: Optional[pd.DataFrame] = None
_BOOT_GUESS: Optional[Dict[str, float]] = None


def _init_boot_pool(df: pd.DataFrame, guess: Dict[str, float]) -> None:
    """Initialiser called once per worker process."""
    global _BOOT_DF, _BOOT_GUESS
    _BOOT_DF = df
    _BOOT_GUESS = guess


def _boot_worker(seed: int) -> Optional[Dict[str, float]]:
    """Single bootstrap iteration (must be module-level for pickling)."""
    with threadpool_limits(limits=1):
        rng = np.random.RandomState(seed)
        n = len(_BOOT_DF)
        idx = rng.choice(n, size=n, replace=True)
        sample = _BOOT_DF.iloc[idx].reset_index(drop=True)
        try:
            r, _ = calculate_radii_trf(sample, max_iter=10,
                                       initial_guess=_BOOT_GUESS)
            return r
        except Exception:
            return None


def bootstrap_uncertainty(df: pd.DataFrame,
                          n_bootstrap: int = 1000,
                          random_state: int = 42,
                          initial_guess: Optional[Dict[str, float]] = None,
                          n_jobs: Optional[int] = None,
                          ) -> Dict:
    """Estimate uncertainty in atomic radii via bootstrap resampling.

    Resamples compounds with replacement, re-optimises for each sample,
    and returns per-element mean, std, and 95 % confidence intervals.
    Uses multiprocessing to parallelise across CPU cores.
    """
    if len(df) == 0:
        return {}

    if initial_guess is None:
        initial_guess, _ = calculate_radii_trf(df)

    elements = sorted(initial_guess.keys())

    if n_jobs is None:
        n_jobs = min(os.cpu_count() or 1, 8)

    rng = np.random.RandomState(random_state)
    seeds = rng.randint(0, 2**31, size=n_bootstrap).tolist()

    boot_radii: Dict[str, List[float]] = {el: [] for el in elements}
    n_done = 0

    with multiprocessing.Pool(
        n_jobs, initializer=_init_boot_pool,
        initargs=(df, initial_guess),
    ) as pool:
        for r in pool.imap_unordered(_boot_worker, seeds, chunksize=4):
            n_done += 1
            if n_done % 100 == 0:
                print(f"  bootstrap {n_done}/{n_bootstrap}")
            if r is not None:
                for el in elements:
                    if el in r:
                        boot_radii[el].append(r[el])

    out: Dict = {"mean": {}, "std": {},
                 "ci_lower": {}, "ci_upper": {},
                 "n_success": 0}
    for el in elements:
        vals = np.array(boot_radii[el])
        if len(vals) > 0:
            out["mean"][el] = float(np.mean(vals))
            out["std"][el] = float(np.std(vals))
            out["ci_lower"][el] = float(np.percentile(vals, 2.5))
            out["ci_upper"][el] = float(np.percentile(vals, 97.5))
            out["n_success"] = max(out["n_success"], len(vals))
    return out


def _calc_lattice_constant(stype: str, rA: float, rB: float,
                           count_A: int, count_B: int) -> float:
    """Compute lattice constant from radii using max contact model."""
    if stype == "B2":
        return max(2 * rA, 2 * rB, (2 / np.sqrt(3)) * (rA + rB))
    else:  # L1_2
        if count_A > count_B:
            r_major, r_minor = rA, rB
        else:
            r_major, r_minor = rB, rA
        return max(2 * np.sqrt(2) * r_major,
                   np.sqrt(2) * (r_major + r_minor),
                   2 * r_minor)


def compare_lattice_constants(df: pd.DataFrame,
                              radii: Dict[str, float]) -> pd.DataFrame:
    """Compare DFT/OQMD lattice constants vs calculated from radii."""
    rows = []
    stype = df["structure_type"].iloc[0] if len(df) > 0 else "B2"
    for _, row in df.iterrows():
        eA, eB = row["element_A"], row["element_B"]
        if eA not in radii or eB not in radii:
            continue
        rA, rB = radii[eA], radii[eB]
        a_ref = row["lattice_constant"]
        a_calc = _calc_lattice_constant(stype, rA, rB,
                                        row["count_A"], row["count_B"])
        error = a_calc - a_ref
        rel_error = abs(error) / a_ref * 100
        rows.append({
            "formula": row["formula"],
            "a_ref": a_ref,
            "a_calc": a_calc,
            "error": error,
            "rel_error_pct": rel_error,
            "energy_above_hull": row["energy_above_hull"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

CASE_LABELS = {
    "MP_B2":   "MP — B2",
    "MP_L12":  "MP — L1$_2$",
    "OQMD_B2": "OQMD — B2",
    "OQMD_L12": "OQMD — L1$_2$",
}

CASE_COLORS = {
    "MP_B2":    "#1f77b4",
    "MP_L12":   "#ff7f0e",
    "OQMD_B2":  "#2ca02c",
    "OQMD_L12": "#d62728",
}


def run_analysis(api_key: str, fig_dir: str):
    os.makedirs(fig_dir, exist_ok=True)

    # ==================================================================
    # Step 1: Extract data for all 4 cases
    # ==================================================================
    print("=" * 70)
    print("Step 1: Extracting data")
    print("=" * 70)

    datasets: Dict[str, pd.DataFrame] = {}

    print("\n  [MP — B2 + L1$_2$ (single query)]")
    datasets["MP_B2"], datasets["MP_L12"] = extract_mp_all(api_key)

    print("\n  [OQMD — B2]")
    datasets["OQMD_B2"] = extract_oqmd_compounds("B2")
    print(f"    => {len(datasets['OQMD_B2'])} compounds")

    print("\n  [OQMD — L1$_2$]")
    datasets["OQMD_L12"] = extract_oqmd_compounds("L1$_2$")
    print(f"    => {len(datasets['OQMD_L12'])} compounds")

    # ==================================================================
    # Step 2: Calculate radii for each case
    # ==================================================================
    print("\n" + "=" * 70)
    print("Step 2: Calculating effective atomic radii (TRF)")
    print("=" * 70)

    results: Dict[str, dict] = {}
    for key, df in datasets.items():
        radii, stats = calculate_radii_trf(df)
        comp = compare_lattice_constants(df, radii)
        lat_rmse = np.sqrt(np.mean(comp["error"] ** 2)) if len(comp) > 0 else float("nan")
        lat_mae = np.mean(np.abs(comp["error"])) if len(comp) > 0 else float("nan")
        mean_rel = comp["rel_error_pct"].mean() if len(comp) > 0 else float("nan")
        med_rel = comp["rel_error_pct"].median() if len(comp) > 0 else float("nan")
        max_rel = comp["rel_error_pct"].max() if len(comp) > 0 else float("nan")

        # Add calculated lattice constant to the dataset DataFrame
        stype = df["structure_type"].iloc[0] if len(df) > 0 else "B2"
        a_calc_list = []
        for _, row in df.iterrows():
            eA, eB = row["element_A"], row["element_B"]
            if eA in radii and eB in radii:
                a_calc_list.append(
                    _calc_lattice_constant(stype, radii[eA], radii[eB],
                                           row["count_A"], row["count_B"])
                )
            else:
                a_calc_list.append(float("nan"))
        df["lattice_constant_calc"] = a_calc_list

        results[key] = {
            "df": df,
            "radii": radii,
            "stats": stats,
            "comparison": comp,
            "lat_rmse": lat_rmse,
            "lat_mae": lat_mae,
            "mean_rel": mean_rel,
            "med_rel": med_rel,
            "max_rel": max_rel,
        }
        print(f"\n  [{CASE_LABELS[key]}]")
        print(f"    Compounds: {stats['n_compounds']}, Elements: {stats['n_elements']}")
        n_viol = stats.get("n_constraint_violations", 0)
        print(f"    Radii  RMSE={stats['rmse']:.4f} A, MAE={stats['mae']:.4f} A")
        print(f"    Lattice RMSE={lat_rmse:.4f} A, MAE={lat_mae:.4f} A")
        print(f"    Rel Error: mean={mean_rel:.2f}%, median={med_rel:.2f}%, max={max_rel:.2f}%")
        print(f"    Constraint violations: {n_viol}")

        # Save radii
        r_df = pd.DataFrame([{"element": el, "radius": r} for el, r in sorted(radii.items())])
        r_df.to_csv(os.path.join(fig_dir, f"radii_{key}.csv"), index=False)

    # ==================================================================
    # Step 2b: Bootstrap uncertainty estimation
    # ==================================================================
    print("\n" + "=" * 70)
    print("Step 2b: Bootstrap uncertainty estimation (1000 resamples)")
    print("=" * 70)

    boot_results: Dict[str, Dict] = {}
    for key, df in datasets.items():
        print(f"\n  [{CASE_LABELS[key]}]")
        radii = results[key]["radii"]
        boot = bootstrap_uncertainty(df, n_bootstrap=1000,
                                     initial_guess=radii)
        boot_results[key] = boot

        # Save radii with uncertainty
        rows_out = []
        for el in sorted(radii.keys()):
            rows_out.append({
                "element": el,
                "radius": radii[el],
                "std": boot["std"].get(el, float("nan")),
                "ci_lower": boot["ci_lower"].get(el, float("nan")),
                "ci_upper": boot["ci_upper"].get(el, float("nan")),
            })
        pd.DataFrame(rows_out).to_csv(
            os.path.join(fig_dir, f"radii_{key}_bootstrap.csv"), index=False)
        print(f"    bootstrap success: {boot.get('n_success', 0)}/1000")

    # Save CSVs (after radii calculation so lattice_constant_calc is included)
    for key, df in datasets.items():
        df.to_csv(os.path.join(fig_dir, f"compounds_{key}.csv"), index=False)

    # ==================================================================
    # Step 3: Figures
    # ==================================================================
    print("\n" + "=" * 70)
    print("Step 3: Generating comparison figures")
    print("=" * 70)

    # --- Figure 1: Dataset summary bar chart ---
    fig, ax = plt.subplots(figsize=(10, 6))
    keys = list(CASE_LABELS.keys())
    counts = [len(datasets[k]) for k in keys]
    bars = ax.bar([CASE_LABELS[k] for k in keys], counts,
                  color=[CASE_COLORS[k] for k in keys], edgecolor="black", alpha=0.85)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(cnt), ha="center", fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of Compounds")
    ax.set_title("Dataset Size by Source and Structure Type")
    plt.tight_layout()
    fig1_path = os.path.join(fig_dir, "fig1_dataset_summary.png")
    plt.savefig(fig1_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig1_path}")

    # --- Figure 2: 2x2 parity plots ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    for ax, key in zip(axes.flat, keys):
        comp = results[key]["comparison"]
        if len(comp) == 0:
            ax.set_title(f"{CASE_LABELS[key]}\n(no data)")
            continue
        ehull = comp["energy_above_hull"].values
        sc = ax.scatter(comp["a_ref"], comp["a_calc"], c=ehull,
                        cmap="RdYlGn_r", edgecolors="black", linewidth=0.5,
                        alpha=0.8, s=50)
        lims = [min(comp["a_ref"].min(), comp["a_calc"].min()) - 0.2,
                max(comp["a_ref"].max(), comp["a_calc"].max()) + 0.2]
        ax.plot(lims, lims, "k--", linewidth=1.5, label="y = x")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Reference Lattice Constant (A)")
        ax.set_ylabel("Calculated Lattice Constant (A)")
        rmse = results[key]["lat_rmse"]
        ax.set_title(f"{CASE_LABELS[key]}\nN={len(comp)}, RMSE={rmse:.4f} A")
        ax.legend(loc="upper left")
        plt.colorbar(sc, ax=ax, label="$E_{\\mathrm{hull}}$ / stability (eV/atom)")
    plt.tight_layout()
    fig2_path = os.path.join(fig_dir, "fig2_parity_2x2.png")
    plt.savefig(fig2_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig2_path}")

    # --- Figure 3: Error metrics bar chart (RMSE, MAE, mean rel error) ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    labels = [CASE_LABELS[k] for k in keys]
    colors = [CASE_COLORS[k] for k in keys]

    # RMSE
    vals = [results[k]["lat_rmse"] for k in keys]
    axes[0].bar(labels, vals, color=colors, edgecolor="black", alpha=0.85)
    axes[0].set_ylabel("Lattice Constant RMSE (A)")
    axes[0].set_title("(a) RMSE")
    for i, v in enumerate(vals):
        axes[0].text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=12)

    # MAE
    vals = [results[k]["lat_mae"] for k in keys]
    axes[1].bar(labels, vals, color=colors, edgecolor="black", alpha=0.85)
    axes[1].set_ylabel("Lattice Constant MAE (A)")
    axes[1].set_title("(b) MAE")
    for i, v in enumerate(vals):
        axes[1].text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=12)

    # Mean relative error
    vals = [results[k]["mean_rel"] for k in keys]
    axes[2].bar(labels, vals, color=colors, edgecolor="black", alpha=0.85)
    axes[2].set_ylabel("Mean Relative Error (%)")
    axes[2].set_title("(c) Mean Relative Error")
    for i, v in enumerate(vals):
        axes[2].text(i, v + 0.05, f"{v:.2f}%", ha="center", fontsize=12)

    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    fig3_path = os.path.join(fig_dir, "fig3_error_metrics.png")
    plt.savefig(fig3_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig3_path}")

    # --- Figure 4: Radius comparison MP vs OQMD per structure type ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, (mp_key, oq_key, stype_label) in zip(
        axes, [("MP_B2", "OQMD_B2", "B2"), ("MP_L12", "OQMD_L12", "L1$_2$")]
    ):
        r_mp = results[mp_key]["radii"]
        r_oq = results[oq_key]["radii"]
        common = sorted(set(r_mp) & set(r_oq))
        if len(common) == 0:
            ax.set_title(f"{stype_label}: no common elements")
            continue
        mp_vals = np.array([r_mp[el] for el in common])
        oq_vals = np.array([r_oq[el] for el in common])
        diff = oq_vals - mp_vals

        ax.scatter(mp_vals, oq_vals, c=CASE_COLORS[mp_key],
                   edgecolors="black", s=60, alpha=0.8)
        for i, el in enumerate(common):
            if abs(diff[i]) > 0.03:
                ax.annotate(el, (mp_vals[i], oq_vals[i]), fontsize=10,
                            xytext=(5, 5), textcoords="offset points")
        lims = [min(mp_vals.min(), oq_vals.min()) - 0.1,
                max(mp_vals.max(), oq_vals.max()) + 0.1]
        ax.plot(lims, lims, "k--", linewidth=1.5)
        ax.set_xlabel("MP Radius (A)")
        ax.set_ylabel("OQMD Radius (A)")
        rmsd = np.sqrt(np.mean(diff ** 2))
        ax.set_title(f"{stype_label}: MP vs OQMD (N={len(common)}, RMSD={rmsd:.4f} A)")

    plt.tight_layout()
    fig4_path = os.path.join(fig_dir, "fig4_radius_mp_vs_oqmd.png")
    plt.savefig(fig4_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig4_path}")

    # --- Figure 5: E_hull / stability distribution for each case ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for ax, key in zip(axes.flat, keys):
        df = datasets[key]
        if len(df) == 0 or "energy_above_hull" not in df.columns:
            ax.set_title(f"{CASE_LABELS[key]} (no data)")
            continue
        ehull = df["energy_above_hull"].dropna().values
        if len(ehull) == 0:
            ax.set_title(f"{CASE_LABELS[key]} (no data)")
            continue
        ax.hist(ehull, bins=40, color=CASE_COLORS[key], edgecolor="black", alpha=0.8)
        ax.axvline(0.1, color="red", linestyle="--", linewidth=2,
                   label="0.1 eV/atom")
        ax.set_xlabel("$E_{\\mathrm{hull}}$ / stability (eV/atom)")
        ax.set_ylabel("Count")
        ax.set_title(f"{CASE_LABELS[key]} (N={len(ehull)})")
        ax.legend()
    plt.tight_layout()
    fig5_path = os.path.join(fig_dir, "fig5_ehull_distributions.png")
    plt.savefig(fig5_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig5_path}")

    # --- Figure 6: Error vs E_hull scatter for each case ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for ax, key in zip(axes.flat, keys):
        comp = results[key]["comparison"]
        if len(comp) == 0:
            ax.set_title(f"{CASE_LABELS[key]} (no data)")
            continue
        ax.scatter(comp["energy_above_hull"], comp["rel_error_pct"],
                   c=CASE_COLORS[key], edgecolors="black", alpha=0.6, s=40)
        ax.axvline(0.1, color="red", linestyle="--", linewidth=2)
        ax.set_xlabel("$E_{\\mathrm{hull}}$ / stability (eV/atom)")
        ax.set_ylabel("Relative Error (%)")
        ax.set_title(f"{CASE_LABELS[key]}")
    plt.tight_layout()
    fig6_path = os.path.join(fig_dir, "fig6_error_vs_ehull.png")
    plt.savefig(fig6_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig6_path}")

    # ==================================================================
    # Step 4: Generate Markdown report
    # ==================================================================
    print("\n" + "=" * 70)
    print("Step 4: Generating Markdown report")
    print("=" * 70)

    lines = []
    lines.append("# 4ケース比較レポート: MP/OQMD $\\times$ B2/L1$_2$\n")

    lines.append("## 1. 概要\n")
    lines.append("Materials Project (MP) および OQMD から B2 (CsCl型) と L1$_2$ (AuCu$_3$型) の")
    lines.append("二元系化合物を抽出し、$E_{\\mathrm{hull}}$ フィルタなしで有効原子半径を")
    lines.append("TRF 最適化により算出した。4ケースの比較を行う。\n")

    # Dataset summary
    lines.append("## 2. データセット比較\n")
    lines.append("| ケース | データソース | 構造タイプ | 化合物数 | 元素数 |")
    lines.append("|:---|:---:|:---:|:---:|:---:|")
    for key in keys:
        s = results[key]["stats"]
        lines.append(f"| {CASE_LABELS[key]} | {'MP' if 'MP' in key else 'OQMD'} "
                     f"| {'B2' if 'B2' in key else 'L1$_2$'} "
                     f"| {s['n_compounds']} | {s['n_elements']} |")
    lines.append("")

    lines.append("![データセット概要](fig1_dataset_summary.png)\n")
    lines.append("**図1**: ケース別の化合物数。\n")

    # Stability distribution
    lines.append("## 3. 安定性分布\n")
    lines.append("![安定性分布](fig5_ehull_distributions.png)\n")
    lines.append("**図5**: 各ケースの $E_{\\mathrm{hull}}$ / stability 分布。赤破線は 0.1 eV/atom。\n")

    # Radii results
    lines.append("## 4. 有効原子半径と格子定数の再現性\n")
    lines.append("| ケース | 半径 RMSE (A) | 半径 MAE (A) | 格子定数 RMSE (A) "
                 "| 格子定数 MAE (A) | 平均相対誤差 (%) | 中央値相対誤差 (%) |")
    lines.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
    for key in keys:
        r = results[key]
        s = r["stats"]
        lines.append(f"| {CASE_LABELS[key]} | {s['rmse']:.4f} | {s['mae']:.4f} "
                     f"| {r['lat_rmse']:.4f} | {r['lat_mae']:.4f} "
                     f"| {r['mean_rel']:.2f} | {r['med_rel']:.2f} |")
    lines.append("")

    lines.append("![パリティプロット](fig2_parity_2x2.png)\n")
    lines.append("**図2**: 各ケースの格子定数パリティプロット。色は $E_{\\mathrm{hull}}$ / stability。\n")

    lines.append("![誤差指標](fig3_error_metrics.png)\n")
    lines.append("**図3**: (a) 格子定数 RMSE, (b) MAE, (c) 平均相対誤差 の比較。\n")

    # MP vs OQMD radius comparison
    lines.append("## 5. MP vs OQMD 有効原子半径の比較\n")
    for mp_key, oq_key, stype_label in [("MP_B2", "OQMD_B2", "B2"),
                                          ("MP_L12", "OQMD_L12", "L1$_2$")]:
        r_mp = results[mp_key]["radii"]
        r_oq = results[oq_key]["radii"]
        common = sorted(set(r_mp) & set(r_oq))
        if len(common) == 0:
            lines.append(f"\n### {stype_label}: 共通元素なし\n")
            continue
        mp_vals = np.array([r_mp[el] for el in common])
        oq_vals = np.array([r_oq[el] for el in common])
        diff = oq_vals - mp_vals
        rmsd = np.sqrt(np.mean(diff ** 2))
        lines.append(f"\n### {stype_label}\n")
        lines.append(f"- 共通元素数: {len(common)}")
        lines.append(f"- 半径 RMSD (MP vs OQMD): {rmsd:.4f} A")
        lines.append(f"- 最大差: {np.max(np.abs(diff)):.4f} A "
                     f"({common[np.argmax(np.abs(diff))]})")

        # Top 10 differences
        sorted_idx = np.argsort(np.abs(diff))[::-1]
        top_n = min(10, len(common))
        lines.append(f"\n| 元素 | $r_{{\\mathrm{{MP}}}}$ (A) | $r_{{\\mathrm{{OQMD}}}}$ (A) "
                     f"| $\\Delta r$ (A) |")
        lines.append("|:---:|:---:|:---:|:---:|")
        for i in sorted_idx[:top_n]:
            sign = "+" if diff[i] >= 0 else ""
            lines.append(f"| {common[i]} | {mp_vals[i]:.4f} | {oq_vals[i]:.4f} "
                         f"| {sign}{diff[i]:.4f} |")
        lines.append("")

    lines.append("![半径比較](fig4_radius_mp_vs_oqmd.png)\n")
    lines.append("**図4**: MP vs OQMD の有効原子半径比較。B2 (左) と L1$_2$ (右)。\n")

    # Error vs stability
    lines.append("## 6. 予測誤差と安定性の関係\n")
    lines.append("![誤差 vs 安定性](fig6_error_vs_ehull.png)\n")
    lines.append("**図6**: 各ケースの格子定数相対誤差 vs $E_{\\mathrm{hull}}$ / stability。\n")

    # Conclusions
    lines.append("## 7. 結論\n")
    lines.append("1. **データ量**: OQMD は B2 構造で MP より多くのデータを提供するが、"
                 "L1$_2$ は MP の方が豊富。")
    lines.append("2. **精度比較**: 4ケース間の格子定数 RMSE と相対誤差を比較し、"
                 "データソースと構造タイプによる精度差を定量化した。")
    lines.append("3. **半径整合性**: MP と OQMD で得られた有効原子半径の一致度を評価した。")
    lines.append("4. **安定性の影響**: 不安定化合物を含めた場合の精度への影響を各ケースで確認した。\n")

    report_path = os.path.join(fig_dir, "four_case_comparison_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Report saved to: {report_path}")

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        print("Error: MP_API_KEY environment variable not set")
        sys.exit(1)

    fig_dir = "four_case_output/figures"
    run_analysis(api_key, fig_dir)
