#!/usr/bin/env python3
"""
Vegard則 vs 構造依存有効半径の比較分析
=========================================

目的:
  1. 有効原子半径 (Vegard式δr) が構造情報を吸収できないことを示す
  2. 原子体積から求められた半径が構造情報を吸収できることを示す
  3. A₃B, B₃A の L1₂ 構造と AB の B2 構造を網羅的に整理

理論:
  - Vegard則: V = Σ cᵢVᵢ → 元素固有の体積のみ、構造無依存
  - DFT体積: V_actual(A₃B) ≠ V_actual(B₃A) → 構造依存性あり
  - 有効半径: r = (3V/4π)^{1/3}
  - δr: 100 × √[Σ cᵢ(1 - rᵢ/r̄)²] → 構造無依存

Author: Satoshi Minamoto (NIMS) / Devin
"""

import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

warnings.filterwarnings("ignore")

# ── Font setup ──────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "figure.titlesize": 20,
    "font.family": "sans-serif",
    "font.sans-serif": ["IPAPGothic", "IPAGothic", "WenQuanYi Zen Hei", "DejaVu Sans"],
})

# ── Pure element atomic volumes (King 1966, Å³) ────────────────────
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

# ── Pure element metallic radii (Goldschmidt CN12, Å) ──────────────
METALLIC_RADII = {
    "Li":1.52,"Be":1.12,"Na":1.86,"Mg":1.60,"Al":1.43,"Si":1.32,
    "K":2.27,"Ca":1.97,"Sc":1.61,"Ti":1.45,"V":1.32,"Cr":1.25,
    "Mn":1.37,"Fe":1.26,"Co":1.25,"Ni":1.25,"Cu":1.28,"Zn":1.37,
    "Ga":1.35,"Ge":1.39,"As":1.48,"Se":1.60,"Rb":2.48,"Sr":2.15,
    "Y":1.81,"Zr":1.60,"Nb":1.43,"Mo":1.36,"Ru":1.34,"Rh":1.34,
    "Pd":1.37,"Ag":1.44,"Cd":1.49,"In":1.63,"Sn":1.41,"Sb":1.61,
    "Te":1.43,"Cs":2.65,"Ba":2.17,"La":1.87,"Ce":1.82,"Pr":1.83,
    "Nd":1.82,"Sm":1.81,"Eu":2.04,"Gd":1.79,"Tb":1.77,"Dy":1.77,
    "Ho":1.76,"Er":1.75,"Tm":1.74,"Yb":1.94,"Lu":1.73,"Hf":1.58,
    "Ta":1.43,"W":1.37,"Re":1.37,"Os":1.34,"Ir":1.36,"Pt":1.38,
    "Au":1.44,"Hg":1.50,"Tl":1.71,"Pb":1.75,"Bi":1.55,"Th":1.80,
    "U":1.38,"Pu":1.64,"Pa":1.63,"Np":1.56,
}

# ── HEA-relevant elements ──────────────────────────────────────────
HEA_ELEMENTS = [
    "Al","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Zr","Nb","Mo","Hf","Ta","W","Re","Pd","Pt","Au","Ag",
    "Sc","Y","Mg","Si","Ge","Sn","Ru","Rh","Os","Ir",
]

OUT = Path("vegard_comparison_output")
OUT.mkdir(exist_ok=True)


# =====================================================================
# Data Loading
# =====================================================================
def load_all_data():
    """Load all DFT compound data from VASP, MP, OQMD."""
    data_dir = Path("data")
    dfs = []

    for src in ["VASP", "MP", "OQMD"]:
        for struct in ["B2", "L12"]:
            f = data_dir / f"compounds_{src}_{struct}.csv"
            if f.exists():
                df = pd.read_csv(f)
                df["db"] = src
                df["stype"] = struct
                dfs.append(df)

    # Also load user-uploaded OQMD data
    for name, stype in [("oqmd_b2_data.csv", "B2"), ("oqmd_l12_data.csv", "L12")]:
        f = data_dir / name
        if f.exists():
            df = pd.read_csv(f)
            df["db"] = "OQMD_new"
            df["stype"] = stype
            if "lattice_constant_a" in df.columns and "lattice_constant" not in df.columns:
                df["lattice_constant"] = df["lattice_constant_a"]
            dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    # Filter valid entries
    combined = combined[combined["lattice_constant"] > 2.0]
    combined = combined[combined["lattice_constant"] < 8.0]
    return combined


# =====================================================================
# 1. Compute Vegard-predicted atomic volume per atom
# =====================================================================
def vegard_volume(elA, elB, cA, cB):
    """Vegard則による平均原子体積 (Å³)."""
    vA = KING_ATOMIC_VOLUMES.get(elA, np.nan)
    vB = KING_ATOMIC_VOLUMES.get(elB, np.nan)
    return cA * vA + cB * vB


def vegard_radius(elA, elB, cA, cB):
    """Vegard則による有効半径 (Å)."""
    v = vegard_volume(elA, elB, cA, cB)
    return (3 * v / (4 * np.pi)) ** (1/3)


def vegard_lattice_constant(elA, elB, cA, cB, Z):
    """Vegard則から予測される格子定数."""
    v = vegard_volume(elA, elB, cA, cB)
    return (v * Z) ** (1/3)


# =====================================================================
# 2. DFT atomic volume-derived radius
# =====================================================================
def dft_volume_per_atom(a, Z):
    """DFT格子定数から原子あたり体積を計算."""
    return a**3 / Z


def volume_to_radius(v):
    """原子体積から等価球半径 r = (3V/4π)^{1/3}."""
    return (3 * v / (4 * np.pi)) ** (1/3)


# =====================================================================
# 3. Build comprehensive comparison table
# =====================================================================
def build_comparison_table(compound_df):
    """
    A₃B, B₃A (L1₂) と AB (B2) の網羅的比較テーブルを構築.
    
    各ペア(X,Y)に対して:
      - L1₂ X₃Y: a_DFT, V_DFT, r_DFT(=volume-derived), V_Vegard, r_Vegard
      - L1₂ Y₃X: a_DFT, V_DFT, r_DFT, V_Vegard, r_Vegard
      - B2  XY:   a_DFT, V_DFT, r_DFT, V_Vegard, r_Vegard
    """
    # Separate by structure
    l12_df = compound_df[compound_df["stype"] == "L12"].copy()
    b2_df = compound_df[compound_df["stype"] == "B2"].copy()

    # For L1₂: build dict (elA, elB) → best lattice constant (prefer VASP)
    l12_data = {}
    for _, row in l12_df.iterrows():
        elA = row["element_A"]
        elB = row["element_B"]
        a = row["lattice_constant"]
        db = row.get("db", "")
        cA = row.get("count_A", 3)
        cB = row.get("count_B", 1)
        if pd.isna(elA) or pd.isna(elB) or pd.isna(a):
            continue
        key = (elA, elB, int(cA), int(cB))
        if key not in l12_data or db == "VASP":
            l12_data[key] = {"a": a, "db": db, "cA": cA, "cB": cB}

    # For B2: build dict sorted_pair → best lattice constant
    b2_data = {}
    for _, row in b2_df.iterrows():
        elA = row["element_A"]
        elB = row["element_B"]
        a = row["lattice_constant"]
        db = row.get("db", "")
        if pd.isna(elA) or pd.isna(elB) or pd.isna(a):
            continue
        pair = tuple(sorted([elA, elB]))
        if pair not in b2_data or db == "VASP":
            b2_data[pair] = {"a": a, "db": db}

    # Build comparison rows
    rows = []
    all_elements = set()
    for key in l12_data:
        all_elements.add(key[0])
        all_elements.add(key[1])

    # Find pairs with both A₃B and B₃A in L1₂
    for elX in sorted(all_elements):
        for elY in sorted(all_elements):
            if elX >= elY:
                continue
            if elX not in KING_ATOMIC_VOLUMES or elY not in KING_ATOMIC_VOLUMES:
                continue

            # L1₂ X₃Y
            key_X3Y = (elX, elY, 3, 1)
            # L1₂ Y₃X
            key_Y3X = (elY, elX, 3, 1)
            # B2 XY
            pair_b2 = tuple(sorted([elX, elY]))

            has_X3Y = key_X3Y in l12_data
            has_Y3X = key_Y3X in l12_data
            has_B2 = pair_b2 in b2_data

            if not (has_X3Y or has_Y3X):
                continue

            row = {"elX": elX, "elY": elY}

            # Pure element volumes
            vX = KING_ATOMIC_VOLUMES[elX]
            vY = KING_ATOMIC_VOLUMES[elY]
            rX_pure = volume_to_radius(vX)
            rY_pure = volume_to_radius(vY)
            row["V_X_pure"] = vX
            row["V_Y_pure"] = vY
            row["r_X_pure"] = rX_pure
            row["r_Y_pure"] = rY_pure

            # L1₂ X₃Y (X=majority 75%, Y=minority 25%)
            if has_X3Y:
                d = l12_data[key_X3Y]
                a = d["a"]
                v_actual = dft_volume_per_atom(a, 4)
                v_vegard = vegard_volume(elX, elY, 0.75, 0.25)
                r_vegard = volume_to_radius(v_vegard)
                r_actual = volume_to_radius(v_actual)
                omega_sf = (v_actual - v_vegard) / v_vegard
                row["a_X3Y"] = a
                row["V_X3Y_DFT"] = v_actual
                row["V_X3Y_Vegard"] = v_vegard
                row["r_X3Y_DFT"] = r_actual
                row["r_X3Y_Vegard"] = r_vegard
                row["Omega_X3Y"] = omega_sf
                row["db_X3Y"] = d["db"]

            # L1₂ Y₃X (Y=majority 75%, X=minority 25%)
            if has_Y3X:
                d = l12_data[key_Y3X]
                a = d["a"]
                v_actual = dft_volume_per_atom(a, 4)
                v_vegard = vegard_volume(elY, elX, 0.75, 0.25)
                r_vegard = volume_to_radius(v_vegard)
                r_actual = volume_to_radius(v_actual)
                omega_sf = (v_actual - v_vegard) / v_vegard
                row["a_Y3X"] = a
                row["V_Y3X_DFT"] = v_actual
                row["V_Y3X_Vegard"] = v_vegard
                row["r_Y3X_DFT"] = r_actual
                row["r_Y3X_Vegard"] = r_vegard
                row["Omega_Y3X"] = omega_sf
                row["db_Y3X"] = d["db"]

            # B2 XY (50:50)
            if has_B2:
                d = b2_data[pair_b2]
                a = d["a"]
                v_actual = dft_volume_per_atom(a, 2)
                v_vegard = vegard_volume(elX, elY, 0.5, 0.5)
                r_vegard = volume_to_radius(v_vegard)
                r_actual = volume_to_radius(v_actual)
                omega_sf = (v_actual - v_vegard) / v_vegard
                row["a_B2"] = a
                row["V_B2_DFT"] = v_actual
                row["V_B2_Vegard"] = v_vegard
                row["r_B2_DFT"] = r_actual
                row["r_B2_Vegard"] = r_vegard
                row["Omega_B2"] = omega_sf
                row["db_B2"] = d["db"]

            rows.append(row)

    df = pd.DataFrame(rows)
    return df


# =====================================================================
# 4. Vegard radius asymmetry analysis
# =====================================================================
def analyze_vegard_asymmetry(table):
    """
    Vegard則の有効半径がA₃B vs B₃Aで対称性を持つことを示す.
    
    Vegard則: V_Vegard(X₃Y) = 0.75·V_X + 0.25·V_Y
              V_Vegard(Y₃X) = 0.75·V_Y + 0.25·V_X
    → r_Vegard(X₃Y) と r_Vegard(Y₃X) の差は
      純元素体積差 |V_X - V_Y| のみに依存（構造情報なし）
    """
    both = table.dropna(subset=["r_X3Y_Vegard", "r_Y3X_Vegard"])

    results = {
        "dr_Vegard": both["r_X3Y_Vegard"] - both["r_Y3X_Vegard"],
        "dr_DFT": both["r_X3Y_DFT"] - both["r_Y3X_DFT"],
        "dV_Vegard": both["V_X3Y_Vegard"] - both["V_Y3X_Vegard"],
        "dV_DFT": both["V_X3Y_DFT"] - both["V_Y3X_DFT"],
        "dOmega": both["Omega_X3Y"] - both["Omega_Y3X"],
    }
    return both, results


# =====================================================================
# 5. Structure-dependent effective radius estimation
# =====================================================================
def estimate_structure_dependent_radii(compound_df):
    """
    各構造(L1₂ A₃B, L1₂ B₃A, B2 AB) から元素別の有効原子体積を推定.
    
    L1₂ X₃Y: V_per_atom = a³/4 = 0.75·V_eff(X, in X₃Y) + 0.25·V_eff(Y, in X₃Y)
    B2 XY:    V_per_atom = a³/2 = 0.5·V_eff(X, in XY) + 0.5·V_eff(Y, in XY)
    
    最小二乗法で各元素の構造ごとの有効体積を推定.
    """
    l12_df = compound_df[compound_df["stype"] == "L12"].copy()
    b2_df = compound_df[compound_df["stype"] == "B2"].copy()

    # Only use elements with King volumes
    valid_elements = set(KING_ATOMIC_VOLUMES.keys())

    # ── L1₂: separate majority (A₃B) and minority (AB₃) roles ──
    l12_majority = defaultdict(list)  # element → list of V contributions as majority
    l12_minority = defaultdict(list)  # element → list of V contributions as minority

    for _, row in l12_df.iterrows():
        elA = row["element_A"]
        elB = row["element_B"]
        a = row["lattice_constant"]
        cA = row.get("count_A", 3)
        if pd.isna(a) or a <= 2 or a >= 8:
            continue
        if elA not in valid_elements or elB not in valid_elements:
            continue
        if elA == elB:
            continue

        v_per_atom = a**3 / 4
        if cA == 3:
            l12_majority[elA].append((elB, v_per_atom))
            l12_minority[elB].append((elA, v_per_atom))
        elif cA == 1:
            l12_minority[elA].append((elB, v_per_atom))
            l12_majority[elB].append((elA, v_per_atom))

    # ── B2: symmetric roles ──
    b2_dict = defaultdict(list)
    for _, row in b2_df.iterrows():
        elA = row["element_A"]
        elB = row["element_B"]
        a = row["lattice_constant"]
        if pd.isna(a) or a <= 2 or a >= 8:
            continue
        if elA not in valid_elements or elB not in valid_elements:
            continue
        if elA == elB:
            continue

        v_per_atom = a**3 / 2
        pair = tuple(sorted([elA, elB]))
        b2_dict[pair].append(v_per_atom)

    return l12_majority, l12_minority, b2_dict


# =====================================================================
# 6. Solve for element-wise effective volumes
# =====================================================================
def solve_effective_volumes(l12_majority, l12_minority, b2_dict):
    """
    各構造における元素の有効体積を最小二乗法で推定.
    
    Joint optimization with 2*n_el unknowns for L1₂:
      V_per_atom = 0.75·V_maj(A) + 0.25·V_min(B)
    where V_maj and V_min are SEPARATE parameter vectors.
    
    B2 equation:  V_per_atom = 0.50·V_eff_b2(A) + 0.50·V_eff_b2(B)
    """
    # Collect all elements
    all_elements = sorted(
        set(l12_majority.keys()) | set(l12_minority.keys()) |
        {e for pair in b2_dict for e in pair}
    )
    all_elements = [e for e in all_elements if e in KING_ATOMIC_VOLUMES]
    el_to_idx = {e: i for i, e in enumerate(all_elements)}
    n_el = len(all_elements)

    results = {}

    # ── L1₂ joint optimization: V_maj[0..n_el-1], V_min[n_el..2*n_el-1] ──
    # For each compound A₃B: V = 0.75·V_maj(A) + 0.25·V_min(B)
    equations_l12 = []
    targets_l12 = []
    for elA, partners in l12_majority.items():
        if elA not in el_to_idx:
            continue
        for elB, v_per_atom in partners:
            if elB not in el_to_idx:
                continue
            # A is majority, B is minority
            equations_l12.append((el_to_idx[elA], el_to_idx[elB], v_per_atom))
            targets_l12.append(v_per_atom)

    if equations_l12:
        # 2*n_el columns: [V_maj(0)..V_maj(n-1), V_min(0)..V_min(n-1)]
        A_mat = np.zeros((len(equations_l12), 2 * n_el))
        b_vec = np.array(targets_l12)
        for i, (iA, iB, _) in enumerate(equations_l12):
            A_mat[i, iA] = 0.75            # V_maj(A)
            A_mat[i, n_el + iB] = 0.25     # V_min(B)

        x0_king = np.array([KING_ATOMIC_VOLUMES.get(e, 15) for e in all_elements])
        x0 = np.concatenate([x0_king, x0_king])
        lb = np.full(2 * n_el, 1.0)
        ub = np.full(2 * n_el, 100.0)
        res = least_squares(lambda x: A_mat @ x - b_vec, x0, bounds=(lb, ub))

        v_eff_l12_majority = {e: res.x[i] for i, e in enumerate(all_elements)}
        v_eff_l12_minority = {e: res.x[n_el + i] for i, e in enumerate(all_elements)}
        results["L12_majority"] = v_eff_l12_majority
        results["L12_minority"] = v_eff_l12_minority
        results["L12_joint_rmse"] = np.sqrt(np.mean(res.fun**2))
        results["L12_majority_rmse"] = results["L12_joint_rmse"]
        results["L12_minority_rmse"] = results["L12_joint_rmse"]

    # ── B2 ──
    equations_b2 = []
    targets_b2 = []
    for pair, vols in b2_dict.items():
        elA, elB = pair
        if elA not in el_to_idx or elB not in el_to_idx:
            continue
        v_mean = np.mean(vols)
        equations_b2.append((el_to_idx[elA], el_to_idx[elB], v_mean))
        targets_b2.append(v_mean)

    if equations_b2:
        A_mat = np.zeros((len(equations_b2), n_el))
        b_vec = np.array(targets_b2)
        for i, (iA, iB, _) in enumerate(equations_b2):
            A_mat[i, iA] = 0.5
            A_mat[i, iB] = 0.5

        x0 = np.array([KING_ATOMIC_VOLUMES.get(e, 15) for e in all_elements])
        res = least_squares(lambda x: A_mat @ x - b_vec, x0, bounds=(1, 100))
        v_eff_b2 = {e: res.x[i] for i, e in enumerate(all_elements)}
        results["B2"] = v_eff_b2
        results["B2_rmse"] = np.sqrt(np.mean(res.fun**2))

    results["elements"] = all_elements
    return results


# =====================================================================
# 7. Figures
# =====================================================================
def fig01_l12_asymmetry(table):
    """図1: L1₂構造のA₃B vs B₃Aの非対称性."""
    both = table.dropna(subset=["a_X3Y", "a_Y3X"]).copy()
    if len(both) == 0:
        return

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    # (a) Lattice constant A₃B vs B₃A
    ax = axes[0]
    ax.scatter(both["a_X3Y"], both["a_Y3X"], alpha=0.3, s=20, c="steelblue")
    lim = [min(both["a_X3Y"].min(), both["a_Y3X"].min()) - 0.2,
           max(both["a_X3Y"].max(), both["a_Y3X"].max()) + 0.2]
    ax.plot(lim, lim, "k--", lw=1.5, label="$a_{X_3Y} = a_{Y_3X}$")
    ax.set_xlabel(r"$a$ (L1$_2$ $X_3Y$) [Å]")
    ax.set_ylabel(r"$a$ (L1$_2$ $Y_3X$) [Å]")
    ax.set_title(r"(a) L1$_2$格子定数: $X_3Y$ vs $Y_3X$")
    ax.legend()
    ax.set_xlim(lim)
    ax.set_ylim(lim)

    # (b) Volume asymmetry: DFT vs Vegard
    ax = axes[1]
    dv_dft = both["V_X3Y_DFT"] - both["V_Y3X_DFT"]
    dv_veg = both["V_X3Y_Vegard"] - both["V_Y3X_Vegard"]
    ax.scatter(dv_veg, dv_dft, alpha=0.3, s=20, c="darkorange")
    xlim = max(abs(dv_veg.min()), abs(dv_veg.max())) * 1.1
    ylim = max(abs(dv_dft.min()), abs(dv_dft.max())) * 1.1
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.plot([-xlim, xlim], [-xlim, xlim], "k--", lw=1.5)
    ax.set_xlabel(r"$\Delta V_{Vegard}$ ($X_3Y - Y_3X$) [Å$^3$]")
    ax.set_ylabel(r"$\Delta V_{DFT}$ ($X_3Y - Y_3X$) [Å$^3$]")
    ax.set_title("(b) 体積非対称性: DFT vs Vegard")

    # (c) Radius asymmetry: DFT vs Vegard
    ax = axes[2]
    dr_dft = both["r_X3Y_DFT"] - both["r_Y3X_DFT"]
    dr_veg = both["r_X3Y_Vegard"] - both["r_Y3X_Vegard"]
    ax.scatter(dr_veg, dr_dft, alpha=0.3, s=20, c="forestgreen")
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    xlim_r = max(abs(dr_veg.min()), abs(dr_veg.max())) * 1.1
    ax.plot([-xlim_r, xlim_r], [-xlim_r, xlim_r], "k--", lw=1.5)
    ax.set_xlabel(r"$\Delta r_{Vegard}$ ($X_3Y - Y_3X$) [Å]")
    ax.set_ylabel(r"$\Delta r_{DFT}$ ($X_3Y - Y_3X$) [Å]")
    ax.set_title("(c) 半径非対称性: DFT vs Vegard")

    corr_v = np.corrcoef(dv_veg.values, dv_dft.values)[0, 1]
    corr_r = np.corrcoef(dr_veg.values, dr_dft.values)[0, 1]
    axes[1].text(0.05, 0.95, f"$r$ = {corr_v:.3f}", transform=axes[1].transAxes,
                 fontsize=14, va="top")
    axes[2].text(0.05, 0.95, f"$r$ = {corr_r:.3f}", transform=axes[2].transAxes,
                 fontsize=14, va="top")

    fig.tight_layout()
    fig.savefig(OUT / "fig01_l12_asymmetry.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig01: N={len(both)}, corr_V={corr_v:.3f}, corr_r={corr_r:.3f}")


def fig02_omega_sf_asymmetry(table):
    """図2: Ω_sfのA₃B vs B₃Aの非対称性."""
    both = table.dropna(subset=["Omega_X3Y", "Omega_Y3X"]).copy()
    if len(both) == 0:
        return

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    # (a) Ω_sf X₃Y vs Y₃X
    ax = axes[0]
    ax.scatter(both["Omega_X3Y"], both["Omega_Y3X"], alpha=0.3, s=20, c="crimson")
    lim = max(abs(both["Omega_X3Y"]).max(), abs(both["Omega_Y3X"]).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1.5, label="symmetric")
    ax.set_xlabel(r"$\Omega_{sf}$ (L1$_2$ $X_3Y$)")
    ax.set_ylabel(r"$\Omega_{sf}$ (L1$_2$ $Y_3X$)")
    ax.set_title(r"(a) $\Omega_{sf}$: $X_3Y$ vs $Y_3X$")
    ax.legend()

    # (b) Ω_sf histogram
    ax = axes[1]
    d_omega = both["Omega_X3Y"] - both["Omega_Y3X"]
    ax.hist(d_omega, bins=80, color="salmon", edgecolor="black", alpha=0.7)
    ax.axvline(0, color="black", lw=2)
    ax.set_xlabel(r"$\Delta\Omega_{sf}$ ($X_3Y - Y_3X$)")
    ax.set_ylabel("Count")
    ax.set_title(r"(b) $\Omega_{sf}$非対称性の分布")
    ax.text(0.95, 0.95, f"mean={d_omega.mean():.4f}\nstd={d_omega.std():.4f}",
            transform=ax.transAxes, va="top", ha="right", fontsize=13)

    # (c) |Ω_sf asymmetry| vs |V_X - V_Y|
    ax = axes[2]
    dv_pure = abs(both["V_X_pure"] - both["V_Y_pure"])
    ax.scatter(dv_pure, abs(d_omega), alpha=0.3, s=20, c="purple")
    ax.set_xlabel(r"|$V_X^{pure} - V_Y^{pure}$| [Å$^3$]")
    ax.set_ylabel(r"|$\Delta\Omega_{sf}$| ($X_3Y - Y_3X$)")
    ax.set_title(r"(c) $\Omega_{sf}$非対称性 vs 純元素体積差")
    corr = np.corrcoef(dv_pure.values, abs(d_omega).values)[0, 1]
    ax.text(0.05, 0.95, f"$r$ = {corr:.3f}", transform=ax.transAxes,
            fontsize=14, va="top")

    fig.tight_layout()
    fig.savefig(OUT / "fig02_omega_sf_asymmetry.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig02: N={len(both)}, mean_dOmega={d_omega.mean():.4f}, std={d_omega.std():.4f}")


def fig03_vegard_cannot_distinguish(table):
    """図3: Vegard則が構造を区別できないことの直接的証明."""
    both = table.dropna(subset=["r_X3Y_DFT", "r_Y3X_DFT", "r_X3Y_Vegard", "r_Y3X_Vegard"]).copy()
    has_b2 = both.dropna(subset=["r_B2_DFT", "r_B2_Vegard"])

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # (a) Vegard radius: same for all structures (trivially)
    ax = axes[0, 0]
    ax.scatter(both["r_X3Y_Vegard"], both["r_Y3X_Vegard"], alpha=0.3, s=20, c="blue", label=r"L1$_2$")
    if len(has_b2) > 0:
        ax.scatter(has_b2["r_X3Y_Vegard"], has_b2["r_B2_Vegard"], alpha=0.3, s=20, c="red",
                   marker="^", label="B2")
    lim = [both["r_X3Y_Vegard"].min() - 0.05, both["r_X3Y_Vegard"].max() + 0.05]
    ax.plot(lim, lim, "k--", lw=1.5)
    ax.set_xlabel(r"$r_{Vegard}$ ($X_3Y$) [Å]")
    ax.set_ylabel(r"$r_{Vegard}$ ($Y_3X$ or B2) [Å]")
    ax.set_title("(a) Vegard有効半径: 構造間の関係")
    ax.legend()

    # (b) DFT radius: different for each structure
    ax = axes[0, 1]
    ax.scatter(both["r_X3Y_DFT"], both["r_Y3X_DFT"], alpha=0.3, s=20, c="blue", label=r"L1$_2$")
    if len(has_b2) > 0:
        ax.scatter(has_b2["r_X3Y_DFT"], has_b2["r_B2_DFT"], alpha=0.3, s=20, c="red",
                   marker="^", label="B2")
    lim = [min(both["r_X3Y_DFT"].min(), both["r_Y3X_DFT"].min()) - 0.05,
           max(both["r_X3Y_DFT"].max(), both["r_Y3X_DFT"].max()) + 0.05]
    ax.plot(lim, lim, "k--", lw=1.5)
    ax.set_xlabel(r"$r_{DFT}$ ($X_3Y$) [Å]")
    ax.set_ylabel(r"$r_{DFT}$ ($Y_3X$ or B2) [Å]")
    ax.set_title("(b) DFT有効半径: 構造間の関係")
    ax.legend()

    # (c) Residual from Vegard: should be ~0 for Vegard, large for DFT
    ax = axes[1, 0]
    resid_veg_X3Y = both["r_X3Y_DFT"] - both["r_X3Y_Vegard"]
    resid_veg_Y3X = both["r_Y3X_DFT"] - both["r_Y3X_Vegard"]
    ax.hist(resid_veg_X3Y, bins=60, alpha=0.6, label=r"$X_3Y$", color="blue")
    ax.hist(resid_veg_Y3X, bins=60, alpha=0.6, label=r"$Y_3X$", color="orange")
    ax.axvline(0, color="black", lw=2)
    ax.set_xlabel(r"$r_{DFT} - r_{Vegard}$ [Å]")
    ax.set_ylabel("Count")
    ax.set_title(r"(c) Vegard残差分布 (L1$_2$)")
    ax.legend()

    # (d) Ω_sf comparison: X₃Y vs Y₃X vs B2
    ax = axes[1, 1]
    if len(has_b2) > 0:
        ax.scatter(has_b2["Omega_X3Y"], has_b2["Omega_B2"], alpha=0.3, s=20, c="red",
                   label=r"$X_3Y$ vs B2")
        ax.scatter(has_b2["Omega_Y3X"], has_b2["Omega_B2"], alpha=0.3, s=20, c="green",
                   marker="^", label=r"$Y_3X$ vs B2")
        lim = max(abs(has_b2["Omega_X3Y"]).max(), abs(has_b2["Omega_B2"]).max()) * 1.1
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=1.5)
    ax.set_xlabel(r"$\Omega_{sf}$ (L1$_2$)")
    ax.set_ylabel(r"$\Omega_{sf}$ (B2)")
    ax.set_title(r"(d) $\Omega_{sf}$: L1$_2$ vs B2")
    ax.legend()

    fig.tight_layout()
    fig.savefig(OUT / "fig03_vegard_cannot_distinguish.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig03: N={len(both)}, N_with_B2={len(has_b2)}")


def fig04_effective_radius_table(radii_results):
    """図4: 構造ごとの有効半径を周期表風にプロット."""
    elements = radii_results["elements"]
    v_l12_maj = radii_results.get("L12_majority", {})
    v_l12_min = radii_results.get("L12_minority", {})
    v_b2 = radii_results.get("B2", {})

    # Compute radii
    common = [e for e in elements if e in v_l12_maj and e in v_l12_min and e in v_b2
              and e in KING_ATOMIC_VOLUMES]
    if not common:
        return

    rows = []
    for e in sorted(common):
        r_pure = volume_to_radius(KING_ATOMIC_VOLUMES[e])
        r_l12_maj = volume_to_radius(v_l12_maj[e])
        r_l12_min = volume_to_radius(v_l12_min[e])
        r_b2 = volume_to_radius(v_b2[e])
        rows.append({
            "Element": e,
            "r_pure": r_pure,
            "r_L12_maj": r_l12_maj,
            "r_L12_min": r_l12_min,
            "r_B2": r_b2,
            "dr_L12_maj": r_l12_maj - r_pure,
            "dr_L12_min": r_l12_min - r_pure,
            "dr_B2": r_b2 - r_pure,
            "asymmetry_L12": r_l12_maj - r_l12_min,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "effective_radii_by_structure.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(20, 14))

    # (a) Effective radius comparison
    ax = axes[0, 0]
    x = np.arange(len(df))
    w = 0.2
    ax.bar(x - 1.5*w, df["r_pure"], w, label="Pure", color="gray", alpha=0.8)
    ax.bar(x - 0.5*w, df["r_L12_maj"], w, label=r"L1$_2$ majority", color="blue", alpha=0.8)
    ax.bar(x + 0.5*w, df["r_L12_min"], w, label=r"L1$_2$ minority", color="orange", alpha=0.8)
    ax.bar(x + 1.5*w, df["r_B2"], w, label="B2", color="red", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Element"], rotation=90, fontsize=10)
    ax.set_ylabel("Effective radius [Å]")
    ax.set_title("(a) 構造別有効半径")
    ax.legend(fontsize=11)

    # (b) Deviation from pure
    ax = axes[0, 1]
    ax.bar(x - w, df["dr_L12_maj"], w, label=r"$\Delta r$ L1$_2$ maj", color="blue", alpha=0.8)
    ax.bar(x, df["dr_L12_min"], w, label=r"$\Delta r$ L1$_2$ min", color="orange", alpha=0.8)
    ax.bar(x + w, df["dr_B2"], w, label=r"$\Delta r$ B2", color="red", alpha=0.8)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Element"], rotation=90, fontsize=10)
    ax.set_ylabel(r"$r_{eff} - r_{pure}$ [Å]")
    ax.set_title("(b) 純元素からの偏差")
    ax.legend(fontsize=11)

    # (c) L1₂ asymmetry
    ax = axes[1, 0]
    colors = ["blue" if v >= 0 else "red" for v in df["asymmetry_L12"]]
    ax.bar(x, df["asymmetry_L12"], color=colors, alpha=0.7)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Element"], rotation=90, fontsize=10)
    ax.set_ylabel(r"$r_{L1_2,maj} - r_{L1_2,min}$ [Å]")
    ax.set_title(r"(c) L1$_2$非対称性 (majority − minority)")

    # (d) r_B2 vs r_L12
    ax = axes[1, 1]
    ax.scatter(df["r_L12_maj"], df["r_B2"], c="blue", s=60, label="majority", zorder=5)
    ax.scatter(df["r_L12_min"], df["r_B2"], c="orange", s=60, marker="^", label="minority", zorder=5)
    lim = [df[["r_L12_maj", "r_L12_min", "r_B2"]].min().min() - 0.05,
           df[["r_L12_maj", "r_L12_min", "r_B2"]].max().max() + 0.05]
    ax.plot(lim, lim, "k--", lw=1.5)
    for _, row in df.iterrows():
        ax.annotate(row["Element"], (row["r_L12_maj"], row["r_B2"]),
                    fontsize=8, ha="center", va="bottom")
    ax.set_xlabel(r"$r_{eff}$ (L1$_2$) [Å]")
    ax.set_ylabel(r"$r_{eff}$ (B2) [Å]")
    ax.set_title(r"(d) B2 vs L1$_2$ 有効半径")
    ax.legend()

    fig.tight_layout()
    fig.savefig(OUT / "fig04_effective_radius_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig04: {len(df)} elements")


def fig05_three_structure_comparison(table):
    """図5: A₃B, B₃A, B2 の3構造比較."""
    has_all = table.dropna(subset=["a_X3Y", "a_Y3X", "a_B2"]).copy()
    if len(has_all) == 0:
        return

    fig, axes = plt.subplots(2, 3, figsize=(24, 14))

    # Filter HEA elements for cleaner visualization
    hea = has_all[has_all["elX"].isin(HEA_ELEMENTS) & has_all["elY"].isin(HEA_ELEMENTS)]
    plot_df = hea if len(hea) >= 20 else has_all

    # (a) Lattice constants: X₃Y vs Y₃X vs B2
    ax = axes[0, 0]
    ax.scatter(plot_df["a_X3Y"], plot_df["a_B2"], alpha=0.4, s=30, c="red", label=r"$X_3Y$ vs B2")
    ax.scatter(plot_df["a_Y3X"], plot_df["a_B2"], alpha=0.4, s=30, c="blue", marker="^",
               label=r"$Y_3X$ vs B2")
    lim = [2.5, 7.0]
    ax.plot(lim, lim, "k--", lw=1.5)
    ax.set_xlabel(r"$a$ (L1$_2$) [Å]")
    ax.set_ylabel("$a$ (B2) [Å]")
    ax.set_title(r"(a) 格子定数: L1$_2$ vs B2")
    ax.legend()

    # (b) Ω_sf across structures
    ax = axes[0, 1]
    ax.scatter(plot_df["Omega_X3Y"], plot_df["Omega_B2"], alpha=0.4, s=30, c="red",
               label=r"$\Omega_{sf}(X_3Y)$")
    ax.scatter(plot_df["Omega_Y3X"], plot_df["Omega_B2"], alpha=0.4, s=30, c="blue",
               marker="^", label=r"$\Omega_{sf}(Y_3X)$")
    lim_o = 0.5
    ax.plot([-lim_o, lim_o], [-lim_o, lim_o], "k--", lw=1.5)
    ax.set_xlabel(r"$\Omega_{sf}$ (L1$_2$)")
    ax.set_ylabel(r"$\Omega_{sf}$ (B2)")
    ax.set_title(r"(b) $\Omega_{sf}$: L1$_2$ vs B2")
    ax.legend()

    # (c) Asymmetry index: |a_X3Y - a_Y3X| / a_B2
    ax = axes[0, 2]
    asym = abs(plot_df["a_X3Y"] - plot_df["a_Y3X"]) / plot_df["a_B2"] * 100
    dr_pure = abs(plot_df["r_X_pure"] - plot_df["r_Y_pure"])
    ax.scatter(dr_pure, asym, alpha=0.4, s=30, c="green")
    ax.set_xlabel(r"|$r_X^{pure} - r_Y^{pure}$| [Å]")
    ax.set_ylabel(r"|$a_{X_3Y} - a_{Y_3X}$| / $a_{B2}$ [%]")
    ax.set_title(r"(c) L1$_2$非対称度 vs 原子サイズ差")
    corr = np.corrcoef(dr_pure.values, asym.values)[0, 1]
    ax.text(0.05, 0.95, f"$r$ = {corr:.3f}", transform=ax.transAxes, fontsize=14, va="top")

    # (d) V_DFT vs V_Vegard for all three structures
    ax = axes[1, 0]
    for struct, col_dft, col_veg, color, marker, label in [
        ("X3Y", "V_X3Y_DFT", "V_X3Y_Vegard", "blue", "o", r"L1$_2$ $X_3Y$"),
        ("Y3X", "V_Y3X_DFT", "V_Y3X_Vegard", "orange", "^", r"L1$_2$ $Y_3X$"),
        ("B2", "V_B2_DFT", "V_B2_Vegard", "red", "s", "B2"),
    ]:
        sub = plot_df.dropna(subset=[col_dft, col_veg])
        ax.scatter(sub[col_veg], sub[col_dft], alpha=0.3, s=20, c=color, marker=marker, label=label)
    lim = [5, 55]
    ax.plot(lim, lim, "k--", lw=1.5)
    ax.set_xlabel(r"$V_{Vegard}$ [Å$^3$]")
    ax.set_ylabel(r"$V_{DFT}$ [Å$^3$]")
    ax.set_title("(d) 原子体積: DFT vs Vegard")
    ax.legend(fontsize=11)

    # (e) Ω_sf distribution by structure
    ax = axes[1, 1]
    for col, color, label in [
        ("Omega_X3Y", "blue", r"L1$_2$ $X_3Y$"),
        ("Omega_Y3X", "orange", r"L1$_2$ $Y_3X$"),
        ("Omega_B2", "red", "B2"),
    ]:
        vals = plot_df[col].dropna()
        ax.hist(vals, bins=50, alpha=0.5, color=color, label=label, edgecolor="black", lw=0.5)
    ax.axvline(0, color="black", lw=2)
    ax.set_xlabel(r"$\Omega_{sf}$")
    ax.set_ylabel("Count")
    ax.set_title(r"(e) $\Omega_{sf}$ 分布（構造別）")
    ax.legend()

    # (f) Ratio a_L12 / a_B2 vs composition
    ax = axes[1, 2]
    ratio_X3Y = plot_df["a_X3Y"] / plot_df["a_B2"]
    ratio_Y3X = plot_df["a_Y3X"] / plot_df["a_B2"]
    ax.hist(ratio_X3Y, bins=40, alpha=0.5, color="blue", label=r"$a_{X_3Y}/a_{B2}$",
            edgecolor="black", lw=0.5)
    ax.hist(ratio_Y3X, bins=40, alpha=0.5, color="orange", label=r"$a_{Y_3X}/a_{B2}$",
            edgecolor="black", lw=0.5)
    # Ideal ratios: L12 FCC (a_fcc) vs B2 BCC (a_bcc)
    # a_fcc/a_bcc = (4/Z_fcc)^(1/3) / (2/Z_bcc)^(1/3) = 4^(1/3) / 2^(1/3) ≈ 1.26
    ax.axvline(2**(1/3), color="black", lw=2, ls="--", label=f"$2^{{1/3}}$ = {2**(1/3):.3f}")
    ax.set_xlabel(r"$a_{L1_2} / a_{B2}$")
    ax.set_ylabel("Count")
    ax.set_title(r"(f) 格子定数比 L1$_2$/B2 分布")
    ax.legend()

    fig.tight_layout()
    fig.savefig(OUT / "fig05_three_structure_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig05: N_all3={len(has_all)}, N_HEA={len(hea)}")


def fig06_hea_element_focus(table):
    """図6: HEA主要元素に限定した詳細比較."""
    hea_pairs = table[table["elX"].isin(HEA_ELEMENTS) & table["elY"].isin(HEA_ELEMENTS)].copy()
    both = hea_pairs.dropna(subset=["Omega_X3Y", "Omega_Y3X"])

    if len(both) == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # (a) Ω_sf heatmap for L1₂ X₃Y
    elements_in_data = sorted(set(both["elX"]) | set(both["elY"]))
    n = len(elements_in_data)
    el_idx = {e: i for i, e in enumerate(elements_in_data)}

    omega_X3Y = np.full((n, n), np.nan)
    omega_Y3X = np.full((n, n), np.nan)
    omega_B2_mat = np.full((n, n), np.nan)

    for _, row in both.iterrows():
        i, j = el_idx[row["elX"]], el_idx[row["elY"]]
        omega_X3Y[i, j] = row["Omega_X3Y"]
        omega_Y3X[i, j] = row["Omega_Y3X"]
        if not pd.isna(row.get("Omega_B2", np.nan)):
            omega_B2_mat[i, j] = row["Omega_B2"]
            omega_B2_mat[j, i] = row["Omega_B2"]

    # Asymmetry matrix
    asym_mat = omega_X3Y - omega_Y3X

    vmax = np.nanmax(np.abs(asym_mat)) * 0.8

    ax = axes[0, 0]
    im = ax.imshow(omega_X3Y, cmap="RdBu_r", vmin=-0.3, vmax=0.3, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_xticklabels(elements_in_data, rotation=90, fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(elements_in_data, fontsize=9)
    ax.set_title(r"(a) $\Omega_{sf}$ L1$_2$ $X_3Y$ (row=$X$, col=$Y$)")
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[0, 1]
    im = ax.imshow(omega_Y3X, cmap="RdBu_r", vmin=-0.3, vmax=0.3, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_xticklabels(elements_in_data, rotation=90, fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(elements_in_data, fontsize=9)
    ax.set_title(r"(b) $\Omega_{sf}$ L1$_2$ $Y_3X$ (row=$X$, col=$Y$)")
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[1, 0]
    im = ax.imshow(asym_mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_xticklabels(elements_in_data, rotation=90, fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(elements_in_data, fontsize=9)
    ax.set_title(r"(c) $\Delta\Omega_{sf}$ ($X_3Y - Y_3X$)")
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[1, 1]
    mask = ~np.isnan(omega_B2_mat)
    if mask.any():
        im = ax.imshow(omega_B2_mat, cmap="RdBu_r", vmin=-0.3, vmax=0.3, aspect="auto")
        ax.set_xticks(range(n))
        ax.set_xticklabels(elements_in_data, rotation=90, fontsize=9)
        ax.set_yticks(range(n))
        ax.set_yticklabels(elements_in_data, fontsize=9)
        ax.set_title(r"(d) $\Omega_{sf}$ B2 (symmetric)")
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.tight_layout()
    fig.savefig(OUT / "fig06_hea_element_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig06: {len(elements_in_data)} HEA elements, {len(both)} pairs")


def fig07_specific_examples(table):
    """図7: 代表的ペアの詳細比較."""
    # Select illustrative pairs
    examples = [
        ("Al", "Ni"), ("Co", "Ti"), ("Cu", "Zn"), ("Fe", "Ni"),
        ("Cr", "Fe"), ("Al", "Ti"), ("Nb", "Ti"), ("Mo", "W"),
        ("Co", "Ni"), ("Cu", "Au"), ("Ag", "Cu"), ("Fe", "Co"),
    ]

    available = []
    for elX, elY in examples:
        pair = tuple(sorted([elX, elY]))
        row = table[(table["elX"] == pair[0]) & (table["elY"] == pair[1])]
        if len(row) > 0 and not row.iloc[0][["a_X3Y", "a_Y3X"]].isna().any():
            available.append((pair[0], pair[1], row.iloc[0]))

    if not available:
        return

    n_pairs = min(len(available), 12)
    ncols = 4
    nrows = (n_pairs + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(24, 5 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)

    for idx, (elX, elY, row) in enumerate(available[:n_pairs]):
        ax = axes[idx // ncols, idx % ncols]

        structures = [f"L1$_2$ {elX}$_3${elY}", f"L1$_2$ {elY}$_3${elX}"]
        a_vals = [row["a_X3Y"], row["a_Y3X"]]
        v_dft = [row["V_X3Y_DFT"], row["V_Y3X_DFT"]]
        v_veg = [row["V_X3Y_Vegard"], row["V_Y3X_Vegard"]]
        colors_dft = ["steelblue", "darkorange"]
        colors_veg = ["lightblue", "lightsalmon"]

        if not pd.isna(row.get("a_B2", np.nan)):
            structures.append(f"B2 {elX}{elY}")
            a_vals.append(row["a_B2"])
            v_dft.append(row["V_B2_DFT"])
            v_veg.append(row["V_B2_Vegard"])
            colors_dft.append("firebrick")
            colors_veg.append("mistyrose")

        x = np.arange(len(structures))
        w = 0.35
        ax.bar(x - w/2, v_dft, w, color=colors_dft, edgecolor="black", label="DFT")
        ax.bar(x + w/2, v_veg, w, color=colors_veg, edgecolor="black", label="Vegard")
        ax.set_xticks(x)
        ax.set_xticklabels(structures, fontsize=10)
        ax.set_ylabel(r"$V$ [Å$^3$/atom]")
        ax.set_title(f"{elX}-{elY}")
        if idx == 0:
            ax.legend(fontsize=10)

    # Hide unused axes
    for idx in range(n_pairs, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle("代表的元素ペアの構造別原子体積比較 (DFT vs Vegard)", fontsize=20, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig07_specific_examples.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig07: {n_pairs} example pairs")


def fig08_delta_r_invariance(table):
    """図8: δrが構造に対して不変であることの直接的証明."""
    both = table.dropna(subset=["r_X3Y_Vegard", "r_Y3X_Vegard"]).copy()
    has_b2 = both.dropna(subset=["r_B2_Vegard"])

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    # For each pair, compute δr (size mismatch parameter)
    # For binary X-Y at composition c_X:
    # r_bar = c_X*r_X + c_Y*r_Y
    # δr = 100 * sqrt(c_X*(1-r_X/r_bar)² + c_Y*(1-r_Y/r_bar)²)
    def delta_r(elX, elY, cX, cY):
        rX = METALLIC_RADII.get(elX, np.nan)
        rY = METALLIC_RADII.get(elY, np.nan)
        if np.isnan(rX) or np.isnan(rY):
            return np.nan
        r_bar = cX * rX + cY * rY
        return 100 * np.sqrt(cX * (1 - rX/r_bar)**2 + cY * (1 - rY/r_bar)**2)

    both["delta_r_X3Y"] = both.apply(lambda r: delta_r(r["elX"], r["elY"], 0.75, 0.25), axis=1)
    both["delta_r_Y3X"] = both.apply(lambda r: delta_r(r["elY"], r["elX"], 0.75, 0.25), axis=1)
    both["delta_r_B2"] = both.apply(lambda r: delta_r(r["elX"], r["elY"], 0.5, 0.5), axis=1)

    # (a) δr: X₃Y vs Y₃X — should be identical
    ax = axes[0]
    mask = both["delta_r_X3Y"].notna() & both["delta_r_Y3X"].notna()
    sub = both[mask]
    ax.scatter(sub["delta_r_X3Y"], sub["delta_r_Y3X"], alpha=0.3, s=20, c="steelblue")
    lim = [0, max(sub["delta_r_X3Y"].max(), sub["delta_r_Y3X"].max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=2, label=r"$\delta_r(X_3Y) = \delta_r(Y_3X)$")
    ax.set_xlabel(r"$\delta_r$ ($X_3Y$, $c_X=0.75$) [%]")
    ax.set_ylabel(r"$\delta_r$ ($Y_3X$, $c_Y=0.75$) [%]")
    ax.set_title(r"(a) $\delta_r$: 組成で変わるが構造情報なし")
    ax.legend()
    rmse = np.sqrt(np.mean((sub["delta_r_X3Y"] - sub["delta_r_Y3X"])**2))
    ax.text(0.05, 0.95, f"RMSE = {rmse:.4f}%\n(理論的に同一)", transform=ax.transAxes,
            fontsize=13, va="top")

    # (b) δr X₃Y vs B2
    ax = axes[1]
    sub2 = both.dropna(subset=["delta_r_X3Y", "delta_r_B2"])
    ax.scatter(sub2["delta_r_X3Y"], sub2["delta_r_B2"], alpha=0.3, s=20, c="darkorange")
    lim = [0, max(sub2["delta_r_X3Y"].max(), sub2["delta_r_B2"].max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=1.5)
    ax.set_xlabel(r"$\delta_r$ ($X_3Y$, $c_X=0.75$) [%]")
    ax.set_ylabel(r"$\delta_r$ (B2, $c_X=0.50$) [%]")
    ax.set_title(r"(b) $\delta_r$: L1$_2$ vs B2")
    ax.text(0.05, 0.95, "組成依存のみ\n構造情報なし", transform=ax.transAxes,
            fontsize=13, va="top")

    # (c) Contrast: Ω_sf DOES depend on structure
    ax = axes[2]
    sub3 = both.dropna(subset=["Omega_X3Y", "Omega_Y3X"])
    ax.scatter(sub3["Omega_X3Y"], sub3["Omega_Y3X"], alpha=0.3, s=20, c="forestgreen")
    lim = max(abs(sub3["Omega_X3Y"]).max(), abs(sub3["Omega_Y3X"]).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1.5)
    ax.set_xlabel(r"$\Omega_{sf}$ ($X_3Y$)")
    ax.set_ylabel(r"$\Omega_{sf}$ ($Y_3X$)")
    ax.set_title(r"(c) $\Omega_{sf}$: 構造情報を含む")
    corr = np.corrcoef(sub3["Omega_X3Y"].values, sub3["Omega_Y3X"].values)[0, 1]
    ax.text(0.05, 0.95, f"$r$ = {corr:.3f}\n(大きな散布 → 構造依存)", transform=ax.transAxes,
            fontsize=13, va="top")

    fig.tight_layout()
    fig.savefig(OUT / "fig08_delta_r_invariance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig08: N={len(both)}")


def fig09_volume_radius_structure_dependence(table, radii_results):
    """図9: 原子体積由来の半径が構造情報を吸収することの証明."""
    v_l12_maj = radii_results.get("L12_majority", {})
    v_l12_min = radii_results.get("L12_minority", {})
    v_b2 = radii_results.get("B2", {})

    common = [e for e in radii_results["elements"]
              if e in v_l12_maj and e in v_l12_min and e in v_b2
              and e in KING_ATOMIC_VOLUMES and e in HEA_ELEMENTS]

    if not common:
        return

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    r_pure = [volume_to_radius(KING_ATOMIC_VOLUMES[e]) for e in common]
    r_l12_maj = [volume_to_radius(v_l12_maj[e]) for e in common]
    r_l12_min = [volume_to_radius(v_l12_min[e]) for e in common]
    r_b2_list = [volume_to_radius(v_b2[e]) for e in common]

    # (a) r_L12_majority vs r_pure
    ax = axes[0]
    ax.scatter(r_pure, r_l12_maj, c="blue", s=80, zorder=5, label=r"L1$_2$ majority")
    ax.scatter(r_pure, r_l12_min, c="orange", s=80, marker="^", zorder=5, label=r"L1$_2$ minority")
    ax.scatter(r_pure, r_b2_list, c="red", s=80, marker="s", zorder=5, label="B2")
    lim = [min(r_pure) - 0.05, max(r_pure) + 0.05]
    ax.plot(lim, lim, "k--", lw=1.5)
    for i, e in enumerate(common):
        ax.annotate(e, (r_pure[i], r_l12_maj[i]), fontsize=9, ha="left", va="bottom")
    ax.set_xlabel(r"$r_{pure}$ [Å]")
    ax.set_ylabel(r"$r_{eff}$ [Å]")
    ax.set_title("(a) 有効半径 vs 純元素半径")
    ax.legend()

    # (b) Deviation: r_eff - r_pure
    ax = axes[1]
    x = np.arange(len(common))
    w = 0.25
    dr_maj = [r_l12_maj[i] - r_pure[i] for i in range(len(common))]
    dr_min = [r_l12_min[i] - r_pure[i] for i in range(len(common))]
    dr_b2 = [r_b2_list[i] - r_pure[i] for i in range(len(common))]
    ax.bar(x - w, dr_maj, w, label=r"L1$_2$ maj", color="blue", alpha=0.8)
    ax.bar(x, dr_min, w, label=r"L1$_2$ min", color="orange", alpha=0.8)
    ax.bar(x + w, dr_b2, w, label="B2", color="red", alpha=0.8)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(common, rotation=90, fontsize=11)
    ax.set_ylabel(r"$r_{eff} - r_{pure}$ [Å]")
    ax.set_title("(b) 純元素からの偏差（構造別）")
    ax.legend()

    # (c) Spread: std of r_eff across structures
    ax = axes[2]
    spreads = [np.std([r_l12_maj[i], r_l12_min[i], r_b2_list[i]]) for i in range(len(common))]
    ax.bar(x, spreads, color="purple", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(common, rotation=90, fontsize=11)
    ax.set_ylabel(r"std($r_{eff}$) across structures [Å]")
    ax.set_title("(c) 有効半径の構造依存バラつき")
    ax.text(0.05, 0.95, "大きいほど構造情報\nを含む",
            transform=ax.transAxes, fontsize=13, va="top")

    fig.tight_layout()
    fig.savefig(OUT / "fig09_volume_radius_structure.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig09: {len(common)} HEA elements")


# =====================================================================
# Summary statistics
# =====================================================================
def compute_statistics(table, radii_results):
    """統計情報のサマリーを計算."""
    stats = {}

    # Data coverage
    stats["total_pairs"] = len(table)
    stats["pairs_with_X3Y"] = table["a_X3Y"].notna().sum()
    stats["pairs_with_Y3X"] = table["a_Y3X"].notna().sum()
    stats["pairs_with_B2"] = table["a_B2"].notna().sum()
    stats["pairs_with_all_3"] = table.dropna(subset=["a_X3Y", "a_Y3X", "a_B2"]).shape[0]
    stats["pairs_with_both_L12"] = table.dropna(subset=["a_X3Y", "a_Y3X"]).shape[0]

    # HEA elements
    hea = table[table["elX"].isin(HEA_ELEMENTS) & table["elY"].isin(HEA_ELEMENTS)]
    stats["hea_pairs_total"] = len(hea)
    stats["hea_pairs_all_3"] = hea.dropna(subset=["a_X3Y", "a_Y3X", "a_B2"]).shape[0]

    # L12 asymmetry
    both = table.dropna(subset=["a_X3Y", "a_Y3X"])
    diff_a = both["a_X3Y"] - both["a_Y3X"]
    stats["l12_mean_abs_diff_a"] = diff_a.abs().mean()
    stats["l12_max_abs_diff_a"] = diff_a.abs().max()
    stats["l12_median_abs_diff_a"] = diff_a.abs().median()

    diff_omega = both["Omega_X3Y"] - both["Omega_Y3X"]
    stats["omega_mean_abs_diff"] = diff_omega.abs().mean() if "Omega_X3Y" in both else np.nan
    stats["omega_std_diff"] = diff_omega.std() if "Omega_X3Y" in both else np.nan

    # Vegard residuals
    both2 = table.dropna(subset=["V_X3Y_DFT", "V_X3Y_Vegard", "V_Y3X_DFT", "V_Y3X_Vegard"])
    resid_X3Y = both2["V_X3Y_DFT"] - both2["V_X3Y_Vegard"]
    resid_Y3X = both2["V_Y3X_DFT"] - both2["V_Y3X_Vegard"]
    stats["vegard_rmse_X3Y"] = np.sqrt(np.mean(resid_X3Y**2))
    stats["vegard_rmse_Y3X"] = np.sqrt(np.mean(resid_Y3X**2))

    # Structure-dependent radii
    if "L12_majority" in radii_results and "L12_minority" in radii_results and "B2" in radii_results:
        common = [e for e in radii_results["elements"]
                  if e in radii_results["L12_majority"] and e in radii_results["L12_minority"]
                  and e in radii_results["B2"] and e in KING_ATOMIC_VOLUMES]
        dr_maj = [volume_to_radius(radii_results["L12_majority"][e]) - volume_to_radius(KING_ATOMIC_VOLUMES[e])
                  for e in common]
        dr_min = [volume_to_radius(radii_results["L12_minority"][e]) - volume_to_radius(KING_ATOMIC_VOLUMES[e])
                  for e in common]
        dr_b2 = [volume_to_radius(radii_results["B2"][e]) - volume_to_radius(KING_ATOMIC_VOLUMES[e])
                 for e in common]
        stats["n_elements_all_structures"] = len(common)
        stats["mean_abs_dr_L12_maj"] = np.mean(np.abs(dr_maj))
        stats["mean_abs_dr_L12_min"] = np.mean(np.abs(dr_min))
        stats["mean_abs_dr_B2"] = np.mean(np.abs(dr_b2))
        stats["rmse_L12_majority"] = radii_results.get("L12_majority_rmse", np.nan)
        stats["rmse_L12_minority"] = radii_results.get("L12_minority_rmse", np.nan)
        stats["rmse_B2"] = radii_results.get("B2_rmse", np.nan)

    return stats


# =====================================================================
# Main
# =====================================================================
def main():
    print("=" * 70)
    print("Vegard則 vs 構造依存有効半径の比較分析")
    print("=" * 70)

    # 1. Load data
    print("\n[1] データ読み込み...")
    compound_df = load_all_data()
    print(f"  Total compounds: {len(compound_df)}")
    print(f"  B2: {(compound_df['stype']=='B2').sum()}")
    print(f"  L12: {(compound_df['stype']=='L12').sum()}")
    print(f"  Sources: {compound_df['db'].value_counts().to_dict()}")

    # 2. Build comparison table
    print("\n[2] 比較テーブル構築...")
    table = build_comparison_table(compound_df)
    print(f"  Total pairs: {len(table)}")
    print(f"  Pairs with X₃Y + Y₃X: {table.dropna(subset=['a_X3Y', 'a_Y3X']).shape[0]}")
    print(f"  Pairs with B2: {table['a_B2'].notna().sum()}")
    print(f"  Pairs with all 3 structures: {table.dropna(subset=['a_X3Y', 'a_Y3X', 'a_B2']).shape[0]}")

    # Save table
    table.to_csv(OUT / "comparison_table.csv", index=False)
    print(f"  Saved to {OUT / 'comparison_table.csv'}")

    # 3. Estimate structure-dependent effective radii
    print("\n[3] 構造別有効半径の推定...")
    l12_maj, l12_min, b2_dict = estimate_structure_dependent_radii(compound_df)
    print(f"  L1₂ majority elements: {len(l12_maj)}")
    print(f"  L1₂ minority elements: {len(l12_min)}")
    print(f"  B2 pairs: {len(b2_dict)}")

    radii_results = solve_effective_volumes(l12_maj, l12_min, b2_dict)
    print(f"  Elements with all 3 structures: {sum(1 for e in radii_results['elements'] if e in radii_results.get('L12_majority', {}) and e in radii_results.get('L12_minority', {}) and e in radii_results.get('B2', {}))}")
    for key in ["L12_majority_rmse", "L12_minority_rmse", "B2_rmse"]:
        if key in radii_results:
            print(f"  {key}: {radii_results[key]:.4f} Å³")

    # 4. Generate figures
    print("\n[4] 図の生成...")
    fig01_l12_asymmetry(table)
    fig02_omega_sf_asymmetry(table)
    fig03_vegard_cannot_distinguish(table)
    fig04_effective_radius_table(radii_results)
    fig05_three_structure_comparison(table)
    fig06_hea_element_focus(table)
    fig07_specific_examples(table)
    fig08_delta_r_invariance(table)
    fig09_volume_radius_structure_dependence(table, radii_results)

    # 5. Statistics
    print("\n[5] 統計サマリー...")
    stats = compute_statistics(table, radii_results)
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Save statistics
    pd.Series(stats).to_csv(OUT / "statistics.csv")

    print(f"\n全出力: {OUT}/")
    print("完了")


if __name__ == "__main__":
    main()
