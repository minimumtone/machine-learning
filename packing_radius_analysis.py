#!/usr/bin/env python3
"""
DFT格子定数 + パッキング制約条件から有効原子半径を求め、
構造情報を吸収できるか検証する.

パッキング制約:
  L1₂ (FCC): 最近接距離 d_nn = a / √2
    → A₃B: majority A–minority B 接触: r_A + r_B = a / √2
    → A₃B: majority A–A 接触:        2 r_A = a / √2
  B2 (BCC): 最近接距離 d_nn = a√3 / 2
    → AB:  r_A + r_B = a√3 / 2

検証:
  1. 各構造から独立にパッキング半径を推定
  2. 構造間の整合性（同じ元素ペアで r_A, r_B が一致するか）
  3. 体積由来半径との比較
"""

import warnings
import io
import base64
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "font.family": "sans-serif",
    "font.sans-serif": ["IPAPGothic", "IPAGothic", "DejaVu Sans"],
})

OUT = Path("vegard_comparison_output")

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
    "Sm":33.0,"Ho":30.9,"Gd":33.0,"Nd":34.2,"Pr":34.6,
}


def r_from_V(V):
    """atomic volume → equivalent sphere radius."""
    return (3 * V / (4 * np.pi))**(1.0/3.0)


def main():
    print("=" * 70)
    print("パッキング制約条件付きDFT有効半径の構造情報吸収能力の検証")
    print("=" * 70)

    table = pd.read_csv(OUT / "comparison_table.csv")
    both = table.dropna(subset=["a_X3Y", "a_Y3X", "a_B2"]).copy()
    print(f"全3構造データあり: {len(both)} ペア")

    # ── 1. ペアごとにパッキング半径を計算 ──
    results = []
    for _, row in both.iterrows():
        elX, elY = row["elX"], row["elY"]
        a_x3y = row["a_X3Y"]  # L1₂ X₃Y: X majority
        a_y3x = row["a_Y3X"]  # L1₂ Y₃X: Y majority
        a_b2 = row["a_B2"]    # B2 XY

        # --- L1₂ X₃Y packing constraints ---
        # X is majority (face), Y is minority (corner)
        # Option A: majority-majority contact: 2*r_X = a/√2
        r_X_from_X3Y_AA = a_x3y / (2 * np.sqrt(2))
        # Option B: majority-minority contact: r_X + r_Y = a/√2
        d_nn_x3y = a_x3y / np.sqrt(2)
        # If we use AA contact for majority: r_Y = d_nn - r_X
        r_Y_from_X3Y = d_nn_x3y - r_X_from_X3Y_AA  # = a/(2√2)

        # --- L1₂ Y₃X packing constraints ---
        # Y is majority, X is minority
        r_Y_from_Y3X_AA = a_y3x / (2 * np.sqrt(2))
        d_nn_y3x = a_y3x / np.sqrt(2)
        r_X_from_Y3X = d_nn_y3x - r_Y_from_Y3X_AA

        # --- B2 packing constraints ---
        # A-B contact: r_X + r_Y = a√3/2
        d_nn_b2 = a_b2 * np.sqrt(3) / 2
        # We need additional info to separate r_X and r_Y
        # Use volume ratio as weight:
        vX = KING_ATOMIC_VOLUMES.get(elX, 15)
        vY = KING_ATOMIC_VOLUMES.get(elY, 15)
        ratio = vX / (vX + vY)
        r_X_from_B2 = d_nn_b2 * ratio
        r_Y_from_B2 = d_nn_b2 * (1 - ratio)

        # --- Volume-derived radii ---
        r_X3Y_vol = r_from_V(row["V_X3Y_DFT"])
        r_Y3X_vol = r_from_V(row["V_Y3X_DFT"])
        r_B2_vol = r_from_V(row["V_B2_DFT"])

        # --- Consistency check ---
        # From X₃Y: r_X ≈ a/(2√2)
        # From Y₃X: r_X ≈ d_nn(Y₃X) - a(Y₃X)/(2√2) 
        # These should be equal if packing gives a unique radius

        results.append({
            "elX": elX, "elY": elY,
            "a_X3Y": a_x3y, "a_Y3X": a_y3x, "a_B2": a_b2,
            # Contact distances
            "d_nn_X3Y": d_nn_x3y, "d_nn_Y3X": d_nn_y3x, "d_nn_B2": d_nn_b2,
            # r_X from different structures
            "r_X_from_X3Y": r_X_from_X3Y_AA,  # X as majority
            "r_X_from_Y3X": r_X_from_Y3X,      # X as minority
            "r_X_from_B2": r_X_from_B2,
            # r_Y from different structures
            "r_Y_from_X3Y": r_Y_from_X3Y,      # Y as minority
            "r_Y_from_Y3X": r_Y_from_Y3X_AA,  # Y as majority
            "r_Y_from_B2": r_Y_from_B2,
            # Volume-derived
            "r_X3Y_vol": r_X3Y_vol,
            "r_Y3X_vol": r_Y3X_vol,
            "r_B2_vol": r_B2_vol,
            # Omega_sf
            "Omega_X3Y": row["Omega_X3Y"],
            "Omega_Y3X": row["Omega_Y3X"],
            "Omega_B2": row["Omega_B2"],
        })

    df = pd.DataFrame(results)

    # ── 2. Statistics ──
    # r_X consistency across structures
    df["dr_X_maj_vs_min"] = df["r_X_from_X3Y"] - df["r_X_from_Y3X"]
    df["dr_X_maj_vs_B2"] = df["r_X_from_X3Y"] - df["r_X_from_B2"]
    df["dr_Y_maj_vs_min"] = df["r_Y_from_Y3X"] - df["r_Y_from_X3Y"]
    df["dr_Y_maj_vs_B2"] = df["r_Y_from_Y3X"] - df["r_Y_from_B2"]

    # d_nn consistency
    df["d_nn_diff_L12"] = df["d_nn_X3Y"] - df["d_nn_Y3X"]
    df["d_nn_diff_X3Y_B2"] = df["d_nn_X3Y"] - df["d_nn_B2"]

    print("\n=== パッキング半径の構造間整合性 ===")
    print(f"r_X (majority vs minority): mean|Δ|={df['dr_X_maj_vs_min'].abs().mean():.4f} Å, "
          f"max|Δ|={df['dr_X_maj_vs_min'].abs().max():.4f} Å")
    print(f"r_X (majority vs B2):       mean|Δ|={df['dr_X_maj_vs_B2'].abs().mean():.4f} Å, "
          f"max|Δ|={df['dr_X_maj_vs_B2'].abs().max():.4f} Å")
    print(f"r_Y (majority vs minority): mean|Δ|={df['dr_Y_maj_vs_min'].abs().mean():.4f} Å, "
          f"max|Δ|={df['dr_Y_maj_vs_min'].abs().max():.4f} Å")

    print(f"\nd_nn(X₃Y) vs d_nn(Y₃X): mean|Δ|={df['d_nn_diff_L12'].abs().mean():.4f} Å")
    print(f"d_nn(X₃Y) vs d_nn(B2):  mean|Δ|={df['d_nn_diff_X3Y_B2'].abs().mean():.4f} Å")

    # ── 3. Global least-squares: fit single r per element from ALL packing constraints ──
    all_elements = sorted(set(df["elX"]) | set(df["elY"]))
    all_elements = [e for e in all_elements if e in KING_ATOMIC_VOLUMES]
    el_to_idx = {e: i for i, e in enumerate(all_elements)}
    n_el = len(all_elements)

    # Build equations: d_nn = r_X + r_Y for all structures
    equations = []
    targets = []

    for _, row in df.iterrows():
        eX, eY = row["elX"], row["elY"]
        if eX not in el_to_idx or eY not in el_to_idx:
            continue
        iX, iY = el_to_idx[eX], el_to_idx[eY]
        # X₃Y: d_nn = a/√2
        equations.append((iX, iY, row["d_nn_X3Y"], "X3Y"))
        targets.append(row["d_nn_X3Y"])
        # Y₃X: d_nn = a/√2
        equations.append((iX, iY, row["d_nn_Y3X"], "Y3X"))
        targets.append(row["d_nn_Y3X"])
        # B2: d_nn = a√3/2
        equations.append((iX, iY, row["d_nn_B2"], "B2"))
        targets.append(row["d_nn_B2"])

    A_mat = np.zeros((len(equations), n_el))
    b_vec = np.array(targets)
    struct_labels = []
    for i, (iX, iY, _, s) in enumerate(equations):
        A_mat[i, iX] = 1.0
        A_mat[i, iY] = 1.0
        struct_labels.append(s)

    struct_labels = np.array(struct_labels)

    # Pure element r for initial guess
    x0 = np.array([r_from_V(KING_ATOMIC_VOLUMES.get(e, 15)) for e in all_elements])
    res_single = least_squares(lambda x: A_mat @ x - b_vec, x0, bounds=(0.5, 3.0))
    r_single = {e: res_single.x[i] for i, e in enumerate(all_elements)}
    residuals_single = res_single.fun
    rmse_single = np.sqrt(np.mean(residuals_single**2))

    # Per-structure RMSE
    for s in ["X3Y", "Y3X", "B2"]:
        mask = struct_labels == s
        rmse_s = np.sqrt(np.mean(residuals_single[mask]**2))
        print(f"  Single-set packing RMSE ({s}): {rmse_s:.4f} Å")
    print(f"  Single-set packing RMSE (all): {rmse_single:.4f} Å")

    # ── 4. Structure-specific least-squares: separate r per structure ──
    # 3*n_el unknowns: r_X3Y_maj[i], r_Y3X_min[i], r_B2[i]
    # But this is complex. Instead fit separately per structure type.
    for s, label in [("X3Y", "L1₂ X₃Y"), ("Y3X", "L1₂ Y₃X"), ("B2", "B2")]:
        mask = struct_labels == s
        A_s = A_mat[mask]
        b_s = b_vec[mask]
        res_s = least_squares(lambda x: A_s @ x - b_s, x0, bounds=(0.5, 3.0))
        rmse_s = np.sqrt(np.mean(res_s.fun**2))
        print(f"  Structure-specific packing RMSE ({label}): {rmse_s:.4f} Å")

    # ── 5. The KEY question: can packing radii predict lattice constants? ──
    # Use single-set packing radii to predict a for each structure
    pred_results = []
    for _, row in df.iterrows():
        eX, eY = row["elX"], row["elY"]
        if eX not in r_single or eY not in r_single:
            continue
        rX = r_single[eX]
        rY = r_single[eY]
        d_nn_pred = rX + rY

        # Predict a from d_nn
        a_pred_l12 = d_nn_pred * np.sqrt(2)  # L1₂: a = d_nn * √2
        a_pred_b2 = d_nn_pred * 2 / np.sqrt(3)  # B2: a = d_nn * 2/√3

        pred_results.append({
            "elX": eX, "elY": eY,
            "a_X3Y_DFT": row["a_X3Y"], "a_Y3X_DFT": row["a_Y3X"], "a_B2_DFT": row["a_B2"],
            "a_X3Y_pred": a_pred_l12, "a_Y3X_pred": a_pred_l12, "a_B2_pred": a_pred_b2,
            "err_X3Y": a_pred_l12 - row["a_X3Y"],
            "err_Y3X": a_pred_l12 - row["a_Y3X"],
            "err_B2": a_pred_b2 - row["a_B2"],
        })

    pred_df = pd.DataFrame(pred_results)

    print("\n=== パッキング半径（単一セット）による格子定数予測 ===")
    print(f"  RMSE a(X₃Y): {np.sqrt(np.mean(pred_df['err_X3Y']**2)):.4f} Å")
    print(f"  RMSE a(Y₃X): {np.sqrt(np.mean(pred_df['err_Y3X']**2)):.4f} Å")
    print(f"  RMSE a(B2):  {np.sqrt(np.mean(pred_df['err_B2']**2)):.4f} Å")

    # Critical: single packing radii predict a(X₃Y) = a(Y₃X) ALWAYS
    # because d_nn = r_X + r_Y is symmetric
    print(f"\n  a(X₃Y)_pred == a(Y₃X)_pred: ALWAYS (r_X + r_Y is symmetric)")
    print(f"  But DFT: mean|a(X₃Y) - a(Y₃X)| = {(pred_df['a_X3Y_DFT'] - pred_df['a_Y3X_DFT']).abs().mean():.4f} Å")
    print(f"  → 単一パッキング半径はL1₂非対称性を原理的に説明できない")

    # ── 6. FIGURES ──
    fig = plt.figure(figsize=(28, 20))

    # (a) d_nn consistency: X₃Y vs Y₃X
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.scatter(df["d_nn_X3Y"], df["d_nn_Y3X"], s=12, alpha=0.6, c="steelblue")
    lim = [df[["d_nn_X3Y","d_nn_Y3X"]].min().min()*0.95,
           df[["d_nn_X3Y","d_nn_Y3X"]].max().max()*1.02]
    ax1.plot(lim, lim, "k--", lw=1)
    ax1.set_xlabel(r"$d_{\rm nn}$ (L1$_2$ X$_3$Y) [Å]")
    ax1.set_ylabel(r"$d_{\rm nn}$ (L1$_2$ Y$_3$X) [Å]")
    ax1.set_title(r"(a) 最近接距離: X$_3$Y vs Y$_3$X", fontweight="bold")
    r_dnn = df["d_nn_X3Y"].corr(df["d_nn_Y3X"])
    mean_diff_dnn = df["d_nn_diff_L12"].abs().mean()
    ax1.text(0.05, 0.92, f"r = {r_dnn:.3f}\nmean|Δ| = {mean_diff_dnn:.3f} Å\n"
             f"d_nn は構造で異なる",
             transform=ax1.transAxes, fontsize=12, color="navy",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#e0e8ff", edgecolor="navy", alpha=0.9))

    # (b) r_X: majority (from X₃Y) vs minority (from Y₃X)
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.scatter(df["r_X_from_X3Y"], df["r_X_from_Y3X"], s=12, alpha=0.6, c="crimson")
    rlim = [0.8, 2.5]
    ax2.plot(rlim, rlim, "k--", lw=1)
    ax2.set_xlabel(r"$r_X^{\rm pack}$ from X$_3$Y (X=majority) [Å]")
    ax2.set_ylabel(r"$r_X^{\rm pack}$ from Y$_3$X (X=minority) [Å]")
    ax2.set_title(r"(b) パッキング半径 $r_X$: majority vs minority", fontweight="bold")
    ax2.set_xlim(rlim); ax2.set_ylim(rlim)
    mean_dr = df["dr_X_maj_vs_min"].abs().mean()
    ax2.text(0.05, 0.92, f"mean|Δr| = {mean_dr:.3f} Å\n→ 構造によって\nパッキング半径も変わる",
             transform=ax2.transAxes, fontsize=12, color="red",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffe0e0", edgecolor="red", alpha=0.9))

    # (c) Predicted a(X₃Y) vs a(Y₃X) from single packing set
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.scatter(pred_df["a_X3Y_DFT"], pred_df["a_Y3X_DFT"], s=12, alpha=0.5,
                c="darkorange", label="DFT (実際)")
    ax3.scatter(pred_df["a_X3Y_pred"], pred_df["a_Y3X_pred"], s=12, alpha=0.5,
                c="blue", label="Packing pred (予測)", marker="x")
    alim = [2.5, 7.5]
    ax3.plot(alim, alim, "k--", lw=1)
    ax3.set_xlabel(r"$a$ (X$_3$Y) [Å]")
    ax3.set_ylabel(r"$a$ (Y$_3$X) [Å]")
    ax3.set_title("(c) 単一パッキング半径の限界", fontweight="bold")
    ax3.legend(fontsize=11)
    ax3.text(0.05, 0.92,
             "パッキング半径 → 必ず\n$a(X_3Y)_{pred} = a(Y_3X)_{pred}$\n"
             "(対角線上に拘束)\n\n"
             "DFT → $a(X_3Y) \\neq a(Y_3X)$\n(大きくずれる)",
             transform=ax3.transAxes, fontsize=11, color="darkred",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff0e0", edgecolor="darkorange", alpha=0.9))

    # (d) Histogram of packing residuals per structure
    ax4 = fig.add_subplot(2, 3, 4)
    for s, color, label in [("X3Y", "green", r"L1$_2$ X$_3$Y"),
                             ("Y3X", "blue", r"L1$_2$ Y$_3$X"),
                             ("B2", "red", "B2")]:
        mask = struct_labels == s
        ax4.hist(residuals_single[mask], bins=60, alpha=0.5, color=color,
                 label=label, edgecolor="black", lw=0.3)
    ax4.axvline(0, color="black", lw=2)
    ax4.set_xlabel(r"$d_{\rm nn}^{\rm pred} - d_{\rm nn}^{\rm DFT}$ [Å]")
    ax4.set_ylabel("Count")
    ax4.set_title("(d) 単一パッキング半径の残差分布", fontweight="bold")
    ax4.legend()
    ax4.text(0.05, 0.92, f"RMSE = {rmse_single:.4f} Å\n→ 構造間の\n不整合が大きい",
             transform=ax4.transAxes, fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#e0ffe0", edgecolor="green", alpha=0.9))

    # (e) Conceptual: why packing radius also fails for L1₂ asymmetry
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.set_xlim(0, 10); ax5.set_ylim(0, 10); ax5.axis("off")
    ax5.set_title("(e) パッキング半径の本質的限界", fontweight="bold")

    box_blue = dict(boxstyle="round,pad=0.4", facecolor="#d0e8ff", edgecolor="navy", lw=2)
    box_red = dict(boxstyle="round,pad=0.4", facecolor="#ffd0d0", edgecolor="red", lw=2)
    box_green = dict(boxstyle="round,pad=0.4", facecolor="#d0ffd0", edgecolor="green", lw=2)

    ax5.text(5, 9, "パッキング制約\n$r_X + r_Y = d_{\\rm nn}(構造)$",
             fontsize=13, ha="center", va="center", bbox=box_blue)
    ax5.text(2.5, 6.5,
             "単一半径セット $(r_X, r_Y)$ で\n"
             "全構造を説明しようとすると:",
             fontsize=12, ha="center", va="center")
    ax5.text(5, 4.5,
             "$d_{\\rm nn}(X_3Y) = d_{\\rm nn}(Y_3X)$ を強制\n"
             "→ $a(X_3Y) = a(Y_3X)$ を予測\n"
             "しかし実際は $a(X_3Y) \\neq a(Y_3X)$",
             fontsize=12, ha="center", va="center", bbox=box_red)
    ax5.text(5, 2,
             "結論: 単一パッキング半径では\nL1₂非対称性を説明不可能",
             fontsize=14, ha="center", va="center", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff0f0", edgecolor="red", lw=3))

    # (f) Comparison: volume-derived vs packing-derived
    ax6 = fig.add_subplot(2, 3, 6)
    # Use element-level comparison
    eff_radii = pd.read_csv(OUT / "effective_radii_by_structure.csv")
    common = [e for e in eff_radii["Element"] if e in r_single]
    r_pack = [r_single[e] for e in common]
    r_pure = [eff_radii[eff_radii["Element"]==e]["r_pure"].values[0] for e in common]
    r_vol_maj = [eff_radii[eff_radii["Element"]==e]["r_L12_maj"].values[0] for e in common]
    r_vol_min = [eff_radii[eff_radii["Element"]==e]["r_L12_min"].values[0] for e in common]

    ax6.scatter(r_pure, r_pack, s=50, c="purple", marker="s", label="Packing (単一)", alpha=0.7)
    ax6.scatter(r_pure, r_vol_maj, s=50, c="blue", marker="o", label="Volume (majority)", alpha=0.7)
    ax6.scatter(r_pure, r_vol_min, s=50, c="orange", marker="^", label="Volume (minority)", alpha=0.7)
    rlim2 = [0.8, 2.4]
    ax6.plot(rlim2, rlim2, "k--", lw=1)
    ax6.set_xlabel(r"$r_{\rm pure}$ (King) [Å]")
    ax6.set_ylabel(r"$r_{\rm eff}$ [Å]")
    ax6.set_title("(f) 純元素半径 vs 各手法の有効半径", fontweight="bold")
    ax6.legend(fontsize=10)
    ax6.text(0.05, 0.92,
             "Packing: 1点のみ\nVolume: maj ≠ min\n→ 体積法のみ\n構造依存性を保持",
             transform=ax6.transAxes, fontsize=11,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0e0ff", edgecolor="purple", alpha=0.9))

    fig.suptitle(
        "DFTパッキング制約付き有効半径は構造情報を吸収できるか？",
        fontsize=20, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fig_path = OUT / "fig_packing_radius_analysis.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {fig_path}")

    # Save as base64 for HTML
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close(fig)

    # ── HTML output ──
    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>DFTパッキング制約付き有効半径の構造情報吸収能力</title>
<script>
  MathJax = {{ tex: {{ inlineMath: [['$','$']] }}, svg: {{ fontCache: 'global' }} }};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
<style>
  body {{
    font-family: "Hiragino Kaku Gothic Pro", "Yu Gothic", "Meiryo", sans-serif;
    max-width: 1200px; margin: 0 auto; padding: 25px;
    line-height: 1.8; color: #222; background: #fafaf8;
  }}
  h1 {{ font-size: 26px; color: #1a3a5c; border-bottom: 3px solid #1a3a5c; padding-bottom: 10px; }}
  h2 {{ font-size: 20px; color: #2c5f8a; margin-top: 30px; border-left: 5px solid #2c5f8a; padding-left: 10px; }}
  h3 {{ font-size: 17px; color: #3a7ab5; margin-top: 20px; }}
  .fig {{ width: 100%; max-width: 1200px; margin: 15px 0; border: 1px solid #ccc; }}
  .caption {{ text-align: center; font-size: 13px; color: #555; margin: 5px 0 20px; }}
  .key {{ background: #fff3cd; border-left: 5px solid #ffc107; padding: 12px 16px; margin: 15px 0; border-radius: 0 6px 6px 0; }}
  .red {{ background: #f8d7da; border-left: 5px solid #dc3545; padding: 12px 16px; margin: 15px 0; border-radius: 0 6px 6px 0; }}
  .green {{ background: #d4edda; border-left: 5px solid #28a745; padding: 12px 16px; margin: 15px 0; border-radius: 0 6px 6px 0; }}
  .eq {{ background: #eef4fa; border: 1px solid #b0ccdf; border-radius: 8px; padding: 12px 18px; margin: 12px 0; text-align: center; font-size: 17px; }}
  table {{ border-collapse: collapse; margin: 10px 0; font-size: 14px; }}
  th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: center; }}
  th {{ background: #e8eef4; }}
</style>
</head>
<body>

<h1>DFTパッキング制約付き有効半径は構造情報を吸収できるか？</h1>

<p style="text-align:right; color:#666;">905元素ペア / 全3構造 (L1$_2$ X$_3$Y, Y$_3$X, B2 XY)</p>

<h2>1. 問いの定式化</h2>

<p>$\\delta r$（純元素半径）は構造情報を含まないことを前回証明した。
では、<strong>DFT格子定数からパッキング制約で求めた有効半径</strong>はどうか？</p>

<div class="eq">
L1$_2$: $d_{{\\rm nn}} = a / \\sqrt{{2}}$, 接触条件: $r_X + r_Y = d_{{\\rm nn}}$<br>
B2: $d_{{\\rm nn}} = a\\sqrt{{3}} / 2$, 接触条件: $r_X + r_Y = d_{{\\rm nn}}$
</div>

<p>DFT格子定数は構造情報を含む → パッキング半径もDFT由来 → 構造情報を吸収できるのでは？</p>

<h2>2. 結果</h2>

<img class="fig" src="data:image/png;base64,{b64}" alt="パッキング半径分析">
<p class="caption">Figure 1. DFTパッキング制約付き有効半径の構造情報吸収能力の検証</p>

<h3>2.1 パッキング半径の構造間不整合（パネル a, b）</h3>

<p>DFT最近接距離 $d_{{\\rm nn}}$ 自体が構造によって異なる（パネル a: mean|Δ| = {mean_diff_dnn:.3f} Å）。
したがって、同じ元素Xのパッキング半径も、Xが多数派のとき(X$_3$Y)と少数派のとき(Y$_3$X)で異なる
（パネル b: mean|Δr| = {mean_dr:.3f} Å）。</p>

<div class="key">
<strong>重要:</strong> DFT格子定数からパッキング半径を構造ごとに独立に求めれば、
構造情報は<strong>吸収される</strong>（DFT由来なので当然）。
問題は「<strong>単一の半径セット</strong>で全構造を説明できるか」である。
</div>

<h3>2.2 単一パッキング半径セットの限界（パネル c, d, e）</h3>

<p>全構造の最近接距離を一つの $(r_X, r_Y)$ で最小二乗フィットすると:</p>

<table>
<tr><th>構造</th><th>RMSE [Å]</th></tr>
<tr><td>L1$_2$ X$_3$Y</td><td>{np.sqrt(np.mean(residuals_single[struct_labels=="X3Y"]**2)):.4f}</td></tr>
<tr><td>L1$_2$ Y$_3$X</td><td>{np.sqrt(np.mean(residuals_single[struct_labels=="Y3X"]**2)):.4f}</td></tr>
<tr><td>B2</td><td>{np.sqrt(np.mean(residuals_single[struct_labels=="B2"]**2)):.4f}</td></tr>
<tr><td><strong>全体</strong></td><td><strong>{rmse_single:.4f}</strong></td></tr>
</table>

<div class="red">
<strong>致命的な限界:</strong> 単一パッキング半径 $(r_X, r_Y)$ からは $d_{{\\rm nn}} = r_X + r_Y$ が一意に決まる。
これはX$_3$YとY$_3$Xで<strong>同じ値</strong>になるため、$a(X_3Y) = a(Y_3X)$ を予測する。
しかし実際は $|a(X_3Y) - a(Y_3X)|$ の平均が {(pred_df['a_X3Y_DFT'] - pred_df['a_Y3X_DFT']).abs().mean():.3f} Å（パネル c）。
<br><br>
<strong>根本原因:</strong> $r_X + r_Y$ は加法的・対称的操作であり、
XとYの「役割」（多数派/少数派）の情報が消失する。
</div>

<h3>2.3 体積由来の半径との比較（パネル f）</h3>

<p>体積由来の有効半径は、majority / minority で<strong>異なる値</strong>を持つ（パネル f: 青丸 vs 橙三角が分離）。
パッキング半径（紫四角）は構造によらず<strong>1点のみ</strong>。</p>

<h2>3. なぜパッキング半径では不十分か</h2>

<div class="eq">
パッキング: $d_{{\\rm nn}} = r_X + r_Y$ → 対称操作 → $d_{{\\rm nn}}(X_3Y) = d_{{\\rm nn}}(Y_3X)$ を強制
</div>

<div class="eq">
体積: $V = 0.75 \\cdot V_{{\\rm maj}}(X) + 0.25 \\cdot V_{{\\rm min}}(Y)$ → 非対称 → $V(X_3Y) \\neq V(Y_3X)$ を許容
</div>

<p>パッキングモデルの問題は、最近接距離が「元素対の和」として記述される点にある。
$r_X + r_Y$ は $(X,Y)$ の順序に依存しない対称量であり、
XがmajorityかminorityかというL1$_2$の<strong>役割の非対称性</strong>を表現できない。</p>

<p>一方、体積モデルでは $V_{{\\rm maj}}(X) \\neq V_{{\\rm min}}(X)$ が許容されるため、
同じ元素でも役割が変われば異なる体積（半径）を取り得る。</p>

<h2>4. 構造ごとにパッキング半径を分けたらどうか？</h2>

<p>構造ごとに独立な半径セットを用意すれば（$r_X^{{\\rm maj}}$, $r_X^{{\\rm min}}$, $r_X^{{\\rm B2}}$）、
構造情報は吸収できる。しかしこれは実質的に<strong>体積由来の半径と同等</strong>になる:</p>

<ul>
<li>どちらもDFT格子定数から導出</li>
<li>どちらも構造×元素のマトリクスパラメータ</li>
<li>パッキングモデルは追加の幾何学的仮定（球の接触）を置くため、
    体積モデルより制約が強く、フィットの自由度が低い</li>
</ul>

<div class="green">
<strong>結論:</strong><br>
<strong>構造ごとに分けたパッキング半径 → 可能だが体積法と本質的に同等</strong><br>
<strong>単一パッキング半径 → L1$_2$非対称性を原理的に説明不可能</strong><br><br>
体積由来の半径は、追加の幾何学的仮定なしに構造情報を自然に吸収でき、
かつmajority/minority の役割の非対称性も表現可能。
したがって、<strong>原子体積からの有効半径が最も情報量の多い記述子</strong>である。
</div>

</body>
</html>"""

    html_path = OUT / "fig_packing_radius_analysis.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved HTML: {html_path}")


if __name__ == "__main__":
    main()
