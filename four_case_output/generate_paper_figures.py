#!/usr/bin/env python3
"""
Generate all figures for the IMRAD report from current 4-case comparison data.
Figures match the filenames referenced in imrad_report_elsevier.tex.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Reference radii ──
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

GOLDSCHMIDT_RADII = {
    "Li": 1.57, "Be": 1.12, "Na": 1.91, "Mg": 1.60, "Al": 1.43, "K": 2.38,
    "Ca": 1.97, "Sc": 1.64, "Ti": 1.47, "V": 1.36, "Cr": 1.28, "Mn": 1.31,
    "Fe": 1.27, "Co": 1.25, "Ni": 1.25, "Cu": 1.28, "Zn": 1.37, "Ga": 1.41,
    "Ge": 1.37, "Rb": 2.51, "Sr": 2.15, "Y": 1.81, "Zr": 1.60, "Nb": 1.47,
    "Mo": 1.40, "Tc": 1.36, "Ru": 1.34, "Rh": 1.34, "Pd": 1.37, "Ag": 1.44,
    "Cd": 1.52, "In": 1.67, "Sn": 1.58, "Sb": 1.61, "Cs": 2.70, "Ba": 2.24,
    "La": 1.88, "Ce": 1.82, "Pr": 1.83, "Nd": 1.82, "Sm": 1.80, "Eu": 2.04,
    "Gd": 1.80, "Tb": 1.78, "Dy": 1.77, "Ho": 1.76, "Er": 1.75, "Tm": 1.74,
    "Yb": 1.93, "Lu": 1.74, "Hf": 1.59, "Ta": 1.47, "W": 1.41, "Re": 1.37,
    "Os": 1.35, "Ir": 1.36, "Pt": 1.39, "Au": 1.44, "Hg": 1.55, "Tl": 1.71,
    "Pb": 1.75, "Bi": 1.82, "Th": 1.80, "Pa": 1.63, "U": 1.54,
}

# ── Load data ──
radii, compounds = {}, {}
for key in ["MP_B2", "MP_L12", "OQMD_B2", "OQMD_L12"]:
    radii[key] = pd.read_csv(f"{DATA_DIR}/radii_{key}.csv")
    compounds[key] = pd.read_csv(f"{DATA_DIR}/compounds_{key}.csv")

def get_radii_dict(key):
    df = radii[key]
    return dict(zip(df["element"], df["radius"]))


# ═══════════════════════════════════════════════════════════
# Figure 1: gs_03_pauling_vs_goldschmidt.png
# ═══════════════════════════════════════════════════════════
def fig_pauling_vs_goldschmidt():
    common = sorted(set(PAULING_RADII) & set(GOLDSCHMIDT_RADII))
    p = [PAULING_RADII[e] for e in common]
    g = [GOLDSCHMIDT_RADII[e] for e in common]
    alkali = ["Li", "Na", "K", "Rb", "Cs"]
    fig, ax = plt.subplots(figsize=(7, 7))
    px_na, py_na, px_a, py_a, labels_a = [], [], [], [], []
    for e, pi, gi in zip(common, p, g):
        if e in alkali:
            px_a.append(gi); py_a.append(pi); labels_a.append(e)
        else:
            px_na.append(gi); py_na.append(pi)
    ax.scatter(px_na, py_na, c="steelblue", s=50, alpha=0.7, label="Other elements")
    ax.scatter(px_a, py_a, c="red", s=80, marker="^", label="Alkali metals", zorder=5)
    for e, x, y in zip(labels_a, px_a, py_a):
        ax.annotate(e, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=11)
    mn, mx = 0.8, 2.8
    ax.plot([mn, mx], [mn, mx], "k--", alpha=0.5, label="Perfect agreement")
    slope, intercept, r, _, _ = stats.linregress(g, p)
    rmse = np.sqrt(np.mean((np.array(p) - np.array(g))**2))
    ax.set_xlabel("Goldschmidt radius (\u00c5)")
    ax.set_ylabel("Pauling radius (\u00c5)")
    ax.set_title("Pauling vs Goldschmidt Metallic Radii")
    ax.legend(loc="upper left")
    ax.text(0.95, 0.05, f"$R^2$ = {r**2:.3f}\nRMSE = {rmse:.3f} \u00c5\nN = {len(common)}",
            transform=ax.transAxes, ha="right", va="bottom",
            bbox=dict(boxstyle="round", fc="wheat", alpha=0.8))
    ax.set_xlim(mn, mx); ax.set_ylim(mn, mx)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/gs_03_pauling_vs_goldschmidt.png", dpi=200)
    plt.close(fig)
    print("  gs_03_pauling_vs_goldschmidt.png")


# ═══════════════════════════════════════════════════════════
# Figure 2: 00a_dataset_size.png
# ═══════════════════════════════════════════════════════════
def fig_dataset_size():
    labels = ["MP\nB2", "MP\nL1$_2$", "OQMD\nB2", "OQMD\nL1$_2$"]
    counts = [len(compounds["MP_B2"]), len(compounds["MP_L12"]),
              len(compounds["OQMD_B2"]), len(compounds["OQMD_L12"])]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, counts, color=colors, edgecolor="black", linewidth=0.5)
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                str(c), ha="center", va="bottom", fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of compounds")
    ax.set_title("Dataset Size Comparison")
    ax.set_ylim(0, max(counts) * 1.15)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/00a_dataset_size.png", dpi=200)
    plt.close(fig)
    print("  00a_dataset_size.png")


# ═══════════════════════════════════════════════════════════
# Figure 3: 22_structure_3d_multiview.png
# ═══════════════════════════════════════════════════════════
def fig_structure_3d():
    fig = plt.figure(figsize=(14, 6))
    # --- B2 unit cell ---
    ax1 = fig.add_subplot(121, projection="3d")
    corners = np.array([[i, j, k] for i in [0,1] for j in [0,1] for k in [0,1]], dtype=float)
    ax1.scatter(*corners.T, s=200, c="royalblue", edgecolors="black", linewidth=0.5, label="A (corner)", zorder=5)
    ax1.scatter([0.5], [0.5], [0.5], s=250, c="crimson", edgecolors="black", linewidth=0.5, label="B (body center)", zorder=5)
    for i in range(2):
        for j in range(2):
            ax1.plot([i, i], [j, j], [0, 1], "gray", alpha=0.4, linewidth=0.8)
            ax1.plot([i, i], [0, 1], [j, j], "gray", alpha=0.4, linewidth=0.8)
            ax1.plot([0, 1], [i, i], [j, j], "gray", alpha=0.4, linewidth=0.8)
    ax1.plot([0, 0.5], [0, 0.5], [0, 0.5], "k--", alpha=0.6, linewidth=1.5)
    ax1.set_title("B2 (CsCl-type)\nCN = 8", fontsize=16)
    ax1.legend(loc="upper left", fontsize=10)
    ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")
    ax1.set_xlim(-0.2, 1.2); ax1.set_ylim(-0.2, 1.2); ax1.set_zlim(-0.2, 1.2)
    # --- L1_2 unit cell ---
    ax2 = fig.add_subplot(122, projection="3d")
    l12_corners = np.array([[i, j, k] for i in [0,1] for j in [0,1] for k in [0,1]], dtype=float)
    ax2.scatter(*l12_corners.T, s=200, c="crimson", edgecolors="black", linewidth=0.5, label="B (corner)", zorder=5)
    face_centers = np.array([
        [0.5, 0.5, 0], [0.5, 0.5, 1],
        [0.5, 0, 0.5], [0.5, 1, 0.5],
        [0, 0.5, 0.5], [1, 0.5, 0.5],
    ], dtype=float)
    ax2.scatter(*face_centers.T, s=250, c="royalblue", edgecolors="black", linewidth=0.5, label="A (face center)", zorder=5)
    for i in range(2):
        for j in range(2):
            ax2.plot([i, i], [j, j], [0, 1], "gray", alpha=0.4, linewidth=0.8)
            ax2.plot([i, i], [0, 1], [j, j], "gray", alpha=0.4, linewidth=0.8)
            ax2.plot([0, 1], [i, i], [j, j], "gray", alpha=0.4, linewidth=0.8)
    ax2.plot([0.5, 0], [0.5, 0], [0, 0.5], "b--", alpha=0.5, linewidth=1.5)
    ax2.set_title("L1$_2$ (Cu$_3$Au-type)\nCN = 12", fontsize=16)
    ax2.legend(loc="upper left", fontsize=10)
    ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("z")
    ax2.set_xlim(-0.2, 1.2); ax2.set_ylim(-0.2, 1.2); ax2.set_zlim(-0.2, 1.2)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/22_structure_3d_multiview.png", dpi=200)
    plt.close(fig)
    print("  22_structure_3d_multiview.png")


# ═══════════════════════════════════════════════════════════
# Figure 4: 12_model_performance.png  (compute RMSE from data)
# ═══════════════════════════════════════════════════════════
def fig_model_performance():
    rmse_vals = []
    for key in ["MP_B2", "MP_L12", "OQMD_B2", "OQMD_L12"]:
        df = compounds[key]
        obs = df["lattice_constant"].values
        pred = df["lattice_constant_calc"].values
        rmse_vals.append(np.sqrt(np.mean((obs - pred)**2)))
    cases = ["MP\nB2", "MP\nL1$_2$", "OQMD\nB2", "OQMD\nL1$_2$"]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(cases, rmse_vals, color=colors, edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, rmse_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{v:.3f}", ha="center", va="bottom", fontsize=13)
    ax.set_ylabel("RMSE (\u00c5)")
    ax.set_title("Lattice Constant Prediction Accuracy")
    ax.set_ylim(0, max(rmse_vals) * 1.25)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/12_model_performance.png", dpi=200)
    plt.close(fig)
    print("  12_model_performance.png")


# ═══════════════════════════════════════════════════════════
# Figure 5 & 6: 03_B2_parity_lattice.png & 04_L12_parity_lattice.png
# ═══════════════════════════════════════════════════════════
def fig_parity_plots():
    configs = [
        ("MP_B2", "MP B2", "03_B2_parity_lattice.png"),
        ("MP_L12", "MP L1$_2$", "04_L12_parity_lattice.png"),
    ]
    for key, title_str, fname in configs:
        df = compounds[key]
        obs = df["lattice_constant"].values
        pred = df["lattice_constant_calc"].values
        r2 = 1 - np.sum((obs - pred)**2) / np.sum((obs - np.mean(obs))**2)
        rmse = np.sqrt(np.mean((obs - pred)**2))
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(obs, pred, s=25, alpha=0.6, c="steelblue", edgecolors="none")
        mn = min(obs.min(), pred.min()) - 0.1
        mx = max(obs.max(), pred.max()) + 0.1
        ax.plot([mn, mx], [mn, mx], "k--", alpha=0.5)
        ax.set_xlabel("DFT lattice constant (\u00c5)")
        ax.set_ylabel("Predicted lattice constant (\u00c5)")
        ax.set_title(f"{title_str}: Observed vs Predicted")
        ax.text(0.05, 0.95, f"$R^2$ = {r2:.3f}\nRMSE = {rmse:.3f} \u00c5\nN = {len(df)}",
                transform=ax.transAxes, ha="left", va="top",
                bbox=dict(boxstyle="round", fc="wheat", alpha=0.8))
        ax.set_xlim(mn, mx); ax.set_ylim(mn, mx)
        ax.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(f"{OUT_DIR}/{fname}", dpi=200)
        plt.close(fig)
        print(f"  {fname}")


# ═══════════════════════════════════════════════════════════
# Figure 7: gs_02_L12_vs_goldschmidt.png
# ═══════════════════════════════════════════════════════════
def fig_l12_vs_goldschmidt():
    rd = get_radii_dict("MP_L12")
    common = sorted(set(rd) & set(GOLDSCHMIDT_RADII))
    opt = [rd[e] for e in common]
    gs = [GOLDSCHMIDT_RADII[e] for e in common]
    slope, intercept, r, _, _ = stats.linregress(gs, opt)
    rmse = np.sqrt(np.mean((np.array(opt) - np.array(gs))**2))
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(gs, opt, s=50, c="green", alpha=0.7, edgecolors="none")
    for e, x, y in zip(common, gs, opt):
        if e in ["Li", "Na", "K", "Rb", "Cs", "Eu", "Ba"]:
            ax.annotate(e, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=10)
    mn, mx = 0.8, 2.8
    ax.plot([mn, mx], [mn, mx], "k--", alpha=0.5, label="Perfect agreement")
    ax.set_xlabel("Goldschmidt radius (\u00c5)")
    ax.set_ylabel("Optimized L1$_2$ radius (\u00c5)")
    ax.set_title("L1$_2$ Radii vs Goldschmidt Radii (MP)")
    ax.text(0.95, 0.05, f"$R^2$ = {r**2:.3f}\nRMSE = {rmse:.3f} \u00c5\nN = {len(common)}",
            transform=ax.transAxes, ha="right", va="bottom",
            bbox=dict(boxstyle="round", fc="wheat", alpha=0.8))
    ax.set_xlim(mn, mx); ax.set_ylim(mn, mx)
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/gs_02_L12_vs_goldschmidt.png", dpi=200)
    plt.close(fig)
    print("  gs_02_L12_vs_goldschmidt.png")


# ═══════════════════════════════════════════════════════════
# Figure 8: 05_B2_vs_pauling.png
# ═══════════════════════════════════════════════════════════
def fig_b2_vs_pauling():
    rd = get_radii_dict("MP_B2")
    common = sorted(set(rd) & set(PAULING_RADII))
    opt = [rd[e] for e in common]
    paul = [PAULING_RADII[e] for e in common]
    alkali_set = {"K", "Rb", "Cs"}
    fig, ax = plt.subplots(figsize=(7, 7))
    ox_n, oy_n, ox_o, oy_o, labels_o = [], [], [], [], []
    for e, x, y in zip(common, paul, opt):
        if e in alkali_set:
            ox_o.append(x); oy_o.append(y); labels_o.append(e)
        else:
            ox_n.append(x); oy_n.append(y)
    ax.scatter(ox_n, oy_n, s=50, c="steelblue", alpha=0.7, label="Other elements")
    ax.scatter(ox_o, oy_o, s=80, c="red", marker="^", label="Alkali metals (K, Rb, Cs)", zorder=5)
    for e, x, y in zip(labels_o, ox_o, oy_o):
        ax.annotate(e, (x, y), textcoords="offset points", xytext=(5, -10), fontsize=11, color="red")
    mn, mx = 0.8, 2.8
    ax.plot([mn, mx], [mn, mx], "k--", alpha=0.5, label="Perfect agreement")
    slope, intercept, r_all, _, _ = stats.linregress(paul, opt)
    paul_filt = [x for e, x in zip(common, paul) if e not in alkali_set]
    opt_filt = [y for e, y in zip(common, opt) if e not in alkali_set]
    _, _, r_filt, _, _ = stats.linregress(paul_filt, opt_filt)
    ax.set_xlabel("Pauling radius (\u00c5)")
    ax.set_ylabel("Optimized B2 radius (\u00c5)")
    ax.set_title("B2 Radii vs Pauling Radii (MP)")
    ax.text(0.95, 0.05,
            f"All: $R^2$ = {r_all**2:.3f}\nExcl. alkali: $R^2$ = {r_filt**2:.3f}\nN = {len(common)}",
            transform=ax.transAxes, ha="right", va="bottom",
            bbox=dict(boxstyle="round", fc="wheat", alpha=0.8))
    ax.set_xlim(mn, mx); ax.set_ylim(mn, mx)
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/05_B2_vs_pauling.png", dpi=200)
    plt.close(fig)
    print("  05_B2_vs_pauling.png")


# ═══════════════════════════════════════════════════════════
# Figure 9: 07_B2_vs_L12_scatter.png
# ═══════════════════════════════════════════════════════════
def fig_b2_vs_l12_scatter():
    b2 = get_radii_dict("MP_B2")
    l12 = get_radii_dict("MP_L12")
    common = sorted(set(b2) & set(l12))
    alkali = {"Li", "Na", "K", "Rb", "Cs"}
    fig, ax = plt.subplots(figsize=(7, 7))
    xn, yn, xa, ya, la = [], [], [], [], []
    for e in common:
        if e in alkali:
            xa.append(l12[e]); ya.append(b2[e]); la.append(e)
        else:
            xn.append(l12[e]); yn.append(b2[e])
    ax.scatter(xn, yn, s=40, c="steelblue", alpha=0.6, label="Other elements")
    ax.scatter(xa, ya, s=80, c="red", marker="^", label="Alkali metals", zorder=5)
    for e, x, y in zip(la, xa, ya):
        ax.annotate(e, (x, y), textcoords="offset points", xytext=(5, -10), fontsize=11, color="red")
    mn, mx = 0.8, 2.8
    ax.plot([mn, mx], [mn, mx], "k--", alpha=0.5, label="Equal radii")
    ax.plot([mn, mx], [mn*0.97, mx*0.97], "r:", alpha=0.5, label="3% contraction (Goldschmidt)")
    ax.set_xlabel("L1$_2$ radius (\u00c5)")
    ax.set_ylabel("B2 radius (\u00c5)")
    ax.set_title("B2 vs L1$_2$ Effective Radii (MP)")
    ax.set_xlim(mn, mx); ax.set_ylim(mn, mx)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/07_B2_vs_L12_scatter.png", dpi=200)
    plt.close(fig)
    print("  07_B2_vs_L12_scatter.png")


# ═══════════════════════════════════════════════════════════
# Figure 10: gs_05_alkali_comparison.png
# ═══════════════════════════════════════════════════════════
def fig_alkali_comparison():
    alkali = ["Li", "Na", "K", "Rb", "Cs"]
    b2 = get_radii_dict("MP_B2")
    l12 = get_radii_dict("MP_L12")
    avail = [e for e in alkali if e in b2 and e in l12]
    x = np.arange(len(avail))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 6))
    paul_vals = [PAULING_RADII.get(e, 0) for e in avail]
    gs_vals = [GOLDSCHMIDT_RADII.get(e, 0) for e in avail]
    b2_vals = [b2[e] for e in avail]
    l12_vals = [l12[e] for e in avail]
    ax.bar(x - 1.5*width, paul_vals, width, label="Pauling", color="gray", edgecolor="black", linewidth=0.5)
    ax.bar(x - 0.5*width, gs_vals, width, label="Goldschmidt", color="steelblue", edgecolor="black", linewidth=0.5)
    ax.bar(x + 0.5*width, b2_vals, width, label="B2 (this work)", color="crimson", edgecolor="black", linewidth=0.5)
    ax.bar(x + 1.5*width, l12_vals, width, label="L1$_2$ (this work)", color="forestgreen", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Element")
    ax.set_ylabel("Atomic radius (\u00c5)")
    ax.set_title("Alkali Metal Radii: Classical vs Optimized")
    ax.set_xticks(x)
    ax.set_xticklabels(avail)
    ax.legend()
    ax.set_ylim(0, max(paul_vals + gs_vals + l12_vals) * 1.15)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/gs_05_alkali_comparison.png", dpi=200)
    plt.close(fig)
    print("  gs_05_alkali_comparison.png")


# ═══════════════════════════════════════════════════════════
# Figure 11: gs_04_B2_contraction.png
# ═══════════════════════════════════════════════════════════
def fig_b2_contraction():
    b2 = get_radii_dict("MP_B2")
    l12 = get_radii_dict("MP_L12")
    common = sorted(set(b2) & set(l12))
    contractions, els = [], []
    for e in common:
        contr = (b2[e] - l12[e]) / l12[e] * 100
        contractions.append(contr)
        els.append(e)
    alkali = {"Li", "Na", "K", "Rb", "Cs"}
    fig, ax = plt.subplots(figsize=(9, 5))
    other_c = [c for e, c in zip(els, contractions) if e not in alkali]
    alkali_c = [c for e, c in zip(els, contractions) if e in alkali]
    ax.hist(other_c, bins=30, color="steelblue", alpha=0.7, label="Other elements", edgecolor="black", linewidth=0.5)
    ax.hist(alkali_c, bins=10, color="red", alpha=0.7, label="Alkali metals", edgecolor="black", linewidth=0.5)
    ax.axvline(x=-3, color="red", linestyle=":", linewidth=2, label="Goldschmidt 3% rule")
    ax.set_xlabel("B2$-$L1$_2$ radius change (%)")
    ax.set_ylabel("Number of elements")
    ax.set_title("Distribution of B2$-$L1$_2$ Radius Contraction (MP)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/gs_04_B2_contraction.png", dpi=200)
    plt.close(fig)
    print("  gs_04_B2_contraction.png")


# ═══════════════════════════════════════════════════════════
# Figure 12 & 13: a7_common_B2.png & a7_common_L12.png
# ═══════════════════════════════════════════════════════════
def fig_common_compounds():
    for struct, fname in [("B2", "a7_common_B2.png"), ("L12", "a7_common_L12.png")]:
        mp_df = compounds[f"MP_{struct}"]
        oqmd_df = compounds[f"OQMD_{struct}"]
        mp_dict = dict(zip(mp_df["formula"], mp_df["lattice_constant"]))
        oqmd_dict = dict(zip(oqmd_df["formula"], oqmd_df["lattice_constant"]))
        common_formulas = sorted(set(mp_dict) & set(oqmd_dict))
        if len(common_formulas) == 0:
            print(f"  {fname}: No common compounds found, skipping")
            continue
        mp_lc = [mp_dict[f] for f in common_formulas]
        oqmd_lc = [oqmd_dict[f] for f in common_formulas]
        slope, intercept, r, _, _ = stats.linregress(mp_lc, oqmd_lc)
        rmse = np.sqrt(np.mean((np.array(mp_lc) - np.array(oqmd_lc))**2))
        struct_label = "B2" if struct == "B2" else "L1$_2$"
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(mp_lc, oqmd_lc, s=30, alpha=0.6, c="steelblue", edgecolors="none")
        mn = min(min(mp_lc), min(oqmd_lc)) - 0.1
        mx = max(max(mp_lc), max(oqmd_lc)) + 0.1
        ax.plot([mn, mx], [mn, mx], "k--", alpha=0.5)
        ax.set_xlabel("MP lattice constant (\u00c5)")
        ax.set_ylabel("OQMD lattice constant (\u00c5)")
        ax.set_title(f"{struct_label}: MP vs OQMD Lattice Constants")
        ax.text(0.05, 0.95, f"$R^2$ = {r**2:.4f}\nRMSE = {rmse:.4f} \u00c5\nN = {len(common_formulas)}",
                transform=ax.transAxes, ha="left", va="top",
                bbox=dict(boxstyle="round", fc="wheat", alpha=0.8))
        ax.set_xlim(mn, mx); ax.set_ylim(mn, mx)
        ax.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(f"{OUT_DIR}/{fname}", dpi=200)
        plt.close(fig)
        print(f"  {fname}")


# ═══════════════════════════════════════════════════════════
# Figure 14: oqmd_mp_common_parity.png (2x2 panel)
# ═══════════════════════════════════════════════════════════
def fig_common_parity_2x2():
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    titles = ["MP B2", "MP L1$_2$", "OQMD B2", "OQMD L1$_2$"]
    keys = ["MP_B2", "MP_L12", "OQMD_B2", "OQMD_L12"]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    for idx, (key, title_str, color) in enumerate(zip(keys, titles, colors)):
        ax = axes[idx // 2][idx % 2]
        df = compounds[key]
        obs = df["lattice_constant"].values
        pred = df["lattice_constant_calc"].values
        r2 = 1 - np.sum((obs - pred)**2) / np.sum((obs - np.mean(obs))**2)
        rmse = np.sqrt(np.mean((obs - pred)**2))
        ax.scatter(obs, pred, s=15, alpha=0.5, c=color, edgecolors="none")
        mn = min(obs.min(), pred.min()) - 0.1
        mx = max(obs.max(), pred.max()) + 0.1
        ax.plot([mn, mx], [mn, mx], "k--", alpha=0.5)
        ax.set_xlabel("DFT lattice constant (\u00c5)")
        ax.set_ylabel("Predicted lattice constant (\u00c5)")
        ax.set_title(title_str)
        ax.text(0.05, 0.95, f"$R^2$ = {r2:.3f}\nRMSE = {rmse:.3f} \u00c5\nN = {len(df)}",
                transform=ax.transAxes, ha="left", va="top",
                bbox=dict(boxstyle="round", fc="wheat", alpha=0.8), fontsize=13)
        ax.set_xlim(mn, mx); ax.set_ylim(mn, mx)
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/oqmd_mp_common_parity.png", dpi=200)
    plt.close(fig)
    print("  oqmd_mp_common_parity.png")


# ═══════════════════════════════════════════════════════════
# Run all
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating paper figures...")
    fig_pauling_vs_goldschmidt()
    fig_dataset_size()
    fig_structure_3d()
    fig_model_performance()
    fig_parity_plots()
    fig_l12_vs_goldschmidt()
    fig_b2_vs_pauling()
    fig_b2_vs_l12_scatter()
    fig_alkali_comparison()
    fig_b2_contraction()
    fig_common_compounds()
    fig_common_parity_2x2()
    print(f"\nAll figures saved to {OUT_DIR}/")
    print("Files:")
    for f in sorted(os.listdir(OUT_DIR)):
        if f.endswith(".png"):
            print(f"  {f}")
