#!/usr/bin/env python3
"""Reproduce the Chen et al. (2023) external SQS cross-check.

Chen et al., Nat. Commun. 14, 2856 (2023), DOI: 10.5281/zenodo.7633180.
The 197 MB archive is downloaded from Zenodo only when the cache is absent.
"""
from pathlib import Path
import json
import hashlib
import os
import re
import stat
import sys
import urllib.request
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
PAPER = Path(__file__).resolve().parent
DATA = ROOT / "data" / "sqs_results.csv"
CACHE = Path(os.environ.get("CHEN2023_CACHE_DIR", "/home/ubuntu/chen2023_compare"))
URL = "https://zenodo.org/api/records/7633180/files/wch3n/mpea_stability-v0.1.zip/content"
DOI = "10.5281/zenodo.7633180"
KNOWN_ARCHIVE_SHA256 = "c2bc82b6f983fb3b9bc4e8e703829ba9a59fc70498b549ff9ca9a2ce89f665ae"
EXCLUDE = {"Gd", "Ce", "La", "Pr", "Nd", "Sm", "Eu", "Tb", "Dy", "Ho",
           "Er", "Tm", "Yb", "Lu", "Y"}
EN = {"Ag": 1.93, "Al": 1.61, "Au": 2.54, "Be": 1.57, "Ca": 1.00,
      "Co": 1.88, "Cr": 1.66, "Cu": 1.90, "Fe": 1.83, "Hf": 1.30,
      "Ir": 2.20, "Mg": 1.31, "Mn": 1.55, "Mo": 2.16, "Nb": 1.60,
      "Ni": 1.91, "Os": 2.20, "Pb": 2.33, "Pd": 2.20, "Pt": 2.28,
      "Re": 1.90, "Rh": 2.28, "Ru": 2.20, "Sc": 1.36, "Si": 1.90,
      "Sn": 1.96, "Ta": 1.50, "Ti": 1.54, "V": 1.63, "W": 2.36,
      "Zn": 1.65, "Zr": 1.33}
VEC = {"Ag": 11, "Al": 3, "Au": 11, "Be": 2, "Ca": 2, "Co": 9,
       "Cr": 6, "Cu": 11, "Fe": 8, "Hf": 4, "Ir": 9, "Mg": 2,
       "Mn": 7, "Mo": 6, "Nb": 5, "Ni": 10, "Os": 8, "Pb": 4,
       "Pd": 10, "Pt": 10, "Re": 7, "Rh": 9, "Ru": 8, "Sc": 3,
       "Si": 4, "Sn": 4, "Ta": 5, "Ti": 4, "V": 5, "W": 6,
       "Zn": 12, "Zr": 4}
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "font.size": 15})
LABELS = {
    "dh_chen": r"$\Delta H_{\mathrm{mix}}$ (eV/atom)",
    "dh_local": r"$\Delta H_{\mathrm{mix}}$ (eV/atom)",
    "omega_sf": r"$\Omega_{\mathrm{sf}}$",
    "omega_sf_dft": r"$\Omega_{\mathrm{sf}}$ (DFT端点基準)",
    "omega_sf_king": r"$\Omega_{\mathrm{sf}}$ (King基準)",
    "radius_diff_A": r"$|\Delta r|$ (\AA)",
}


def chen_data_root():
    """Return the extracted Chen archive root, downloading it only if absent."""
    files = list((CACHE / "chen2023_data").glob("*/model_params/omegas.json"))
    if files:
        return files[0].parents[1]

    CACHE.mkdir(parents=True, exist_ok=True)
    archive = CACHE / "mpea_stability-v0.1.zip"
    if not archive.exists():
        urllib.request.urlretrieve(URL, archive)
    digest_context = hashlib.sha256()
    with open(archive, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest_context.update(chunk)
    digest = digest_context.hexdigest()
    if digest != KNOWN_ARCHIVE_SHA256:
        archive.unlink()
        raise RuntimeError(f"Zenodo archive SHA-256 mismatch: {digest}")
    target = CACHE / "chen2023_data"
    target.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        target = target.resolve()
        for info in z.infolist():
            member = Path(info.filename)
            if stat.S_ISLNK(info.external_attr >> 16):
                raise RuntimeError(f"Unsafe symbolic-link archive member: {info.filename}")
            if member.is_absolute() or ".." in member.parts:
                raise RuntimeError(f"Unsafe archive member: {info.filename}")
            destination = (target / member).resolve()
            if destination != target and target not in destination.parents:
                raise RuntimeError(f"Archive member escapes target: {info.filename}")
            if info.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(destination, "wb") as dst:
                dst.write(src.read())
    files = list(target.glob("*/model_params/omegas.json"))
    if not files:
        raise RuntimeError("Chen archive extraction produced no omegas.json")
    return files[0].parents[1]


def chen_omega():
    path = chen_data_root() / "model_params" / "omegas.json"
    return json.load(open(path)), path


def pair(s):
    m = re.fullmatch(r"([A-Z][a-z]?)(\d+)([A-Z][a-z]?)(\d+)", str(s))
    if not m:
        return None
    a, na, b, nb = m.groups()
    if a == b:
        return (a, a, True)
    if na != nb:
        return None
    return (*sorted((a, b)), False)


def local(root, natoms, statuses):
    d = pd.read_csv(DATA)
    d = d[(d.structure_root == root) & (d.natoms == natoms) &
          d.status.isin(statuses) &
          d.relax_converged.astype(str).str.lower().eq("yes")].copy()
    parsed = d.dir.map(pair)
    pure = parsed.map(lambda x: x is not None and x[2])
    energy = {}
    for _, r in d[pure].iterrows():
        x = pair(r.dir)[0]
        try:
            energy[x] = float(r.energy_eV) / natoms
        except (TypeError, ValueError):
            pass
    d["pair"] = parsed.map(lambda x: None if x is None or x[2] else f"{x[0]}-{x[1]}")
    d = d[d.pair.notna()].copy()
    d["dh_local"] = np.nan
    for i, r in d.iterrows():
        a, b = r.pair.split("-")
        if a in energy and b in energy:
            d.loc[i, "dh_local"] = float(r.energy_eV) / natoms - (energy[a] + energy[b]) / 2
    return d


def table(chen, lattice, local_df, formal_pairs, omega_dft, omega_king, pure_vol):
    c = pd.DataFrame([{"pair": k, "dh_chen": v / 4}
                      for k, v in chen["omegas"][lattice].items()])
    d = local_df.merge(c, on="pair", how="outer")
    d = d[d.pair.isin(formal_pairs)].copy()
    d["omega_sf_dft"] = d.pair.map(omega_dft)
    d["omega_sf_king"] = d.pair.map(omega_king)
    d["omega_sf"] = d["omega_sf_dft"]
    d["radius_diff_A"] = d.pair.map(lambda p: abs(
        (3 * pure_vol[p.split("-")[0]] / (4 * np.pi)) ** (1 / 3) -
        (3 * pure_vol[p.split("-")[1]] / (4 * np.pi)) ** (1 / 3))
        if all(x in pure_vol for x in p.split("-")) else np.nan)
    d["en_diff"] = d.pair.map(lambda p: abs(EN[p.split("-")[0]] - EN[p.split("-")[1]])
                               if all(x in EN for x in p.split("-")) else np.nan)
    d["vec_diff"] = d.pair.map(lambda p: abs(VEC[p.split("-")[0]] - VEC[p.split("-")[1]])
                                if all(x in VEC for x in p.split("-")) else np.nan)
    return d


def metric(d, x, y):
    z = d.dropna(subset=[x, y])
    lr = linregress(z[x], z[y])
    return {"n": len(z), "pearson_r": pearsonr(z[x], z[y]).statistic,
            "spearman_rho": spearmanr(z[x], z[y]).statistic,
            "slope": lr.slope, "intercept": lr.intercept,
            "mae": np.mean(np.abs(z[y] - z[x])),
            "rmse": np.sqrt(np.mean((z[y] - z[x]) ** 2))}


def regression(d, omega_col="omega_sf"):
    z = d.dropna(subset=[omega_col, "dh_chen", "radius_diff_A", "en_diff", "vec_diff"])
    X = np.column_stack([np.ones(len(z)), z[["dh_chen", "radius_diff_A"]]])
    omega = z[omega_col]
    coef = np.linalg.lstsq(X, omega, rcond=None)[0]
    pred = X @ coef
    r2 = 1 - np.sum((omega - pred) ** 2) / np.sum((omega - omega.mean()) ** 2)
    def pc(x, control):
        A = np.column_stack([np.ones(len(z)), z[[control]]])
        rx = z[x] - A @ np.linalg.lstsq(A, z[x], rcond=None)[0]
        ry = omega - A @ np.linalg.lstsq(A, omega, rcond=None)[0]
        return pearsonr(rx, ry).statistic
    return {"n": len(z), "a_dh": coef[1], "b_radius": coef[2], "intercept": coef[0],
            "r2": r2, "partial_dh": pc("dh_chen", "radius_diff_A"),
            "partial_radius": pc("radius_diff_A", "dh_chen"),
            "r_radius": pearsonr(omega, z.radius_diff_A).statistic,
            "n_radius": len(z), "r_dh": pearsonr(omega, z.dh_chen).statistic,
            "n_dh": len(z), "r_en": pearsonr(omega, z.en_diff).statistic,
            "n_en": len(z), "r_vec": pearsonr(omega, z.vec_diff).statistic,
            "n_vec": len(z)}


def scatter(d, name, x, y, color=None):
    z = d.dropna(subset=[x, y] + ([] if color is None else [color]))
    fig, ax = plt.subplots(figsize=(9, 7))
    if color:
        h = ax.scatter(z[x], z[y], c=z[color], cmap="coolwarm", s=32, alpha=.85)
        fig.colorbar(h, ax=ax, label=LABELS[color])
    else:
        ax.scatter(z[x], z[y], s=32, alpha=.8)
    lr = linregress(z[x], z[y])
    xx = np.linspace(z[x].min(), z[x].max(), 100)
    ax.plot(xx, lr.intercept + lr.slope * xx, "k-", lw=1.7)
    ax.set_xlabel(LABELS[x], fontsize=16); ax.set_ylabel(LABELS[y], fontsize=16)
    ax.text(0.04, 0.96, f"$n={len(z)}$, $r={pearsonr(z[x], z[y]).statistic:.3f}$\n"
            f"$y={lr.slope:.3f}x{lr.intercept:+.3f}$",
            transform=ax.transAxes, va="top", ha="left",
            bbox={"facecolor": "white", "alpha": .82, "edgecolor": "0.5"})
    ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(PAPER / name, dpi=150); plt.close(fig)


def parity(bcc, fcc):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 7), sharex=True, sharey=True)
    for ax, d, lattice in zip(axes, [bcc, fcc], ["BCC", "FCC"]):
        z = d.dropna(subset=["dh_chen", "dh_local"]).copy()
        ax.scatter(z.dh_chen, z.dh_local, s=28, alpha=.7, color="#4472c4")
        special = {"Fe-Hf", "Au-Cr", "Ag-Cr"}
        for _, row in z[z.pair.isin(special)].iterrows():
            ax.scatter(row.dh_chen, row.dh_local, s=50, color="#c00000", zorder=3)
            ax.annotate(row.pair, (row.dh_chen, row.dh_local),
                        xytext=(4, 4), textcoords="offset points", fontsize=11)
        lo = min(z.dh_chen.min(), z.dh_local.min())
        hi = max(z.dh_chen.max(), z.dh_local.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.4)
        lr = linregress(z.dh_chen, z.dh_local)
        ax.text(0.04, 0.96, f"$n={len(z)}$, $r={pearsonr(z.dh_chen, z.dh_local).statistic:.3f}$\n"
                f"MAE={np.mean(np.abs(z.dh_local-z.dh_chen)):.3f} eV/atom",
                transform=ax.transAxes, va="top", ha="left",
                bbox={"facecolor": "white", "alpha": .82, "edgecolor": "0.5"})
        ax.set_title(lattice, fontsize=17)
        ax.set_xlabel(r"Chen $\Delta H_{\mathrm{mix}}$ (eV/atom)", fontsize=16)
        ax.grid(alpha=.2)
    axes[0].set_ylabel(r"当方（本研究） $\Delta H_{\mathrm{mix}}$ (eV/atom)", fontsize=16)
    fig.tight_layout()
    fig.savefig(PAPER / "fig_chen2023_dh_parity.png", dpi=150)
    plt.close(fig)


def main():
    chen, source = chen_omega()
    sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(PAPER))
    import generate_all_figures as gf
    src = gf.load_sqs_data()
    bcc_dft = {"-".join(p): v for p, v in src["omega_dft"].items()}
    bcc_king = {"-".join(p): v for p, v in src["omega_king"].items()}
    fcc_dft = {"-".join(p): v for p, v in src["fcc_omega_dft"].items()}
    fcc_king = {"-".join(p): v for p, v in src["fcc_omega_king"].items()}
    bcc = local("BCC_SQS", 16, {"OK"})
    fcc = local("FCC_SQS", 32, {"OK", "SKIP"})
    b = table(chen, "BCC", bcc, set(bcc_dft) | set(bcc_king),
              bcc_dft, bcc_king, src["pure_vol"])
    f = table(chen, "FCC", fcc, set(fcc_dft) | set(fcc_king),
              fcc_dft, fcc_king, src["fcc_pure_vol"])
    b.to_csv(PAPER / "results_chen2023_bcc.csv", index=False)
    f.to_csv(PAPER / "results_chen2023_fcc.csv", index=False)
    rows = []
    for lat, d, omega_dft, omega_king in [
            ("BCC", b, bcc_dft, bcc_king), ("FCC", f, fcc_dft, fcc_king)]:
        for basis, omega_col, pairs in [
                ("formal_dft", "omega_sf_dft", set(omega_dft)),
                ("formal_king", "omega_sf_king", set(omega_king))]:
            z = d[d.pair.isin(pairs)].copy()
            rows.append({"lattice": lat, "subset": basis,
                         "comparison": "Chen_dH_vs_omega_sf",
                         "omega_reference": "DFT endpoints" if basis.endswith("dft") else "King",
                         **metric(z, "dh_chen", omega_col)})
            rows.append({"lattice": lat, "subset": basis,
                         "comparison": "size_regression",
                         "omega_reference": "DFT endpoints" if basis.endswith("dft") else "King",
                         **regression(z, omega_col)})
            rows.append({"lattice": lat, "subset": basis,
                         "comparison": "local_dH_vs_Chen_dH",
                         "omega_reference": "DFT endpoints" if basis.endswith("dft") else "King",
                         **metric(z, "dh_chen", "dh_local")})
    pd.DataFrame(rows).to_csv(PAPER / "results_chen2023_crosscheck.csv", index=False)
    json.dump({"doi": DOI, "source_url": URL, "archive_sha256": KNOWN_ARCHIVE_SHA256,
               "bcc_dft_pairs": len(bcc_dft), "bcc_king_pairs": len(bcc_king),
               "fcc_dft_pairs": len(fcc_dft), "fcc_king_pairs": len(fcc_king),
               "metrics": rows},
              open(PAPER / "chen2023_crosscheck_metrics.json", "w"), indent=2)
    scatter(b[b.pair.isin(bcc_dft)], "fig_chen2023_dh_vs_omega_bcc.png",
            "dh_chen", "omega_sf_dft")
    scatter(f[f.pair.isin(fcc_dft)], "fig_chen2023_dh_vs_omega_fcc.png",
            "dh_chen", "omega_sf_dft")
    scatter(b[b.pair.isin(bcc_dft)], "fig_chen2023_radius_vs_omega_bcc.png",
            "radius_diff_A", "omega_sf_dft", "dh_chen")
    parity(b[b.pair.isin(bcc_king)], f[f.pair.isin(fcc_king)])


if __name__ == "__main__":
    main()
