#!/usr/bin/env python3
"""Compute 1NN Warren-Cowley SRO for Ni-sublattice defects in B2 supercells.

Used to quantify how "random" each relaxed vacancy / antisite configuration is.
Outputs a CSV with SRO parameters that can be used to filter outliers.
"""
import os
import re
import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import read
from ase.neighborlist import neighbor_list
from scipy.spatial import cKDTree

BASE = os.path.dirname(os.path.abspath(__file__))
RELAX = os.path.join(BASE, "relax")
AN = os.path.join(BASE, "analysis")


def make_ideal_b2_supercell(rep=4, a=2.88):
    """Ideal B2 4x4x4 supercell with 128 sites.

    Returns an ASE Atoms object with Ni-sublattice and Al-sublattice indices.
    """
    symbols = []
    pos = []
    ni_sites = []
    al_sites = []
    idx = 0
    for i in range(rep):
        for j in range(rep):
            for k in range(rep):
                # Ni sublattice
                x, y, z = i / rep, j / rep, k / rep
                pos.append([x, y, z])
                symbols.append("Ni")
                ni_sites.append(idx)
                idx += 1
                # Al sublattice
                pos.append([
                    (i + 0.5) / rep,
                    (j + 0.5) / rep,
                    (k + 0.5) / rep,
                ])
                symbols.append("Al")
                al_sites.append(idx)
                idx += 1
    at = Atoms(symbols=symbols, scaled_positions=pos, cell=[a * rep] * 3, pbc=True)
    return at, ni_sites, al_sites


def sro_for_file(path, max_dr=1.1):
    """Return dict with composition and 1NN Warren-Cowley SRO.

    max_dr: maximum fractional distance for mapping a relaxed atom to an
            ideal site (roughly half the nearest ideal site spacing).
    """
    at = read(path)
    # Determine B2 supercell rep (Ni sublattice sites per edge).
    rep = at.info.get("rep")
    if rep is None:
        n = len(at)
        # For perfect/antisite n = 2*rep^3; for vacancy use the stored n_sites
        n_sites = at.info.get("n_sites")
        if n_sites is not None:
            rep = int(round((n_sites / 2.0) ** (1.0 / 3.0)))
        else:
            # Fallback for 4x4x4 and similar cubic B2 cells
            rep = int(round((n / 2.0) ** (1.0 / 3.0)))
    ideal_atoms, ni_sites, al_sites = make_ideal_b2_supercell(rep=rep)
    # Use fractional coordinates for assignment, which is robust to cell changes.
    ideal_scaled = ideal_atoms.get_scaled_positions()
    atoms_scaled = at.get_scaled_positions()
    symbols = at.get_chemical_symbols()
    tree = cKDTree(ideal_scaled)
    # Build occupancy vector on the ideal Ni sublattice.
    # 0 = Ni (or absent host), 1 = defect (vacancy or antisite)
    # Start with 1 = defect; mark as 0 when an atom maps there.
    occupied = np.ones(len(ni_sites), dtype=int)  # 1 = defect
    for s, p in zip(symbols, atoms_scaled):
        dist, idx_in_ideal = tree.query(p)
        if dist > max_dr:
            continue
        if idx_in_ideal in ni_sites:
            local_idx = ni_sites.index(idx_in_ideal)
            if s == "Ni":
                occupied[local_idx] = 0
            elif s == "Al":
                # antisite Al on Ni sublattice
                occupied[local_idx] = 1
    c = occupied.mean()
    # 1NN within Ni sublattice: simple-cubic neighbors, cutoff ~ a_eff
    cell = at.get_cell().diagonal()
    a_eff_from_cell = cell.mean() / rep
    cutoff = a_eff_from_cell * 1.05
    # Build a dummy atoms object containing only the Ni sublattice sites.
    ni_pos_cart = (ideal_atoms.get_positions()[ni_sites])
    ni_atoms = Atoms(["Ni"] * len(ni_sites), positions=ni_pos_cart, cell=at.get_cell(), pbc=True)
    i, j = neighbor_list("ij", ni_atoms, cutoff)
    n_both_defect = 0
    n_total = 0
    for m, n in zip(i, j):
        if m < n:  # count each pair once
            n_total += 1
            if occupied[m] == 1 and occupied[n] == 1:
                n_both_defect += 1
    if n_total == 0 or c == 0 or c == 1:
        alpha = np.nan
    else:
        p_def_def = n_both_defect / n_total
        alpha = 1.0 - p_def_def / (c * c)
    return {
        "path": path,
        "n_Ni_sublattice": int((~occupied.astype(bool)).sum()),
        "n_defect_sublattice": int(occupied.sum()),
        "c_defect": c,
        "n_1nn_pairs": n_total,
        "n_defect_defect_pairs": n_both_defect,
        "alpha_1nn_WC": alpha,
        "a_eff_A": a_eff_from_cell,
    }


def main():
    rows = []
    for fn in sorted(os.listdir(RELAX)):
        if fn.endswith(".extxyz") and (
            "vacNi_alrich" in fn or "antisiteAl_dense" in fn
        ):
            path = os.path.join(RELAX, fn)
            try:
                d = sro_for_file(path)
                # parse x target from filename
                m = re.search(r"x0\.\d{3}", fn)
                x_target = float(m.group(0)[1:]) if m else np.nan
                m2 = re.search(r"_s(\d+)", fn)
                seed = int(m2.group(1)) if m2 else np.nan
                branch = "vacancy" if "vacNi" in fn else "antisite"
                d.update({"structure_id": fn, "branch": branch, "x_Al_target": x_target, "seed": seed})
                rows.append(d)
            except Exception as e:
                print(f"skip {fn}: {e}")
    df = pd.DataFrame(rows)
    out = os.path.join(AN, "b2_highAl_sqs_sro.csv")
    df.to_csv(out, index=False)
    print(f"Wrote {out} ({len(df)} rows)")
    # Summary table for vacancy / antisite in 0.70-0.78
    print(df[(df.x_Al_target >= 0.68) & (df.x_Al_target <= 0.80)][[
        "structure_id", "x_Al_target", "branch", "c_defect", "alpha_1nn_WC"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
