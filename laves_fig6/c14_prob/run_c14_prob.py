#!/usr/bin/env python3
"""Probabilistic C14-Nb(Ni1-xAlx)2 composition sweep (Fig. 6(b), Yamanouchi 2018).

Dense x_Al sweep of the C14 Laves B-sublattice (2a + 6h sites) using random
Ni/Al site occupations (3 seeds per composition) in a 2x2x1 supercell
(48 atoms, 32 B sites), relaxed with MACE-MP-0. The random occupations sample
both the 2a/6h partitioning and the in-sublattice arrangement, so V-bar(x) is
obtained as a distribution (mean +/- std) rather than one ordered structure.

Outputs:
  analysis/c14_prob_volumes.csv
  relax/*.extxyz
"""
import os
import time

import numpy as np
import pandas as pd
from ase.filters import FrechetCellFilter
from ase.io import write as ase_write
from ase.optimize import LBFGS
from ase.spacegroup import crystal
from mace.calculators import mace_mp

BASE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.join(BASE, "analysis")
RELAX = os.path.join(BASE, "relax")
for d in (AN, RELAX):
    os.makedirs(d, exist_ok=True)

FMAX = 0.02
NSEEDS = 3
X6H = 0.8306
Z4F = 0.0630

CALC = mace_mp(model="medium", default_dtype="float64", device="cpu")


def make_c14(a=4.81, c=7.85):
    at = crystal(["Nb", "Ni", "Cu"],
                 basis=[(1 / 3, 2 / 3, Z4F), (0, 0, 0), (X6H, 2 * X6H - 1, 0.25)],
                 spacegroup=194, cellpar=[a, a, c, 90, 90, 120])
    sites = [{"Nb": "4f", "Ni": "2a", "Cu": "6h"}[s] for s in at.get_chemical_symbols()]
    at.set_chemical_symbols(["Nb" if t == "4f" else "Ni" for t in sites])
    at.new_array("site", np.array(sites, dtype="U3"))
    return at


def relax(atoms, name):
    atoms.calc = CALC
    opt = LBFGS(FrechetCellFilter(atoms), logfile=None)
    t0 = time.time()
    opt.run(fmax=FMAX, steps=500)
    conv = bool(opt.converged())
    e = float(atoms.get_potential_energy())
    v = float(atoms.get_volume())
    print(f"[relax] {name}: N={len(atoms)} E={e:.4f} eV V/atom={v/len(atoms):.4f} "
          f"conv={conv} ({time.time()-t0:.1f}s)", flush=True)
    ase_write(os.path.join(RELAX, name + ".extxyz"), atoms)
    return e, v, conv


X_TARGETS = [i / 16 for i in range(1, 16)]  # 0.0625 ... 0.9375

rows = []
for x in X_TARGETS:
    for seed in range(NSEEDS):
        rng = np.random.default_rng(2000 * seed + int(round(x * 10000)))
        at = make_c14().repeat((2, 2, 1))
        sites = at.arrays["site"]
        b_idx = np.where(sites != "4f")[0]
        n_al = int(round(x * len(b_idx)))
        syms = np.array(at.get_chemical_symbols())
        pick = rng.choice(b_idx, size=n_al, replace=False)
        syms[pick] = "Al"
        at.set_chemical_symbols(list(syms))
        n_al_2a = int(np.sum(sites[pick] == "2a"))
        name = f"c14_x{x:.4f}_rand_s{seed}"
        e, v, conv = relax(at, name)
        rows.append(dict(
            structure_id=name, x_Al_target=x, x_Al=n_al / len(b_idx), seed=seed,
            n_atoms=len(at), n_B_sites=len(b_idx), n_Al=n_al, n_Al_on_2a=n_al_2a,
            energy_eV=e, volume_A3=v, V_per_atom_A3=v / len(at), converged=conv,
        ))

pd.DataFrame(rows).to_csv(os.path.join(AN, "c14_prob_volumes.csv"), index=False)
print("done:", len(rows), "configurations")
