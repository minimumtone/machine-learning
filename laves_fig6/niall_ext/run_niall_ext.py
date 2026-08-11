#!/usr/bin/env python3
"""Extended Ni(Al) fcc solid solution + B2 ordering-degree study (MACE-MP-0).

Part 1: fcc Ni(Al) SQS-like random solid solution extended to high Al content
        (x_Al = 0.3125 ... 1.0), to compare with the Vegard line between
        fcc-Ni and fcc-Al.
Part 2: B2 ordering degree at x_Al = 0.5: long-range order parameter eta is
        varied by swapping a fraction of Ni/Al pairs between sublattices
        (eta = 1 perfect B2 ... eta = 0 random bcc), 3 seeds per eta.

Outputs:
  analysis/niall_fcc_ext.csv
  analysis/b2_order_param.csv
  relax/*.extxyz
"""
import os
import time

import numpy as np
import pandas as pd
from ase.build import bulk
from ase.filters import FrechetCellFilter
from ase.io import write as ase_write
from ase.optimize import LBFGS
from mace.calculators import mace_mp

BASE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.join(BASE, "analysis")
RELAX = os.path.join(BASE, "relax")
for d in (AN, RELAX):
    os.makedirs(d, exist_ok=True)

FMAX = 0.02
NSEEDS = 3
CALC = mace_mp(model="medium", default_dtype="float64", device="cpu")


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


# --- Part 1: fcc Ni(Al) extended to high Al ---------------------------------
rows = []
X_FCC = [0.3125, 0.375, 0.4375, 0.5, 0.5625, 0.625, 0.75, 0.875, 1.0]
for x in X_FCC:
    n_al = int(round(32 * x))
    seeds = range(NSEEDS) if 0 < n_al < 32 else [0]
    for seed in seeds:
        rng = np.random.default_rng(3000 * seed + int(round(x * 10000)))
        at = bulk("Ni", "fcc", a=3.6, cubic=True).repeat((2, 2, 2))  # 32 atoms
        syms = np.array(at.get_chemical_symbols())
        syms[rng.choice(len(at), size=n_al, replace=False)] = "Al"
        at.set_chemical_symbols(list(syms))
        name = f"fcc_NiAl_x{x:.4f}_ext_s{seed}"
        e, v, conv = relax(at, name)
        rows.append(dict(structure_id=name, x_Al=n_al / 32, seed=seed,
                         n_atoms=len(at), energy_eV=e, volume_A3=v,
                         V_per_atom_A3=v / len(at),
                         a_fcc_A=(4 * v / len(at)) ** (1 / 3), converged=conv))
pd.DataFrame(rows).to_csv(os.path.join(AN, "niall_fcc_ext.csv"), index=False)
print("fcc ext done:", len(rows))

# --- Part 2: B2 ordering degree at x_Al = 0.5 --------------------------------
REP = 4
NCELL = REP ** 3


def make_b2():
    at = bulk("NiAl", crystalstructure="cesiumchloride", a=2.88, cubic=True)
    return at.repeat((REP, REP, REP))


rows = []
# eta = 1 - 2 * (fraction of swapped pairs); swap m Ni<->Al pairs
for m in (0, 8, 16, 24, 32):  # of 64 per sublattice -> eta = 1, 0.75, 0.5, 0.25, 0
    eta = 1.0 - 2.0 * m / NCELL
    seeds = range(NSEEDS) if m > 0 else [0]
    for seed in seeds:
        rng = np.random.default_rng(4000 * seed + m)
        at = make_b2()
        syms = np.array(at.get_chemical_symbols())
        ni_idx = np.where(syms == "Ni")[0]
        al_idx = np.where(syms == "Al")[0]
        si = rng.choice(ni_idx, size=m, replace=False)
        sj = rng.choice(al_idx, size=m, replace=False)
        syms[si] = "Al"
        syms[sj] = "Ni"
        at.set_chemical_symbols(list(syms))
        name = f"b2_eta{eta:.2f}_s{seed}"
        e, v, conv = relax(at, name)
        rows.append(dict(structure_id=name, eta=eta, n_swaps=m, seed=seed,
                         n_atoms=len(at), energy_eV=e, volume_A3=v,
                         V_per_atom_A3=v / len(at),
                         a_eff_A=(v / NCELL) ** (1 / 3), converged=conv))
pd.DataFrame(rows).to_csv(os.path.join(AN, "b2_order_param.csv"), index=False)
print("order-param done:", len(rows))
