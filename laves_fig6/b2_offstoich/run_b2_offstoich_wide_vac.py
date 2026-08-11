#!/usr/bin/env python3
"""Wide-composition vacancy-branch B2 configurations + supercell-size check.

Extends the vacancy branch over a wide x_Al range:
  Ni-rich side (Al vacancies, Ni count fixed): x_Al = 0.30 ... 0.40
  Al-rich side (Ni vacancies, Al count fixed): x_Al = 0.60 ... 0.66
with 3 random defect placements each (4x4x4 supercell, 128 sites), and adds a
5x5x5 (250-site) supercell check at representative compositions to quantify
the finite-size dependence of the defect-defect image interaction.

Outputs:
  analysis/b2_offstoich_volumes_wide_vac.csv   (merged by make_figures.py)
  analysis/b2_offstoich_sizecheck.csv          4x4x4 vs 5x5x5 comparison
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

FMAX = 0.02
NSEEDS = 3

X_NI_RICH = [0.30, 0.32, 0.34, 0.36, 0.38, 0.40]   # Al vacancies
X_AL_RICH = [0.60, 0.62, 0.64, 0.66]               # Ni vacancies
SIZECHECK = [(0.46, "Al"), (0.54, "Ni")]           # (x_Al, vacancy species)

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


def make_b2(rep, a=2.88):
    at = bulk("NiAl", crystalstructure="cesiumchloride", a=a, cubic=True)
    return at.repeat((rep, rep, rep))


mu = {}
for el, struc, a0 in (("Ni", "fcc", 3.52), ("Al", "fcc", 4.05)):
    at = bulk(el, struc, a=a0, cubic=True).repeat((2, 2, 2))
    e, v, conv = relax(at, f"pure_{el}_{struc}_wide")
    mu[el] = e / len(at)


def run_vacancy(x, seed, rep, tag=""):
    ncell = rep ** 3
    rng = np.random.default_rng(1000 * seed + int(round(x * 1000)) + 7 * rep)
    base = make_b2(rep)
    syms = base.get_chemical_symbols()
    if x < 0.5:  # Al vacancies, Ni count fixed
        idx = [i for i, s in enumerate(syms) if s == "Al"]
        n_keep = int(round(ncell * x / (1.0 - x)))
        n_ni, n_al = ncell, n_keep
        vac_species = "Al"
    else:        # Ni vacancies, Al count fixed
        idx = [i for i, s in enumerate(syms) if s == "Ni"]
        n_keep = int(round(ncell * (1.0 - x) / x))
        n_ni, n_al = n_keep, ncell
        vac_species = "Ni"
    n_vac = ncell - n_keep
    at = base.copy()
    del at[[int(i) for i in rng.choice(idx, size=n_vac, replace=False)]]
    name = f"b2_x{x:.3f}_vac{vac_species}_r{rep}_s{seed}{tag}"
    e, v, conv = relax(at, name)
    n = n_ni + n_al
    return dict(
        structure_id=name, branch="vacancy", x_Al_target=x,
        x_Al=round(n_al / n, 6), seed=seed, n_Ni=n_ni, n_Al=n_al, n_atoms=n,
        n_sites=2 * ncell, energy_eV=e, volume_A3=v,
        V_per_atom_A3=v / n, a_eff_A=(v / ncell) ** (1.0 / 3.0),
        E_form_eV_atom=(e - n_ni * mu["Ni"] - n_al * mu["Al"]) / n,
        converged=conv, rep=rep,
    )


rows = []
for x in X_NI_RICH + X_AL_RICH:
    for seed in range(NSEEDS):
        rows.append(run_vacancy(x, seed, rep=4))
pd.DataFrame(rows).drop(columns=["rep"]).to_csv(
    os.path.join(AN, "b2_offstoich_volumes_wide_vac.csv"), index=False)
print("wide-range done:", len(rows), "configurations")

size_rows = []
for x, _sp in SIZECHECK:
    for rep in (4, 5):
        for seed in range(NSEEDS):
            size_rows.append(run_vacancy(x, seed, rep=rep, tag="_szchk"))
pd.DataFrame(size_rows).to_csv(
    os.path.join(AN, "b2_offstoich_sizecheck.csv"), index=False)
print("size-check done:", len(size_rows), "configurations")
