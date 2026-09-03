#!/usr/bin/env python3
"""Additional seeds for the high-Al Ni-vacancy branch (0.70-0.78).

Runs seeds 3-7 for the compositions that showed the largest scatter in the
3-seed run, so the mean/standard-deviation estimates are more robust.

Outputs:
  analysis/b2_offstoich_volumes_alrich_vac_extra_seeds.csv
  relax/*.extxyz
"""
import json
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
REP = 4
NCELL = REP ** 3
SEED_START = 3
SEED_END = 7

X_TARGETS = [0.70, 0.72, 0.74, 0.76, 0.78]

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


def make_b2(a=2.88):
    at = bulk("NiAl", crystalstructure="cesiumchloride", a=a, cubic=True)
    return at.repeat((REP, REP, REP))


with open(os.path.join(AN, "b2_offstoich_summary.json")) as f:
    summary = json.load(f)
mu_Ni = summary["mu_Ni_eV"]
mu_Al = summary["mu_Al_eV"]

rows = []
for x in X_TARGETS:
    for seed in range(SEED_START, SEED_END + 1):
        rng = np.random.default_rng(
            5000 * seed + int(round(x * 1000)) + 17 * REP
        )
        base = make_b2()
        ni_idx = [i for i, s in enumerate(base.get_chemical_symbols()) if s == "Ni"]

        n_ni_v = int(round(NCELL * (1.0 - x) / x))
        n_vac = NCELL - n_ni_v
        n_ni, n_al = n_ni_v, NCELL
        n = n_ni + n_al

        at = base.copy()
        del at[[int(i) for i in rng.choice(ni_idx, size=n_vac, replace=False)]]
        name = f"b2_x{x:.3f}_vacNi_alrich_s{seed}"
        e, v, conv = relax(at, name)

        rows.append(dict(
            structure_id=name, branch="vacancy", x_Al_target=x,
            x_Al=round(n_al / n, 6), seed=seed, n_Ni=n_ni, n_Al=n_al, n_atoms=n,
            n_sites=2 * NCELL, energy_eV=e, volume_A3=v,
            V_per_atom_A3=v / n, a_eff_A=(v / NCELL) ** (1.0 / 3.0),
            E_form_eV_atom=(e - n_ni * mu_Ni - n_al * mu_Al) / n,
            converged=conv,
        ))

pd.DataFrame(rows).to_csv(
    os.path.join(AN, "b2_offstoich_volumes_alrich_vac_extra_seeds.csv"), index=False)
print("done:", len(rows), "configurations")
