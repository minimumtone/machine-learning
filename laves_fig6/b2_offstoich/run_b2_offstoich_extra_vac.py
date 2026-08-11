#!/usr/bin/env python3
"""Additional Ni-rich vacancy-branch B2 configurations (x_Al <= 0.5).

Densifies the Al-vacancy branch of the off-stoichiometric B2 study with
extra compositions (3 random defect placements each), reusing the same
supercell, relaxation settings and CSV schema as run_b2_offstoich.py.

Outputs:
  analysis/b2_offstoich_volumes_extra_vac.csv  (merged by make_figures.py)
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
REP = 4
NCELL = REP ** 3
NSEEDS = 3

X_TARGETS = [0.41, 0.43, 0.45, 0.47, 0.49]

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


# pure references for formation energies (same protocol as main script)
mu = {}
for el, struc, a0 in (("Ni", "fcc", 3.52), ("Al", "fcc", 4.05)):
    at = bulk(el, struc, a=a0, cubic=True).repeat((2, 2, 2))
    e, v, conv = relax(at, f"pure_{el}_{struc}_extra")
    mu[el] = e / len(at)

rows = []
for x in X_TARGETS:
    for seed in range(NSEEDS):
        rng = np.random.default_rng(1000 * seed + int(round(x * 1000)))
        base = make_b2()
        al_idx = [i for i, s in enumerate(base.get_chemical_symbols()) if s == "Al"]

        # Al vacancies (Ni count fixed = 64)
        n_al_v = int(round(NCELL * x / (1.0 - x)))
        n_vac = NCELL - n_al_v
        at = base.copy()
        del at[[int(i) for i in rng.choice(al_idx, size=n_vac, replace=False)]]
        name = f"b2_x{x:.3f}_vacAl_s{seed}"
        e, v, conv = relax(at, name)
        n_ni, n_al = NCELL, n_al_v
        n = n_ni + n_al
        rows.append(dict(
            structure_id=name, branch="vacancy", x_Al_target=x,
            x_Al=round(n_al / n, 6), seed=seed, n_Ni=n_ni, n_Al=n_al, n_atoms=n,
            n_sites=2 * NCELL, energy_eV=e, volume_A3=v,
            V_per_atom_A3=v / n, a_eff_A=(v / NCELL) ** (1.0 / 3.0),
            E_form_eV_atom=(e - n_ni * mu["Ni"] - n_al * mu["Al"]) / n,
            converged=conv,
        ))

pd.DataFrame(rows).to_csv(
    os.path.join(AN, "b2_offstoich_volumes_extra_vac.csv"), index=False)
print("done:", len(rows), "configurations")
