#!/usr/bin/env python3
"""Relax the x_Al=0.50 point as an explicit vacancy-branch configuration.

At exactly 1:1 Ni:Al the vacancy model has zero vacancies, so the resulting
structure is identical to perfect B2.  This adds a duplicate row under branch
'vacancy' so that the vacancy curve naturally meets the perfect-B2 point at
x_Al=0.50 in the B2 volume / lattice-constant figures.
"""
import json
import os
import time

import numpy as np
from ase.build import bulk
from ase.filters import FrechetCellFilter
from ase.io import write as ase_write
from ase.optimize import LBFGS
from mace.calculators import mace_mp

BASE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.join(BASE, "analysis")
RELAX = os.path.join(BASE, "relax")
os.makedirs(AN, exist_ok=True)
os.makedirs(RELAX, exist_ok=True)

FMAX = 0.02
REP = 4
NCELL = REP ** 3
CALC = mace_mp(model="medium", default_dtype="float64", device="cpu")

with open(os.path.join(AN, "b2_offstoich_summary.json")) as f:
    summary = json.load(f)
MU_NI = summary["mu_Ni_eV"]
MU_AL = summary["mu_Al_eV"]


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


at = bulk("NiAl", crystalstructure="cesiumchloride", a=2.88, cubic=True).repeat((REP, REP, REP))
name = "b2_x0.500_vacNi_perfect"
e, v, conv = relax(at, name)

x_al = 0.5
n_ni = NCELL
n_al = NCELL
n = n_ni + n_al
row = dict(
    structure_id=name, branch="vacancy", x_Al_target=x_al,
    x_Al=x_al, seed=0, n_Ni=n_ni, n_Al=n_al, n_atoms=n,
    n_sites=2 * NCELL, energy_eV=e, volume_A3=v,
    V_per_atom_A3=v / n, a_eff_A=(v / NCELL) ** (1.0 / 3.0),
    E_form_eV_atom=(e - n_ni * MU_NI - n_al * MU_AL) / n,
    converged=conv,
)

import pandas as pd
out = os.path.join(AN, "b2_offstoich_volumes_vacancy_stoichiometric.csv")
pd.DataFrame([row]).to_csv(out, index=False)
print(f"wrote {out}")
