#!/usr/bin/env python3
"""Additional B2 antisite configurations for Fig. 6(a) energy diagram.

Extends the antisite branch into highly off-stoichiometric ranges:
  Ni-rich side (x_Al < 0.5): Ni antisites on the Al sublattice for x in [0.20, 0.40]
  Al-rich side (x_Al > 0.5): Al antisites on the Ni sublattice for x in [0.60, 0.80]

Reuses the same 4x4x4 B2 supercell, MACE-MP-0 settings and CSV schema as
run_b2_offstoich.py.

Outputs:
  analysis/b2_offstoich_volumes_antisite_extra.csv
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
REP = 4
NCELL = REP ** 3
NSEEDS = 3

X_NI_RICH = [0.20, 0.30, 0.40]
X_AL_RICH = [0.60, 0.70, 0.80]

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


# pure references (same 2x2x2 fcc protocol as main script)
mu = {}
for el, struc, a0 in (("Ni", "fcc", 3.52), ("Al", "fcc", 4.05)):
    at = bulk(el, struc, a=a0, cubic=True).repeat((2, 2, 2))
    e, v, conv = relax(at, f"pure_{el}_{struc}_antisite_extra")
    mu[el] = e / len(at)

rows = []

for x in X_NI_RICH:
    for seed in range(NSEEDS):
        rng = np.random.default_rng(1000 * seed + int(round(x * 1000)))
        base = make_b2()
        al_idx = [i for i, s in enumerate(base.get_chemical_symbols()) if s == "Al"]

        n_al = int(round(2 * NCELL * x))
        n_anti = NCELL - n_al
        at = base.copy()
        for i in rng.choice(al_idx, size=n_anti, replace=False):
            at[i].symbol = "Ni"
        name = f"b2_x{x:.3f}_antisiteNi_s{seed}"
        e, v, conv = relax(at, name)
        n_ni, n_al_final = NCELL + n_anti, n_al
        n = n_ni + n_al_final
        rows.append(dict(
            structure_id=name, branch="antisite", x_Al_target=x,
            x_Al=round(n_al_final / n, 6), seed=seed, n_Ni=n_ni, n_Al=n_al_final,
            n_atoms=n, n_sites=2 * NCELL, energy_eV=e, volume_A3=v,
            V_per_atom_A3=v / n, a_eff_A=(v / NCELL) ** (1.0 / 3.0),
            E_form_eV_atom=(e - n_ni * mu["Ni"] - n_al_final * mu["Al"]) / n,
            converged=conv,
        ))

for x in X_AL_RICH:
    for seed in range(NSEEDS):
        rng = np.random.default_rng(1000 * seed + int(round(x * 1000)))
        base = make_b2()
        ni_idx = [i for i, s in enumerate(base.get_chemical_symbols()) if s == "Ni"]

        n_ni = int(round(2 * NCELL * (1.0 - x)))
        n_anti = NCELL - n_ni
        at = base.copy()
        for i in rng.choice(ni_idx, size=n_anti, replace=False):
            at[i].symbol = "Al"
        name = f"b2_x{x:.3f}_antisiteAl_s{seed}"
        e, v, conv = relax(at, name)
        n_ni_final, n_al = n_ni, NCELL + n_anti
        n = n_ni_final + n_al
        rows.append(dict(
            structure_id=name, branch="antisite", x_Al_target=x,
            x_Al=round(n_al / n, 6), seed=seed, n_Ni=n_ni_final, n_Al=n_al,
            n_atoms=n, n_sites=2 * NCELL, energy_eV=e, volume_A3=v,
            V_per_atom_A3=v / n, a_eff_A=(v / NCELL) ** (1.0 / 3.0),
            E_form_eV_atom=(e - n_ni_final * mu["Ni"] - n_al * mu["Al"]) / n,
            converged=conv,
        ))

pd.DataFrame(rows).to_csv(
    os.path.join(AN, "b2_offstoich_volumes_antisite_extra.csv"), index=False)
print("done:", len(rows), "configurations")
