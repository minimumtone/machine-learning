#!/usr/bin/env python3
"""Second batch of 4x4x4 antisite seeds (seeds 8-12) for 0.71-0.75 Al."""
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
SEED_START = 8
SEED_END = 12
X = [0.71, 0.73, 0.75]

CALC = mace_mp(model="medium", default_dtype="float64", device="cpu")

with open(os.path.join(AN, "b2_offstoich_summary.json")) as f:
    summary = json.load(f)
mu = {"Ni": summary["mu_Ni_eV"], "Al": summary["mu_Al_eV"]}


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


def sublattice_indices(atoms):
    ni = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "Ni"]
    al = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "Al"]
    return ni, al


rows = []
for x in X:
    for seed in range(SEED_START, SEED_END + 1):
        rng = np.random.default_rng(1000 * seed + int(round(x * 1000)))
        base = make_b2()
        ni_idx, al_idx = sublattice_indices(base)

        n_ni = int(round(2 * NCELL * (1.0 - x)))
        n_anti = NCELL - n_ni
        at = base.copy()
        for i in rng.choice(ni_idx, size=n_anti, replace=False):
            at[i].symbol = "Al"
        name = f"b2_x{x:.3f}_antisiteAl_dense_s{seed}"
        e, v, conv = relax(at, name)
        n_al = NCELL + n_anti
        n = n_ni + n_al
        x_actual = n_al / n
        rows.append(dict(
            structure_id=name, branch="antisite", x_Al_target=x,
            x_Al=round(x_actual, 6), seed=seed, n_Ni=n_ni, n_Al=n_al,
            n_atoms=n, n_sites=2*NCELL, energy_eV=e, volume_A3=v,
            V_per_atom_A3=v/n, a_eff_A=(v/NCELL)**(1.0/3.0),
            E_form_eV_atom=(e - n_ni*mu["Ni"] - n_al*mu["Al"])/n,
            converged=conv,
        ))

df = pd.DataFrame(rows)
out = os.path.join(AN, "b2_offstoich_volumes_antisite_alrich_dense_extra_seeds2.csv")
df.to_csv(out, index=False)
print(f"Wrote {out} with {len(df)} rows")
