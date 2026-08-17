#!/usr/bin/env python3
"""Dense B2 competition scan near x_Al = 0.5."""
import json, os, time
import numpy as np, pandas as pd
from ase.build import bulk
from ase.filters import FrechetCellFilter
from ase.io import write as ase_write
from ase.optimize import LBFGS
from mace.calculators import mace_mp

BASE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.join(BASE, "analysis")
RELAX = os.path.join(BASE, "relax")
os.makedirs(AN, exist_ok=True); os.makedirs(RELAX, exist_ok=True)

FMAX = 0.02
REP = 4
NCELL = REP**3
NSEEDS = 3
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


def record(rows, name, branch, x_al_target, seed, e, v, n_ni, n_al, conv):
    n = n_ni + n_al
    x_al = n_al / n
    a_eff = (v / NCELL) ** (1.0 / 3.0)
    e_form = (e - n_ni * mu["Ni"] - n_al * mu["Al"]) / n
    rows.append(dict(
        structure_id=name, branch=branch, x_Al_target=x_al_target,
        x_Al=round(x_al, 6), seed=seed, n_Ni=n_ni, n_Al=n_al, n_atoms=n,
        n_sites=2*NCELL, energy_eV=e, volume_A3=v,
        V_per_atom_A3=v/n, a_eff_A=a_eff, E_form_eV_atom=e_form,
        converged=conv,
    ))


rows = []
X_TARGETS = [0.47, 0.49, 0.51, 0.53]

for x in X_TARGETS:
    for seed in range(NSEEDS):
        rng = np.random.default_rng(1000 * seed + int(round(x * 1000)))
        base = make_b2()
        ni_idx, al_idx = sublattice_indices(base)

        if x < 0.5:
            # Ni antisites on Al sublattice
            n_al = int(round(2 * NCELL * x))
            n_anti = NCELL - n_al
            at = base.copy()
            for i in rng.choice(al_idx, size=n_anti, replace=False):
                at[i].symbol = "Ni"
            name = f"b2_x{x:.3f}_antisiteNi_s{seed}"
            e, v, conv = relax(at, name)
            record(rows, name, "antisite", x, seed, e, v, NCELL + n_anti, n_al, conv)

            # Al vacancies
            n_al_v = int(round(NCELL * x / (1.0 - x)))
            n_vac = NCELL - n_al_v
            at = base.copy()
            del at[[int(i) for i in rng.choice(al_idx, size=n_vac, replace=False)]]
            name = f"b2_x{x:.3f}_vacAl_s{seed}"
            e, v, conv = relax(at, name)
            record(rows, name, "vacancy", x, seed, e, v, NCELL, n_al_v, conv)
        else:
            # Ni vacancies
            n_ni_v = int(round(NCELL * (1.0 - x) / x))
            n_vac = NCELL - n_ni_v
            at = base.copy()
            del at[[int(i) for i in rng.choice(ni_idx, size=n_vac, replace=False)]]
            name = f"b2_x{x:.3f}_vacNi_s{seed}"
            e, v, conv = relax(at, name)
            record(rows, name, "vacancy", x, seed, e, v, n_ni_v, NCELL, conv)

            # Al antisites on Ni sublattice
            n_ni = int(round(2 * NCELL * (1.0 - x)))
            n_anti = NCELL - n_ni
            at = base.copy()
            for i in rng.choice(ni_idx, size=n_anti, replace=False):
                at[i].symbol = "Al"
            name = f"b2_x{x:.3f}_antisiteAl_s{seed}"
            e, v, conv = relax(at, name)
            record(rows, name, "antisite", x, seed, e, v, n_ni, NCELL + n_anti, conv)

df = pd.DataFrame(rows)
out = os.path.join(AN, "b2_offstoich_volumes_50competition.csv")
df.to_csv(out, index=False)
print(f"Wrote {out} with {len(df)} rows")
