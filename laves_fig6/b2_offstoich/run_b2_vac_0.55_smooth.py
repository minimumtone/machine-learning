#!/usr/bin/env python3
"""4×4×4 intermediate Ni-vacancy points to smooth the Gibbs curve around x_Al≈0.55."""
import json, os, time, math
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)
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


rows = []
# Choose n_vac values that fall between the existing x=0.54 (n_vac=9) and x=0.56 (n_vac=14)
# and also between 0.52 (n_vac=5) and 0.56 to densify the discrete composition grid.
N_VACS = [10, 11, 12, 13]

for n_vac in N_VACS:
    n_ni = NCELL - n_vac
    x_actual = NCELL / (2.0 * NCELL - n_vac)  # n_Al=NCELL, n_sites=2*NCELL
    x_target = round(x_actual, 6)
    for seed in range(NSEEDS):
        rng = np.random.default_rng(10000 + n_vac * 10 + seed)
        base = make_b2()
        ni_idx, al_idx = sublattice_indices(base)
        at = base.copy()
        del at[[int(i) for i in rng.choice(ni_idx, size=n_vac, replace=False)]]
        name = f"b2_x{x_target:.3f}_vacNi_alrich_smooth_{REP}x{REP}x{REP}_s{seed}"
        e, v, conv = relax(at, name)
        n_al = NCELL
        n = n_ni + n_al
        rows.append(dict(
            structure_id=name, branch="vacancy", x_Al_target=x_target,
            x_Al=round(n_al / n, 6), seed=seed, n_Ni=n_ni, n_Al=n_al, n_atoms=n,
            n_sites=2 * NCELL, energy_eV=e, volume_A3=v,
            V_per_atom_A3=v / n, a_eff_A=(v / NCELL) ** (1.0 / 3.0),
            E_form_eV_atom=(e - n_ni * mu["Ni"] - n_al * mu["Al"]) / n,
            converged=conv,
            sqs_cell=f"{REP}x{REP}x{REP}"
        ))

out = os.path.join(AN, "b2_offstoich_volumes_vacancy_055_smooth.csv")
pd.DataFrame(rows).to_csv(out, index=False)
print("Wrote", out)
