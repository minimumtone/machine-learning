#!/usr/bin/env python3
"""Re-check Al-rich fcc Ni(Al) near 70 % Al using 3x3x3 (108-atom) supercells."""
import os, time, numpy as np, pandas as pd
from ase.build import bulk
from ase.filters import FrechetCellFilter
from ase.io import write as ase_write
from ase.optimize import LBFGS
from mace.calculators import mace_mp

BASE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.join(BASE, "analysis")
RELAX = os.path.join(BASE, "relax")
for d in (AN, RELAX): os.makedirs(d, exist_ok=True)

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


rows = []
N = 108
X = [0.625, 0.6875, 0.75, 0.875]
for x_t in X:
    n_al = round(N * x_t)
    for seed in range(NSEEDS):
        rng = np.random.default_rng(7000 * seed + round(x_t * 10000))
        at = bulk("Ni", "fcc", a=3.6, cubic=True).repeat((3, 3, 3))
        idx = rng.choice(len(at), size=n_al, replace=False)
        syms = np.array(at.get_chemical_symbols())
        syms[idx] = "Al"
        at.set_chemical_symbols(list(syms))
        x = n_al / N
        name = f"fcc_NiAl_x{x:.4f}_3x3x3_s{seed}"
        e, v, conv = relax(at, name)
        rows.append({"structure_id": name, "x_Al": x, "seed": seed,
                     "n_atoms": N, "energy_eV": e, "volume_A3": v,
                     "V_per_atom_A3": v / N, "a_fcc_A": (4 * v / N) ** (1 / 3),
                     "converged": conv})

pd.DataFrame(rows).to_csv(os.path.join(AN, "niall_fcc_3x3x3.csv"), index=False)
print("done:", len(rows))
