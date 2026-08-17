#!/usr/bin/env python3
"""FCC solid-solution X(Al) at x_Al=0.5 for X = Co, Pd, Rh, Ir.

Used for reproducing the fcc-derived atomic-size part of Yamanouchi & Miura
Table 4.  2x2x2 fcc supercell (32 atoms), 1 random seed per system.
"""
import os, time
import numpy as np
import pandas as pd
from ase.build import bulk
from ase.filters import FrechetCellFilter
from ase.io import write as ase_write
from ase.optimize import LBFGS
from mace.calculators import mace_mp
import torch

torch.set_num_threads(1)

BASE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.join(BASE, "analysis")
RELAX = os.path.join(BASE, "relax")
os.makedirs(AN, exist_ok=True); os.makedirs(RELAX, exist_ok=True)

FMAX = 0.02
CALC = mace_mp(model="medium", default_dtype="float64", device="cpu")

SYSTEMS = [
    ("Co", "CoAl_fcc50"),
    ("Pd", "PdAl_fcc50"),
    ("Rh", "RhAl_fcc50"),
    ("Ir", "IrAl_fcc50"),
]

rows = []
for elem, name in SYSTEMS:
    atoms = bulk(elem, crystalstructure="fcc", a=3.6, cubic=True).repeat((2, 2, 2))
    # replace 16 of 32 atoms with Al
    idx = np.arange(len(atoms))
    rng = np.random.default_rng(hash(name) % 2**31)
    al_idx = rng.choice(idx, size=16, replace=False)
    for i in al_idx:
        atoms[i].symbol = "Al"
    atoms.calc = CALC
    opt = LBFGS(FrechetCellFilter(atoms), logfile=None)
    t0 = time.time()
    conv = opt.run(fmax=FMAX, steps=500)
    e = float(atoms.get_potential_energy())
    v = float(atoms.get_volume())
    syms = atoms.get_chemical_symbols()
    n_X = syms.count(elem)
    n_Al = syms.count("Al")
    n = len(atoms)
    rows.append({
        "system": name, "X": elem, "n_X": n_X, "n_Al": n_Al, "n_atoms": n,
        "energy_eV": e, "energy_per_atom_eV": e / n,
        "volume_A3": v, "volume_per_atom_A3": v / n,
        "converged": conv, "time_s": time.time() - t0,
    })
    ase_write(os.path.join(RELAX, f"{name}.extxyz"), atoms)
    print(f"{name}: E/n={e/n:.4f} V/n={v/n:.3f} a={v**(1/3):.4f} conv={conv} ({rows[-1]['time_s']:.1f}s)")

df = pd.DataFrame(rows)
df.to_csv(os.path.join(AN, "fcc_xal_sqs_volumes.csv"), index=False)
print("Wrote", os.path.join(AN, "fcc_xal_sqs_volumes.csv"))
