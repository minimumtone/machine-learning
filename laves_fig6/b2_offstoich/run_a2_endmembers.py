#!/usr/bin/env python3
"""Compute A2 (bcc) Ni and Al end-member energies for the 4SL/8SL B2 model."""
import os
import time
import numpy as np
import pandas as pd
from ase.io import write
from ase.build import bulk
from ase.optimize import LBFGS
from ase.filters import FrechetCellFilter
from mace.calculators import mace_mp

BASE = os.path.dirname(os.path.abspath(__file__))
CALC = mace_mp(model="medium", default_dtype="float64", device="cpu")
FMAX = 0.02


def relax_a2(element, a_guess, rep=(2,2,2)):
    atoms = bulk(element, "bcc", a=a_guess, cubic=True) * rep
    atoms.calc = CALC
    atoms = FrechetCellFilter(atoms)
    opt = LBFGS(atoms, logfile=None)
    t0 = time.time()
    opt.run(fmax=FMAX, steps=300)
    relaxed = atoms.atoms
    e = float(relaxed.get_potential_energy())
    v = float(relaxed.get_volume())
    a = (v * 2 / len(relaxed)) ** (1.0 / 3.0)  # bcc conventional a: 2 atoms per cube
    e_atom = e / len(relaxed)
    print(f"[A2-{element}] a={a:.4f} Å E/atom={e_atom:.6f} eV conv={opt.converged()} ({time.time()-t0:.1f}s)", flush=True)
    fn = os.path.join(BASE, "relax", f"a2_{element.lower()}.extxyz")
    write(fn, relaxed)
    return {
        "element": element,
        "a_A": a,
        "E_total_eV": e,
        "n_atoms": len(relaxed),
        "E_atom_eV": e_atom,
        "V_atom_A3": v / len(relaxed),
        "converged": opt.converged(),
        "file": fn,
    }


rows = [
    relax_a2("Ni", 2.82, rep=(4, 4, 4)),
    relax_a2("Al", 3.25, rep=(4, 4, 4)),
]

df = pd.DataFrame(rows)
out = os.path.join(BASE, "analysis", "a2_endmember_energies.csv")
df.to_csv(out, index=False)
print(df.to_string(index=False))
