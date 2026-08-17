#!/usr/bin/env python3
"""MACE-relax MP reference structures for Yamanouchi Table 4 analogues."""
import json, os, time
import numpy as np
import pandas as pd
from ase.filters import FrechetCellFilter
from ase.io import read, write as ase_write
from ase.optimize import LBFGS
from mace.calculators import mace_mp
import torch

torch.set_num_threads(1)

BASE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.join(BASE, "analysis")

CALC = mace_mp(model="medium", default_dtype="float64", device="cpu")
FMAX = 0.02

with open(os.path.join(AN, "mp_table4_structures.json")) as f:
    refs = {r["label"]: r for r in json.load(f)}

# elemental references: Ni/Al from existing MACE benchmark; relax Co/Pd/Rh/Ir fcc now.
mace_ref = pd.read_csv(os.path.join(AN, "mace_mp_ref_results.csv"))
mu = {
    "Ni": float(mace_ref[mace_ref.label == "Ni"].energy_per_atom_eV.values[0]),
    "Al": float(mace_ref[mace_ref.label == "Al"].energy_per_atom_eV.values[0]),
}


def relax_atoms(atoms):
    atoms.calc = CALC
    opt = LBFGS(FrechetCellFilter(atoms), logfile=None)
    conv = opt.run(fmax=FMAX, steps=500)
    return float(atoms.get_potential_energy()), float(atoms.get_volume()), conv


# pass 1: fcc end members
for name in ["fcc_Co", "fcc_Pd", "fcc_Rh", "fcc_Ir"]:
    r = refs[name]
    atoms = read(r["structure_file"])
    e, v, conv = relax_atoms(atoms)
    elem = r["formula"]
    mu[elem] = e / len(atoms)
    out = os.path.join(AN, f"mace_{name}.extxyz")
    ase_write(out, atoms)
    print(f"{name}: {elem} mu={mu[elem]:.4f} eV V/n={v/len(atoms):.3f} conv={conv}")

# pass 2: compounds (B2 + C14_NbNi2)
rows = []
for name in ["B2_CoAl", "B2_PdAl", "B2_RhAl", "B2_IrAl", "C14_NbNi2"]:
    r = refs.get(name)
    if not r:
        continue
    atoms = read(r["structure_file"])
    e, v, conv = relax_atoms(atoms)
    syms = atoms.get_chemical_symbols()
    counts = {}
    for s in syms:
        counts[s] = counts.get(s, 0) + 1
    n = len(syms)
    missing = [el for el in counts if el not in mu]
    if missing:
        e_form = np.nan  # e.g. C14_NbNi2 lacks a Nb reference here
    else:
        e_form = (e - sum(counts[el] * mu[el] for el in counts)) / n
    rows.append({
        "label": name, "mp_id": r["mp_id"], "formula": r["formula"],
        "n_atoms": n,
        "energy_eV": e, "energy_per_atom_eV": e / n,
        "volume_A3": v, "volume_per_atom_A3": v / n,
        "formation_energy_per_atom_eV": e_form,
        "mp_energy_per_atom": r["energy_per_atom_eV"],
        "mp_formation_energy_per_atom": r["formation_energy_per_atom_eV"],
        "converged": conv,
    })
    out = os.path.join(AN, f"mace_{name}.extxyz")
    ase_write(out, atoms)
    e_form_str = f"{e_form:.3f}" if not (isinstance(e_form, float) and np.isnan(e_form)) else "NaN"
    print(f"{name}: E/n={e/n:.4f} V/n={v/n:.3f} Ef={e_form_str} conv={conv}")

df = pd.DataFrame(rows)
df.to_csv(os.path.join(AN, "mace_table4_results.csv"), index=False)
print("Wrote", os.path.join(AN, "mace_table4_results.csv"))
