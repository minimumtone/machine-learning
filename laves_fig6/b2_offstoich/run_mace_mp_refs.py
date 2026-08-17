#!/usr/bin/env python3
"""Re-relax the MP reference Ni-Al structures with MACE-MP-0 medium.

Produces `analysis/mace_mp_ref_results.csv` for benchmark against MP DFT and
experimental values.
"""
import json, os, time
import pandas as pd
from ase.filters import FrechetCellFilter
from ase.io import read, write as ase_write
from ase.optimize import LBFGS
from mace.calculators import mace_mp
import torch

torch.set_num_threads(2)

BASE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.join(BASE, "analysis")

CALC = mace_mp(model="medium", default_dtype="float64", device="cpu")
FMAX = 0.02

with open(os.path.join(AN, "mp_reference_structures.json")) as f:
    refs = json.load(f)

with open(os.path.join(AN, "b2_offstoich_summary.json")) as f:
    summary = json.load(f)
mu_Ni = summary["mu_Ni_eV"]
mu_Al = summary["mu_Al_eV"]

rows = []
for r in refs:
    atoms = read(r["structure_file"])
    atoms.calc = CALC
    name = r["label"]
    t0 = time.time()
    opt = LBFGS(FrechetCellFilter(atoms), logfile=None)
    converged = opt.run(fmax=FMAX, steps=500)
    e = float(atoms.get_potential_energy())
    v = float(atoms.get_volume())
    n_Ni = sum(1 for s in atoms.get_chemical_symbols() if s == "Ni")
    n_Al = sum(1 for s in atoms.get_chemical_symbols() if s == "Al")
    n = n_Ni + n_Al
    symbol_dict = {"Ni": n_Ni, "Al": n_Al}
    rows.append({
        "label": name,
        "mp_id": r["mp_id"],
        "formula": r["formula"],
        "n_atoms": n, "n_Ni": n_Ni, "n_Al": n_Al,
        "energy_eV": e, "energy_per_atom_eV": e / n,
        "volume_A3": v, "volume_per_atom_A3": v / n,
        "formation_energy_per_atom_eV": (e - n_Ni * mu_Ni - n_Al * mu_Al) / n,
        "mp_energy_per_atom": r["energy_per_atom_eV"],
        "mp_formation_energy_per_atom": r["formation_energy_per_atom_eV"],
        "time_s": time.time() - t0,
        "converged": converged,
    })
    out = os.path.join(AN, f"mace_{name}.extxyz")
    ase_write(out, atoms)
    print(f"{name}: E/n={e/n:.4f} V/n={v/n:.3f} conv={converged} ({rows[-1]['time_s']:.1f}s)")

df = pd.DataFrame(rows)
df.to_csv(os.path.join(AN, "mace_mp_ref_results.csv"), index=False)
print("Wrote", os.path.join(AN, "mace_mp_ref_results.csv"))
