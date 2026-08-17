#!/usr/bin/env python3
"""Fetch, relax with MACE, and append Ni3Al4 reference data."""
import os, json, time, requests
import numpy as np
import pandas as pd
from ase import Atoms
from ase.filters import FrechetCellFilter
from ase.io import write as ase_write
from ase.optimize import LBFGS
from mace.calculators import mace_mp
import torch

torch.set_num_threads(2)

BASE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.join(BASE, "analysis")
API_KEY = os.environ["MP_API_KEY"]
URL = "https://api.materialsproject.org/materials"

def mp_get(endpoint, params):
    r = requests.get(f"{URL}/{endpoint}/", params=params,
                     headers={"X-API-KEY": API_KEY}, timeout=60)
    r.raise_for_status()
    return r.json()["data"]

# fetch Ni3Al4 thermo + core
data = mp_get("thermo", {"formula": "Ni3Al4",
                         "_fields": "material_id,formula_pretty,energy_per_atom,formation_energy_per_atom,energy_above_hull",
                         "_limit": 20})
d = min(data, key=lambda x: x.get("energy_above_hull", 1e9))
mpid = d["material_id"]
print("MP Ni3Al4:", mpid, "Ef=", d["formation_energy_per_atom"])
sdata = mp_get("core", {"material_ids": mpid, "_fields": "material_id,structure", "_limit": 5})
struct = sdata[0]["structure"]
cell = np.array(struct["lattice"]["matrix"])
symbols, frac = [], []
for site in struct["sites"]:
    spec = site["species"][0]
    symbols.append(spec["element"])
    frac.append(site["abc"])
atoms = Atoms(symbols=symbols, scaled_positions=frac, cell=cell, pbc=True)
print("atoms", len(atoms), symbols.count("Ni"), symbols.count("Al"))

# Store the MP structure separately before MACE relaxation so that
# mp_reference_structures.json has a consistent MP-origin volume/structure.
n_Ni = symbols.count("Ni")
n_Al = symbols.count("Al")
n = n_Ni + n_Al
mp_volume = atoms.get_volume()
mp_xyz = os.path.join(AN, "mp_Ni3Al4.extxyz")
ase_write(mp_xyz, atoms)

# relax with MACE
atoms_mace = atoms.copy()
calc = mace_mp(model="medium", default_dtype="float64", device="cpu")
atoms_mace.calc = calc
t0 = time.time()
opt = LBFGS(FrechetCellFilter(atoms_mace), logfile=None)
conv = opt.run(fmax=0.02, steps=500)
e = atoms_mace.get_potential_energy()
v = atoms_mace.get_volume()

with open(os.path.join(AN, "b2_offstoich_summary.json")) as f:
    summary = json.load(f)
mu_Ni = summary["mu_Ni_eV"]; mu_Al = summary["mu_Al_eV"]
Ef = (e - n_Ni*mu_Ni - n_Al*mu_Al) / n

mace_xyz = os.path.join(AN, "mace_Ni3Al4.extxyz")
ase_write(mace_xyz, atoms_mace)
print(f"MACE Ef={Ef:.4f} V/n={v/n:.3f} conv={conv}")

# append to mace_mp_ref_results.csv
csv = os.path.join(AN, "mace_mp_ref_results.csv")
df = pd.read_csv(csv)
if "Ni3Al4" not in df.label.values:
    df.loc[len(df)] = {
        "label": "Ni3Al4", "mp_id": mpid, "formula": "Ni3Al4",
        "n_atoms": n, "n_Ni": n_Ni, "n_Al": n_Al,
        "energy_eV": e, "energy_per_atom_eV": e/n,
        "volume_A3": v, "volume_per_atom_A3": v/n,
        "formation_energy_per_atom_eV": Ef,
        "mp_energy_per_atom": d["energy_per_atom"],
        "mp_formation_energy_per_atom": d["formation_energy_per_atom"],
        "time_s": time.time()-t0, "converged": conv
    }
    df.to_csv(csv, index=False)
    print("appended to mace_mp_ref_results.csv")

# append to mp_reference_structures.json
jsf = os.path.join(AN, "mp_reference_structures.json")
with open(jsf) as f:
    refs = json.load(f)
if not any(r["label"]=="Ni3Al4" for r in refs):
    refs.append({
        "label": "Ni3Al4", "formula": "Ni3Al4", "mp_id": mpid,
        "energy_per_atom_eV": d["energy_per_atom"],
        "formation_energy_per_atom_eV": d["formation_energy_per_atom"],
        "energy_above_hull_eV": d.get("energy_above_hull", 0.0),
        "n_atoms": n,
        "volume_A3": mp_volume,
        "volume_per_atom_A3": mp_volume / n,
        "mace_volume_A3": v,
        "mace_volume_per_atom_A3": v / n,
        "mace_structure_file": mace_xyz,
        "structure_file": mp_xyz,
    })
    with open(jsf, "w") as f:
        json.dump(refs, f, indent=2)
    print("appended to mp_reference_structures.json")
