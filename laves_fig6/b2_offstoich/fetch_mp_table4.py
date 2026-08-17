#!/usr/bin/env python3
"""Fetch MP reference structures for Yamanouchi & Miura Table 4 analogues.

Systems: B2-type aluminides (CoAl, PdAl, RhAl, IrAl) and corresponding
fcc solid-solution end members, plus Nb-Ni2 / Nb-Al2 for C14 reference.
"""
import json, os, requests
import numpy as np
from ase import Atoms
from ase.io import write as ase_write

BASE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.join(BASE, "analysis")
os.makedirs(AN, exist_ok=True)

API_KEY = os.environ["MP_API_KEY"]
BASE_URL = "https://api.materialsproject.org/materials"


def mp_get(endpoint, params):
    r = requests.get(f"{BASE_URL}/{endpoint}/", params=params,
                     headers={"X-API-KEY": API_KEY}, timeout=60)
    r.raise_for_status()
    return r.json()["data"]


def structure_from_mp(s):
    cell = np.array(s["lattice"]["matrix"])
    symbols = []
    frac = []
    for site in s["sites"]:
        symbols.append(site["species"][0]["element"])
        frac.append(site["abc"])
    return Atoms(symbols=symbols, scaled_positions=frac, cell=cell, pbc=True)


TARGETS = [
    ("B2_CoAl", "CoAl"),
    ("B2_PdAl", "PdAl"),
    ("B2_RhAl", "RhAl"),
    ("B2_IrAl", "IrAl"),
    ("fcc_Co", "Co"),
    ("fcc_Pd", "Pd"),
    ("fcc_Rh", "Rh"),
    ("fcc_Ir", "Ir"),
    ("C14_NbNi2", "NbNi2"),
]

rows = []
for name, formula in TARGETS:
    thermo = mp_get("thermo", {"formula": formula,
                                "_fields": "material_id,formula_pretty,energy_per_atom,formation_energy_per_atom,energy_above_hull",
                                "_limit": 20})
    d = min(thermo, key=lambda x: x.get("energy_above_hull", 1e9))
    mpid = d["material_id"]
    struct_data = mp_get("core", {"material_ids": mpid,
                                  "_fields": "material_id,structure",
                                  "_limit": 5})
    if not struct_data:
        print(f"{name}: no structure for {mpid}")
        continue
    atoms = structure_from_mp(struct_data[0]["structure"])
    out = os.path.join(AN, f"mp_{name}.extxyz")
    ase_write(out, atoms)
    rows.append({
        "label": name, "formula": formula, "mp_id": mpid,
        "energy_per_atom_eV": d["energy_per_atom"],
        "formation_energy_per_atom_eV": d["formation_energy_per_atom"],
        "energy_above_hull_eV": d["energy_above_hull"],
        "n_atoms": len(atoms),
        "volume_per_atom_A3": atoms.get_volume() / len(atoms),
        "structure_file": out,
    })
    print(f"{name}: {mpid} hull={d['energy_above_hull']:.4f} V/atom={rows[-1]['volume_per_atom_A3']:.3f}")

out_json = os.path.join(AN, "mp_table4_structures.json")
with open(out_json, "w") as f:
    json.dump(rows, f, indent=2)
print(f"Wrote {out_json}")
