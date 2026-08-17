#!/usr/bin/env python3
"""Fetch Materials Project (PBE) structures and energies for Ni-Al compounds.

Outputs `analysis/mp_reference_structures.json` and relaxed-geometry `.extxyz`
files that can be read directly by ASE for MACE re-relaxation.
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


def structure_from_mp(d):
    s = d["structure"]
    cell = np.array(s["lattice"]["matrix"])
    symbols = []
    frac = []
    for site in s["sites"]:
        spec = site["species"][0]
        symbols.append(spec["element"])
        frac.append(site["abc"])
    return Atoms(symbols=symbols, scaled_positions=frac, cell=cell, pbc=True)


TARGETS = {
    "Ni": {"formula": "Ni", "proto": "fcc"},
    "Al": {"formula": "Al", "proto": "fcc"},
    "B2_NiAl": {"formula": "NiAl", "proto": "B2"},
    "L12_Ni3Al": {"formula": "Ni3Al", "proto": "L1_2"},
    "Ni5Al3": {"formula": "Ni5Al3", "proto": ""},
    "Ni2Al3": {"formula": "Ni2Al3", "proto": ""},
    "NiAl3": {"formula": "NiAl3", "proto": ""},
}

rows = []
for name, info in TARGETS.items():
    data = mp_get("thermo", {"formula": info["formula"],
                             "_fields": "material_id,formula_pretty,energy_per_atom,formation_energy_per_atom,energy_above_hull",
                             "_limit": 20})
    # pick lowest energy_above_hull (stable if zero)
    d = min(data, key=lambda x: x.get("energy_above_hull", 1e9))
    mpid = d["material_id"]
    struct_data = mp_get("core", {"material_ids": mpid,
                                  "_fields": "material_id,structure",
                                  "_limit": 5})
    if not struct_data:
        print(f"  {name}: no structure returned for {mpid}")
        continue
    d["structure"] = struct_data[0]["structure"]
    atoms = structure_from_mp(d)
    out_xyz = os.path.join(AN, f"mp_{name}.extxyz")
    ase_write(out_xyz, atoms)
    rows.append({
        "label": name,
        "formula": info["formula"],
        "mp_id": d["material_id"],
        "energy_per_atom_eV": d["energy_per_atom"],
        "formation_energy_per_atom_eV": d["formation_energy_per_atom"],
        "energy_above_hull_eV": d["energy_above_hull"],
        "n_atoms": len(atoms),
        "volume_A3": atoms.get_volume(),
        "volume_per_atom_A3": atoms.get_volume() / len(atoms),
        "structure_file": out_xyz,
    })
    print(f"{name}: {d['material_id']} E hull={d['energy_above_hull']:.4f} eV V/atom={rows[-1]['volume_per_atom_A3']:.3f}")

out_json = os.path.join(AN, "mp_reference_structures.json")
with open(out_json, "w") as f:
    json.dump(rows, f, indent=2)
print(f"Wrote {out_json}")
