#!/usr/bin/env python3
"""FCC-SRO vs FCC-SQS comparison for Ni(Al) near 70 % Al.

Generate short-range ordered (SRO) fcc configurations that maximise Ni-Al
nearest-neighbour bonds by simulated annealing on the bond-counting energy,
then relax with MACE-MP-0 and compare with SQS/random configurations.
"""
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


def unlike_bonds(atoms, cutoff=2.65):
    """Count Ni-Al nearest-neighbour pairs for 32-atom fcc supercell (a~3.6)."""
    d = atoms.get_all_distances(mic=True)
    n = len(atoms)
    syms = np.array(atoms.get_chemical_symbols())
    cnt = 0
    for i in range(n):
        for j in range(i + 1, n):
            if d[i, j] < cutoff and syms[i] != syms[j]:
                cnt += 1
    return cnt


def make_sro(n_al, n_steps=20000, seed=0, T0=3.0, T1=0.01):
    at = bulk("Ni", "fcc", a=3.6, cubic=True).repeat((2, 2, 2))
    n_atoms = len(at)
    syms = np.array(["Ni"] * n_atoms)
    rng = np.random.default_rng(seed)
    al_idx = rng.choice(n_atoms, size=n_al, replace=False)
    syms[al_idx] = "Al"
    at.set_chemical_symbols(list(syms))

    ni_pos = np.where(syms == "Ni")[0]
    al_pos = np.where(syms == "Al")[0]
    if len(ni_pos) == 0 or len(al_pos) == 0:
        return at

    current_E = -unlike_bonds(at)
    for step in range(n_steps):
        T = T0 * ((T1 / T0) ** (step / n_steps))
        i = rng.choice(ni_pos)
        j = rng.choice(al_pos)
        at[i].symbol = "Al"
        at[j].symbol = "Ni"
        new_E = -unlike_bonds(at)
        dE = new_E - current_E
        if dE < 0 or rng.random() < np.exp(-dE / max(T, 1e-6)):
            current_E = new_E
            ni_pos = np.where(np.array(at.get_chemical_symbols()) == "Ni")[0]
            al_pos = np.where(np.array(at.get_chemical_symbols()) == "Al")[0]
        else:
            # revert
            at[i].symbol = "Ni"
            at[j].symbol = "Al"
    return at


rows = []
N_ATOMS = 32
NSEED = 3
X = [0.5, 0.625, 0.6875, 0.75, 0.875]
for x_t in X:
    n_al = round(N_ATOMS * x_t)
    nseed = 1 if n_al in (0, N_ATOMS) else NSEED
    for seed in range(nseed):
        at = make_sro(n_al, seed=8000 * seed + round(x_t * 10000))
        n_al_actual = np.sum(np.array(at.get_chemical_symbols()) == "Al")
        x = n_al_actual / N_ATOMS
        name = f"fcc_NiAl_x{x:.4f}_SRO_s{seed}"
        e, v, conv = relax(at, name)
        rows.append({"structure_id": name, "x_Al": x, "seed": seed,
                     "n_atoms": N_ATOMS, "kind": "SRO",
                     "energy_eV": e, "volume_A3": v,
                     "V_per_atom_A3": v / N_ATOMS,
                     "a_fcc_A": (4 * v / N_ATOMS) ** (1 / 3),
                     "unlike_bonds": unlike_bonds(at), "converged": conv})

pd.DataFrame(rows).to_csv(os.path.join(AN, "niall_fcc_sro.csv"), index=False)
print("done:", len(rows))
