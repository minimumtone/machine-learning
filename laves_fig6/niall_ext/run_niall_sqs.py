#!/usr/bin/env python3
"""SQS fcc Ni(Al) full-composition sweep to test Vegard line and stability.

Uses icet generate_sqs_from_supercells on the same 32-atom fcc primitive
supercell as the original run_fig6_pipeline.py.  Computes both volume and
energy, and compares with the existing random-seed 2×2×2 data so we can
judge whether a truly random/ideal fcc solution (no Ni-Ni clustering) is
energetically stable.
"""
import os
import time

import pandas as pd
from ase.build import bulk
from ase.filters import FrechetCellFilter
from ase.io import write as ase_write
from ase.optimize import LBFGS
from icet import ClusterSpace
from icet.tools.structure_generation import generate_sqs_from_supercells
from mace.calculators import mace_mp

BASE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.join(BASE, "analysis")
RELAX = os.path.join(BASE, "relax")
for d in (AN, RELAX):
    os.makedirs(d, exist_ok=True)

FMAX = 0.02
CALC = mace_mp(model="medium", default_dtype="float64", device="cpu")

fcc_prim = bulk("Ni", "fcc", a=3.52)
fcc_cs = ClusterSpace(fcc_prim, cutoffs=[7.0, 4.5], chemical_symbols=["Ni", "Al"])
fcc_super = [fcc_prim.repeat((2, 4, 4))]  # 32 atoms

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
# n_al of 32 -> x = n_al / 32; pick same x grid as random sweep where possible
N_AL_LIST = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
NSEED = 2
SQS_STEPS = 15000

for n_al in N_AL_LIST:
    x = n_al / 32.0
    nseed = 1 if n_al in (0, 32) else NSEED
    for s in range(nseed):
        if n_al in (0, 32):
            sym = "Al" if n_al == 32 else "Ni"
            at = bulk(sym, "fcc", a=3.6 if sym == "Ni" else 4.05, cubic=True).repeat((2, 2, 2))
            rep = (2, 2, 2)
            sqs_id = "-"
        else:
            rep = (2, 4, 4)
            at = generate_sqs_from_supercells(
                fcc_cs, [sc.copy() for sc in fcc_super], n_steps=SQS_STEPS,
                random_seed=1000 * n_al + s,
                target_concentrations={"A": {"Ni": 1 - x, "Al": x}})
            sqs_id = f"sqs_s{s}"
        name = f"fcc_NiAl_x{x:.4f}_SQS{sqs_id}"
        e, v, conv = relax(at, name)
        rows.append({"structure_id": name, "x_Al": x, "n_al": n_al, "seed": s,
                         "n_atoms": len(at), "sqs_id": sqs_id, "energy_eV": e,
                         "volume_A3": v, "V_per_atom_A3": v / len(at),
                         "a_conv_A": (4 * v / len(at)) ** (1 / 3),
                         "converged": conv})

pd.DataFrame(rows).to_csv(os.path.join(AN, "niall_fcc_sqs.csv"), index=False)
print("done:", len(rows))
