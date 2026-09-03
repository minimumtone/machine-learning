#!/usr/bin/env python3
"""Extra 5x5x5 SQS high-Al vacancy seeds (seeds 3-9) to reduce scatter."""
import os, time, json, math
import numpy as np
import pandas as pd
from ase.build import bulk
from ase.io import write as ase_write
from ase.filters import FrechetCellFilter
from ase.optimize import LBFGS
from icet import ClusterSpace
from icet.tools.structure_generation import generate_sqs_from_supercells
from mace.calculators import mace_mp

BASE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.join(BASE, "analysis")
RELAX = os.path.join(BASE, "relax")
os.makedirs(RELAX, exist_ok=True)

REP = 5
NCELL = REP ** 3
X_TARGETS = [0.68, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80]
SEEDS = list(range(3, 10))  # 3-9
FMAX = 0.02

with open(os.path.join(AN, "b2_offstoich_summary.json")) as f:
    summary = json.load(f)
MU_NI = summary["mu_Ni_eV"]
MU_AL = summary["mu_Al_eV"]

prim = bulk("NiAl", crystalstructure="cesiumchloride", a=2.88, cubic=True)
cs = ClusterSpace(prim, [5.0], [["Ni", "X"], ["Al"]])
supercell = prim.repeat((REP, REP, REP))
calc = mace_mp(model="medium", default_dtype="float64", device="cpu")

rows = []
for x in X_TARGETS:
    n_ni = int(round(NCELL * (1.0 - x) / x))
    n_ni = max(0, min(n_ni, NCELL))
    n_vac = NCELL - n_ni
    n_atoms = n_ni + NCELL
    x_actual_target = NCELL / (NCELL + n_ni)
    for seed in SEEDS:
        t0 = time.time()
        sqs = generate_sqs_from_supercells(
            cs, [supercell],
            {"A": {"Ni": n_ni / NCELL, "X": n_vac / NCELL}, "B": {"Al": 1.0}},
            n_steps=5000, random_seed=seed
        )
        keep = [i for i, s in enumerate(sqs.get_chemical_symbols()) if s != "X"]
        at = sqs[keep]
        name = f"b2_x{x:.3f}_vacNi_alrich_sqs_{REP}x{REP}x{REP}_s{seed}"
        at.set_pbc(True)
        at.info["rep"] = REP
        at.info["n_sites"] = 2 * NCELL
        at.calc = calc
        opt = LBFGS(FrechetCellFilter(at), logfile=None)
        opt.run(fmax=FMAX, steps=500)
        conv = opt.converged()
        v = at.get_volume()
        e = at.get_potential_energy()
        syms = at.get_chemical_symbols()
        n_al = syms.count("Al")
        n_ni_actual = syms.count("Ni")
        n = len(syms)
        rows.append(dict(
            structure_id=name, branch="vacancy", x_Al_target=x,
            x_Al=round(n_al / n, 6), seed=seed, n_Ni=n_ni_actual, n_Al=n_al, n_atoms=n,
            n_sites=2 * NCELL, energy_eV=e, volume_A3=v,
            V_per_atom_A3=v / n, a_eff_A=(v / NCELL) ** (1.0 / 3.0),
            E_form_eV_atom=(e - n_ni_actual * MU_NI - n_al * MU_AL) / n,
            converged=conv,
            sqs_cell=f"{REP}x{REP}x{REP}"
        ))
        ase_write(os.path.join(RELAX, name + ".extxyz"), at)
        print(f"{name}: x={n_al/n:.4f} V={v/n:.4f} conv={conv} time={time.time()-t0:.1f}s")

out = os.path.join(AN, "b2_offstoich_volumes_alrich_vac_5x5x5_sqs_extra.csv")
pd.DataFrame(rows).to_csv(out, index=False)
print("Wrote", out)
