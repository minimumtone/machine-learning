#!/usr/bin/env python3
"""Additional random seeds for the Ni-vacancy model near x_Al = 0.54 (4x4x4 cell).

The existing 3-seed means at 9 and 10 Ni vacancies (x_Al = 0.538, 0.542) scatter
because the energy depends on how many 2NN V_Ni-V_Ni pairs (+0.15 eV each) a
placement happens to contain.  Seeds 3-12 are added for 8, 9 and 10 vacancies
(x_Al = 0.533, 0.538, 0.542); x_Al_target follows the existing rows so that
make_figures.py pools them with the old seeds.

Usage:
  python run_b2_vacNi_054_extra_seeds.py run <worker> <n_workers>
  python run_b2_vacNi_054_extra_seeds.py collect

Outputs:
  analysis/b2_offstoich_volumes_vacNi_054_extra_seeds.csv
  analysis/vacNi_054_extra_seeds/*.json, relax/*.extxyz
"""
import glob
import json
import os
import sys
import time

import numpy as np
from ase.build import bulk

BASE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.join(BASE, "analysis")
RELAX = os.path.join(BASE, "relax")
OUT = os.path.join(AN, "vacNi_054_extra_seeds")
FMAX = 0.02
REP = 4
NCELL = REP ** 3
SEEDS = range(3, 13)
# n_vac -> x_Al_target used by the existing rows (8 vacancies had none)
TARGETS = {8: 0.533333, 9: 0.54, 10: 0.542373}


def jobs():
    return [(n, s) for n in TARGETS for s in SEEDS]


def run(worker, n_workers):
    import torch
    torch.set_num_threads(2)
    from ase.filters import FrechetCellFilter
    from ase.io import write as ase_write
    from ase.optimize import LBFGS
    from mace.calculators import mace_mp
    calc = mace_mp(model="medium", default_dtype="float64", device="cpu")
    with open(os.path.join(AN, "b2_offstoich_summary.json")) as f:
        s = json.load(f)
    os.makedirs(OUT, exist_ok=True)
    for k, (n_vac, seed) in enumerate(jobs()):
        if k % n_workers != worker:
            continue
        name = f"b2_x{NCELL / (2 * NCELL - n_vac):.3f}_vacNi_n{n_vac}_extra_s{seed}"
        out = os.path.join(OUT, name + ".json")
        if os.path.exists(out):
            continue
        at = bulk("NiAl", crystalstructure="cesiumchloride", a=2.88, cubic=True).repeat(REP)
        ni_idx = [i for i, x in enumerate(at.get_chemical_symbols()) if x == "Ni"]
        rng = np.random.default_rng(7000 * seed + 31 * n_vac)
        del at[[int(i) for i in rng.choice(ni_idx, size=n_vac, replace=False)]]
        at.calc = calc
        opt = LBFGS(FrechetCellFilter(at), logfile=None)
        t0 = time.time()
        opt.run(fmax=FMAX, steps=500)
        e, v = float(at.get_potential_energy()), float(at.get_volume())
        n_ni, n_al = NCELL - n_vac, NCELL
        n = n_ni + n_al
        ase_write(os.path.join(RELAX, name + ".extxyz"), at)
        row = dict(structure_id=name, branch="vacancy", x_Al_target=TARGETS[n_vac],
                   x_Al=round(n_al / n, 6), seed=seed, n_Ni=n_ni, n_Al=n_al, n_atoms=n,
                   n_sites=2 * NCELL, energy_eV=e, volume_A3=v, V_per_atom_A3=v / n,
                   a_eff_A=(v / NCELL) ** (1.0 / 3.0),
                   E_form_eV_atom=(e - n_ni * s["mu_Ni_eV"] - n_al * s["mu_Al_eV"]) / n,
                   converged=bool(opt.converged()))
        with open(out, "w") as f:
            json.dump(row, f)
        print(f"[w{worker}] {name}: E={e:.4f} conv={row['converged']} "
              f"({time.time() - t0:.0f}s)", flush=True)


def collect():
    import pandas as pd
    rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(OUT, "*.json")))]
    df = pd.DataFrame(rows).sort_values(["n_Ni", "seed"], ascending=[False, True])
    p = os.path.join(AN, "b2_offstoich_volumes_vacNi_054_extra_seeds.csv")
    df.to_csv(p, index=False)
    print(f"wrote {p} ({len(df)} rows)")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(int(sys.argv[2]), int(sys.argv[3]))
    else:
        collect()
