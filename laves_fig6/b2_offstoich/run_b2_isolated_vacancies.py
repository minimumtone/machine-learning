#!/usr/bin/env python3
"""Dispersed (mutually non-adjacent) single vacancies in B2-NiAl (MACE-MP-0).

Complements run_b2_defect_clusters.py: the same point defect (V_Ni or V_Al) at
several concentrations, with the vacancies kept as far apart as possible so
that every vacancy is an isolated mono-vacancy.

  set A: 4x4x4 cell, n = 2..8 vacancies, 3 placements each.  From 400 random
         placements the three with the largest minimum vacancy-vacancy distance
         (minimum-image) are kept.
  set B: one vacancy in 3x3x3, 5x5x5 and 6x6x6 cells.

Usage:
  python run_b2_isolated_vacancies.py list
  python run_b2_isolated_vacancies.py run <worker> <n_workers>
  python run_b2_isolated_vacancies.py collect
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
RELAX = os.path.join(BASE, "relax_isolated_vacancies")
OUT = os.path.join(AN, "isolated_vacancies")
FMAX = 0.02
A0 = 2.8825
N_LIST = range(2, 9)
N_PLACE = 3
N_TRIAL = 400
REPS_SINGLE = (3, 5, 6)


def perfect(rep):
    return bulk("NiAl", crystalstructure="cesiumchloride", a=A0, cubic=True).repeat(rep)


def enumerate_configs():
    cfgs = []
    at = perfect(4)
    sym = np.array(at.get_chemical_symbols())
    dm = at.get_all_distances(mic=True)
    for host in ("Ni", "Al"):
        sites = np.where(sym == host)[0]
        for rep in REPS_SINGLE:
            s = np.array(perfect(rep).get_chemical_symbols())
            cfgs.append(dict(name=f"isoV_{host}_n1_rep{rep}", defect=f"V_{host}", rep=rep,
                             n_vac=1, sites=[int(np.where(s == host)[0][0])],
                             d_min_A=float(rep * A0)))
        for n in N_LIST:
            rng = np.random.default_rng(1000 * n + (0 if host == "Ni" else 1))
            best = {}
            for _ in range(N_TRIAL):
                pick = tuple(sorted(int(i) for i in rng.choice(sites, n, replace=False)))
                sub = dm[np.ix_(pick, pick)]
                best[pick] = float(sub[np.triu_indices(n, 1)].min())
            top = sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))[:N_PLACE]
            for k, (pick, dmin) in enumerate(top):
                cfgs.append(dict(name=f"isoV_{host}_n{n}_p{k}", defect=f"V_{host}", rep=4,
                                 n_vac=n, sites=list(pick), d_min_A=dmin))
    return cfgs


def run(worker, n_workers):
    import torch
    torch.set_num_threads(max(1, os.cpu_count() // n_workers))
    from ase.filters import FrechetCellFilter
    from ase.io import write as ase_write
    from ase.optimize import LBFGS
    from mace.calculators import mace_mp
    calc = mace_mp(model="medium", default_dtype="float64", device="cpu")
    os.makedirs(RELAX, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    for n, cfg in enumerate(enumerate_configs()):
        if n % n_workers != worker:
            continue
        out = os.path.join(OUT, cfg["name"] + ".json")
        if os.path.exists(out):
            continue
        at = perfect(cfg["rep"])
        for i in sorted(cfg["sites"], reverse=True):
            del at[i]
        at.calc = calc
        opt = LBFGS(FrechetCellFilter(at), logfile=None)
        t0 = time.time()
        opt.run(fmax=FMAX, steps=500)
        sym = at.get_chemical_symbols()
        res = dict(cfg, n_Ni=sym.count("Ni"), n_Al=sym.count("Al"), n_atoms=len(at),
                   energy_eV=float(at.get_potential_energy()),
                   volume_A3=float(at.get_volume()),
                   converged=bool(opt.converged()), n_steps=int(opt.get_number_of_steps()),
                   time_s=time.time() - t0)
        ase_write(os.path.join(RELAX, cfg["name"] + ".extxyz"), at)
        with open(out, "w") as f:
            json.dump(res, f)
        print(f"[w{worker}] {cfg['name']}: E={res['energy_eV']:.4f} conv={res['converged']} "
              f"steps={res['n_steps']} ({res['time_s']:.0f}s)", flush=True)


def collect():
    import pandas as pd
    with open(os.path.join(AN, "b2_offstoich_summary.json")) as f:
        s = json.load(f)
    rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(OUT, "*.json")))]
    for r in rows:
        r["x_Al"] = r["n_Al"] / r["n_atoms"]
        r["E_form_eV_atom"] = (r["energy_eV"] - r["n_Ni"] * s["mu_Ni_eV"]
                               - r["n_Al"] * s["mu_Al_eV"]) / r["n_atoms"]
        r["sites"] = " ".join(map(str, r["sites"]))
    df = pd.DataFrame(rows).sort_values(["defect", "n_vac", "rep", "name"])
    out = os.path.join(AN, "b2_isolated_vacancies.csv")
    df.to_csv(out, index=False)
    print(f"wrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "list":
        c = enumerate_configs()
        for x in c:
            print(x["name"], x["n_vac"], round(x["d_min_A"], 2))
        print(len(c))
    elif cmd == "run":
        run(int(sys.argv[2]), int(sys.argv[3]))
    elif cmd == "collect":
        collect()
