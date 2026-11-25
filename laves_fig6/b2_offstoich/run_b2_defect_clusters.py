#!/usr/bin/env python3
"""Isolated point defects, defect pairs and defect triples in B2-NiAl (MACE-MP-0).

4x4x4 B2 supercell (128 sites), same relaxation settings as the other
b2_offstoich runs (FrechetCellFilter + LBFGS, fmax = 0.02 eV/A).

Point defects: V_Ni, Al_Ni (Ni sublattice), V_Al, Ni_Al (Al sublattice).
Pairs  : all 10 type combinations, every neighbour shell up to 5NN, plus the
         farthest separation available in the cell.
Triples: all 20 type combinations, every symmetry-distinct connected cluster
         (bond = 1NN or 2NN distance).

Usage:
  python run_b2_defect_clusters.py list
  python run_b2_defect_clusters.py run <worker> <n_workers>
  python run_b2_defect_clusters.py collect
"""
import glob
import itertools
import json
import os
import sys
import time

import numpy as np
from ase.build import bulk
from ase.io import read as ase_read

BASE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.join(BASE, "analysis")
RELAX = os.path.join(BASE, "relax_defect_clusters")
OUT = os.path.join(AN, "defect_clusters")

FMAX = 0.02
REP = 4
A0 = 2.8825
TYPES = ["V_Ni", "Al_Ni", "V_Al", "Ni_Al"]
SUBLATTICE = {"V_Ni": "Ni", "Al_Ni": "Ni", "V_Al": "Al", "Ni_Al": "Al"}
PAIR_CUT = 1.75 * A0   # includes 5NN (sqrt(3) a)
BOND_CUT = 1.05 * A0   # 1NN (sqrt(3)/2 a) and 2NN (a)


def perfect():
    return bulk("NiAl", crystalstructure="cesiumchloride", a=A0, cubic=True).repeat(REP)


def shell_name(d):
    r2 = round(4 * (d / A0) ** 2)
    return {3: "1NN", 4: "2NN", 8: "3NN", 11: "4NN", 12: "5NN"}.get(r2, f"r2x4={r2}")


def fingerprint(types, idx, dm):
    edges = []
    for i, j in itertools.combinations(range(len(idx)), 2):
        t = tuple(sorted((types[i], types[j])))
        edges.append((t, round(float(dm[idx[i], idx[j]]), 2)))
    return tuple(sorted(edges))


def enumerate_configs():
    at = perfect()
    sym = np.array(at.get_chemical_symbols())
    dm = at.get_all_distances(mic=True)
    sites = {s: np.where(sym == s)[0] for s in ("Ni", "Al")}
    cfgs = [dict(name="perfect", types=[], sites=[], kind="perfect")]
    for t in TYPES:
        cfgs.append(dict(name=f"iso_{t}", types=[t], sites=[int(sites[SUBLATTICE[t]][0])],
                         kind="isolated"))
    # pairs
    for t1, t2 in itertools.combinations_with_replacement(TYPES, 2):
        i0 = int(sites[SUBLATTICE[t1]][0])
        cand = [int(j) for j in sites[SUBLATTICE[t2]] if j != i0]
        seen, mult = {}, {}
        for j in cand:
            d = round(float(dm[i0, j]), 2)
            mult[d] = mult.get(d, 0) + 1
            if d <= PAIR_CUT and d not in seen:
                seen[d] = j
        dmax = max(round(float(dm[i0, j]), 2) for j in cand)
        jfar = [j for j in cand if round(float(dm[i0, j]), 2) == dmax][0]
        for d, j in sorted(seen.items()):
            cfgs.append(dict(name=f"pair_{t1}+{t2}_{shell_name(d)}", types=[t1, t2],
                             sites=[i0, j], kind="pair", geometry=shell_name(d),
                             multiplicity=mult[d]))
        cfgs.append(dict(name=f"pair_{t1}+{t2}_far", types=[t1, t2], sites=[i0, jfar],
                         kind="pair_far", geometry=f"far({dmax:.2f}A)", multiplicity=mult[dmax]))
    # triples: connected clusters
    for ts in itertools.combinations_with_replacement(TYPES, 3):
        seen, mult = {}, {}
        i0 = int(sites[SUBLATTICE[ts[0]]][0])
        for j in sites[SUBLATTICE[ts[1]]]:
            for k in sites[SUBLATTICE[ts[2]]]:
                idx = [i0, int(j), int(k)]
                if len(set(idx)) < 3:
                    continue
                d = [dm[idx[a], idx[b]] for a, b in ((0, 1), (0, 2), (1, 2))]
                if sum(x <= BOND_CUT for x in d) < 2:
                    continue
                fp = fingerprint(ts, idx, dm)
                seen.setdefault(fp, idx)
                mult[fp] = mult.get(fp, 0) + 1
        for n, (fp, idx) in enumerate(sorted(seen.items())):
            geo = "/".join(f"{a[0]}-{a[1]}:{shell_name(dd)}" if shell_name(dd)[0] != "r"
                           else f"{a[0]}-{a[1]}:{dd:.2f}A" for a, dd in fp)
            cfgs.append(dict(name="tri_" + "+".join(ts) + f"_g{n}", types=list(ts),
                             sites=idx, kind="triple", geometry=geo, multiplicity=mult[fp]))
    return cfgs


def build(cfg):
    at = perfect()
    kill = []
    for t, i in zip(cfg["types"], cfg["sites"]):
        if t.startswith("V_"):
            kill.append(i)
        else:
            at[i].symbol = t.split("_")[0]
    for i in sorted(kill, reverse=True):
        del at[i]
    return at


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
        at = build(cfg)
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


def final_defects(cfg):
    """Defect content after relaxation, from site occupancy relative to perfect B2."""
    ref = perfect()
    at = ase_read(os.path.join(RELAX, cfg["name"] + ".extxyz"))
    scale = (at.get_volume() / ref.get_volume()) ** (1 / 3)
    ref.set_cell(at.cell, scale_atoms=True)
    from ase.geometry import get_distances
    _, d = get_distances(ref.positions, at.positions, cell=at.cell, pbc=True)
    owner = d.argmin(axis=0)          # nearest ideal site of every atom
    occ = {}
    for a, s in enumerate(owner):
        occ.setdefault(int(s), []).append(at[a].symbol)
    out = []
    for s in range(len(ref)):
        o = occ.get(s, [])
        host = ref[s].symbol
        if len(o) == 0:
            out.append(f"V_{host}")
        elif len(o) == 1 and o[0] != host:
            out.append(f"{o[0]}_{host}")
        elif len(o) > 1:
            out.append(f"multi({','.join(o)})_{host}")
    return sorted(out), float(d.min(axis=0).max() / scale)


def collect():
    import pandas as pd
    with open(os.path.join(AN, "b2_offstoich_summary.json")) as f:
        s = json.load(f)
    mu_ni, mu_al = s["mu_Ni_eV"], s["mu_Al_eV"]
    rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(OUT, "*.json")))]
    by = {r["name"]: r for r in rows}
    mult = {c["name"]: c.get("multiplicity") for c in enumerate_configs()}
    for r in rows:
        r["multiplicity"] = mult[r["name"]]   # ordered placements with defect 1 fixed
    e0 = by["perfect"]["energy_eV"]
    mu_fu = e0 / (REP ** 3)            # mu_Ni + mu_Al in stoichiometric B2
    iso = {t: by[f"iso_{t}"]["energy_eV"] - e0 for t in TYPES}   # raw dE vs perfect
    for r in rows:
        k = len(r["types"])
        n = r["n_atoms"]
        r["x_Al"] = r["n_Al"] / n
        r["E_form_eV_atom"] = (r["energy_eV"] - r["n_Ni"] * mu_ni - r["n_Al"] * mu_al) / n
        r["defects"] = "+".join(r["types"])
        fd, dmax = final_defects(r)
        r["final_defects"] = "+".join(fd)
        r["max_disp_A"] = dmax
        # changed defect content, or an atom left midway between two sites (> 0.8 A)
        r["transformed"] = sorted(r["types"]) != fd or dmax > 0.8
        if k >= 2:
            r["E_bind_eV"] = (r["energy_eV"] - e0) - sum(iso[t] for t in r["types"])
        # excess atoms relative to perfect cell
        r["dN_Ni"] = r["n_Ni"] - REP ** 3
        r["dN_Al"] = r["n_Al"] - REP ** 3
        r.pop("sites", None)
        r.pop("types", None)
    df = pd.DataFrame(rows)
    df["mu_NiAl_fu_eV"] = mu_fu
    order = {"perfect": 0, "isolated": 1, "pair": 2, "pair_far": 3, "triple": 4}
    df = df.sort_values(by=["kind", "defects", "name"], key=lambda c: c.map(order) if c.name == "kind" else c)
    out = os.path.join(AN, "b2_defect_clusters.csv")
    df.to_csv(out, index=False)
    print(f"wrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "list":
        c = enumerate_configs()
        for x in c:
            print(x["name"], x.get("geometry", ""))
        from collections import Counter
        print(Counter(x["kind"] for x in c), len(c))
    elif cmd == "run":
        run(int(sys.argv[2]), int(sys.argv[3]))
    elif cmd == "collect":
        collect()
