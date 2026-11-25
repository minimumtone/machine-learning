#!/usr/bin/env python3
"""Cell-size check of the key defect clusters: 6x6x6 B2 supercell (432 sites).

Re-uses run_b2_defect_clusters.py (same enumeration, names and relaxation
settings) with REP = 6 for all 117 configurations (the key ones in SELECT are
run first), and compares E_bind with the 4x4x4 values in
analysis/b2_defect_clusters.csv.  "pair_*_far" is the farthest separation of
each cell, so its distance differs between the two cell sizes.

Usage:
  python run_b2_defect_clusters_6x6x6.py list
  python run_b2_defect_clusters_6x6x6.py run <worker> <n_workers>
  python run_b2_defect_clusters_6x6x6.py collect
"""
import glob
import json
import os
import sys

import run_b2_defect_clusters as m

m.REP = 6
m.RELAX = os.path.join(m.BASE, "relax_defect_clusters_6x6x6")
m.OUT = os.path.join(m.AN, "defect_clusters_6x6x6")

SELECT = ["perfect", "iso_V_Ni", "iso_Al_Ni", "iso_V_Al", "iso_Ni_Al",
          "pair_V_Ni+V_Ni_2NN", "pair_V_Ni+V_Ni_3NN", "pair_V_Ni+V_Ni_5NN",
          "pair_V_Ni+V_Al_1NN", "pair_V_Ni+V_Al_4NN",
          "pair_V_Ni+Ni_Al_1NN", "pair_V_Ni+Ni_Al_4NN",
          "pair_Al_Ni+Ni_Al_1NN", "pair_V_Al+V_Al_2NN",
          "tri_V_Ni+V_Ni+Ni_Al_g0", "tri_V_Ni+V_Ni+Ni_Al_g1", "tri_V_Ni+V_Ni+Ni_Al_g2",
          "tri_V_Ni+V_Ni+V_Al_g1", "tri_V_Ni+V_Ni+V_Al_g2", "tri_V_Ni+V_Al+V_Al_g3",
          "tri_V_Ni+V_Ni+V_Ni_g0", "tri_V_Ni+V_Ni+V_Ni_g1"]



def fingerprint(types, idx, dm):
    """As m.fingerprint, but distances are snapped to the ideal shell value so that
    2a = 5.765 A is not split into 5.76 / 5.77 by round-off (it is not in 4x4x4)."""
    edges = []
    for i, j in m.itertools.combinations(range(len(idx)), 2):
        t = tuple(sorted((types[i], types[j])))
        r2 = round(4 * (float(dm[idx[i], idx[j]]) / m.A0) ** 2)
        edges.append((t, round(m.A0 * r2 ** 0.5 / 2 - 1e-9, 2)))
    return tuple(sorted(edges))


m.fingerprint = fingerprint
_all = m.enumerate_configs


def enumerate_configs():
    cfgs = _all()
    by = {c["name"]: c for c in cfgs}
    return [by[n] for n in SELECT] + [c for c in cfgs if c["name"] not in SELECT]


m.enumerate_configs = enumerate_configs


def collect():
    import pandas as pd
    rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(m.OUT, "*.json")))]
    by = {r["name"]: r for r in rows}
    e0 = by["perfect"]["energy_eV"]
    iso = {t: by[f"iso_{t}"]["energy_eV"] - e0 for t in m.TYPES}
    ref = pd.read_csv(os.path.join(m.AN, "b2_defect_clusters.csv")).set_index("name")
    out = []
    for r in rows:
        fd, dmax = m.final_defects(r)
        d = dict(name=r["name"], kind=r["kind"], geometry=r.get("geometry", ""),
                 n_atoms=r["n_atoms"], energy_eV=r["energy_eV"], converged=r["converged"],
                 n_steps=r["n_steps"], final_defects="+".join(fd), max_disp_A=dmax,
                 transformed=sorted(r["types"]) != fd or dmax > 0.8,
                 dE_6x6x6_eV=r["energy_eV"] - e0,
                 dE_4x4x4_eV=ref.loc[r["name"], "energy_eV"] - ref.loc["perfect", "energy_eV"])
        if len(r["types"]) >= 2:
            d["E_bind_6x6x6_eV"] = d["dE_6x6x6_eV"] - sum(iso[t] for t in r["types"])
            d["E_bind_4x4x4_eV"] = ref.loc[r["name"], "E_bind_eV"]
        out.append(d)
    order = [c["name"] for c in _all() if c["name"] in by]
    df = pd.DataFrame(out).set_index("name").loc[order].reset_index()
    p = os.path.join(m.AN, "b2_defect_clusters_6x6x6.csv")
    df.to_csv(p, index=False)
    print(df.drop(columns=["energy_eV", "final_defects"]).round(3).to_string())
    print(f"wrote {p} ({len(df)} rows)")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "list":
        for c in enumerate_configs():
            print(c["name"], c.get("geometry", ""), c["sites"])
    elif cmd == "run":
        m.run(int(sys.argv[2]), int(sys.argv[3]))
    elif cmd == "collect":
        collect()
