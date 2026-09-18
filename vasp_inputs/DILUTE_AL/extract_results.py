#!/usr/bin/env python3
"""Extract volumes/energies from completed dilute-Al VASP calculations."""

import os
import csv
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def read_contcar(path):
    with open(path) as f:
        lines = f.readlines()
    scale = float(lines[1].strip())
    vecs = [[float(x) for x in lines[i].split()] for i in (2, 3, 4)]
    v = [[c * scale for c in row] for row in vecs]
    vol = (v[0][0] * (v[1][1] * v[2][2] - v[1][2] * v[2][1])
           - v[0][1] * (v[1][0] * v[2][2] - v[1][2] * v[2][0])
           + v[0][2] * (v[1][0] * v[2][1] - v[1][1] * v[2][0]))
    counts = [int(x) for x in lines[6].split()]
    return abs(vol), sum(counts)


def read_energy(path):
    with open(path) as f:
        lines = f.readlines()
    for line in reversed(lines):
        if "F=" in line:
            parts = line.split()
            return float(parts[parts.index("F=") + 1])
    return None


def main():
    rows, not_run = [], []
    for d in sorted(os.listdir(BASE_DIR)):
        dp = os.path.join(BASE_DIR, d)
        if not os.path.isdir(dp):
            continue
        m = re.fullmatch(r"(fcc|bcc)_([A-Z][a-z]?)_(n(\d+)|imp\d+)", d)
        if not m:
            continue
        contcar = os.path.join(dp, "CONTCAR")
        if not os.path.exists(contcar) or os.path.getsize(contcar) == 0:
            not_run.append(d)
            continue
        vol, natoms = read_contcar(contcar)
        with open(os.path.join(dp, "POSCAR")) as f:
            pl = f.readlines()
        cnts = [int(x) for x in pl[6].split()]
        els = pl[5].split()
        n_al = cnts[els.index("Al")] if "Al" in els else 0
        osz = os.path.join(dp, "OSZICAR")
        e = read_energy(osz) if os.path.exists(osz) else None
        conv = "yes" if os.path.exists(os.path.join(dp, "OUTCAR")) and \
            "reached required accuracy" in open(
                os.path.join(dp, "OUTCAR"), errors="ignore").read() else "no"
        rows.append(dict(
            directory=d, lattice=m.group(1), host=m.group(2),
            n_Al=n_al, natoms=natoms, c_Al=n_al / natoms,
            volume_A3=vol, volume_per_atom_A3=vol / natoms,
            energy_eV=e, converged=conv))

    out = os.path.join(BASE_DIR, "dilute_al_results.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                           ["directory"])
        w.writeheader()
        w.writerows(rows)
    print(f"Extracted {len(rows)} results -> {out}")
    if not_run:
        print(f"Not yet run: {len(not_run)}")


if __name__ == "__main__":
    main()
