#!/usr/bin/env python3
"""Parse Henkelman ACF.dat into per-atom Bader volume/charge data."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from ase.io import read


def parse_acf(path: Path):
    rows = []
    in_table = False
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text.startswith("--------"):
            in_table = not in_table
            continue
        if not in_table or not text or text.startswith("NUMBER"):
            continue
        fields = text.split()
        if len(fields) < 7:
            continue
        try:
            index = int(fields[0])
            charge = float(fields[4])
            volume = float(fields[6])
        except ValueError:
            continue
        rows.append((index, charge, volume))
    return rows


def main() -> None:
    calc_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    acf = calc_dir / "ACF.dat"
    poscar = calc_dir / "POSCAR"
    if not acf.exists() or not poscar.exists():
        raise FileNotFoundError("ACF.dat and POSCAR are required")
    atoms = read(poscar, format="vasp")
    rows = parse_acf(acf)
    if len(rows) != len(atoms):
        raise ValueError(f"ACF.dat has {len(rows)} atoms, POSCAR has {len(atoms)}")
    output = calc_dir / "bader_per_atom.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "seed", "atom_index", "element", "V_bader_A3", "charge_e"])
        for (index, charge, volume), element in zip(rows, atoms.get_chemical_symbols()):
            writer.writerow([calc_dir.name, 0, index - 1, element, f"{volume:.8f}", f"{charge:.8f}"])
    print(output)


if __name__ == "__main__":
    main()
