#!/usr/bin/env python3
"""Parse Henkelman ACF.dat into per-atom Bader volume/charge data."""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

from ase.io import read


ZVAL_PATTERN = re.compile(r"\bZVAL\s*=\s*([0-9]+(?:\.[0-9]*)?)")


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
            n_bader = float(fields[4])
            volume = float(fields[6])
        except ValueError:
            continue
        rows.append((index, n_bader, volume))
    return rows


def read_zvals(path: Path, species: list[str]) -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is required to compute dQ_e; generate POTCAR before parsing ACF.dat"
        )
    zvals = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = ZVAL_PATTERN.search(line)
        if match:
            zvals.append(float(match.group(1)))
    if len(zvals) != len(species):
        raise ValueError(
            f"POTCAR has {len(zvals)} ZVAL blocks, POSCAR has {len(species)} species"
        )
    return dict(zip(species, zvals))


def main() -> None:
    calc_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    acf = calc_dir / "ACF.dat"
    poscar = calc_dir / "POSCAR"
    if not acf.exists() or not poscar.exists():
        raise FileNotFoundError("ACF.dat and POSCAR are required")
    atoms = read(poscar, format="vasp")
    species = list(dict.fromkeys(atoms.get_chemical_symbols()))
    zval_by_element = read_zvals(calc_dir / "POTCAR", species)
    rows = parse_acf(acf)
    if len(rows) != len(atoms):
        raise ValueError(f"ACF.dat has {len(rows)} atoms, POSCAR has {len(atoms)}")
    output = calc_dir / "bader_per_atom.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        handle.write("# dQ_e = zval_e - n_bader_e; positive dQ_e means electron loss (cation-like).\n")
        writer.writerow(
            [
                "label",
                "seed",
                "atom_index",
                "element",
                "V_bader_A3",
                "n_bader_e",
                "zval_e",
                "dQ_e",
            ]
        )
        for (index, n_bader, volume), element in zip(rows, atoms.get_chemical_symbols()):
            zval = zval_by_element[element]
            d_q = zval - n_bader
            writer.writerow(
                [
                    calc_dir.name,
                    0,
                    index - 1,
                    element,
                    f"{volume:.8f}",
                    f"{n_bader:.8f}",
                    f"{zval:.8f}",
                    f"{d_q:.8f}",
                ]
            )
    print(output)


if __name__ == "__main__":
    main()
