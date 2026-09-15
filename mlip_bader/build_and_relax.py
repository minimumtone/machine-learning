#!/usr/bin/env python3
"""Build and relax 128-atom BCC cells with MACE-MP-0 small."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
from ase.build import bulk
from ase.filters import FrechetCellFilter
from ase.io import write
from ase.optimize import FIRE
from mace.calculators import mace_mp


ROOT = Path(__file__).resolve().parent
STRUCTURES = ROOT / "structures"
SUMMARY = ROOT / "relax_results.csv"
LOG = ROOT / "build_and_relax_output.txt"
ELEMENTS = ["Hf", "Nb", "Ta", "Ti", "Zr", "Al", "V"]
REFRACTORY = ["Hf", "Nb", "Ta", "Ti", "Zr"]
AL_NB_TI_V = ["Al", "Nb", "Ti", "V"]
HEA_COMPOSITIONS = {
    "HfNbTaTiZr": {"Hf": 26, "Nb": 26, "Ta": 26, "Ti": 25, "Zr": 25},
    "AlNbTiV": {"Al": 32, "Nb": 32, "Ti": 32, "V": 32},
}

# Initial BCC guesses only; the values used for all mixed cells come from
# the preceding MACE-relaxed pure-element calculations.
INITIAL_PURE_A = {
    "Hf": 3.54,
    "Nb": 3.30,
    "Ta": 3.30,
    "Ti": 3.30,
    "Zr": 3.64,
    "Al": 2.98,
    "V": 3.03,
}


def log_line(text: str) -> None:
    print(text, flush=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(text + "\n")


def bcc_cell(element: str, a: float):
    return bulk(element, "bcc", a=a, cubic=True).repeat((4, 4, 4))


def weighted_lattice(composition: dict[str, int], pure_a: dict[str, float]) -> float:
    total = sum(composition.values())
    return sum(pure_a[element] * count for element, count in composition.items()) / total


def set_random_occupation(atoms, composition: dict[str, int], seed: int) -> None:
    symbols = [element for element, count in composition.items() for _ in range(count)]
    if len(symbols) != len(atoms):
        raise ValueError(f"occupation has {len(symbols)} atoms, cell has {len(atoms)}")
    np.random.default_rng(seed).shuffle(symbols)
    atoms.set_chemical_symbols(symbols)


def relax(atoms, calculator, label: str, seed: int) -> dict:
    atoms.calc = calculator
    filter_atoms = FrechetCellFilter(atoms, hydrostatic_strain=False)
    optimizer = FIRE(filter_atoms, logfile=None, maxstep=0.15)
    started = time.perf_counter()
    optimizer.run(fmax=0.01, steps=500)
    if not optimizer.converged():
        optimizer.run(fmax=0.01, steps=1500)
    elapsed = time.perf_counter() - started
    converged = bool(optimizer.converged())
    nsteps = int(getattr(optimizer, "nsteps", 0))
    path = STRUCTURES / f"{label}_s{seed}.extxyz"
    write(path, atoms, format="extxyz")
    volume = float(atoms.get_volume())
    a_bcc = float((2.0 * volume / len(atoms)) ** (1.0 / 3.0))
    energy = float(atoms.get_potential_energy())
    fmax_final = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
    result = {
        "label": label,
        "seed": seed,
        "natoms": len(atoms),
        "volume_A3": volume,
        "a_bcc_A": a_bcc,
        "energy_eV": energy,
        "converged": converged,
        "nsteps": nsteps,
        "fmax_final": fmax_final,
    }
    log_line(
        f"{label:15s} seed={seed} converged={converged} "
        f"steps={nsteps:3d} volume={volume:.6f} a={a_bcc:.6f} "
        f"energy={energy:.6f} elapsed={elapsed:.1f}s"
    )
    return result


def pair_labels(elements: list[str]) -> list[tuple[str, str]]:
    return [(elements[i], elements[j]) for i in range(len(elements)) for j in range(i + 1, len(elements))]


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    STRUCTURES.mkdir(parents=True, exist_ok=True)
    LOG.write_text(
        "MACE-MP-0 small, float64, CPU; FrechetCellFilter hydrostatic_strain=False; "
        "FIRE fmax=0.01 eV/A, max 500+1500 steps for unconverged cells\n",
        encoding="utf-8",
    )
    started = time.perf_counter()
    log_line("Loading MACE-MP-0 small (the model is cached by mace-torch).")
    calculator = mace_mp(model="small", default_dtype="float64", device="cpu")

    results = []
    pure_a: dict[str, float] = {}
    log_line("Relaxing seven pure BCC references.")
    for element in ELEMENTS:
        atoms = bcc_cell(element, INITIAL_PURE_A[element])
        result = relax(atoms, calculator, element, 0)
        results.append(result)
        pure_a[element] = result["a_bcc_A"]

    log_line("Relaxing 15 random 50:50 binary cells, three seeds each.")
    pairs = list(dict.fromkeys(pair_labels(REFRACTORY) + pair_labels(AL_NB_TI_V)))
    for first, second in pairs:
        label = f"{first}-{second}"
        composition = {first: 64, second: 64}
        a = weighted_lattice(composition, pure_a)
        for seed in range(3):
            atoms = bcc_cell(first, a)
            set_random_occupation(atoms, composition, seed)
            results.append(relax(atoms, calculator, label, seed))

    log_line("Relaxing two multicomponent HEA cells, three seeds each.")
    for label, composition in HEA_COMPOSITIONS.items():
        a = weighted_lattice(composition, pure_a)
        for seed in range(3):
            first = next(iter(composition))
            atoms = bcc_cell(first, a)
            set_random_occupation(atoms, composition, seed)
            results.append(relax(atoms, calculator, label, seed))

    with SUMMARY.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "label",
                "seed",
                "natoms",
                "volume_A3",
                "a_bcc_A",
                "energy_eV",
                "converged",
                "nsteps",
                "fmax_final",
            ],
        )
        writer.writeheader()
        writer.writerows(results)
    elapsed = time.perf_counter() - started
    log_line(f"Wrote {len(results)} relaxation rows to {SUMMARY}.")
    log_line(f"Total relaxation wall time: {elapsed:.1f} s ({elapsed / 3600:.2f} h).")


if __name__ == "__main__":
    main()
