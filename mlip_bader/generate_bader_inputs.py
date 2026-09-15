#!/usr/bin/env python3
"""Generate VASP single-point/Bader inputs from selected MACE relaxed cells."""

from __future__ import annotations

import os
from pathlib import Path

from ase.io import read, write


ROOT = Path(__file__).resolve().parent
STRUCTURES = ROOT / "structures"
OUT = ROOT / "vasp_bader"

POTCAR_VARIANTS = {
    "Al": "Al",
    "Hf": "Hf_pv",
    "Nb": "Nb_pv",
    "Ta": "Ta_pv",
    "Ti": "Ti_pv",
    "V": "V_sv",
    "Zr": "Zr_sv",
}
MAGMOM = {"Al": 0.0, "Hf": 0.0, "Nb": 0.0, "Ta": 0.0, "Ti": 0.0, "V": 0.0, "Zr": 0.0}
SELECTED = [
    "HfNbTaTiZr",
    "AlNbTiV",
    "Al-Nb",
    "Al-V",
    "Nb-Ti",
    "Hf-Nb",
    "Ta-Zr",
    "Hf",
    "Nb",
    "Ta",
    "Ti",
    "Zr",
    "Al",
    "V",
]


def incar(atoms, label: str) -> str:
    magmom = " ".join(f"{MAGMOM[element]:.1f}" for element in atoms.get_chemical_symbols())
    return "\n".join(
        [
            f"SYSTEM = {label} MACE relaxed Bader single point",
            "ENCUT  = 520",
            "PREC   = Accurate",
            "EDIFF  = 1E-6",
            "NELM   = 300",
            "LREAL  = .FALSE.",
            "IBRION = -1",
            "NSW    = 0",
            "ISIF   = 2",
            "ISMEAR = 1",
            "SIGMA  = 0.1",
            "GGA    = PE",
            "ALGO   = Normal",
            "ISPIN  = 2",
            f"MAGMOM = {magmom}",
            "LORBIT = 11",
            "LCHARG = .TRUE.",
            "LAECHG = .TRUE.",
            "LWAVE  = .FALSE.",
            "NGXF   = 320",
            "NGYF   = 320",
            "NGZF   = 320",
            "NCORE  = 4",
            "",
            "# NGXF/YF/ZF are explicitly doubled relative to the default estimate.",
            "# Values are 320 for the approximately 13.3 A, 128-atom cells.",
            "",
        ]
    )


def kpoints() -> str:
    return "Gamma-centered 2x2x2\n0\nGamma\n2 2 2\n0 0 0\n"


def write_input(label: str) -> list[str]:
    source = STRUCTURES / f"{label}_s0.extxyz"
    if not source.exists():
        raise FileNotFoundError(source)
    atoms = read(source)
    target = OUT / label
    target.mkdir(parents=True, exist_ok=True)
    write(target / "POSCAR", atoms, format="vasp", direct=True, sort=True, vasp5=True)
    (target / "INCAR").write_text(incar(atoms, label), encoding="utf-8")
    (target / "KPOINTS").write_text(kpoints(), encoding="utf-8")
    return sorted(set(atoms.get_chemical_symbols()))


def write_make_potcar(records: dict[str, list[str]]) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "# Generate POTCAR files using the repository's PBE POTCAR variants.",
        'set -euo pipefail',
        'if [[ -z "${VASP_PP_PATH:-}" ]]; then echo "Set VASP_PP_PATH first"; exit 1; fi',
        'cd "$(dirname "${BASH_SOURCE[0]}")"',
        'PP_DIR="$VASP_PP_PATH/potpaw_PBE"',
        "",
    ]
    for label, elements in records.items():
        parts = " ".join(f'"$PP_DIR"/{POTCAR_VARIANTS[element]}/POTCAR' for element in elements)
        lines.append(f'cat {parts} > "{label}/POTCAR"')
    path = OUT / "make_potcar.sh"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = {}
    for label in SELECTED:
        records[label] = write_input(label)
    write_make_potcar(records)
    print(f"Wrote {len(records)} VASP input directories under {OUT}.")


if __name__ == "__main__":
    main()
