#!/usr/bin/env python3
"""Generate 16-atom Be8Co8 magnetic and fixed-volume E--V VASP inputs.

The calculations target two hypotheses for the Be-Co BCC-SQS discrepancy:

* a very flat E--V curve, where the sign of Omega_sf near zero is
  sensitive to the ionic/cell relaxation tolerance; and
* competing Co magnetic states selected by the initial MAGMOM.

Only the alloy is generated here.  The Be and Co endpoint volumes already
agree within 0.05% between the available 16- and 128-atom data, so endpoint
recalculations are intentionally excluded.

The source POSCAR is the existing Be8Co8 SQS input.  Its first eight sites
are Be and its last eight sites are Co; all generated MAGMOM vectors follow
that ordering.  The AFM assignment is based on the periodic nearest-neighbor
graph of the eight actual Co fractional coordinates, not on an arbitrary
alternation of the input order.
"""

from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE_POSCAR = ROOT / "SQS_RECALC_2x2x2" / "BCC_HIGHDEV" / "Be8Co8" / "POSCAR"
OUTPUT = ROOT / "BECO_MAGMOM_EV"
# User-provided clean Be8Co8 rerun (CLEANED:5), not data/sqs_results.csv:399.
# Its 150.27357974 A^3 is 0.33% above the repository's old 149.78262846 A^3
# (CLEANED:0), while its total energy is lower by 0.0586 eV per 16-atom cell.
# PR #474 replaced only >0.5% volume discrepancies, so the old row remains;
# this deeper-relaxation rerun is the physically preferred E-V reference.
REFERENCE_CELL_VOLUME = 150.27357974
EV_FACTORS = (0.94, 0.96, 0.98, 1.00, 1.02, 1.04, 1.06)
KPOINTS = "Automatic mesh\n0\nGamma\n  6 6 6\n  0 0 0\n"
POTCAR_VARIANTS = {"Be": "Be", "Co": "Co"}

MAG_CONFIGS = {
    "NM": None,
    "FM_low": "8*0.0 8*0.5",
    "FM_ref": "8*0.0 8*1.5",
    "FM_high": "8*0.0 8*3.0",
}


def read_poscar(path: Path) -> tuple[list[str], float, list[list[float]], list[str]]:
    lines = path.read_text().splitlines()
    if len(lines) < 24:
        raise ValueError(f"POSCAR is unexpectedly short: {path}")
    scale = float(lines[1].split()[0])
    vectors = [[float(x) for x in lines[i].split()[:3]] for i in range(2, 5)]
    if scale == 0:
        raise ValueError(f"POSCAR scale factor must not be zero: {path}")
    if scale < 0:
        determinant = abs(lattice_determinant(vectors))
        if determinant == 0:
            raise ValueError(f"POSCAR lattice vectors have zero volume: {path}")
        scale = (abs(scale) / determinant) ** (1.0 / 3.0)
    elements = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    if elements != ["Be", "Co"] or counts != [8, 8]:
        raise ValueError(
            "The source POSCAR must contain element order Be Co with counts 8 8"
        )
    coord_start = 8
    coords = lines[coord_start : coord_start + sum(counts)]
    return elements, scale, vectors, coords


def lattice_determinant(vectors: list[list[float]]) -> float:
    a, b, c = vectors
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def cell_volume(scale: float, vectors: list[list[float]]) -> float:
    return abs(lattice_determinant(vectors)) * abs(scale) ** 3


def scaled_vectors(
    scale: float, vectors: list[list[float]], target_volume: float
) -> tuple[float, list[list[float]]]:
    current = cell_volume(scale, vectors)
    factor = (target_volume / current) ** (1.0 / 3.0)
    return scale, [[x * factor for x in row] for row in vectors]


def write_poscar(
    path: Path,
    title: str,
    scale: float,
    vectors: list[list[float]],
    coords: list[str],
) -> None:
    lines = [
        title,
        f"{scale:.16g}",
        *("  " + "  ".join(f"{x:.12f}" for x in row) for row in vectors),
        "  Be  Co",
        "  8  8",
        "Direct",
        *coords,
    ]
    path.write_text("\n".join(lines) + "\n")


def write_incar(path: Path, system: str, magmom: str | None, isif: int) -> None:
    lines = [
        f"SYSTEM = {system}",
        "",
        "ENCUT  = 520",
        "PREC   = Accurate",
        "EDIFF  = 1E-6",
        "NELM   = 300",
        "LREAL  = .FALSE.",
        "",
        "IBRION = 2",
        f"ISIF   = {isif}",
        "NSW    = 200",
        "EDIFFG = -0.005",
        "POTIM  = 0.02",
        "",
        "ISMEAR = 1",
        "SIGMA  = 0.1",
        "",
        "GGA    = PE",
        "ALGO   = Normal",
        "",
    ]
    if magmom is None:
        lines.append("ISPIN  = 1")
    else:
        lines.extend(["ISPIN  = 2", f"MAGMOM = {magmom}"])
    lines.extend(
        [
            "",
            "LORBIT = 11",
            "LWAVE  = .FALSE.",
            "LCHARG = .FALSE.",
            "",
            "NCORE  = 4",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def verify_poscar_volume(path: Path, target_volume: float) -> None:
    """Fail loudly if a written POSCAR does not hold the requested volume."""
    _, scale, vectors, _ = read_poscar(path)
    actual = cell_volume(scale, vectors)
    if abs(actual - target_volume) > 1e-6 * target_volume:
        raise ValueError(
            f"{path} has volume {actual:.8f} A^3, expected "
            f"{target_volume:.8f} A^3"
        )


def write_case(
    name: str,
    title: str,
    magmom: str | None,
    isif: int,
    poscar_data: tuple[float, list[list[float]], list[str]],
) -> None:
    case = OUTPUT / name
    case.mkdir(parents=True, exist_ok=True)
    scale, vectors, coords = poscar_data
    write_incar(case / "INCAR", title, magmom, isif)
    write_poscar(case / "POSCAR", title, scale, vectors, coords)
    (case / "KPOINTS").write_text(KPOINTS)


def afm_magmom(coords: list[str]) -> str:
    """Color the periodic nearest-neighbor Co graph and format MAGMOM."""
    co = []
    for line in coords[8:]:
        values = [float(x) for x in line.split()[:3]]
        co.append(values)
    distances = []
    for i in range(len(co)):
        for j in range(i):
            delta = [co[i][k] - co[j][k] for k in range(3)]
            delta = [x - round(x) for x in delta]
            distances.append((sum(x * x for x in delta), i, j))
    nearest = min(d for d, _, _ in distances)
    graph = [[] for _ in co]
    for distance, i, j in distances:
        if abs(distance - nearest) < 1e-10:
            graph[i].append(j)
            graph[j].append(i)

    colors = {}
    for start in range(len(co)):
        if start in colors:
            continue
        colors[start] = 1
        queue = [start]
        while queue:
            current = queue.pop(0)
            for neighbor in graph[current]:
                expected = -colors[current]
                if neighbor in colors and colors[neighbor] != expected:
                    raise ValueError("Co nearest-neighbor graph is not bipartite")
                if neighbor not in colors:
                    colors[neighbor] = expected
                    queue.append(neighbor)
    signs = ["1.5" if colors[i] > 0 else "-1.5" for i in range(len(co))]
    return "8*0.0 " + " ".join(signs)


def write_shell_scripts(cases: list[tuple[str, list[str]]]) -> None:
    potcar = [
        "#!/bin/bash",
        "# Generate POTCAR files without storing pseudopotentials in Git.",
        'set -eu',
        'if [ -z "${VASP_PP_PATH:-}" ]; then',
        '  echo "Error: set VASP_PP_PATH to a PAW-PBE directory." >&2',
        "  exit 1",
        "fi",
        'PP_DIR="$VASP_PP_PATH/potpaw_PBE"',
    ]
    for name, elements in cases:
        sources = " ".join(f'"$PP_DIR"/{POTCAR_VARIANTS[e]}/POTCAR' for e in elements)
        potcar.append(f'cat {sources} > "{name}/POTCAR"')
    potcar.append(f'echo "Generated {len(cases)} POTCAR files."')
    potcar_path = OUTPUT / "make_potcar.sh"
    potcar_path.write_text("\n".join(potcar) + "\n")
    os.chmod(potcar_path, 0o755)

    run = [
        "#!/bin/bash",
        "# Run Be-Co MAGMOM and fixed-volume E--V calculations.",
        "# Usage: bash run_all.sh [NPROCS] (default: 8)",
        "set -u",
        'NPROCS="${1:-8}"',
        'VASPBIN="${VASPBIN:?Set VASPBIN to the VASP executable.}"',
        'BASE_DIR="$(cd "$(dirname "$0")" && pwd)"',
        'cd "$BASE_DIR" || exit 1',
        "bash make_potcar.sh || exit 1",
        "",
    ]
    for name, _ in cases:
        run.extend(
            [
                f'cd "$BASE_DIR/{name}" || exit 1',
                'if [ ! -f POTCAR ]; then echo "ERROR: missing POTCAR in '
                f'{name}" >&2; exit 1; fi',
                f'echo "START {name}"',
                'mpirun -np "$NPROCS" "$VASPBIN" > vasp.out 2>&1 || '
                f'echo "FAIL {name}"',
                f'echo "DONE {name}"',
                "",
            ]
        )
    run.append('echo "All calculations finished."')
    run_path = OUTPUT / "run_all.sh"
    run_path.write_text("\n".join(run) + "\n")
    os.chmod(run_path, 0o755)


def main() -> None:
    elements, source_scale, source_vectors, coords = read_poscar(SOURCE_POSCAR)
    mag_configs = dict(MAG_CONFIGS)
    mag_configs["AFM"] = afm_magmom(coords)
    source_volume = cell_volume(source_scale, source_vectors)
    ref_scale, ref_vectors = scaled_vectors(
        source_scale, source_vectors, REFERENCE_CELL_VOLUME
    )

    cases: list[tuple[str, list[str]]] = []
    # Full relaxations intentionally start from the existing POSCAR unchanged.
    for config, magmom in mag_configs.items():
        name = f"MAGMOM/{config}"
        label = "nonmagnetic" if config == "NM" else f"MAGMOM {config}"
        write_case(
            name,
            f"Be8Co8 {label} (ISIF=3)",
            magmom,
            3,
            (source_scale, source_vectors, coords),
        )
        verify_poscar_volume(OUTPUT / name / "POSCAR", source_volume)
        cases.append((name, elements))

    # Fixed-volume E--V runs use the recalc reference cell at factor 1.00.
    for config, magmom in (("FM_ref", mag_configs["FM_ref"]), ("NM", None)):
        for ratio in EV_FACTORS:
            label = f"{ratio:.2f}".replace(".", "p")
            name = f"EV/{config}/V{label}"
            target = REFERENCE_CELL_VOLUME * ratio
            scale, vectors = scaled_vectors(ref_scale, ref_vectors, target)
            write_case(
                name,
                f"Be8Co8 E-V {config} V/Vref={ratio:.2f} (ISIF=4)",
                magmom,
                4,
                (scale, vectors, coords),
            )
            verify_poscar_volume(OUTPUT / name / "POSCAR", target)
            cases.append((name, elements))

    write_shell_scripts(cases)
    print(f"Generated {len(cases)} calculations under {OUTPUT}")
    print(f"Source POSCAR volume: {source_volume:.8f} A^3")
    print(f"Reference volume: {REFERENCE_CELL_VOLUME:.8f} A^3")
    print("AFM Co signs by POSCAR Co order: + - + + - + - -")


if __name__ == "__main__":
    main()
