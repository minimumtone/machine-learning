#!/usr/bin/env python3
"""Generate VASP inputs to COMPLETE the 2x2x2 SQS dataset (16-atom BCC / 32-atom FCC).

Targets (provenance: paper/analyze_sqs_recalc_needs.py, 2026-06):
  1. BCC_PURE   : 16 pure elements whose SQS A8A8 volume deviated >3% from King
                  and were replaced by King/MP in load_sqs_data().
                  Recalculated with explicit MAGMOM (FM + AF variants for Cr/Mn/Fe)
                  and tighter convergence (EDIFF 1e-6, NELM 300, NSW 200).
  2. BCC_MISSING: 51 A8B8 pairs present in B2 data but absent from SQS
                  (incl. Co-Mn needed by test alloy AlCoMnNiV).
  3. BCC_HIGHDEV: 19 A8B8 pairs whose SQS volume deviates >10% from the
                  MP/OQMD B2 volume (verification reruns; see
                  paper/sqs_vs_mp_oqmd_deviation.csv).
  4. FCC_PURE   : 5 pure elements (A16A16) replaced in the FCC reliability check.

Output:
    SQS_RECALC_2x2x2/
    ├── BCC_PURE/{El}8{El}8[_AF]/   (INCAR, POSCAR, KPOINTS)
    ├── BCC_MISSING/{A}8{B}8/
    ├── BCC_HIGHDEV/{A}8{B}8/
    ├── FCC_PURE/{El}16{El}16/
    ├── make_potcar.sh
    └── run_all.sh

Directory naming matches data/sqs_results.csv so extract_vasp_results.py works
unchanged.
"""
import os
from pathlib import Path

BASE = Path(__file__).resolve().parent / "SQS_RECALC_2x2x2"

# --------------------------------------------------------------------
# Target lists (from paper/analyze_sqs_recalc_needs.py)
# --------------------------------------------------------------------
BCC_PURE_RECALC = [
    "Ag", "Au", "Be", "Ca", "Cr", "Fe", "Ir", "Mn",
    "Os", "Pb", "Pd", "Pt", "Re", "Rh", "Ru", "Sn",
]
BCC_MISSING_PAIRS = [
    ("Ag", "Pb"), ("Al", "Pb"), ("Al", "Pt"), ("Al", "Sn"), ("Au", "Mo"),
    ("Au", "Ni"), ("Au", "Pb"), ("Au", "Re"), ("Be", "Pb"), ("Ca", "Co"),
    ("Ca", "Os"), ("Ca", "Pb"), ("Ca", "Rh"), ("Ca", "V"), ("Co", "Mn"),
    ("Co", "Pb"), ("Cr", "Pb"), ("Cr", "Pd"), ("Cr", "Pt"), ("Cu", "Pb"),
    ("Cu", "Sn"), ("Fe", "Pb"), ("Hf", "Pb"), ("Hf", "Sc"), ("Ir", "Pb"),
    ("Ir", "Sn"), ("Mg", "Pb"), ("Mn", "Os"), ("Mn", "Pb"), ("Mo", "Pb"),
    ("Ni", "Pb"), ("Ni", "Pt"), ("Ni", "Sn"), ("Os", "Pb"), ("Pb", "Pd"),
    ("Pb", "Pt"), ("Pb", "Re"), ("Pb", "Rh"), ("Pb", "Ru"), ("Pb", "Sn"),
    ("Pb", "Ta"), ("Pb", "V"), ("Pb", "W"), ("Pb", "Zr"), ("Pd", "Re"),
    ("Pt", "Re"), ("Pt", "Sn"), ("Re", "Ru"), ("Re", "Sn"), ("Sc", "Sn"),
    ("Sn", "Zn"),
]
BCC_HIGHDEV_PAIRS = [
    ("Mg", "Pt"), ("Mg", "Os"), ("Mg", "Ni"), ("Ca", "Nb"), ("Fe", "Sn"),
    ("Rh", "Sc"), ("Ca", "Zr"), ("Co", "Sn"), ("Be", "Ir"), ("Fe", "Mg"),
    ("Ca", "Mo"), ("Be", "Pd"), ("Be", "Os"), ("Be", "Rh"), ("Be", "Re"),
    ("Be", "Co"), ("Mg", "V"), ("Al", "Fe"), ("Be", "W"),
]
FCC_PURE_RECALC = ["Nb", "Os", "Pb", "Pd", "Pt"]

# --------------------------------------------------------------------
# Constants (consistent with generate_l12_sqs_recalc.py)
# --------------------------------------------------------------------
POTCAR_VARIANTS = {
    "Ag": "Ag",    "Al": "Al",    "Au": "Au",    "Be": "Be",
    "Ca": "Ca_sv", "Co": "Co",    "Cr": "Cr_pv", "Cu": "Cu_pv",
    "Fe": "Fe_pv", "Hf": "Hf_pv", "Ir": "Ir",    "Mg": "Mg_pv",
    "Mn": "Mn_pv", "Mo": "Mo_pv", "Nb": "Nb_pv", "Ni": "Ni_pv",
    "Os": "Os_pv", "Pb": "Pb_d",  "Pd": "Pd",    "Pt": "Pt",
    "Re": "Re_pv", "Rh": "Rh_pv", "Ru": "Ru_pv", "Sc": "Sc_sv",
    "Sn": "Sn_d",  "Ta": "Ta_pv", "Ti": "Ti_pv", "V":  "V_sv",
    "W":  "W_sv",  "Zn": "Zn",    "Zr": "Zr_sv",
}
DEFAULT_MAGMOM = {"Cr": 1.5, "Mn": 3.0, "Fe": 2.5, "Co": 1.5, "Ni": 0.6}
MAGNETIC_ELEMENTS = set(DEFAULT_MAGMOM)
AF_ELEMENTS = {"Cr", "Mn"}  # AF variants generated for pure Cr/Mn (and Fe FM ref)

# King experimental atomic volumes (A^3/atom) — initial-guess only
KING_V = {
    "Ag": 17.061, "Al": 16.607, "Au": 16.966, "Be": 8.111, "Ca": 43.630,
    "Co": 11.075, "Cr": 12.008, "Cu": 11.810, "Fe": 11.776, "Hf": 22.317,
    "Ir": 14.155, "Mg": 23.239, "Mn": 12.210, "Mo": 15.583, "Nb": 17.978,
    "Ni": 10.940, "Os": 13.977, "Pb": 30.321, "Pd": 14.716, "Pt": 15.095,
    "Re": 14.712, "Rh": 13.754, "Ru": 13.571, "Sc": 25.004, "Sn": 27.053,
    "Ta": 18.021, "Ti": 17.645, "V": 13.816, "W": 15.855, "Zn": 15.212,
    "Zr": 23.280,
}

# SQS-16 BCC occupation (identical to generate_refractory_sqs.py)
SQS_BCC_OCCUPATION = [0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0]
BCC_POSITIONS = []
for ix in range(2):
    for iy in range(2):
        for iz in range(2):
            BCC_POSITIONS.append((ix / 2.0, iy / 2.0, iz / 2.0))
            BCC_POSITIONS.append(((ix + 0.5) / 2.0, (iy + 0.5) / 2.0,
                                  (iz + 0.5) / 2.0))

FCC_CONV = [(0.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5)]
FCC_POSITIONS = []
for ix in range(2):
    for iy in range(2):
        for iz in range(2):
            for p in FCC_CONV:
                FCC_POSITIONS.append(((p[0] + ix) / 2.0, (p[1] + iy) / 2.0,
                                      (p[2] + iz) / 2.0))


def a_bcc(el):
    return (2.0 * KING_V[el]) ** (1.0 / 3.0)


def a_fcc(el):
    return (4.0 * KING_V[el]) ** (1.0 / 3.0)


def write_incar(dirpath, system, magmom=None):
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
        "ISIF   = 3",
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
        "ISPIN  = 2",
    ]
    if magmom:
        lines.append(f"MAGMOM = {magmom}")
    lines += ["", "LORBIT = 11", "LWAVE  = .FALSE.", "LCHARG = .FALSE.",
              "", "NCORE  = 4", ""]
    (Path(dirpath) / "INCAR").write_text("\n".join(lines))


def write_kpoints(dirpath, kmesh):
    (Path(dirpath) / "KPOINTS").write_text(
        f"Automatic mesh\n0\nGamma\n  {kmesh} {kmesh} {kmesh}\n  0 0 0\n")


def write_poscar_bcc(dirpath, el_a, el_b):
    a_super = 2.0 * 0.5 * (a_bcc(el_a) + a_bcc(el_b))
    if el_a == el_b:
        header = [f"{el_a}16 BCC-SQS16 recalc (2x2x2 pure)", "1.0",
                  f"  {a_super:.6f}  0.0  0.0", f"  0.0  {a_super:.6f}  0.0",
                  f"  0.0  0.0  {a_super:.6f}", f"  {el_a}", "  16", "Direct"]
        coords = BCC_POSITIONS
    else:
        pos_a = [p for p, o in zip(BCC_POSITIONS, SQS_BCC_OCCUPATION) if o == 0]
        pos_b = [p for p, o in zip(BCC_POSITIONS, SQS_BCC_OCCUPATION) if o == 1]
        header = [f"{el_a}8{el_b}8 BCC-SQS16 (2x2x2, 50:50)", "1.0",
                  f"  {a_super:.6f}  0.0  0.0", f"  0.0  {a_super:.6f}  0.0",
                  f"  0.0  0.0  {a_super:.6f}", f"  {el_a}  {el_b}",
                  "  8  8", "Direct"]
        coords = pos_a + pos_b
    lines = header + [f"  {p[0]:.6f}  {p[1]:.6f}  {p[2]:.6f}" for p in coords]
    (Path(dirpath) / "POSCAR").write_text("\n".join(lines) + "\n")


def write_poscar_fcc_pure(dirpath, el):
    a_super = 2.0 * a_fcc(el)
    lines = [f"{el}32 FCC-SQS32 recalc (2x2x2 pure)", "1.0",
             f"  {a_super:.6f}  0.0  0.0", f"  0.0  {a_super:.6f}  0.0",
             f"  0.0  0.0  {a_super:.6f}", f"  {el}", "  32", "Direct"]
    lines += [f"  {p[0]:.6f}  {p[1]:.6f}  {p[2]:.6f}" for p in FCC_POSITIONS]
    (Path(dirpath) / "POSCAR").write_text("\n".join(lines) + "\n")


def magmom_pure(el, n, af=False):
    m = DEFAULT_MAGMOM.get(el, 0.0)
    if m == 0.0:
        return None
    if af:
        return " ".join(f"{m if i % 2 == 0 else -m:.1f}" for i in range(n))
    return f"{n}*{m}"


def magmom_pair(el_a, el_b):
    if el_a not in MAGNETIC_ELEMENTS and el_b not in MAGNETIC_ELEMENTS:
        return None
    m_a = DEFAULT_MAGMOM.get(el_a, 0.0)
    m_b = DEFAULT_MAGMOM.get(el_b, 0.0)
    return f"8*{m_a} 8*{m_b}"


def main():
    calcs = []  # (relpath, [pot_symbols])

    # 1. BCC pure recalcs (FM; + AF variant for Cr/Mn)
    for el in BCC_PURE_RECALC:
        variants = [("", False)]
        if el in AF_ELEMENTS:
            variants.append(("_AF", True))
        for suffix, af in variants:
            name = f"BCC_PURE/{el}8{el}8{suffix}"
            d = BASE / name
            d.mkdir(parents=True, exist_ok=True)
            write_poscar_bcc(d, el, el)
            write_incar(d, f"{el}16 BCC-SQS16 pure{suffix}",
                        magmom_pure(el, 16, af=af))
            write_kpoints(d, 6)
            calcs.append((name, [el]))

    # 2+3. BCC missing + high-deviation pairs
    for group, pairs in [("BCC_MISSING", BCC_MISSING_PAIRS),
                         ("BCC_HIGHDEV", BCC_HIGHDEV_PAIRS)]:
        for el_a, el_b in pairs:
            el_a, el_b = sorted([el_a, el_b])
            name = f"{group}/{el_a}8{el_b}8"
            d = BASE / name
            d.mkdir(parents=True, exist_ok=True)
            write_poscar_bcc(d, el_a, el_b)
            write_incar(d, f"{el_a}8{el_b}8 BCC-SQS16", magmom_pair(el_a, el_b))
            write_kpoints(d, 6)
            calcs.append((name, [el_a, el_b]))

    # 4. FCC pure recalcs
    for el in FCC_PURE_RECALC:
        name = f"FCC_PURE/{el}16{el}16"
        d = BASE / name
        d.mkdir(parents=True, exist_ok=True)
        write_poscar_fcc_pure(d, el)
        write_incar(d, f"{el}32 FCC-SQS32 pure", magmom_pure(el, 32))
        write_kpoints(d, 5)
        calcs.append((name, [el]))

    # make_potcar.sh
    lines = ["#!/bin/bash",
             "# POTCAR generation for SQS_RECALC_2x2x2",
             'if [ -z "$VASP_PP_PATH" ]; then',
             '    echo "Error: set VASP_PP_PATH"; exit 1; fi',
             'PP_DIR="$VASP_PP_PATH/potpaw_PBE"', ""]
    for name, els in calcs:
        pots = " ".join(f'"$PP_DIR"/{POTCAR_VARIANTS[e]}/POTCAR' for e in els)
        lines.append(f"cat {pots} > {name}/POTCAR")
    lines.append(f'echo "Done: {len(calcs)} POTCARs."')
    p = BASE / "make_potcar.sh"
    p.write_text("\n".join(lines) + "\n")
    os.chmod(p, 0o755)

    # run_all.sh (same conventions as BCC_SQS run_all.sh: mpirun, parallel jobs)
    run = f"""#!/bin/bash
# Run all SQS_RECALC_2x2x2 calculations.
# Usage: bash run_all.sh [NJOBS_PARALLEL] [NPROCS_PER_JOB]   (default 8x4)
# Requires: $VASPBIN, $VASP_PP_PATH
set -e
NJOBS=${{1:-8}}
NP=${{2:-4}}
BASEDIR=$(cd "$(dirname "$0")" && pwd)
cd "$BASEDIR"
bash make_potcar.sh

run_one() {{
    d="$1"
    cd "$BASEDIR/$d"
    if [ -f static_OUTCAR ] || grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "SKIP $d"; return 0
    fi
    echo "START $d"
    mpirun -np $NP "$VASPBIN" > vasp.out 2>&1 || {{ echo "FAIL $d"; return 0; }}
    # static run at relaxed geometry
    cp CONTCAR POSCAR
    sed -e 's/NSW    = 200/NSW    = 0/' -e 's/IBRION = 2/IBRION = -1/' INCAR > INCAR.static
    mv INCAR INCAR.relax && mv INCAR.static INCAR
    mpirun -np $NP "$VASPBIN" > vasp_static.out 2>&1 || echo "FAIL-STATIC $d"
    cp OUTCAR static_OUTCAR
    mv INCAR.relax INCAR
    echo "DONE $d"
}}
export -f run_one
export BASEDIR NP VASPBIN

find . -mindepth 2 -maxdepth 2 -type d | sed 's|^\\./||' | \\
    xargs -P $NJOBS -I{{}} bash -c 'run_one "$@"' _ {{}}
echo "All done. Extract with: python ../extract_vasp_results.py"
"""
    p = BASE / "run_all.sh"
    p.write_text(run)
    os.chmod(p, 0o755)

    n_pure = sum(1 for c in calcs if c[0].startswith("BCC_PURE"))
    n_miss = sum(1 for c in calcs if c[0].startswith("BCC_MISSING"))
    n_dev = sum(1 for c in calcs if c[0].startswith("BCC_HIGHDEV"))
    n_fcc = sum(1 for c in calcs if c[0].startswith("FCC_PURE"))
    print(f"Generated {len(calcs)} calculations in {BASE}")
    print(f"  BCC_PURE:    {n_pure} (16 elements, +AF variants for Cr/Mn)")
    print(f"  BCC_MISSING: {n_miss}")
    print(f"  BCC_HIGHDEV: {n_dev}")
    print(f"  FCC_PURE:    {n_fcc}")


if __name__ == "__main__":
    main()
