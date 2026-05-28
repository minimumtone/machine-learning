#!/usr/bin/env python3
"""
VASP input file generator for MISSING B2, L1₂, BCC-SQS, and FCC-SQS calculations.

Reads existing compounds_VASP_B2.csv and compounds_VASP_L12.csv to identify
which target element pairs are already computed, then generates VASP inputs
only for the missing pairs. SQS calculations are generated for all pairs
(no existing SQS data).

Target: 38 elements (Gd, Ce excluded for 4f instability)

Structure types:
  - B2 (CsCl-type, 2 atoms): AB and BA for each pair
  - L1₂ (Cu₃Au-type, 4 atoms): A3B and B3A for each pair
  - BCC-SQS (2×2×2 supercell, 16 atoms): A8B8 for each unordered pair
  - FCC-SQS (2×2×2 supercell, 32 atoms): A16B16 for each unordered pair

Output structure:
    VASP_missing/
    ├── BCC_B2/
    │   ├── AuAg/  (INCAR, POSCAR, KPOINTS)
    │   ├── AgAu/
    │   └── ...
    ├── FCC_L12/
    │   ├── Be3Ag/
    │   ├── Ag3Be/
    │   └── ...
    ├── BCC_SQS/
    │   ├── Ag8Al8/
    │   ├── Ag8Ag8/
    │   └── ...
    ├── FCC_SQS/
    │   ├── Ag16Al16/
    │   ├── Ag16Ag16/
    │   └── ...
    ├── make_potcar.sh
    ├── run_all.sh
    └── summary.txt

Usage:
    python generate_missing_inputs.py \\
        --b2-csv  /path/to/compounds_VASP_B2.csv \\
        --l12-csv /path/to/compounds_VASP_L12.csv

Environment variables:
    VASP_PP_PATH : path to PAW-PBE pseudopotential directory
    VASPBIN      : VASP executable command
"""

import os
import csv
import argparse
import itertools

# =====================================================================
# Target 38 elements
# =====================================================================
ALL_ELEMENTS = sorted([
    'Ag', 'Al', 'Au', 'Be', 'Ca', 'Co', 'Cr', 'Cu', 'Dy', 'Er',
    'Fe', 'Ge', 'Hf', 'Ir', 'La', 'Mg', 'Mn', 'Mo', 'Nb', 'Ni',
    'Os', 'Pb', 'Pd', 'Pt', 'Re', 'Rh', 'Ru', 'Sc', 'Si', 'Sn',
    'Ta', 'Tb', 'Ti', 'V',  'W',  'Y',  'Zn', 'Zr',
])
assert len(ALL_ELEMENTS) == 38

# =====================================================================
# BCC lattice constants (Å) from VASP same-element B2 calculations
# For elements not in existing B2 data, use DFT-PBE estimates
# =====================================================================
ELEMENT_A0_BCC = {
    "Ag": 3.3009, "Al": 3.2254, "Au": 3.240,  "Be": 2.530,
    "Ca": 4.3838, "Co": 2.8010, "Cr": 2.8363, "Cu": 2.8955,
    "Dy": 3.990,  "Er": 3.960,  "Fe": 2.8266, "Ge": 3.3764,
    "Hf": 3.5338, "Ir": 3.060,  "La": 4.2224, "Mg": 3.5789,
    "Mn": 2.7883, "Mo": 3.1486, "Nb": 3.3237, "Ni": 2.7938,
    "Os": 3.020,  "Pb": 3.940,  "Pd": 3.1386, "Pt": 3.120,
    "Re": 3.100,  "Rh": 3.020,  "Ru": 3.0460, "Sc": 3.6771,
    "Si": 3.0942, "Sn": 3.8076, "Ta": 3.3119, "Tb": 4.020,
    "Ti": 3.2396, "V":  2.9821, "W":  3.1724, "Y":  4.0265,
    "Zn": 3.1572, "Zr": 3.5687,
}

# FCC lattice constants (Å) for L1₂ initial guess
ELEMENT_A0_FCC = {
    "Ag": 4.152,  "Al": 4.040,  "Au": 4.160,  "Be": 3.200,
    "Ca": 5.530,  "Co": 3.545,  "Cr": 3.640,  "Cu": 3.635,
    "Dy": 5.030,  "Er": 4.990,  "Fe": 3.570,  "Ge": 4.280,
    "Hf": 4.470,  "Ir": 3.839,  "La": 5.320,  "Mg": 4.520,
    "Mn": 3.630,  "Mo": 3.960,  "Nb": 4.160,  "Ni": 3.524,
    "Os": 3.850,  "Pb": 4.990,  "Pd": 3.890,  "Pt": 3.924,
    "Re": 3.900,  "Rh": 3.803,  "Ru": 3.827,  "Sc": 4.630,
    "Si": 3.870,  "Sn": 4.810,  "Ta": 4.150,  "Tb": 5.070,
    "Ti": 4.100,  "V":  3.820,  "W":  3.980,  "Y":  5.080,
    "Zn": 3.940,  "Zr": 4.540,
}

# =====================================================================
# POTCAR variants (PAW-PBE)
# =====================================================================
POTCAR_VARIANTS = {
    "Ag": "Ag",    "Al": "Al",    "Au": "Au",    "Be": "Be",
    "Ca": "Ca_sv", "Co": "Co",    "Cr": "Cr_pv", "Cu": "Cu_pv",
    "Dy": "Dy_3",  "Er": "Er_3",  "Fe": "Fe_pv", "Ge": "Ge_d",
    "Hf": "Hf_pv", "Ir": "Ir",    "La": "La",    "Mg": "Mg_pv",
    "Mn": "Mn_pv", "Mo": "Mo_pv", "Nb": "Nb_pv", "Ni": "Ni_pv",
    "Os": "Os_pv", "Pb": "Pb_d",  "Pd": "Pd",    "Pt": "Pt",
    "Re": "Re_pv", "Rh": "Rh_pv", "Ru": "Ru_pv", "Sc": "Sc_sv",
    "Si": "Si",    "Sn": "Sn_d",  "Ta": "Ta_pv", "Tb": "Tb_3",
    "Ti": "Ti_pv", "V":  "V_sv",  "W":  "W_sv",  "Y":  "Y_sv",
    "Zn": "Zn",    "Zr": "Zr_sv",
}

# =====================================================================
# SQS-16 BCC configuration (optimized α_1nn ≈ α_2nn ≈ α_3nn ≈ 0)
# =====================================================================
SQS_OCCUPATION = [0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0]

BCC_2x2x2_POSITIONS = []
for ix in range(2):
    for iy in range(2):
        for iz in range(2):
            BCC_2x2x2_POSITIONS.append((ix / 2.0, iy / 2.0, iz / 2.0))
            BCC_2x2x2_POSITIONS.append(
                ((ix + 0.5) / 2.0, (iy + 0.5) / 2.0, (iz + 0.5) / 2.0))

# =====================================================================
# SQS-32 FCC configuration (2×2×2 FCC supercell, 32 atoms)
# Occupation optimized for α_1nn ≈ α_2nn ≈ 0
# =====================================================================
FCC_2x2x2_POSITIONS = []
for ix in range(2):
    for iy in range(2):
        for iz in range(2):
            # 4 FCC basis atoms per unit cell
            FCC_2x2x2_POSITIONS.append((ix / 2.0, iy / 2.0, iz / 2.0))
            FCC_2x2x2_POSITIONS.append((ix / 2.0, (iy + 0.5) / 2.0, (iz + 0.5) / 2.0))
            FCC_2x2x2_POSITIONS.append(((ix + 0.5) / 2.0, iy / 2.0, (iz + 0.5) / 2.0))
            FCC_2x2x2_POSITIONS.append(((ix + 0.5) / 2.0, (iy + 0.5) / 2.0, iz / 2.0))

# SQS occupation for 32-atom FCC: optimized for α₁ ≈ α₂ ≈ α₃ ≈ 0
# 0=A, 1=B, 16 atoms each
# Monte Carlo optimized: α₁=0.000, α₂=0.000, α₃=0.000
FCC_SQS_OCCUPATION = [
    1, 1, 1, 1,  # cell (0,0,0)
    1, 0, 1, 1,  # cell (0,0,1)
    0, 1, 1, 1,  # cell (0,1,0)
    1, 1, 0, 0,  # cell (0,1,1)
    0, 0, 0, 1,  # cell (1,0,0)
    1, 0, 0, 0,  # cell (1,0,1)
    0, 0, 1, 0,  # cell (1,1,0)
    0, 1, 0, 0,  # cell (1,1,1)
]


# =====================================================================
# INCAR / POSCAR / KPOINTS writers
# =====================================================================

def write_incar_b2(dirpath):
    """INCAR for B2 (2 atoms, high-precision)."""
    content = """\
SYSTEM = B2 structure optimization

ENCUT  = 520
PREC   = Accurate
EDIFF  = 1E-6
NELM   = 200
LREAL  = .FALSE.

IBRION = 2
ISIF   = 3
NSW    = 100
EDIFFG = -0.01

ISMEAR = 1
SIGMA  = 0.2

GGA    = PE
ISPIN  = 2

LORBIT = 11
LWAVE  = .FALSE.
LCHARG = .FALSE.
NCORE  = 4
"""
    with open(os.path.join(dirpath, "INCAR"), "w") as f:
        f.write(content)


def write_incar_l12(dirpath):
    """INCAR for L1₂ (4 atoms, high-precision)."""
    content = """\
SYSTEM = L12 structure optimization

ENCUT  = 520
PREC   = Accurate
EDIFF  = 1E-6
NELM   = 200
LREAL  = .FALSE.

IBRION = 2
ISIF   = 3
NSW    = 100
EDIFFG = -0.01

ISMEAR = 1
SIGMA  = 0.2

GGA    = PE
ISPIN  = 2

LORBIT = 11
LWAVE  = .FALSE.
LCHARG = .FALSE.
NCORE  = 4
"""
    with open(os.path.join(dirpath, "INCAR"), "w") as f:
        f.write(content)


def write_incar_sqs(dirpath):
    """INCAR for BCC-SQS (16 atoms). ISIF=7: volume-only relaxation, cubic shape preserved."""
    content = """\
SYSTEM = BCC-SQS structure optimization (cubic constraint)

ENCUT  = 520
PREC   = Accurate
EDIFF  = 1E-6
NELM   = 200
LREAL  = .FALSE.

IBRION = 2
ISIF   = 7
NSW    = 100
EDIFFG = -0.01

ISMEAR = 1
SIGMA  = 0.2

GGA    = PE
ISPIN  = 2

LORBIT = 11
LWAVE  = .FALSE.
LCHARG = .FALSE.
NCORE  = 4
"""
    with open(os.path.join(dirpath, "INCAR"), "w") as f:
        f.write(content)


def write_incar_fcc_sqs(dirpath):
    """INCAR for FCC-SQS (32 atoms). ISIF=7: volume-only relaxation, cubic shape preserved."""
    content = """\
SYSTEM = FCC-SQS structure optimization (cubic constraint)

ENCUT  = 520
PREC   = Accurate
EDIFF  = 1E-6
NELM   = 200
LREAL  = .FALSE.

IBRION = 2
ISIF   = 7
NSW    = 100
EDIFFG = -0.01

ISMEAR = 1
SIGMA  = 0.2

GGA    = PE
ISPIN  = 2

LORBIT = 11
LWAVE  = .FALSE.
LCHARG = .FALSE.
NCORE  = 4
"""
    with open(os.path.join(dirpath, "INCAR"), "w") as f:
        f.write(content)


def write_poscar_b2(dirpath, el_corner, el_body, a0):
    """POSCAR for B2 (CsCl-type, 2 atoms)."""
    content = f"""\
{el_corner}{el_body} B2 (Pm-3m, CsCl-type)
1.0
  {a0:.6f}  0.000000  0.000000
  0.000000  {a0:.6f}  0.000000
  0.000000  0.000000  {a0:.6f}
  {el_corner}  {el_body}
  1  1
Direct
  0.000000  0.000000  0.000000
  0.500000  0.500000  0.500000
"""
    with open(os.path.join(dirpath, "POSCAR"), "w") as f:
        f.write(content)


def write_poscar_l12(dirpath, el_face, el_corner, a0):
    """POSCAR for L1₂ (Cu₃Au-type, 4 atoms). el_face = majority (3), el_corner = minority (1)."""
    content = f"""\
{el_face}3{el_corner} L12 (Pm-3m)
1.0
  {a0:.6f}  0.000000  0.000000
  0.000000  {a0:.6f}  0.000000
  0.000000  0.000000  {a0:.6f}
  {el_face}  {el_corner}
  3  1
Direct
  0.000000  0.500000  0.500000
  0.500000  0.000000  0.500000
  0.500000  0.500000  0.000000
  0.000000  0.000000  0.000000
"""
    with open(os.path.join(dirpath, "POSCAR"), "w") as f:
        f.write(content)


def write_poscar_sqs(dirpath, el_a, el_b, a_super):
    """POSCAR for BCC-SQS16 (2×2×2, A8B8 or pure A16)."""
    if el_a == el_b:
        lines = [
            f"{el_a}16 BCC-SQS16 (2x2x2, pure reference)",
            "1.0",
            f"  {a_super:.6f}  0.000000  0.000000",
            f"  0.000000  {a_super:.6f}  0.000000",
            f"  0.000000  0.000000  {a_super:.6f}",
            f"  {el_a}",
            f"  16",
            "Direct",
        ]
        for pos in BCC_2x2x2_POSITIONS:
            lines.append(f"  {pos[0]:.6f}  {pos[1]:.6f}  {pos[2]:.6f}")
    else:
        pos_a = [BCC_2x2x2_POSITIONS[i] for i, o in enumerate(SQS_OCCUPATION) if o == 0]
        pos_b = [BCC_2x2x2_POSITIONS[i] for i, o in enumerate(SQS_OCCUPATION) if o == 1]
        lines = [
            f"{el_a}8{el_b}8 BCC-SQS16 (2x2x2, 50:50)",
            "1.0",
            f"  {a_super:.6f}  0.000000  0.000000",
            f"  0.000000  {a_super:.6f}  0.000000",
            f"  0.000000  0.000000  {a_super:.6f}",
            f"  {el_a}  {el_b}",
            f"  8  8",
            "Direct",
        ]
        for pos in pos_a:
            lines.append(f"  {pos[0]:.6f}  {pos[1]:.6f}  {pos[2]:.6f}")
        for pos in pos_b:
            lines.append(f"  {pos[0]:.6f}  {pos[1]:.6f}  {pos[2]:.6f}")
    lines.append("")
    with open(os.path.join(dirpath, "POSCAR"), "w") as f:
        f.write("\n".join(lines))


def write_poscar_fcc_sqs(dirpath, el_a, el_b, a_super):
    """POSCAR for FCC-SQS (2×2×2 supercell, 32 atoms)."""
    if el_a == el_b:
        lines = [
            f"{el_a}32 FCC-SQS32 (2x2x2, pure reference)",
            "1.0",
            f"  {a_super:.6f}  0.000000  0.000000",
            f"  0.000000  {a_super:.6f}  0.000000",
            f"  0.000000  0.000000  {a_super:.6f}",
            f"  {el_a}",
            f"  32",
            "Direct",
        ]
        for pos in FCC_2x2x2_POSITIONS:
            lines.append(f"  {pos[0]:.6f}  {pos[1]:.6f}  {pos[2]:.6f}")
    else:
        pos_a = [FCC_2x2x2_POSITIONS[i] for i, o in enumerate(FCC_SQS_OCCUPATION) if o == 0]
        pos_b = [FCC_2x2x2_POSITIONS[i] for i, o in enumerate(FCC_SQS_OCCUPATION) if o == 1]
        lines = [
            f"{el_a}16{el_b}16 FCC-SQS32 (2x2x2, 50:50)",
            "1.0",
            f"  {a_super:.6f}  0.000000  0.000000",
            f"  0.000000  {a_super:.6f}  0.000000",
            f"  0.000000  0.000000  {a_super:.6f}",
            f"  {el_a}  {el_b}",
            f"  16  16",
            "Direct",
        ]
        for pos in pos_a:
            lines.append(f"  {pos[0]:.6f}  {pos[1]:.6f}  {pos[2]:.6f}")
        for pos in pos_b:
            lines.append(f"  {pos[0]:.6f}  {pos[1]:.6f}  {pos[2]:.6f}")
    lines.append("")
    with open(os.path.join(dirpath, "POSCAR"), "w") as f:
        f.write("\n".join(lines))


def write_kpoints(dirpath, kmesh):
    content = f"""\
Automatic mesh
0
Gamma
  {kmesh} {kmesh} {kmesh}
  0 0 0
"""
    with open(os.path.join(dirpath, "KPOINTS"), "w") as f:
        f.write(content)


# =====================================================================
# Script generators
# =====================================================================

def generate_potcar_script(base_dir, all_calcs):
    """Generate unified make_potcar.sh for all structure types."""
    lines = [
        "#!/bin/bash",
        "# POTCAR generation for all missing VASP calculations",
        "# Usage: bash make_potcar.sh",
        "# Requires: $VASP_PP_PATH",
        "",
        'if [ -z "$VASP_PP_PATH" ]; then',
        '    echo "Error: VASP_PP_PATH is not set."',
        '    exit 1',
        'fi',
        "",
        f'echo "Generating POTCAR for {len(all_calcs)} calculations..."',
        "FAIL=0",
        "",
    ]

    for subdir, el_list in all_calcs:
        pots = [POTCAR_VARIANTS.get(el, el) for el in el_list]
        cat_parts = " ".join(f'"$VASP_PP_PATH"/{p}/POTCAR' for p in pots)
        lines.append(f'cat {cat_parts} > {subdir}/POTCAR 2>/dev/null')
        pot_str = "+".join(pots)
        lines.append(
            f'if [ $? -ne 0 ]; then echo "  FAIL: {subdir} ({pot_str})"; '
            f'FAIL=$((FAIL+1)); fi'
        )

    lines.append("")
    lines.append(f'echo "Done. Failed: $FAIL / {len(all_calcs)}"')

    path = os.path.join(base_dir, "make_potcar.sh")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    os.chmod(path, 0o755)


def generate_run_script(base_dir, all_calcs):
    """Generate unified run_all.sh using $VASPBIN."""
    lines = [
        "#!/bin/bash",
        "# Batch execution for all missing VASP calculations",
        "# Usage: bash run_all.sh",
        "# Requires: $VASPBIN",
        "",
        'if [ -z "$VASPBIN" ]; then',
        '    echo "Error: VASPBIN is not set."',
        '    exit 1',
        'fi',
        "",
        'BASE=$(cd "$(dirname "$0")" && pwd)',
        'LOG="$BASE/run_status.log"',
        "",
        f'echo "=== Missing VASP Calculations ===" | tee "$LOG"',
        f'echo "Total: {len(all_calcs)} calculations" | tee -a "$LOG"',
        'echo "VASPBIN=$VASPBIN" | tee -a "$LOG"',
        'echo "Started: $(date)" | tee -a "$LOG"',
        'echo "" | tee -a "$LOG"',
        "",
    ]

    for i, (subdir, _) in enumerate(all_calcs, 1):
        lines.append(f'echo "[{i}/{len(all_calcs)}] {subdir}..." | tee -a "$LOG"')
        lines.append(f'cd "$BASE/{subdir}"')
        lines.append('if [ ! -f POTCAR ]; then')
        lines.append(f'    echo "  SKIP (no POTCAR)" | tee -a "$LOG"')
        lines.append('else')
        lines.append('$VASPBIN > vasp.out 2>&1')
        lines.append('if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then')
        lines.append(f'    echo "  CONVERGED" | tee -a "$LOG"')
        lines.append('else')
        lines.append(f'    echo "  WARNING: not converged" | tee -a "$LOG"')
        lines.append('fi')
        lines.append('fi')
        lines.append(f'cd "$BASE"')
        lines.append("")

    lines.append('echo "" | tee -a "$LOG"')
    lines.append('echo "Finished: $(date)" | tee -a "$LOG"')

    path = os.path.join(base_dir, "run_all.sh")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    os.chmod(path, 0o755)


# =====================================================================
# Main
# =====================================================================

def load_existing_pairs(csv_path):
    """Load (element_A, element_B) pairs from existing CSV."""
    pairs = set()
    if csv_path and os.path.isfile(csv_path):
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                pairs.add((row['element_A'], row['element_B']))
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Generate VASP inputs for missing B2, L1₂, SQS calculations")
    parser.add_argument('--b2-csv', default=None,
                        help='Path to existing compounds_VASP_B2.csv')
    parser.add_argument('--l12-csv', default=None,
                        help='Path to existing compounds_VASP_L12.csv')
    parser.add_argument('--fcc-sqs-csv', default=None,
                        help='Path to existing compounds_VASP_FCC_SQS.csv (if any)')
    parser.add_argument('--output', default=None,
                        help='Output base directory (default: VASP_missing/ in same dir)')
    args = parser.parse_args()

    base_dir = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "VASP_missing")

    b2_dir = os.path.join(base_dir, "BCC_B2")
    l12_dir = os.path.join(base_dir, "FCC_L12")
    sqs_dir = os.path.join(base_dir, "BCC_SQS")
    fcc_sqs_dir = os.path.join(base_dir, "FCC_SQS")

    # Load existing data
    existing_b2 = load_existing_pairs(args.b2_csv)
    existing_l12 = load_existing_pairs(args.l12_csv)
    existing_fcc_sqs = load_existing_pairs(args.fcc_sqs_csv)

    print(f"Existing B2 pairs:      {len(existing_b2)}")
    print(f"Existing L12 pairs:     {len(existing_l12)}")
    print(f"Existing FCC-SQS pairs: {len(existing_fcc_sqs)}")
    print()

    all_calcs = []  # list of (subdir_relative, [elements_for_potcar])

    # ----- B2 missing -----
    b2_count = 0
    for el_a in ALL_ELEMENTS:
        for el_b in ALL_ELEMENTS:
            if (el_a, el_b) in existing_b2:
                continue
            dirname = f"{el_a}1{el_b}1"
            dirpath = os.path.join(b2_dir, dirname)
            os.makedirs(dirpath, exist_ok=True)
            a0 = 0.5 * (ELEMENT_A0_BCC.get(el_a, 3.20) + ELEMENT_A0_BCC.get(el_b, 3.20))
            write_incar_b2(dirpath)
            write_poscar_b2(dirpath, el_a, el_b, a0)
            write_kpoints(dirpath, kmesh=16)
            if el_a == el_b:
                all_calcs.append((f"BCC_B2/{dirname}", [el_a]))
            else:
                all_calcs.append((f"BCC_B2/{dirname}", [el_a, el_b]))
            b2_count += 1

    # ----- L1₂ missing -----
    l12_count = 0
    for el_a in ALL_ELEMENTS:
        for el_b in ALL_ELEMENTS:
            if (el_a, el_b) in existing_l12:
                continue
            # L12: el_b = corner (minority, 1 atom), el_a = face (majority, 3 atoms)
            # Directory convention: corner1face3 (e.g., Al1Ag3)
            dirname = f"{el_b}1{el_a}3"
            dirpath = os.path.join(l12_dir, dirname)
            os.makedirs(dirpath, exist_ok=True)
            a0_a = ELEMENT_A0_FCC.get(el_a, 3.80)
            a0_b = ELEMENT_A0_FCC.get(el_b, 3.80)
            a0 = 0.75 * a0_a + 0.25 * a0_b
            write_incar_l12(dirpath)
            write_poscar_l12(dirpath, el_a, el_b, a0)
            write_kpoints(dirpath, kmesh=12)
            if el_a == el_b:
                all_calcs.append((f"FCC_L12/{dirname}", [el_a]))
            else:
                all_calcs.append((f"FCC_L12/{dirname}", [el_a, el_b]))
            l12_count += 1

    # ----- BCC-SQS (all pairs, no existing data) -----
    sqs_count = 0
    for el_a, el_b in itertools.combinations(ALL_ELEMENTS, 2):
        dirname = f"{el_a}8{el_b}8"
        dirpath = os.path.join(sqs_dir, dirname)
        os.makedirs(dirpath, exist_ok=True)
        a_super = 2.0 * 0.5 * (ELEMENT_A0_BCC.get(el_a, 3.20) + ELEMENT_A0_BCC.get(el_b, 3.20))
        write_incar_sqs(dirpath)
        write_poscar_sqs(dirpath, el_a, el_b, a_super)
        write_kpoints(dirpath, kmesh=12)
        all_calcs.append((f"BCC_SQS/{dirname}", [el_a, el_b]))
        sqs_count += 1

    # Same-element BCC-SQS references
    for el in ALL_ELEMENTS:
        dirname = f"{el}8{el}8"
        dirpath = os.path.join(sqs_dir, dirname)
        os.makedirs(dirpath, exist_ok=True)
        a_super = 2.0 * ELEMENT_A0_BCC.get(el, 3.20)
        write_incar_sqs(dirpath)
        write_poscar_sqs(dirpath, el, el, a_super)
        write_kpoints(dirpath, kmesh=12)
        all_calcs.append((f"BCC_SQS/{dirname}", [el]))
        sqs_count += 1

    # ----- FCC-SQS (skip existing) -----
    fcc_sqs_count = 0
    for el_a, el_b in itertools.combinations(ALL_ELEMENTS, 2):
        if (el_a, el_b) in existing_fcc_sqs or (el_b, el_a) in existing_fcc_sqs:
            continue
        dirname = f"{el_a}16{el_b}16"
        dirpath = os.path.join(fcc_sqs_dir, dirname)
        os.makedirs(dirpath, exist_ok=True)
        a_super = 2.0 * 0.5 * (ELEMENT_A0_FCC.get(el_a, 3.80) + ELEMENT_A0_FCC.get(el_b, 3.80))
        write_incar_fcc_sqs(dirpath)
        write_poscar_fcc_sqs(dirpath, el_a, el_b, a_super)
        write_kpoints(dirpath, kmesh=4)
        all_calcs.append((f"FCC_SQS/{dirname}", [el_a, el_b]))
        fcc_sqs_count += 1

    # Same-element FCC-SQS references
    for el in ALL_ELEMENTS:
        if (el, el) in existing_fcc_sqs:
            continue
        dirname = f"{el}16{el}16"
        dirpath = os.path.join(fcc_sqs_dir, dirname)
        os.makedirs(dirpath, exist_ok=True)
        a_super = 2.0 * ELEMENT_A0_FCC.get(el, 3.80)
        write_incar_fcc_sqs(dirpath)
        write_poscar_fcc_sqs(dirpath, el, el, a_super)
        write_kpoints(dirpath, kmesh=4)
        all_calcs.append((f"FCC_SQS/{dirname}", [el]))
        fcc_sqs_count += 1

    # Generate helper scripts
    generate_potcar_script(base_dir, all_calcs)
    generate_run_script(base_dir, all_calcs)

    # Summary
    total = b2_count + l12_count + sqs_count + fcc_sqs_count
    summary = f"""\
=== VASP Missing Calculations Summary ===

Target elements: {len(ALL_ELEMENTS)}
Existing B2 pairs:  {len(existing_b2)}
Existing L12 pairs: {len(existing_l12)}

Generated:
  B2 (CsCl, 2 atoms):        {b2_count:>5} calculations  [ENCUT=520, k=16×16×16]
  L12 (Cu3Au, 4 atoms):      {l12_count:>5} calculations  [ENCUT=520, k=12×12×12]
  BCC-SQS (2×2×2, 16 atoms): {sqs_count:>5} calculations  [ENCUT=520, k=12×12×12]
  FCC-SQS (2×2×2, 32 atoms): {fcc_sqs_count:>5} calculations  [ENCUT=520, k=4×4×4]
  Total:                      {total:>5} calculations

Output: {base_dir}/

Next steps:
  1. cd {base_dir}
  2. bash make_potcar.sh          # needs $VASP_PP_PATH
  3. bash run_all.sh              # needs $VASPBIN
"""
    print(summary)

    with open(os.path.join(base_dir, "summary.txt"), "w") as f:
        f.write(summary)


if __name__ == "__main__":
    main()
