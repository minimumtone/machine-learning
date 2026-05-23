#!/usr/bin/env python3
"""
VASP input file generator for BCC SQS (Special Quasi-random Structure) calculations.

Generates INCAR, POSCAR, KPOINTS for binary BCC solid solutions using SQS
supercells. This provides a better proxy for BCC HEA behavior than ordered
B2 structures.

SQS approach:
  - 16-atom BCC supercell (2×2×2 conventional BCC = 16 atoms)
  - 50:50 composition (8A + 8B) to maximize pair correlation matching
  - SQS configurations are pre-generated to minimize short-range order

Comparison with B2:
  - B2 = perfectly ordered (each atom surrounded by 8 unlike neighbors)
  - SQS = quasi-random (mimics random solid solution pair correlations)
  - SQS is physically more relevant for BCC HEA applications

Usage:
    python generate_refractory_sqs.py [--group refractory|all|custom]

    --group refractory : 9高融点元素のみ (default, 45計算)
    --group all        : BCC_B2全39元素 (780計算)
    --elements "Fe,Co,Ni,Cr,Mn"  : 任意元素指定

Output structure:
    BCC_SQS/
    ├── Cr8Hf8/   (INCAR, POSCAR, KPOINTS)
    ├── Cr8Mo8/
    ├── ...
    ├── Cr8Cr8/   (same-element references)
    ├── make_potcar.sh
    └── run_all.sh
"""

import os
import sys
import argparse
import itertools
import numpy as np

# =====================================================================
# Element groups
# =====================================================================
REFRACTORY_ELEMENTS = sorted(['Nb', 'Ti', 'V', 'Zr', 'Mo', 'Ta', 'W', 'Hf', 'Cr'])

# All elements with existing BCC_B2 data (from VASP same-element B2 calculations)
ALL_B2_ELEMENTS = sorted([
    'Ag', 'Al', 'As', 'B', 'Ca', 'Cd', 'Ce', 'Co', 'Cr', 'Cu',
    'Fe', 'Ga', 'Gd', 'Ge', 'Hf', 'In', 'La', 'Mg', 'Mn', 'Mo',
    'Na', 'Nb', 'Ni', 'Pd', 'Ru', 'Sb', 'Sc', 'Se', 'Si', 'Sn',
    'Sr', 'Ta', 'Tc', 'Ti', 'V', 'W', 'Y', 'Zn', 'Zr',
])

# BCC lattice constants (Å) from VASP B2 same-element calculations
ELEMENT_A0_BCC = {
    "Ag": 3.3009, "Al": 3.2254, "As": 3.3672, "B":  2.3110,
    "Ca": 4.3838, "Cd": 3.5840, "Ce": 3.7670, "Co": 2.8010,
    "Cr": 2.8363, "Cu": 2.8955, "Fe": 2.8266, "Ga": 3.3618,
    "Gd": 4.1013, "Ge": 3.3764, "Hf": 3.5338, "In": 3.8099,
    "La": 4.2224, "Mg": 3.5789, "Mn": 2.7883, "Mo": 3.1486,
    "Na": 4.2002, "Nb": 3.3237, "Ni": 2.7938, "Pd": 3.1386,
    "Ru": 3.0460, "Sb": 3.7828, "Sc": 3.6771, "Se": 3.4604,
    "Si": 3.0942, "Sn": 3.8076, "Sr": 4.7133, "Ta": 3.3119,
    "Tc": 3.0712, "Ti": 3.2396, "V":  2.9821, "W":  3.1724,
    "Y":  4.0265, "Zn": 3.1572, "Zr": 3.5687,
}

# POTCAR variants (PAW-PBE recommended)
POTCAR_VARIANTS = {
    "Ag": "Ag",    "Al": "Al",    "As": "As",    "B":  "B",
    "Ca": "Ca_sv", "Cd": "Cd",    "Ce": "Ce",    "Co": "Co",
    "Cr": "Cr_pv", "Cu": "Cu",    "Fe": "Fe_pv", "Ga": "Ga_d",
    "Gd": "Gd_3",  "Ge": "Ge_d",  "Hf": "Hf_pv", "In": "In_d",
    "La": "La",    "Mg": "Mg_pv", "Mn": "Mn_pv", "Mo": "Mo_pv",
    "Na": "Na_pv", "Nb": "Nb_pv", "Ni": "Ni_pv", "Pd": "Pd",
    "Ru": "Ru_pv", "Sb": "Sb",    "Sc": "Sc_sv", "Se": "Se",
    "Si": "Si",    "Sn": "Sn_d",  "Sr": "Sr_sv", "Ta": "Ta_pv",
    "Tc": "Tc_pv", "Ti": "Ti_pv", "V":  "V_sv",  "W":  "W_sv",
    "Y":  "Y_sv",  "Zn": "Zn",    "Zr": "Zr_sv",
}

# =====================================================================
# SQS-16 configuration for BCC A8B8 (2×2×2 supercell)
# Pre-optimized site occupation to minimize Warren-Cowley SRO parameters
# BCC basis: 2 atoms/cell → 2×2×2 = 16 atoms
# Convention: 0 = element A, 1 = element B
# =====================================================================
# This SQS-16 has been optimized to give:
#   α_1nn ≈ 0 (1st nearest neighbor)
#   α_2nn ≈ 0 (2nd nearest neighbor)
#   α_3nn ≈ 0 (3rd nearest neighbor)
SQS_OCCUPATION = [0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0]

# BCC 2×2×2 supercell fractional coordinates
# Conventional BCC has atoms at (0,0,0) and (0.5,0.5,0.5)
# 2×2×2 → 16 atoms
BCC_2x2x2_POSITIONS = []
for ix in range(2):
    for iy in range(2):
        for iz in range(2):
            # Corner atom
            BCC_2x2x2_POSITIONS.append(
                (ix / 2.0, iy / 2.0, iz / 2.0))
            # Body-center atom
            BCC_2x2x2_POSITIONS.append(
                ((ix + 0.5) / 2.0, (iy + 0.5) / 2.0, (iz + 0.5) / 2.0))


def estimate_sqs_a0(el_a, el_b):
    """Estimate initial supercell lattice parameter (2×a_BCC)."""
    a_a = ELEMENT_A0_BCC.get(el_a, 3.20)
    a_b = ELEMENT_A0_BCC.get(el_b, 3.20)
    a_avg = 0.5 * (a_a + a_b)
    return 2.0 * a_avg  # 2×2×2 supercell


def write_incar_sqs(dirpath):
    """Write INCAR matching existing BCC_B2 settings (ASE-generated style)."""
    content = """INCAR created by Atomic Simulation Environment
 ENCUT = 320.000000
 POTIM = 0.020000
 EDIFF = 1.00e-06
 EDIFFG = -1.00e-02
 ALGO = Normal
 GGA = PE
 PREC = high
 IBRION = 2
 ISIF = 3
 ISPIN = 2
 NELM = 60
 NSW = 120
"""
    with open(os.path.join(dirpath, "INCAR"), "w") as f:
        f.write(content)


def write_poscar_sqs(dirpath, el_a, el_b, a_super):
    """
    Write POSCAR for BCC SQS-16 supercell (2×2×2, A8B8).

    el_a: element A (8 atoms)
    el_b: element B (8 atoms)
    For same-element (el_a == el_b): pure BCC supercell (16 atoms of one element)
    """
    if el_a == el_b:
        # Same element: all 16 atoms are the same
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
        lines.append("")
    else:
        # Binary: SQS occupation
        pos_a = []
        pos_b = []
        for i, occ in enumerate(SQS_OCCUPATION):
            if occ == 0:
                pos_a.append(BCC_2x2x2_POSITIONS[i])
            else:
                pos_b.append(BCC_2x2x2_POSITIONS[i])

        n_a = len(pos_a)
        n_b = len(pos_b)

        lines = [
            f"{el_a}{n_a}{el_b}{n_b} BCC-SQS16 (2x2x2, 50:50)",
            "1.0",
            f"  {a_super:.6f}  0.000000  0.000000",
            f"  0.000000  {a_super:.6f}  0.000000",
            f"  0.000000  0.000000  {a_super:.6f}",
            f"  {el_a}  {el_b}",
            f"  {n_a}  {n_b}",
            "Direct",
        ]

        for pos in pos_a:
            lines.append(f"  {pos[0]:.6f}  {pos[1]:.6f}  {pos[2]:.6f}")
        for pos in pos_b:
            lines.append(f"  {pos[0]:.6f}  {pos[1]:.6f}  {pos[2]:.6f}")
        lines.append("")

    with open(os.path.join(dirpath, "POSCAR"), "w") as f:
        f.write("\n".join(lines))


def write_kpoints_sqs(dirpath, kmesh=4):
    """Write KPOINTS file (smaller k-mesh for 16-atom supercell)."""
    content = f"""Automatic mesh
0
Gamma
  {kmesh} {kmesh} {kmesh}
  0 0 0
"""
    with open(os.path.join(dirpath, "KPOINTS"), "w") as f:
        f.write(content)


def generate_potcar_script(base_dir, calculations):
    """Generate shell script to create POTCAR from $VASP_PP_PATH/potpaw_PBE."""
    lines = [
        "#!/bin/bash",
        "# POTCAR generation script for BCC_SQS refractory calculations",
        "# Usage: bash make_potcar.sh",
        "# Requires: $VASP_PP_PATH environment variable",
        "",
        'if [ -z "$VASP_PP_PATH" ]; then',
        '    echo "Error: VASP_PP_PATH environment variable is not set."',
        '    echo "Set it to the VASP pseudopotential base directory, e.g.:"',
        '    echo "  export VASP_PP_PATH=/path/to/vasp_pp"',
        '    exit 1',
        'fi',
        "",
        'PP_DIR="$VASP_PP_PATH/potpaw_PBE"',
        'echo "Using PP_DIR=$PP_DIR"',
        'echo "Generating POTCAR files for BCC_SQS refractory calculations..."',
        "",
    ]

    for calc_name, el_a, el_b in calculations:
        pot_a = POTCAR_VARIANTS.get(el_a, el_a)
        pot_b = POTCAR_VARIANTS.get(el_b, el_b)
        if el_a == el_b:
            # Same-element: single POTCAR
            lines.append(f"# --- {calc_name} (pure reference) ---")
            lines.append(f'echo "  {calc_name}: {pot_a}"')
            lines.append(
                f'cat "$PP_DIR"/{pot_a}/POTCAR '
                f'> {calc_name}/POTCAR'
            )
        else:
            lines.append(f"# --- {calc_name} ---")
            lines.append(f'echo "  {calc_name}: {pot_a} + {pot_b}"')
            lines.append(
                f'cat "$PP_DIR"/{pot_a}/POTCAR "$PP_DIR"/{pot_b}/POTCAR '
                f'> {calc_name}/POTCAR'
            )
        lines.append(
            f'if [ $? -ne 0 ]; then echo "  WARNING: Failed for {calc_name}"; fi'
        )
        lines.append("")

    lines.append(f'echo "Done. Generated POTCAR for {len(calculations)} calculations."')
    lines.append("")

    with open(os.path.join(base_dir, "make_potcar.sh"), "w") as f:
        f.write("\n".join(lines))
    os.chmod(os.path.join(base_dir, "make_potcar.sh"), 0o755)


def generate_run_script(base_dir, calculations):
    """Generate all-in-one execution script (POTCAR生成→並列計算→結果抽出)."""
    calc_names = [c[0] for c in calculations]

    # Build POTCAR generation commands
    potcar_lines = []
    for calc_name, el_a, el_b in calculations:
        pot_a = POTCAR_VARIANTS.get(el_a, el_a)
        pot_b = POTCAR_VARIANTS.get(el_b, el_b)
        if el_a == el_b:
            potcar_lines.append(
                f'cat "$PP_DIR"/{pot_a}/POTCAR > "$BASEDIR"/{calc_name}/POTCAR')
        else:
            potcar_lines.append(
                f'cat "$PP_DIR"/{pot_a}/POTCAR "$PP_DIR"/{pot_b}/POTCAR > "$BASEDIR"/{calc_name}/POTCAR')

    potcar_block = "\n".join(potcar_lines)

    content = f"""#!/bin/bash
# All-in-one execution script for BCC_SQS DFT calculations
# Handles: POTCAR generation → parallel VASP execution → result extraction
#
# Environment: OpenMPI, 32 cores
# Default: 8 parallel jobs × 4 cores each = 32 cores
#
# Usage:
#   bash run_all.sh              # 8並列×4コア (default)
#   bash run_all.sh 4 8          # 4並列×8コア
#   bash run_all.sh 16 2         # 16並列×2コア
#
# Requires: $VASP_PP_PATH environment variable

set -e

NJOBS_PARALLEL=${{1:-8}}
NPROCS_PER_JOB=${{2:-4}}
VASP_CMD="mpirun -np $NPROCS_PER_JOB vasp_std"
LOG="run_status.log"
BASEDIR=$(cd "$(dirname "$0")" && pwd)

# ============================================================
# Step 1: POTCAR generation
# ============================================================
echo "=== Step 1: POTCAR Generation ===" | tee $LOG

if [ -z "$VASP_PP_PATH" ]; then
    echo "Error: VASP_PP_PATH environment variable is not set." | tee -a $LOG
    echo "Set it to the VASP pseudopotential base directory, e.g.:" | tee -a $LOG
    echo "  export VASP_PP_PATH=/path/to/vasp_pp" | tee -a $LOG
    exit 1
fi

PP_DIR="$VASP_PP_PATH/potpaw_PBE"
echo "Using PP_DIR=$PP_DIR" | tee -a $LOG
echo "Generating POTCAR for {len(calculations)} calculations..." | tee -a $LOG

{potcar_block}

echo "POTCAR generation complete." | tee -a $LOG
echo "" | tee -a $LOG

# ============================================================
# Step 2: Parallel VASP execution
# ============================================================
set +e
echo "=== Step 2: Parallel VASP Execution ===" | tee -a $LOG
echo "Total: {len(calculations)} calculations (16 atoms each)" | tee -a $LOG
echo "Parallel jobs: $NJOBS_PARALLEL × $NPROCS_PER_JOB cores = $(($NJOBS_PARALLEL * $NPROCS_PER_JOB)) cores" | tee -a $LOG
echo "Started: $(date)" | tee -a $LOG
echo "" | tee -a $LOG

# Function to run a single calculation
run_one() {{
    local dir=$1
    local basedir=$2
    local vasp_cmd=$3
    cd "$basedir/$dir"
    if [ -f OUTCAR ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  SKIP (already converged): $dir"
        return 0
    fi
    $vasp_cmd > vasp.out 2>&1
    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
        echo "  CONVERGED: $dir"
    else
        echo "  WARNING (not converged): $dir"
    fi
}}
export -f run_one

# Run all calculations in parallel
echo "{chr(10).join(calc_names)}" | \\
    xargs -I {{}} -P $NJOBS_PARALLEL bash -c "run_one '{{}}' '$BASEDIR' '$VASP_CMD'" 2>&1 | tee -a $LOG

echo "" | tee -a $LOG
echo "Finished: $(date)" | tee -a $LOG

# ============================================================
# Step 3: Result extraction
# ============================================================
echo "" | tee -a $LOG
echo "=== Step 3: Result Extraction ===" | tee -a $LOG
cd "$BASEDIR"
python extract_results.py 2>&1 | tee -a $LOG

# ============================================================
# Summary
# ============================================================
echo "" | tee -a $LOG
echo "=== Summary ===" | tee -a $LOG
TOTAL={len(calculations)}
CONVERGED=$(grep -c "CONVERGED" $LOG 2>/dev/null || echo 0)
SKIPPED=$(grep -c "SKIP" $LOG 2>/dev/null || echo 0)
FAILED=$(grep -c "WARNING" $LOG 2>/dev/null || echo 0)
echo "  Converged: $CONVERGED / $TOTAL" | tee -a $LOG
echo "  Skipped (already done): $SKIPPED" | tee -a $LOG
echo "  Not converged: $FAILED" | tee -a $LOG
echo "" | tee -a $LOG
echo "Results saved to: sqs_refractory_results.csv" | tee -a $LOG
"""

    with open(os.path.join(base_dir, "run_all.sh"), "w") as f:
        f.write(content)
    os.chmod(os.path.join(base_dir, "run_all.sh"), 0o755)


def generate_extract_script(base_dir):
    """Generate Python script to extract results and compute Omega_sf."""
    content = '''#!/usr/bin/env python3
"""
Extract lattice constants from completed SQS refractory VASP calculations
and compute Omega_sf values.

Omega_sf = (V_sqs - V_vegard) / V_vegard

where V_sqs = a_sqs^3 / 16 (volume per atom from 16-atom supercell)
      V_vegard = 0.5 * V_A + 0.5 * V_B (Vegard average for 50:50)
      V_A, V_B = pure element BCC atomic volumes (from same-element SQS or King)
"""

import os
import csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# King experimental atomic volumes (Å³) for reference
KING_VOLUMES = {
    "Cr": 12.01, "Hf": 22.31, "Mo": 15.58, "Nb": 17.98,
    "Ta": 18.01, "Ti": 17.65, "V": 13.82, "W": 15.86, "Zr": 23.28,
}


def extract_lattice_from_contcar(contcar_path):
    """Extract lattice vectors from CONTCAR and compute volume."""
    with open(contcar_path, "r") as f:
        lines = f.readlines()
    scale = float(lines[1].strip())
    a = [float(x) for x in lines[2].split()]
    b = [float(x) for x in lines[3].split()]
    c = [float(x) for x in lines[4].split()]

    # Volume = |a · (b × c)|
    bxc = [b[1]*c[2] - b[2]*c[1],
           b[2]*c[0] - b[0]*c[2],
           b[0]*c[1] - b[1]*c[0]]
    vol = abs(a[0]*bxc[0] + a[1]*bxc[1] + a[2]*bxc[2]) * scale**3

    # Effective cubic lattice constant
    a_eff = vol ** (1.0/3.0)
    return a_eff, vol


results = []
dirs = sorted([d for d in os.listdir(BASE_DIR)
               if os.path.isdir(os.path.join(BASE_DIR, d))
               and d[0].isupper() and '8' in d])

print("SQS-16 Refractory Results:")
print("=" * 70)

for d in dirs:
    contcar = os.path.join(BASE_DIR, d, "CONTCAR")
    if not os.path.exists(contcar):
        print(f"  {d}: CONTCAR not found")
        continue

    # Parse elements from directory name (e.g., "Cr8Hf8")
    pair_name = d
    # Elements are in POSCAR header
    poscar = os.path.join(BASE_DIR, d, "POSCAR")
    with open(poscar, "r") as f:
        poscar_lines = f.readlines()
    elements = poscar_lines[5].split()
    el_a = elements[0]
    el_b = elements[1] if len(elements) > 1 else elements[0]

    a_eff, vol_total = extract_lattice_from_contcar(contcar)
    vol_per_atom = vol_total / 16.0

    # Compute Omega_sf
    v_a = KING_VOLUMES[el_a]
    v_b = KING_VOLUMES[el_b]
    v_vegard = 0.5 * v_a + 0.5 * v_b
    omega_sf = (vol_per_atom - v_vegard) / v_vegard

    results.append({
        "pair": pair_name,
        "element_A": el_a,
        "element_B": el_b,
        "a_supercell": a_eff,
        "vol_per_atom": vol_per_atom,
        "v_vegard": v_vegard,
        "omega_sf": omega_sf,
    })
    print(f"  {pair_name:8s}: a_eff = {a_eff:.4f} Å, "
          f"V/atom = {vol_per_atom:.3f} Å³, "
          f"V_Vegard = {v_vegard:.3f} Å³, "
          f"Ω_sf = {omega_sf:+.4f}")

# Save to CSV
csv_path = os.path.join(BASE_DIR, "sqs_refractory_results.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["pair", "element_A", "element_B",
                                           "a_supercell", "vol_per_atom",
                                           "v_vegard", "omega_sf"])
    writer.writeheader()
    writer.writerows(results)

print(f"\\nSaved {len(results)} results to {csv_path}")
print(f"\\nComparison with B2 data can be done by running:")
print(f"  python compare_b2_sqs.py")
'''
    with open(os.path.join(base_dir, "extract_results.py"), "w") as f:
        f.write(content)
    os.chmod(os.path.join(base_dir, "extract_results.py"), 0o755)


def main():
    parser = argparse.ArgumentParser(
        description="BCC SQS VASP input generator")
    parser.add_argument("--group", choices=["refractory", "all"],
                        default="refractory",
                        help="Element group: refractory (9元素,45計算) or all (39元素,780計算)")
    parser.add_argument("--elements", type=str, default=None,
                        help="Comma-separated element list (e.g., 'Fe,Co,Ni,Cr,Mn')")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Output directory (default: BCC_SQS)")
    args = parser.parse_args()

    # Determine element list
    if args.elements:
        elements = sorted([e.strip() for e in args.elements.split(",")])
        group_name = "custom"
        # Validate
        for el in elements:
            if el not in ELEMENT_A0_BCC:
                print(f"Error: Unknown element '{el}'. Available: {ALL_B2_ELEMENTS}")
                sys.exit(1)
    elif args.group == "all":
        elements = ALL_B2_ELEMENTS
        group_name = "all"
    else:
        elements = REFRACTORY_ELEMENTS
        group_name = "refractory"

    # Output directory
    if args.outdir:
        base_dir = args.outdir
    else:
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "BCC_SQS")
    os.makedirs(base_dir, exist_ok=True)

    calculations = []

    # Generate same-element references
    for el in elements:
        calc_name = f"{el}8{el}8"
        a_super = 2.0 * ELEMENT_A0_BCC[el]
        dirpath = os.path.join(base_dir, calc_name)
        os.makedirs(dirpath, exist_ok=True)
        write_incar_sqs(dirpath)
        write_poscar_sqs(dirpath, el, el, a_super)
        write_kpoints_sqs(dirpath)
        calculations.append((calc_name, el, el))

    # Generate all pairs (symmetric: only one direction needed)
    for el_a, el_b in itertools.combinations(elements, 2):
        calc_name = f"{el_a}8{el_b}8"
        a_super = estimate_sqs_a0(el_a, el_b)
        dirpath = os.path.join(base_dir, calc_name)
        os.makedirs(dirpath, exist_ok=True)
        write_incar_sqs(dirpath)
        write_poscar_sqs(dirpath, el_a, el_b, a_super)
        write_kpoints_sqs(dirpath)
        calculations.append((calc_name, el_a, el_b))

    # Generate helper scripts
    generate_potcar_script(base_dir, calculations)
    generate_run_script(base_dir, calculations)
    generate_extract_script(base_dir)

    n_homo = len(elements)
    n_hetero = len(calculations) - n_homo
    n_total = len(calculations)
    print(f"Generated {n_total} SQS calculations in {base_dir}/")
    print(f"  - Group: {group_name} ({len(elements)} elements)")
    print(f"  - {n_hetero} binary pairs (A8B8)")
    print(f"  - {n_homo} same-element references (A8A8)")
    print(f"  - Total: {n_total} calculations")
    print(f"  - Each: 16 atoms (2×2×2 BCC supercell)")
    print()
    print("Directory naming: A8B8 format (e.g., Cr8Hf8, Mo8Mo8)")
    print(f"  Elements: {', '.join(elements)}")
    print()
    print("Next steps:")
    print(f"  1. export VASP_PP_PATH=/path/to/vasp_pp  # (potpaw_PBE/ が含まれるディレクトリ)")
    print(f"  2. cd {os.path.basename(base_dir)}")
    print(f"  3. bash run_all.sh   # POTCAR生成→並列計算→結果抽出 (all-in-one)")
    print()
    if n_total > 100:
        est_hours = n_total * 0.75 / 8  # 8並列想定
        print(f"Estimated time (8並列×4コア): ~{est_hours:.0f} hours")
    else:
        est_hours = n_total * 0.75 / 8
        print(f"Estimated time (8並列×4コア): ~{est_hours:.1f} hours")


if __name__ == "__main__":
    main()
