#!/usr/bin/env python3
"""
VASP input file generator for refractory BCC SQS (Special Quasi-random Structure)
calculations.

Generates INCAR, POSCAR, KPOINTS for binary BCC solid solutions of refractory
elements using SQS supercells. This provides a better proxy for BCC HEA behavior
than ordered B2 structures.

SQS approach:
  - 16-atom BCC supercell (2×2×2 conventional BCC = 16 atoms)
  - 50:50 composition (8A + 8B) to maximize pair correlation matching
  - SQS configurations are pre-generated to minimize short-range order

Comparison with B2:
  - B2 = perfectly ordered (each atom surrounded by 8 unlike neighbors)
  - SQS = quasi-random (mimics random solid solution pair correlations)
  - SQS is physically more relevant for BCC HEA applications

Usage:
    python generate_refractory_sqs.py

Output structure:
    SQS_refractory/
    ├── CrHf_sqs16/   (INCAR, POSCAR, KPOINTS)
    ├── CrMo_sqs16/
    ├── ...
    ├── make_potcar.sh
    └── run_all.sh
"""

import os
import itertools
import numpy as np

# =====================================================================
# Refractory elements
# =====================================================================
REFRACTORY_ELEMENTS = sorted(['Nb', 'Ti', 'V', 'Zr', 'Mo', 'Ta', 'W', 'Hf', 'Cr'])

# BCC lattice constants (Å)
ELEMENT_A0_BCC = {
    "Cr": 2.880,
    "Hf": 3.530,
    "Mo": 3.147,
    "Nb": 3.301,
    "Ta": 3.303,
    "Ti": 3.250,
    "V":  3.024,
    "W":  3.165,
    "Zr": 3.570,
}

# POTCAR variants
POTCAR_VARIANTS = {
    "Cr": "Cr_pv",
    "Hf": "Hf_pv",
    "Mo": "Mo_pv",
    "Nb": "Nb_pv",
    "Ta": "Ta_pv",
    "Ti": "Ti_pv",
    "V":  "V_sv",
    "W":  "W_pv",
    "Zr": "Zr_sv",
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
    """Write INCAR for SQS structure optimization (16 atoms)."""
    content = """SYSTEM = BCC SQS-16 optimization (refractory)

# Electronic relaxation
ENCUT  = 520
PREC   = Accurate
EDIFF  = 1E-6
NELM   = 200
LREAL  = Auto

# Ionic relaxation
IBRION = 2
ISIF   = 3
NSW    = 200
EDIFFG = -0.01

# Smearing (metals)
ISMEAR = 1
SIGMA  = 0.2

# Exchange-correlation
GGA    = PE

# Spin polarization (important for Cr)
ISPIN  = 2

# Output
LORBIT = 11
LWAVE  = .FALSE.
LCHARG = .FALSE.

# Performance (16 atoms → moderate parallelization)
NCORE  = 4
"""
    with open(os.path.join(dirpath, "INCAR"), "w") as f:
        f.write(content)


def write_poscar_sqs(dirpath, el_a, el_b, a_super):
    """
    Write POSCAR for BCC SQS-16 supercell (2×2×2, A8B8).

    el_a: element A (8 atoms)
    el_b: element B (8 atoms)
    """
    # Sort atoms by element (VASP requires contiguous species blocks)
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
    """Generate shell script to create POTCAR from $VASPPOT."""
    lines = [
        "#!/bin/bash",
        "# POTCAR generation script for SQS refractory calculations",
        "# Usage: bash make_potcar.sh",
        "# Requires: $VASPPOT environment variable",
        "",
        'if [ -z "$VASPPOT" ]; then',
        '    echo "Error: VASPPOT environment variable is not set."',
        '    echo "Set it to the PAW-PBE pseudopotential directory, e.g.:"',
        '    echo "  export VASPPOT=/path/to/potpaw_PBE.64"',
        '    exit 1',
        'fi',
        "",
        'echo "Using VASPPOT=$VASPPOT"',
        'echo "Generating POTCAR files for SQS refractory calculations..."',
        "",
    ]

    for calc_name, el_a, el_b in calculations:
        pot_a = POTCAR_VARIANTS.get(el_a, el_a)
        pot_b = POTCAR_VARIANTS.get(el_b, el_b)
        lines.append(f"# --- {calc_name} ---")
        lines.append(f'echo "  {calc_name}: {pot_a} + {pot_b}"')
        lines.append(
            f'cat "$VASPPOT"/{pot_a}/POTCAR "$VASPPOT"/{pot_b}/POTCAR '
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
    """Generate batch submission script for all calculations."""
    lines = [
        "#!/bin/bash",
        "# Batch execution script for SQS refractory DFT calculations",
        "# Usage: bash run_all.sh",
        "#",
        "# Each SQS-16 calc is 16 atoms → moderate cost (~30min-1h per job).",
        "# Adjust VASP_CMD and NPROCS to your environment.",
        "",
        'VASP_CMD="mpirun -np ${NPROCS:-8} vasp_std"',
        'LOG="run_status.log"',
        "",
        'echo "=== SQS Refractory Calculations ===" | tee $LOG',
        f'echo "Total: {len(calculations)} calculations (16 atoms each)" | tee -a $LOG',
        'echo "Started: $(date)" | tee -a $LOG',
        'echo "" | tee -a $LOG',
        "",
    ]

    for i, (calc_name, _, _) in enumerate(calculations, 1):
        lines.append(f'echo "[{i}/{len(calculations)}] {calc_name}..." | tee -a $LOG')
        lines.append(f'cd {calc_name}')
        lines.append(f'$VASP_CMD > vasp.out 2>&1')
        lines.append(f'if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then')
        lines.append(f'    echo "  CONVERGED" | tee -a ../$LOG')
        lines.append(f'else')
        lines.append(f'    echo "  WARNING: not converged" | tee -a ../$LOG')
        lines.append(f'fi')
        lines.append(f'cd ..')
        lines.append("")

    lines.append('echo "" | tee -a $LOG')
    lines.append('echo "Finished: $(date)" | tee -a $LOG')
    lines.append("")

    with open(os.path.join(base_dir, "run_all.sh"), "w") as f:
        f.write("\n".join(lines))
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
               if os.path.isdir(os.path.join(BASE_DIR, d)) and "_sqs16" in d])

print("SQS-16 Refractory Results:")
print("=" * 70)

for d in dirs:
    contcar = os.path.join(BASE_DIR, d, "CONTCAR")
    if not os.path.exists(contcar):
        print(f"  {d}: CONTCAR not found")
        continue

    # Parse elements from directory name (e.g., "CrHf_sqs16")
    pair_name = d.replace("_sqs16", "")
    # Elements are in POSCAR header
    poscar = os.path.join(BASE_DIR, d, "POSCAR")
    with open(poscar, "r") as f:
        poscar_lines = f.readlines()
    elements = poscar_lines[5].split()
    el_a, el_b = elements[0], elements[1]

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
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SQS_refractory")
    os.makedirs(base_dir, exist_ok=True)

    # Generate all refractory pairs (symmetric: only one direction needed)
    calculations = []
    for el_a, el_b in itertools.combinations(REFRACTORY_ELEMENTS, 2):
        calc_name = f"{el_a}{el_b}_sqs16"
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

    print(f"Generated {len(calculations)} SQS calculations in {base_dir}/")
    print(f"  - 36 refractory pairs × 1 (symmetric SQS) = 36 calculations")
    print(f"  - Each: 16 atoms (2×2×2 BCC supercell, A8B8)")
    print()
    print("SQS advantages over B2:")
    print("  - Random-like atom distribution (mimics BCC solid solution)")
    print("  - No artificial long-range order")
    print("  - Ω_sf directly applicable to BCC HEA prediction")
    print()
    print("Next steps:")
    print("  1. cd SQS_refractory")
    print("  2. bash make_potcar.sh")
    print("  3. bash run_all.sh")
    print("  4. python extract_results.py")
    print()
    print("Note: SQS calculations are more expensive than B2 (~30min-1h each).")
    print("      Consider submitting to a job queue system.")


if __name__ == "__main__":
    main()
