#!/usr/bin/env python3
"""
VASP input file generator for missing L1₂ Ω_sf pairs.

Generates INCAR, POSCAR, KPOINTS for each A3B and B3A L1₂ compound.
Also generates a POTCAR concatenation script and a batch submission script.

Usage:
    python generate_vasp_inputs.py

Output structure:
    L12_calculations/
    ├── Fe3Mn/   (INCAR, POSCAR, KPOINTS)
    ├── Mn3Fe/
    ├── ...
    ├── make_potcar.sh    # POTCAR generation from $VASPPOT
    └── run_all.sh        # Batch submission script
"""

import os
import math

# =====================================================================
# Missing L1₂ pairs (sorted by HEA usage frequency)
# =====================================================================
MISSING_L12_PAIRS = [
    ("Fe", "Mn", 10),   # used in 10 FCC HEAs
    ("Cr", "Mn", 7),    # used in 7
    ("Al", "Mn", 3),
    ("Cr", "Mo", 3),
    ("Fe", "Mo", 3),
    ("Mo", "Ni", 3),
    ("Ir", "Pd", 2),
    ("Ir", "Ru", 2),
    ("Pd", "Pt", 2),
    ("Pd", "Rh", 2),
    ("Pd", "Ru", 2),
    ("Pt", "Ru", 2),
    ("Ni", "Pd", 2),
    ("Os", "Pd", 1),
    ("Os", "Pt", 1),
    ("Os", "Rh", 1),
    ("Os", "Ru", 1),
    ("Cr", "V",  1),
    ("Fe", "V",  1),
    ("Ni", "V",  1),
    ("Al", "Cr", 1),
]

# =====================================================================
# Approximate pure-element lattice constants (Å) for initial POSCAR
# Used to estimate initial L1₂ lattice parameter via Vegard's law
# Source: experimental values (FCC-equivalent a₀)
# =====================================================================
ELEMENT_A0_FCC = {
    "Al": 4.050, "Cr": 3.640, "Mn": 3.630, "Fe": 3.570, "Co": 3.545,
    "Ni": 3.524, "Cu": 3.615, "Zn": 3.940, "V":  3.820, "Mo": 3.960,
    "Pd": 3.890, "Pt": 3.924, "Rh": 3.803, "Ir": 3.839, "Ru": 3.827,
    "Os": 3.850, "Ti": 4.100, "Nb": 4.160, "Ta": 4.150, "W":  3.980,
    "Hf": 4.470, "Zr": 4.540,
}


def estimate_l12_a0(el_face, el_corner):
    """Estimate initial L1₂ lattice parameter from Vegard's law."""
    a_face = ELEMENT_A0_FCC.get(el_face, 3.80)
    a_corner = ELEMENT_A0_FCC.get(el_corner, 3.80)
    return 0.75 * a_face + 0.25 * a_corner


def write_incar(dirpath):
    """Write INCAR for L1₂ structure optimization."""
    content = """SYSTEM = L12 structure optimization

# Electronic relaxation
ENCUT  = 520
PREC   = Accurate
EDIFF  = 1E-6
NELM   = 200
LREAL  = .FALSE.

# Ionic relaxation
IBRION = 2
ISIF   = 3
NSW    = 100
EDIFFG = -0.01

# Smearing (metals)
ISMEAR = 1
SIGMA  = 0.2

# Exchange-correlation
GGA    = PE

# Spin polarization (important for 3d metals)
ISPIN  = 2

# Output
LORBIT = 11
LWAVE  = .FALSE.
LCHARG = .FALSE.

# Performance
NCORE  = 4
"""
    with open(os.path.join(dirpath, "INCAR"), "w") as f:
        f.write(content)


def write_poscar(dirpath, el_face, el_corner, a0):
    """
    Write POSCAR for L1₂ (Cu3Au-type, Pm-3m) structure.

    Atomic positions:
      Corner atom (1×): (0, 0, 0)
      Face-center atoms (3×): (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)

    el_face:   element on face centers (3 atoms)
    el_corner: element on corners (1 atom)
    => Formula: el_face₃ el_corner₁
    """
    content = f"""{el_face}3{el_corner} L12 (Pm-3m)
1.0
  {a0:.6f}  0.000000  0.000000
  0.000000  {a0:.6f}  0.000000
  0.000000  0.000000  {a0:.6f}
  {el_face}  {el_corner}
  3  1
Direct
  0.000000  0.500000  0.500000  ! {el_face} (face)
  0.500000  0.000000  0.500000  ! {el_face} (face)
  0.500000  0.500000  0.000000  ! {el_face} (face)
  0.000000  0.000000  0.000000  ! {el_corner} (corner)
"""
    with open(os.path.join(dirpath, "POSCAR"), "w") as f:
        f.write(content)


def write_kpoints(dirpath, kmesh=12):
    """Write KPOINTS file (Gamma-centered)."""
    content = f"""Automatic mesh
0
Gamma
  {kmesh} {kmesh} {kmesh}
  0 0 0
"""
    with open(os.path.join(dirpath, "KPOINTS"), "w") as f:
        f.write(content)


# POTCAR recommended pseudopotentials (VASP 5.4+ PAW-PBE)
# Use _pv/_sv variants for elements where semi-core states are important
POTCAR_VARIANTS = {
    "Al": "Al",
    "Cr": "Cr_pv",
    "Mn": "Mn_pv",
    "Fe": "Fe_pv",
    "Co": "Co",
    "Ni": "Ni_pv",
    "Cu": "Cu_pv",
    "V":  "V_sv",
    "Mo": "Mo_pv",
    "Pd": "Pd",
    "Pt": "Pt",
    "Rh": "Rh_pv",
    "Ir": "Ir",
    "Ru": "Ru_pv",
    "Os": "Os_pv",
    "Ti": "Ti_pv",
    "Nb": "Nb_pv",
    "Ta": "Ta_pv",
    "W":  "W_pv",
    "Hf": "Hf_pv",
    "Zr": "Zr_sv",
    "Zn": "Zn",
}


def generate_potcar_script(base_dir, calculations):
    """Generate shell script to create POTCAR from $VASPPOT."""
    lines = [
        "#!/bin/bash",
        "# POTCAR generation script",
        "# Usage: bash make_potcar.sh",
        "# Requires: $VASPPOT environment variable pointing to VASP pseudopotential directory",
        "#   e.g., export VASPPOT=/path/to/potpaw_PBE.64",
        "",
        'if [ -z "$VASPPOT" ]; then',
        '    echo "Error: VASPPOT environment variable is not set."',
        '    echo "Set it to the PAW-PBE pseudopotential directory, e.g.:"',
        '    echo "  export VASPPOT=/path/to/potpaw_PBE.64"',
        '    exit 1',
        'fi',
        "",
        'echo "Using VASPPOT=$VASPPOT"',
        'echo "Generating POTCAR files for L12 calculations..."',
        "",
    ]

    for calc_name, el_face, el_corner in calculations:
        pot_face = POTCAR_VARIANTS.get(el_face, el_face)
        pot_corner = POTCAR_VARIANTS.get(el_corner, el_corner)
        lines.append(f"# --- {calc_name} ---")
        lines.append(f'echo "  {calc_name}: {pot_face} + {pot_corner}"')
        lines.append(
            f'cat "$VASPPOT"/{pot_face}/POTCAR "$VASPPOT"/{pot_corner}/POTCAR '
            f'> {calc_name}/POTCAR'
        )
        lines.append(
            f'if [ $? -ne 0 ]; then echo "  WARNING: Failed for {calc_name}"; fi'
        )
        lines.append("")

    lines.append('echo "Done. Generated POTCAR for all calculations."')
    lines.append("")

    with open(os.path.join(base_dir, "make_potcar.sh"), "w") as f:
        f.write("\n".join(lines))
    os.chmod(os.path.join(base_dir, "make_potcar.sh"), 0o755)


def generate_run_script(base_dir, calculations):
    """Generate batch submission script for all calculations."""
    lines = [
        "#!/bin/bash",
        "# Batch execution script for L12 DFT calculations",
        "# Usage: bash run_all.sh",
        "#",
        "# Adjust VASP_CMD and NPROCS to your environment.",
        "# For NIMS GPU cluster, modify the job submission commands as needed.",
        "",
        "VASP_CMD=\"mpirun -np ${NPROCS:-16} vasp_std\"",
        "BASE_DIR=$(cd $(dirname $0) && pwd)",
        "",
        "# --- Job submission function ---",
        "run_calc() {",
        '    local dir="$1"',
        '    local name=$(basename "$dir")',
        '    cd "$dir"',
        "",
        '    if [ -f "OUTCAR" ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then',
        '        echo "SKIP: $name (already converged)"',
        '        cd "$BASE_DIR"',
        '        return 0',
        "    fi",
        "",
        '    if [ ! -f "POTCAR" ]; then',
        '        echo "ERROR: $name - POTCAR not found. Run make_potcar.sh first."',
        '        cd "$BASE_DIR"',
        '        return 1',
        "    fi",
        "",
        '    echo "RUN: $name"',
        "    $VASP_CMD > vasp.log 2>&1",
        "",
        '    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then',
        '        echo "  OK: $name converged"',
        "    else",
        '        echo "  WARNING: $name may not have converged"',
        "    fi",
        '    cd "$BASE_DIR"',
        "}",
        "",
        "# --- Run all calculations ---",
        'echo "Starting L12 DFT calculations..."',
        f'echo "Total: {len(calculations)} calculations"',
        "",
    ]

    for calc_name, _, _ in calculations:
        lines.append(f'run_calc "$BASE_DIR/{calc_name}"')

    lines.append("")
    lines.append('echo "All calculations completed."')
    lines.append("")

    with open(os.path.join(base_dir, "run_all.sh"), "w") as f:
        f.write("\n".join(lines))
    os.chmod(os.path.join(base_dir, "run_all.sh"), 0o755)


def generate_extract_script(base_dir, calculations):
    """Generate script to extract optimized lattice constants from CONTCAR/OUTCAR."""
    lines = [
        "#!/bin/bash",
        "# Extract optimized lattice constants from L12 calculations",
        "# Usage: bash extract_results.sh > l12_results.csv",
        "",
        'echo "formula,element_A,element_B,count_A,count_B,lattice_constant,converged"',
        "",
    ]

    for calc_name, el_face, el_corner in calculations:
        lines.append(f'# --- {calc_name} ---')
        lines.append(f'DIR="{calc_name}"')
        lines.append('if [ -d "$DIR" ] && [ -f "$DIR/CONTCAR" ]; then')
        lines.append('    A=$(head -3 "$DIR/CONTCAR" | tail -1 | awk \'{print $1}\')')
        lines.append('    CONV="no"')
        lines.append('    if grep -q "reached required accuracy" "$DIR/OUTCAR" 2>/dev/null; then')
        lines.append('        CONV="yes"')
        lines.append('    fi')
        lines.append(f'    echo "{el_face}3{el_corner},{el_face},{el_corner},3,1,$A,$CONV"')
        lines.append('else')
        lines.append(f'    echo "{el_face}3{el_corner},{el_face},{el_corner},3,1,NA,not_run"')
        lines.append('fi')
        lines.append("")

    with open(os.path.join(base_dir, "extract_results.sh"), "w") as f:
        f.write("\n".join(lines))
    os.chmod(os.path.join(base_dir, "extract_results.sh"), 0o755)


def generate_jobscript(base_dir, calculations):
    """Generate a PBS/SLURM job script template for NIMS cluster."""
    # PBS version
    pbs = """#!/bin/bash
#PBS -N L12_omega_sf
#PBS -l select=1:ncpus=16:mpiprocs=16
#PBS -l walltime=48:00:00
#PBS -j oe

# Load modules (adjust to your NIMS environment)
# module load vasp/6.4.0
# module load intel-mpi

cd $PBS_O_WORKDIR

export NPROCS=16
bash run_all.sh 2>&1 | tee run_all.log

echo "Extracting results..."
bash extract_results.sh > l12_results.csv
echo "Done."
"""

    # SLURM version
    slurm = """#!/bin/bash
#SBATCH -J L12_omega_sf
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -t 48:00:00
#SBATCH -o L12_%j.out

# Load modules (adjust to your NIMS environment)
# module load vasp/6.4.0
# module load intel-mpi

cd $SLURM_SUBMIT_DIR

export NPROCS=16
bash run_all.sh 2>&1 | tee run_all.log

echo "Extracting results..."
bash extract_results.sh > l12_results.csv
echo "Done."
"""
    with open(os.path.join(base_dir, "job_pbs.sh"), "w") as f:
        f.write(pbs)
    os.chmod(os.path.join(base_dir, "job_pbs.sh"), 0o755)

    with open(os.path.join(base_dir, "job_slurm.sh"), "w") as f:
        f.write(slurm)
    os.chmod(os.path.join(base_dir, "job_slurm.sh"), 0o755)


def main():
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "L12_calculations")
    os.makedirs(base_dir, exist_ok=True)

    calculations = []  # (dir_name, el_face, el_corner)

    for elA, elB, freq in MISSING_L12_PAIRS:
        # A3B: A at face centers, B at corner
        name1 = f"{elA}3{elB}"
        dir1 = os.path.join(base_dir, name1)
        os.makedirs(dir1, exist_ok=True)
        a0_1 = estimate_l12_a0(elA, elB)
        write_incar(dir1)
        write_poscar(dir1, elA, elB, a0_1)
        write_kpoints(dir1)
        calculations.append((name1, elA, elB))

        # B3A: B at face centers, A at corner
        name2 = f"{elB}3{elA}"
        dir2 = os.path.join(base_dir, name2)
        os.makedirs(dir2, exist_ok=True)
        a0_2 = estimate_l12_a0(elB, elA)
        write_incar(dir2)
        write_poscar(dir2, elB, elA, a0_2)
        write_kpoints(dir2)
        calculations.append((name2, elB, elA))

    # Generate utility scripts
    generate_potcar_script(base_dir, calculations)
    generate_run_script(base_dir, calculations)
    generate_extract_script(base_dir, calculations)
    generate_jobscript(base_dir, calculations)

    # Summary
    print(f"Generated {len(calculations)} L1₂ calculation directories in:")
    print(f"  {base_dir}/")
    print()
    print("Files per directory:")
    print("  INCAR   - Electronic/ionic relaxation parameters")
    print("  POSCAR  - L1₂ structure (Pm-3m, 4 atoms)")
    print("  KPOINTS - 12×12×12 Gamma-centered mesh")
    print()
    print("Utility scripts:")
    print("  make_potcar.sh     - Generate POTCAR from $VASPPOT")
    print("  run_all.sh         - Sequential execution of all calculations")
    print("  extract_results.sh - Extract optimized lattice constants")
    print("  job_pbs.sh         - PBS job submission template")
    print("  job_slurm.sh       - SLURM job submission template")
    print()
    print("Workflow:")
    print("  1. cd L12_calculations")
    print("  2. export VASPPOT=/path/to/potpaw_PBE.64")
    print("  3. bash make_potcar.sh")
    print("  4. qsub job_pbs.sh  (or sbatch job_slurm.sh)")
    print("  5. bash extract_results.sh > l12_results.csv")
    print()

    # Print pair summary table
    print("=" * 60)
    print(f"{'Pair':<10} {'Calculations':<30} {'HEA freq':>8}")
    print("-" * 60)
    for elA, elB, freq in MISSING_L12_PAIRS:
        print(f"{elA}-{elB:<8} {elA}3{elB}, {elB}3{elA:<20} {freq:>8}")
    print("=" * 60)
    print(f"Total: {len(MISSING_L12_PAIRS)} pairs, {len(calculations)} calculations")
    print(f"Estimated wall time: ~{len(calculations) * 0.5:.0f} hours @ 16 cores")


if __name__ == "__main__":
    main()
