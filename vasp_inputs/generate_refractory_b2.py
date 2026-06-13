#!/usr/bin/env python3
"""
VASP input file generator for refractory B2 pair calculations.

Generates INCAR, POSCAR, KPOINTS for all 36 refractory element B2 pairs
(Nb, Ti, V, Zr, Mo, Ta, W, Hf, Cr) to resolve the BCC HEA Ω_sf data gap.

For each pair A-B, both AB and BA B2 structures are generated:
  - AB: A at corner (0,0,0), B at body-center (0.5,0.5,0.5)
  - BA: B at corner, A at body-center

Usage:
    python generate_refractory_b2.py

Output structure:
    B2_refractory/
    ├── CrHf/   (INCAR, POSCAR, KPOINTS)
    ├── HfCr/
    ├── CrMo/
    ├── ...
    ├── make_potcar.sh
    └── run_all.sh
"""

import os
import itertools

# =====================================================================
# Refractory elements for BCC HEA applications
# =====================================================================
REFRACTORY_ELEMENTS = sorted(['Nb', 'Ti', 'V', 'Zr', 'Mo', 'Ta', 'W', 'Hf', 'Cr'])

# =====================================================================
# Approximate BCC lattice constants (Å) for initial POSCAR
# Source: experimental values or DFT-PBE estimates
# =====================================================================
ELEMENT_A0_BCC = {
    "Cr": 2.880,  # exp BCC
    "Hf": 3.530,  # DFT BCC (HCP stable)
    "Mo": 3.147,  # exp BCC
    "Nb": 3.301,  # exp BCC
    "Ta": 3.303,  # exp BCC
    "Ti": 3.250,  # DFT BCC (HCP stable)
    "V":  3.024,  # exp BCC
    "W":  3.165,  # exp BCC
    "Zr": 3.570,  # DFT BCC (HCP stable)
}

# POTCAR recommended pseudopotentials (VASP 5.4+ PAW-PBE)
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


def estimate_b2_a0(el_a, el_b):
    """Estimate initial B2 lattice parameter from Vegard's law."""
    a_a = ELEMENT_A0_BCC.get(el_a, 3.20)
    a_b = ELEMENT_A0_BCC.get(el_b, 3.20)
    return 0.5 * (a_a + a_b)


def write_incar(dirpath):
    """Write INCAR for B2 structure optimization."""
    content = """SYSTEM = B2 structure optimization (refractory)

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

# Spin polarization (important for Cr, Mn)
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


def write_poscar(dirpath, el_corner, el_body, a0):
    """
    Write POSCAR for B2 (CsCl-type, Pm-3m) structure.

    Atomic positions:
      Corner atom (1×): (0, 0, 0)
      Body-center atom (1×): (0.5, 0.5, 0.5)

    el_corner: element on corners (1 atom)
    el_body:   element on body center (1 atom)
    => Formula: el_corner el_body (AB)
    """
    content = f"""{el_corner}{el_body} B2 (Pm-3m, CsCl-type)
1.0
  {a0:.6f}  0.000000  0.000000
  0.000000  {a0:.6f}  0.000000
  0.000000  0.000000  {a0:.6f}
  {el_corner}  {el_body}
  1  1
Direct
  0.000000  0.000000  0.000000  ! {el_corner} (corner)
  0.500000  0.500000  0.500000  ! {el_body} (body-center)
"""
    with open(os.path.join(dirpath, "POSCAR"), "w") as f:
        f.write(content)


def write_kpoints(dirpath, kmesh=16):
    """Write KPOINTS file (Gamma-centered, denser for small B2 cell)."""
    content = f"""Automatic mesh
0
Gamma
  {kmesh} {kmesh} {kmesh}
  0 0 0
"""
    with open(os.path.join(dirpath, "KPOINTS"), "w") as f:
        f.write(content)


def generate_potcar_script(base_dir, calculations):
    """Generate shell script to create POTCAR from $VASP_PP_PATH."""
    lines = [
        "#!/bin/bash",
        "# POTCAR generation script for B2 refractory calculations",
        "# Usage: bash make_potcar.sh",
        "# Requires: $VASP_PP_PATH environment variable pointing to VASP pseudopotential directory",
        "#   e.g., export VASP_PP_PATH=/path/to/potpaw_PBE.64",
        "",
        'if [ -z "$VASP_PP_PATH" ]; then',
        '    echo "Error: VASP_PP_PATH environment variable is not set."',
        '    echo "Set it to the PAW-PBE pseudopotential directory, e.g.:"',
        '    echo "  export VASP_PP_PATH=/path/to/potpaw_PBE.64"',
        '    exit 1',
        'fi',
        "",
        'echo "Using VASP_PP_PATH=$VASP_PP_PATH"',
        'echo "Generating POTCAR files for B2 refractory calculations..."',
        "",
    ]

    for calc_name, el_corner, el_body in calculations:
        pot_corner = POTCAR_VARIANTS.get(el_corner, el_corner)
        pot_body = POTCAR_VARIANTS.get(el_body, el_body)
        lines.append(f"# --- {calc_name} ---")
        lines.append(f'echo "  {calc_name}: {pot_corner} + {pot_body}"')
        lines.append(
            f'cat "$VASP_PP_PATH"/{pot_corner}/POTCAR "$VASP_PP_PATH"/{pot_body}/POTCAR '
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
        "# Batch execution script for B2 refractory DFT calculations",
        "# Usage: bash run_all.sh",
        "#",
        "# Adjust VASP_CMD and NPROCS to your environment.",
        "# Each B2 calc is 2 atoms → very fast (minutes per job).",
        "",
        'VASP_CMD="mpirun -np ${NPROCS:-4} vasp_std"',
        'LOG="run_status.log"',
        "",
        'echo "=== B2 Refractory Calculations ===" | tee $LOG',
        f'echo "Total: {len(calculations)} calculations" | tee -a $LOG',
        'echo "Started: $(date)" | tee -a $LOG',
        'echo "" | tee -a $LOG',
        "",
    ]

    for i, (calc_name, _, _) in enumerate(calculations, 1):
        lines.append(f'echo "[{i}/{len(calculations)}] {calc_name}..." | tee -a $LOG')
        lines.append(f'cd {calc_name}')
        lines.append('$VASP_CMD > vasp.out 2>&1')
        lines.append('if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then')
        lines.append('    echo "  CONVERGED" | tee -a ../$LOG')
        lines.append('else')
        lines.append('    echo "  WARNING: not converged" | tee -a ../$LOG')
        lines.append('fi')
        lines.append('cd ..')
        lines.append("")

    lines.append('echo "" | tee -a $LOG')
    lines.append('echo "Finished: $(date)" | tee -a $LOG')
    lines.append('echo "Check run_status.log for results."')
    lines.append("")

    with open(os.path.join(base_dir, "run_all.sh"), "w") as f:
        f.write("\n".join(lines))
    os.chmod(os.path.join(base_dir, "run_all.sh"), 0o755)


def generate_extract_script(base_dir):
    """Generate Python script to extract results from completed calculations."""
    content = '''#!/usr/bin/env python3
"""Extract lattice constants from completed B2 refractory VASP calculations."""

import os
import csv
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def extract_lattice_from_contcar(contcar_path):
    """Extract lattice constant from CONTCAR."""
    with open(contcar_path, "r") as f:
        lines = f.readlines()
    scale = float(lines[1].strip())
    a_vec = [float(x) for x in lines[2].split()]
    a = (a_vec[0]**2 + a_vec[1]**2 + a_vec[2]**2)**0.5 * scale
    return a

def extract_energy_from_oszicar(oszicar_path):
    """Extract final energy from OSZICAR."""
    with open(oszicar_path, "r") as f:
        lines = f.readlines()
    for line in reversed(lines):
        if "F=" in line:
            parts = line.split()
            idx = parts.index("F=") if "F=" in parts else -1
            if idx >= 0:
                return float(parts[idx + 1])
    return None

results = []
dirs = sorted([d for d in os.listdir(BASE_DIR)
               if os.path.isdir(os.path.join(BASE_DIR, d)) and d[0].isupper()])

for d in dirs:
    contcar = os.path.join(BASE_DIR, d, "CONTCAR")
    oszicar = os.path.join(BASE_DIR, d, "OSZICAR")

    if not os.path.exists(contcar):
        print(f"  {d}: CONTCAR not found (not yet run?)")
        continue

    # Parse element names from directory name
    # Format: AB (e.g., CrHf, HfCr)
    poscar = os.path.join(BASE_DIR, d, "POSCAR")
    with open(poscar, "r") as f:
        poscar_lines = f.readlines()
    elements = poscar_lines[5].split()
    el_corner = elements[0]
    el_body = elements[1]

    a = extract_lattice_from_contcar(contcar)
    e = extract_energy_from_oszicar(oszicar) if os.path.exists(oszicar) else None

    results.append({
        "directory": d,
        "element_A": el_corner,
        "element_B": el_body,
        "lattice_constant": a,
        "energy_per_atom": e / 2 if e else None,
    })
    print(f"  {d}: a = {a:.6f} Å, E/atom = {e/2:.6f} eV" if e else f"  {d}: a = {a:.6f} Å")

# Save to CSV
csv_path = os.path.join(BASE_DIR, "b2_refractory_results.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["directory", "element_A", "element_B",
                                           "lattice_constant", "energy_per_atom"])
    writer.writeheader()
    writer.writerows(results)

print(f"\\nSaved {len(results)} results to {csv_path}")
'''
    with open(os.path.join(base_dir, "extract_results.py"), "w") as f:
        f.write(content)
    os.chmod(os.path.join(base_dir, "extract_results.py"), 0o755)


def main():
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "B2_refractory")
    os.makedirs(base_dir, exist_ok=True)

    # Generate all refractory B2 pairs (both AB and BA)
    calculations = []
    for el_a, el_b in itertools.combinations(REFRACTORY_ELEMENTS, 2):
        # AB: el_a at corner, el_b at body
        calc_name_ab = f"{el_a}{el_b}"
        a0 = estimate_b2_a0(el_a, el_b)
        dirpath_ab = os.path.join(base_dir, calc_name_ab)
        os.makedirs(dirpath_ab, exist_ok=True)
        write_incar(dirpath_ab)
        write_poscar(dirpath_ab, el_a, el_b, a0)
        write_kpoints(dirpath_ab)
        calculations.append((calc_name_ab, el_a, el_b))

        # BA: el_b at corner, el_a at body
        calc_name_ba = f"{el_b}{el_a}"
        dirpath_ba = os.path.join(base_dir, calc_name_ba)
        os.makedirs(dirpath_ba, exist_ok=True)
        write_incar(dirpath_ba)
        write_poscar(dirpath_ba, el_b, el_a, a0)
        write_kpoints(dirpath_ba)
        calculations.append((calc_name_ba, el_b, el_a))

    # Also generate same-element B2 (pure element reference)
    for el in REFRACTORY_ELEMENTS:
        calc_name = f"{el}{el}"
        a0 = ELEMENT_A0_BCC[el]
        dirpath = os.path.join(base_dir, calc_name)
        os.makedirs(dirpath, exist_ok=True)
        write_incar(dirpath)
        write_poscar(dirpath, el, el, a0)
        write_kpoints(dirpath)
        calculations.append((calc_name, el, el))

    # Generate helper scripts
    generate_potcar_script(base_dir, calculations)
    generate_run_script(base_dir, calculations)
    generate_extract_script(base_dir)

    print(f"Generated {len(calculations)} B2 calculations in {base_dir}/")
    print("  - 36 pairs × 2 (AB + BA) = 72 hetero calculations")
    print("  - 9 same-element references")
    print("  - Total: 81 calculations")
    print()
    print("Next steps:")
    print("  1. cd B2_refractory")
    print("  2. bash make_potcar.sh   # generate POTCAR files")
    print("  3. bash run_all.sh       # run all VASP calculations")
    print("  4. python extract_results.py  # extract lattice constants")


if __name__ == "__main__":
    main()
