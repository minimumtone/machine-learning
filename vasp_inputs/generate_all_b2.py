#!/usr/bin/env python3
"""
VASP input file generator for ALL 38-element B2 pair calculations.

Generates INCAR, POSCAR, KPOINTS for all C(38,2)=703 element pairs
plus 38 same-element references (total 703×2 + 38 = 1444 calculations).

For each pair A-B, both AB and BA B2 structures are generated:
  - AB: A at corner (0,0,0), B at body-center (0.5,0.5,0.5)
  - BA: B at corner, A at body-center

Directory structure:
    BCC_B2/
    ├── AgAl/   (INCAR, POSCAR, KPOINTS)
    ├── AlAg/
    ├── ...
    ├── make_potcar.sh
    ├── run_all.sh
    └── extract_results.py

Usage:
    python generate_all_b2.py

Environment variables:
    VASP_PP_PATH : path to PAW-PBE pseudopotential directory
    VASPBIN      : VASP executable command
"""

import os
import itertools

# =====================================================================
# All 38 elements (from delta_parameters.csv; Gd, Ce excluded for 4f instability)
# Sorted alphabetically
# =====================================================================
ALL_ELEMENTS = sorted([
    'Ag', 'Al', 'Au', 'Be', 'Ca', 'Co', 'Cr', 'Cu', 'Dy', 'Er',
    'Fe', 'Ge', 'Hf', 'Ir', 'La', 'Mg', 'Mn', 'Mo', 'Nb', 'Ni',
    'Os', 'Pb', 'Pd', 'Pt', 'Re', 'Rh', 'Ru', 'Sc', 'Si', 'Sn',
    'Ta', 'Tb', 'Ti', 'V',  'W',  'Y',  'Zn', 'Zr',
])
assert len(ALL_ELEMENTS) == 38, f"Expected 38 elements, got {len(ALL_ELEMENTS)}"

# =====================================================================
# Approximate BCC lattice constants (Å) for initial B2 POSCAR
# Source: experimental BCC values or DFT-PBE estimates for non-BCC elements
# For elements without BCC data, estimated from atomic volume:
#   a_BCC ≈ (2 * V_pure)^(1/3)
# =====================================================================
ELEMENT_A0_BCC = {
    # 3d transition metals
    "Sc": 3.630,  # DFT BCC (HCP stable)
    "Ti": 3.250,  # DFT BCC (HCP stable)
    "V":  3.024,  # exp BCC
    "Cr": 2.880,  # exp BCC
    "Mn": 2.900,  # DFT BCC (complex stable)
    "Fe": 2.870,  # exp BCC
    "Co": 2.820,  # DFT BCC (HCP stable)
    "Ni": 2.800,  # DFT BCC (FCC stable)
    "Cu": 2.870,  # DFT BCC (FCC stable)
    "Zn": 3.130,  # DFT BCC (HCP stable)
    # 4d transition metals
    "Y":  4.070,  # DFT BCC (HCP stable)
    "Zr": 3.570,  # DFT BCC (HCP stable)
    "Nb": 3.301,  # exp BCC
    "Mo": 3.147,  # exp BCC
    "Ru": 3.010,  # DFT BCC (HCP stable)
    "Rh": 3.020,  # DFT BCC (FCC stable)
    "Pd": 3.080,  # DFT BCC (FCC stable)
    "Ag": 3.260,  # DFT BCC (FCC stable)
    # 5d transition metals
    "La": 4.220,  # DFT BCC (DHCP stable)
    "Hf": 3.530,  # DFT BCC (HCP stable)
    "Ta": 3.303,  # exp BCC
    "W":  3.165,  # exp BCC
    "Re": 3.100,  # DFT BCC (HCP stable)
    "Os": 3.020,  # DFT BCC (HCP stable)
    "Ir": 3.060,  # DFT BCC (FCC stable)
    "Pt": 3.120,  # DFT BCC (FCC stable)
    "Au": 3.240,  # DFT BCC (FCC stable)
    # Rare earths (4f — only Dy, Er, Tb, Y included; Gd, Ce excluded)
    "Dy": 3.990,  # DFT BCC
    "Er": 3.960,  # DFT BCC
    "Tb": 4.020,  # DFT BCC
    # Alkaline earth / main group
    "Ca": 4.460,  # DFT BCC (FCC stable)
    "Mg": 3.600,  # DFT BCC (HCP stable)
    "Be": 2.530,  # DFT BCC (HCP stable)
    # p-block metals / metalloids
    "Al": 3.230,  # DFT BCC (FCC stable)
    "Si": 3.430,  # DFT BCC (diamond stable)
    "Ge": 3.570,  # DFT BCC (diamond stable)
    "Sn": 3.810,  # DFT BCC (tetragonal stable)
    "Pb": 3.940,  # DFT BCC (FCC stable)
}

# =====================================================================
# POTCAR recommended pseudopotentials (VASP 5.4+ PAW-PBE)
# Elements with semi-core states use _pv or _sv variants
# =====================================================================
POTCAR_VARIANTS = {
    # 3d
    "Sc": "Sc_sv", "Ti": "Ti_pv", "V": "V_sv", "Cr": "Cr_pv",
    "Mn": "Mn_pv", "Fe": "Fe_pv", "Co": "Co",   "Ni": "Ni_pv",
    "Cu": "Cu_pv", "Zn": "Zn",
    # 4d
    "Y": "Y_sv",   "Zr": "Zr_sv", "Nb": "Nb_pv", "Mo": "Mo_pv",
    "Ru": "Ru_pv", "Rh": "Rh_pv", "Pd": "Pd",    "Ag": "Ag",
    # 5d
    "La": "La",    "Hf": "Hf_pv", "Ta": "Ta_pv", "W": "W_pv",
    "Re": "Re_pv", "Os": "Os_pv", "Ir": "Ir",    "Pt": "Pt",
    "Au": "Au",
    # Rare earths
    "Dy": "Dy_3",  "Er": "Er_3",  "Tb": "Tb_3",
    # Alkaline earth / main group
    "Ca": "Ca_sv", "Mg": "Mg_pv", "Be": "Be",
    # p-block
    "Al": "Al",    "Si": "Si",    "Ge": "Ge_d",
    "Sn": "Sn_d",  "Pb": "Pb_d",
}


def estimate_b2_a0(el_a, el_b):
    """Estimate initial B2 lattice parameter from Vegard's law."""
    a_a = ELEMENT_A0_BCC.get(el_a, 3.20)
    a_b = ELEMENT_A0_BCC.get(el_b, 3.20)
    return 0.5 * (a_a + a_b)


def write_incar(dirpath):
    """Write INCAR for B2 structure optimization."""
    content = """\
SYSTEM = B2 structure optimization

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


def write_poscar(dirpath, el_corner, el_body, a0):
    """
    Write POSCAR for B2 (CsCl-type, Pm-3m) structure.

    Atomic positions:
      Corner atom (1×): (0, 0, 0)
      Body-center atom (1×): (0.5, 0.5, 0.5)
    """
    content = f"""\
{el_corner}{el_body} B2 (Pm-3m, CsCl-type)
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
    """Write KPOINTS file (Gamma-centered, dense for small B2 cell)."""
    content = f"""\
Automatic mesh
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
        "# POTCAR generation script for all B2 calculations",
        "# Usage: bash make_potcar.sh",
        "# Requires: $VASP_PP_PATH pointing to PAW-PBE pseudopotential directory",
        "#   e.g., export VASP_PP_PATH=/path/to/potpaw_PBE.64",
        "",
        'if [ -z "$VASP_PP_PATH" ]; then',
        '    echo "Error: VASP_PP_PATH is not set."',
        '    echo "  export VASP_PP_PATH=/path/to/potpaw_PBE.64"',
        '    exit 1',
        'fi',
        "",
        'echo "Using VASP_PP_PATH=$VASP_PP_PATH"',
        f'echo "Generating POTCAR for {len(calculations)} calculations..."',
        "",
        "FAIL=0",
        "",
    ]

    for calc_name, el_corner, el_body in calculations:
        pot_corner = POTCAR_VARIANTS.get(el_corner, el_corner)
        pot_body = POTCAR_VARIANTS.get(el_body, el_body)
        lines.append(
            f'cat "$VASP_PP_PATH"/{pot_corner}/POTCAR "$VASP_PP_PATH"/{pot_body}/POTCAR '
            f'> {calc_name}/POTCAR 2>/dev/null'
        )
        lines.append(
            f'if [ $? -ne 0 ]; then echo "  FAIL: {calc_name} ({pot_corner}+{pot_body})"; FAIL=$((FAIL+1)); fi'
        )

    lines.append("")
    lines.append(f'echo "Done. Failed: $FAIL / {len(calculations)}"')
    lines.append("")

    script_path = os.path.join(base_dir, "make_potcar.sh")
    with open(script_path, "w") as f:
        f.write("\n".join(lines))
    os.chmod(script_path, 0o755)


def generate_run_script(base_dir, calculations):
    """Generate batch execution script using $VASPBIN."""
    lines = [
        "#!/bin/bash",
        "# Batch execution script for all B2 DFT calculations",
        "# Usage: bash run_all.sh",
        "#",
        "# Requires: $VASPBIN environment variable",
        "# Each B2 calc is 2 atoms → typically minutes per job.",
        "",
        'if [ -z "$VASPBIN" ]; then',
        '    echo "Error: VASPBIN is not set."',
        '    echo "  export VASPBIN=/path/to/vasp_std"',
        '    exit 1',
        'fi',
        "",
        'BASE=$(cd "$(dirname "$0")" && pwd)',
        'LOG="$BASE/run_status.log"',
        "",
        'echo "=== B2 All-Element Calculations ===" | tee "$LOG"',
        f'echo "Total: {len(calculations)} calculations" | tee -a "$LOG"',
        'echo "VASPBIN=$VASPBIN" | tee -a "$LOG"',
        'echo "Started: $(date)" | tee -a "$LOG"',
        'echo "" | tee -a "$LOG"',
        "",
    ]

    for i, (calc_name, _, _) in enumerate(calculations, 1):
        lines.append(f'echo "[{i}/{len(calculations)}] {calc_name}..." | tee -a "$LOG"')
        lines.append(f'cd "$BASE/{calc_name}"')
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
    lines.append("")

    script_path = os.path.join(base_dir, "run_all.sh")
    with open(script_path, "w") as f:
        f.write("\n".join(lines))
    os.chmod(script_path, 0o755)


def generate_extract_script(base_dir):
    """Generate Python script to extract results from completed calculations."""
    content = '''\
#!/usr/bin/env python3
"""Extract lattice constants and energies from completed B2 VASP calculations."""

import os
import csv

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


def main():
    results = []
    not_run = []
    dirs = sorted([d for d in os.listdir(BASE_DIR)
                   if os.path.isdir(os.path.join(BASE_DIR, d))])

    for d in dirs:
        contcar = os.path.join(BASE_DIR, d, "CONTCAR")
        oszicar = os.path.join(BASE_DIR, d, "OSZICAR")
        poscar = os.path.join(BASE_DIR, d, "POSCAR")

        if not os.path.exists(contcar) or os.path.getsize(contcar) == 0:
            not_run.append(d)
            continue

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
            "lattice_constant_A": a,
            "energy_per_atom": e / 2 if e else None,
        })

    csv_path = os.path.join(BASE_DIR, "b2_all_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "directory", "element_A", "element_B",
            "lattice_constant_A", "energy_per_atom"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Extracted {len(results)} results -> {csv_path}")
    if not_run:
        print(f"Not yet run: {len(not_run)} directories")


if __name__ == "__main__":
    main()
'''
    script_path = os.path.join(base_dir, "extract_results.py")
    with open(script_path, "w") as f:
        f.write(content)
    os.chmod(script_path, 0o755)


def main():
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "BCC_B2")
    os.makedirs(base_dir, exist_ok=True)

    n_elements = len(ALL_ELEMENTS)
    n_hetero_pairs = n_elements * (n_elements - 1) // 2
    n_hetero_calcs = n_hetero_pairs * 2  # AB + BA
    n_homo_calcs = n_elements             # AA references
    n_total = n_hetero_calcs + n_homo_calcs

    print(f"Elements: {n_elements}")
    print(f"Hetero pairs: {n_hetero_pairs} × 2 (AB+BA) = {n_hetero_calcs}")
    print(f"Homo references: {n_homo_calcs}")
    print(f"Total calculations: {n_total}")
    print()

    calculations = []

    # Hetero pairs: AB and BA
    for el_a, el_b in itertools.combinations(ALL_ELEMENTS, 2):
        a0 = estimate_b2_a0(el_a, el_b)

        # AB: el_a at corner, el_b at body-center
        calc_ab = f"{el_a}1{el_b}1"
        dir_ab = os.path.join(base_dir, calc_ab)
        os.makedirs(dir_ab, exist_ok=True)
        write_incar(dir_ab)
        write_poscar(dir_ab, el_a, el_b, a0)
        write_kpoints(dir_ab)
        calculations.append((calc_ab, el_a, el_b))

        # BA: el_b at corner, el_a at body-center
        calc_ba = f"{el_b}1{el_a}1"
        dir_ba = os.path.join(base_dir, calc_ba)
        os.makedirs(dir_ba, exist_ok=True)
        write_incar(dir_ba)
        write_poscar(dir_ba, el_b, el_a, a0)
        write_kpoints(dir_ba)
        calculations.append((calc_ba, el_b, el_a))

    # Same-element references (AA)
    for el in ALL_ELEMENTS:
        calc_name = f"{el}1{el}1"
        a0 = ELEMENT_A0_BCC.get(el, 3.20)
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
    print(f"  - {n_hetero_pairs} pairs × 2 (AB + BA) = {n_hetero_calcs} hetero")
    print(f"  - {n_homo_calcs} same-element references")
    print(f"  - Total: {len(calculations)} directories")
    print()
    print("Next steps:")
    print("  1. cd BCC_B2")
    print("  2. bash make_potcar.sh          # generate POTCAR (needs $VASP_PP_PATH)")
    print("  3. bash run_all.sh              # run all VASP calculations (needs $VASPBIN)")
    print("  4. python extract_results.py    # extract lattice constants")


if __name__ == "__main__":
    main()
