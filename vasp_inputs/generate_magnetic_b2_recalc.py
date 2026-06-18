#!/usr/bin/env python3
"""
VASP input file generator for magnetic B2 recalculations.

Generates improved INCAR/POSCAR/KPOINTS with:
  - Explicit MAGMOM for magnetic elements (Fe, Co, Ni, Mn, Cr)
  - Enhanced convergence: NSW=300, EDIFFG=-0.005, SIGMA=0.1, NELM=300
  - Proper spin-polarized settings

Directory structure:
    BCC_B2/
    ├── pure_Fe/   (BCC pure element)
    ├── pure_Co/
    ├── ...
    ├── Al1Co1/    (B2 binary)
    ├── Co1Al1/
    ├── ...
    ├── make_potcar.sh
    ├── run_all.sh
    └── extract_results.py

Usage:
    python generate_magnetic_b2_recalc.py
"""

import os
import itertools

# =====================================================================
# Magnetic elements and their default MAGMOM (μ_B)
# =====================================================================
MAGMOM = {
    "Fe": 3.0,
    "Co": 2.0,
    "Ni": 1.0,
    "Mn": 4.0,
    "Cr": 2.0,
}

# All elements involved in magnetic B2 pairs (after 4f+Y exclusion)
# = magnetic elements + their partners in HEA-relevant pairs
MAGNETIC_ELEMENTS = sorted(MAGMOM.keys())

# HEA constituent elements (from training+test dataset)
HEA_ELEMENTS = {
    'Al', 'Co', 'Cr', 'Cu', 'Fe', 'Hf', 'Mn', 'Mo', 'Nb', 'Ni',
    'Pd', 'Sn', 'Ta', 'Ti', 'V', 'W', 'Zn', 'Zr',
}

# All partner elements (excluding 4f+Y and rare non-metals)
PARTNER_ELEMENTS = sorted({
    'Ag', 'Al', 'Au', 'Be', 'Ca', 'Co', 'Cr', 'Cu', 'Fe', 'Ge',
    'Hf', 'Ir', 'Mg', 'Mn', 'Mo', 'Nb', 'Ni', 'Os', 'Pb', 'Pd',
    'Pt', 'Re', 'Rh', 'Ru', 'Sc', 'Si', 'Sn', 'Ta', 'Ti', 'V',
    'W', 'Zn', 'Zr',
})

# =====================================================================
# BCC lattice constants (Å) for initial POSCAR
# =====================================================================
ELEMENT_A0_BCC = {
    "Sc": 3.630, "Ti": 3.250, "V": 3.024, "Cr": 2.880, "Mn": 2.900,
    "Fe": 2.870, "Co": 2.820, "Ni": 2.800, "Cu": 2.870, "Zn": 3.130,
    "Zr": 3.570, "Nb": 3.301, "Mo": 3.147, "Ru": 3.010, "Rh": 3.020,
    "Pd": 3.080, "Ag": 3.260, "Hf": 3.530, "Ta": 3.303, "W": 3.165,
    "Re": 3.100, "Os": 3.020, "Ir": 3.060, "Pt": 3.120, "Au": 3.240,
    "Ca": 4.460, "Mg": 3.600, "Be": 2.530, "Al": 3.230, "Si": 3.430,
    "Ge": 3.570, "Sn": 3.810, "Pb": 3.940,
}

# =====================================================================
# POTCAR variants (PAW-PBE)
# =====================================================================
POTCAR_VARIANTS = {
    "Sc": "Sc_sv", "Ti": "Ti_pv", "V": "V_sv", "Cr": "Cr_pv",
    "Mn": "Mn_pv", "Fe": "Fe_pv", "Co": "Co", "Ni": "Ni_pv",
    "Cu": "Cu_pv", "Zn": "Zn",
    "Zr": "Zr_sv", "Nb": "Nb_pv", "Mo": "Mo_pv",
    "Ru": "Ru_pv", "Rh": "Rh_pv", "Pd": "Pd", "Ag": "Ag",
    "Hf": "Hf_pv", "Ta": "Ta_pv", "W": "W_pv",
    "Re": "Re_pv", "Os": "Os_pv", "Ir": "Ir", "Pt": "Pt", "Au": "Au",
    "Ca": "Ca_sv", "Mg": "Mg_pv", "Be": "Be",
    "Al": "Al", "Si": "Si", "Ge": "Ge_d", "Sn": "Sn_d", "Pb": "Pb_d",
}


def write_incar_b2(dirpath, el_corner, el_body):
    """Write INCAR for B2 with explicit MAGMOM."""
    # Build MAGMOM string: corner(1) body(1)
    mag_corner = MAGMOM.get(el_corner, 0.0)
    mag_body = MAGMOM.get(el_body, 0.0)

    content = f"""\
SYSTEM = B2 {el_corner}{el_body} magnetic recalculation

# Electronic relaxation
ENCUT  = 520
PREC   = Accurate
EDIFF  = 1E-6
NELM   = 300
LREAL  = .FALSE.

# Ionic relaxation
IBRION = 2
ISIF   = 3
NSW    = 300
EDIFFG = -0.005

# Smearing (metals - reduced for magnetic accuracy)
ISMEAR = 1
SIGMA  = 0.1

# Exchange-correlation
GGA    = PE

# Spin polarization
ISPIN  = 2
MAGMOM = {mag_corner:.1f} {mag_body:.1f}

# Output
LORBIT = 11
LWAVE  = .FALSE.
LCHARG = .FALSE.

# Performance
NCORE  = 4
"""
    with open(os.path.join(dirpath, "INCAR"), "w") as f:
        f.write(content)


def write_incar_pure(dirpath, el):
    """Write INCAR for pure BCC element."""
    mag = MAGMOM.get(el, 0.0)

    content = f"""\
SYSTEM = BCC pure {el} magnetic recalculation

# Electronic relaxation
ENCUT  = 520
PREC   = Accurate
EDIFF  = 1E-6
NELM   = 300
LREAL  = .FALSE.

# Ionic relaxation
IBRION = 2
ISIF   = 3
NSW    = 300
EDIFFG = -0.005

# Smearing (metals - reduced for magnetic accuracy)
ISMEAR = 1
SIGMA  = 0.1

# Exchange-correlation
GGA    = PE

# Spin polarization
ISPIN  = 2
MAGMOM = 2*{mag:.1f}

# Output
LORBIT = 11
LWAVE  = .FALSE.
LCHARG = .FALSE.

# Performance
NCORE  = 4
"""
    with open(os.path.join(dirpath, "INCAR"), "w") as f:
        f.write(content)


def write_poscar_b2(dirpath, el_corner, el_body, a0):
    """Write POSCAR for B2 (CsCl-type, Pm-3m)."""
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


def write_poscar_pure(dirpath, el, a0):
    """Write POSCAR for pure BCC (Im-3m, 2 atoms in conventional cell)."""
    content = f"""\
{el} BCC (Im-3m)
1.0
  {a0:.6f}  0.000000  0.000000
  0.000000  {a0:.6f}  0.000000
  0.000000  0.000000  {a0:.6f}
  {el}
  2
Direct
  0.000000  0.000000  0.000000
  0.500000  0.500000  0.500000
"""
    with open(os.path.join(dirpath, "POSCAR"), "w") as f:
        f.write(content)


def write_kpoints(dirpath, kmesh=16):
    """Write KPOINTS (Gamma-centered)."""
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
    """Generate make_potcar.sh."""
    lines = [
        "#!/bin/bash",
        "# POTCAR generation for magnetic B2 recalculations",
        "# Usage: bash make_potcar.sh",
        "# Requires: $VASP_PP_PATH pointing to potpaw_PBE directory",
        "",
        'if [ -z "$VASP_PP_PATH" ]; then',
        '    echo "Error: VASP_PP_PATH not set"',
        '    echo "  export VASP_PP_PATH=/path/to/potpaw_PBE.64"',
        '    exit 1',
        'fi',
        "",
        f'echo "Generating POTCAR for {len(calculations)} calculations..."',
        "FAIL=0",
        "",
    ]
    for calc_name, elements in calculations:
        pots = [POTCAR_VARIANTS.get(el, el) for el in elements]
        cat_args = " ".join(
            [f'"$VASP_PP_PATH"/{p}/POTCAR' for p in pots]
        )
        lines.append(f"cat {cat_args} > {calc_name}/POTCAR 2>/dev/null")
        lines.append(
            f'if [ $? -ne 0 ]; then echo "  FAIL: {calc_name}"; FAIL=$((FAIL+1)); fi'
        )
    lines.append("")
    lines.append(f'echo "Done. Failed: $FAIL / {len(calculations)}"')

    path = os.path.join(base_dir, "make_potcar.sh")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    os.chmod(path, 0o755)


def generate_run_script(base_dir, calculations):
    """Generate run_all.sh."""
    lines = [
        "#!/bin/bash",
        "# Batch execution for magnetic B2 recalculations",
        "# Usage: bash run_all.sh",
        "",
        'if [ -z "$VASPBIN" ]; then',
        '    echo "Error: VASPBIN not set"',
        '    echo "  export VASPBIN=/path/to/vasp_std"',
        '    exit 1',
        'fi',
        "",
        'BASE=$(cd "$(dirname "$0")" && pwd)',
        'LOG="$BASE/run_status.log"',
        "",
        f'echo "=== Magnetic B2 Recalculations ===" | tee "$LOG"',
        f'echo "Total: {len(calculations)} calculations" | tee -a "$LOG"',
        'echo "Started: $(date)" | tee -a "$LOG"',
        "",
    ]
    for i, (calc_name, _) in enumerate(calculations, 1):
        lines.append(f'echo "[{i}/{len(calculations)}] {calc_name}..." | tee -a "$LOG"')
        lines.append(f'cd "$BASE/{calc_name}"')
        lines.append('if [ ! -f POTCAR ]; then')
        lines.append('    echo "  SKIP (no POTCAR)" | tee -a "$LOG"')
        lines.append('else')
        lines.append('    $VASPBIN > vasp.out 2>&1')
        lines.append('    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then')
        lines.append('        echo "  CONVERGED" | tee -a "$LOG"')
        lines.append('    else')
        lines.append('        echo "  WARNING: not converged" | tee -a "$LOG"')
        lines.append('    fi')
        lines.append('fi')
        lines.append('cd "$BASE"')
        lines.append("")
    lines.append('echo "Finished: $(date)" | tee -a "$LOG"')

    path = os.path.join(base_dir, "run_all.sh")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    os.chmod(path, 0o755)


def generate_extract_script(base_dir):
    """Generate extract_results.py."""
    content = '''\
#!/usr/bin/env python3
"""Extract lattice constants and energies from magnetic B2 recalculations."""

import os
import csv
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def extract_lattice_from_contcar(path):
    with open(path) as f:
        lines = f.readlines()
    scale = float(lines[1].strip())
    a_vec = [float(x) for x in lines[2].split()]
    return (a_vec[0]**2 + a_vec[1]**2 + a_vec[2]**2)**0.5 * scale


def extract_energy_from_oszicar(path):
    with open(path) as f:
        lines = f.readlines()
    for line in reversed(lines):
        if "F=" in line:
            parts = line.split()
            idx = parts.index("F=") if "F=" in parts else -1
            if idx >= 0:
                return float(parts[idx + 1])
    return None


def extract_magmom_from_outcar(path):
    """Extract final magnetic moment per atom from OUTCAR."""
    mag = None
    with open(path) as f:
        for line in f:
            if "number of electron" in line and "magnetization" in line:
                parts = line.split()
                mag = float(parts[-1])
    return mag


def main():
    results = []
    not_run = []
    dirs = sorted([d for d in os.listdir(BASE_DIR)
                   if os.path.isdir(os.path.join(BASE_DIR, d))])

    for d in dirs:
        contcar = os.path.join(BASE_DIR, d, "CONTCAR")
        oszicar = os.path.join(BASE_DIR, d, "OSZICAR")
        outcar = os.path.join(BASE_DIR, d, "OUTCAR")
        poscar = os.path.join(BASE_DIR, d, "POSCAR")

        if not os.path.exists(contcar) or os.path.getsize(contcar) == 0:
            not_run.append(d)
            continue

        # Parse elements from POSCAR
        with open(poscar) as f:
            poscar_lines = f.readlines()
        elements = poscar_lines[5].split()

        is_pure = d.startswith("pure_")
        natoms = 2  # both pure BCC (2 atoms) and B2 (2 atoms)

        a = extract_lattice_from_contcar(contcar)
        e = extract_energy_from_oszicar(oszicar) if os.path.exists(oszicar) else None
        mag = extract_magmom_from_outcar(outcar) if os.path.exists(outcar) else None

        results.append({
            "directory": d,
            "type": "pure" if is_pure else "B2",
            "element_A": elements[0],
            "element_B": elements[1] if len(elements) > 1 else elements[0],
            "lattice_constant_A": a,
            "energy_per_atom": e / natoms if e else None,
            "magnetization": mag,
        })

    csv_path = os.path.join(BASE_DIR, "magnetic_b2_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "directory", "type", "element_A", "element_B",
            "lattice_constant_A", "energy_per_atom", "magnetization"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Extracted {len(results)} results -> {csv_path}")
    if not_run:
        print(f"Not yet run: {len(not_run)} directories")


if __name__ == "__main__":
    main()
'''
    path = os.path.join(base_dir, "extract_results.py")
    with open(path, "w") as f:
        f.write(content)
    os.chmod(path, 0o755)


def main():
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "BCC_B2")
    os.makedirs(base_dir, exist_ok=True)

    calculations = []  # (dir_name, [elements])

    # =====================================================================
    # 1. Pure magnetic elements (BCC)
    # =====================================================================
    print("=== Pure magnetic elements ===")
    for el in MAGNETIC_ELEMENTS:
        dir_name = f"pure_{el}"
        dirpath = os.path.join(base_dir, dir_name)
        os.makedirs(dirpath, exist_ok=True)
        a0 = ELEMENT_A0_BCC.get(el, 3.0)
        write_incar_pure(dirpath, el)
        write_poscar_pure(dirpath, el, a0)
        write_kpoints(dirpath)
        calculations.append((dir_name, [el]))
        print(f"  {dir_name}: a0={a0:.3f}, MAGMOM={MAGMOM[el]:.1f}")

    n_pure = len(calculations)

    # =====================================================================
    # 2. B2 binary pairs (at least one magnetic element)
    # =====================================================================
    # Build pair list
    mag_set = set(MAGNETIC_ELEMENTS)
    pairs_a = []  # Priority A: both HEA
    pairs_b = []  # Priority B: one HEA

    for el_a, el_b in itertools.combinations(PARTNER_ELEMENTS, 2):
        if el_a not in mag_set and el_b not in mag_set:
            continue
        if el_a in HEA_ELEMENTS and el_b in HEA_ELEMENTS:
            pairs_a.append((el_a, el_b))
        elif el_a in HEA_ELEMENTS or el_b in HEA_ELEMENTS:
            pairs_b.append((el_a, el_b))

    def gen_b2_pair(el_a, el_b, label):
        """Generate AB and BA B2 directories."""
        a0 = 0.5 * (ELEMENT_A0_BCC.get(el_a, 3.2) + ELEMENT_A0_BCC.get(el_b, 3.2))
        dirs_created = []
        for corner, body in [(el_a, el_b), (el_b, el_a)]:
            dir_name = f"{corner}1{body}1"
            dirpath = os.path.join(base_dir, dir_name)
            os.makedirs(dirpath, exist_ok=True)
            write_incar_b2(dirpath, corner, body)
            write_poscar_b2(dirpath, corner, body, a0)
            write_kpoints(dirpath)
            calculations.append((dir_name, [corner, body]))
            dirs_created.append(dir_name)
        return dirs_created

    print(f"\n=== Priority A: Both HEA elements ({len(pairs_a)} pairs) ===")
    n_before_a = len(calculations)
    for el_a, el_b in pairs_a:
        gen_b2_pair(el_a, el_b, "A")
    n_a = len(calculations) - n_before_a

    print(f"\n=== Priority B: One HEA element ({len(pairs_b)} pairs) ===")
    n_before_b = len(calculations)
    for el_a, el_b in pairs_b:
        gen_b2_pair(el_a, el_b, "B")
    n_b = len(calculations) - n_before_b

    # =====================================================================
    # 3. Generate helper scripts
    # =====================================================================
    generate_potcar_script(base_dir, calculations)
    generate_run_script(base_dir, calculations)
    generate_extract_script(base_dir)

    # =====================================================================
    # Summary
    # =====================================================================
    print(f"\n{'='*60}")
    print(f"Generated {len(calculations)} calculations in {base_dir}/")
    print(f"  Pure magnetic elements: {n_pure}")
    print(f"  Priority A (both HEA): {len(pairs_a)} pairs × 2 = {n_a}")
    print(f"  Priority B (one HEA):  {len(pairs_b)} pairs × 2 = {n_b}")
    print(f"  Total: {len(calculations)} directories")
    print(f"{'='*60}")
    print()
    print("INCAR improvements vs original:")
    print("  NSW:    100 -> 300")
    print("  EDIFFG: -0.01 -> -0.005")
    print("  SIGMA:  0.2 -> 0.1")
    print("  NELM:   200 -> 300")
    print("  MAGMOM: (none) -> explicit per-atom values")
    print()
    print("Next steps:")
    print("  1. cd BCC_B2")
    print("  2. bash make_potcar.sh")
    print("  3. bash run_all.sh")
    print("  4. python extract_results.py")


if __name__ == "__main__":
    main()
