#!/usr/bin/env python3
"""
VASP input file generator for missing Table 6 (tab:delta) entries.

Table 6 has "---" entries for the following elements:
  - L1₂ missing: Be, Mo, Os, Re, W  (need L1₂ A₃B / B₃A calculations)
  - B2 missing:  Ge, Pb              (need B2 AB calculations)

For each missing element, we generate calculations with 3-5 partner elements
that already have reliable δ values. This ensures the least-squares
decomposition Ω_sf(A-B) ≈ δ_A + δ_B is well-conditioned.

Usage:
    python generate_table6_missing.py

Output structure:
    table6_missing/
    ├── L12/
    │   ├── Be3Ni/   (INCAR, POSCAR, KPOINTS)
    │   ├── Ni3Be/
    │   ├── Be3Cu/
    │   ├── ...
    │   ├── Mo3Ni/
    │   ├── ...
    │   ├── W3Ni/
    │   └── ...
    ├── B2/
    │   ├── GeNi/
    │   ├── GeCu/
    │   ├── ...
    │   ├── PbNi/
    │   └── ...
    ├── make_potcar.sh
    └── run_all.sh
"""

import os
import itertools

# =====================================================================
# Missing elements and their required structure types
# =====================================================================
# L1₂ missing: these elements have B2 δ but no L1₂ δ
L12_MISSING = ["Be", "Mo", "Os", "Re", "W"]

# B2 missing: these elements have L1₂ δ but no B2 δ
B2_MISSING = ["Ge", "Pb"]

# =====================================================================
# Partner elements for pairing (well-characterized, reliable δ values)
# Selected to span diverse chemistry and ensure good conditioning
# =====================================================================
# For L1₂: use elements with established L1₂ δ values
L12_PARTNERS = {
    "Be": ["Ni", "Cu", "Pd", "Pt", "Au"],      # FCC metals with known L1₂
    "Mo": ["Ni", "Pd", "Pt", "Al", "Cu"],       # common L1₂ formers
    "Os": ["Ni", "Pd", "Pt", "Ir", "Rh"],       # PGM group
    "Re": ["Ni", "Pd", "Pt", "Al", "Ir"],       # transition metals
    "W":  ["Ni", "Pd", "Pt", "Al", "Cu"],       # common L1₂ formers
}

# For B2: use elements with established B2 δ values
B2_PARTNERS = {
    "Ge": ["Ni", "Cu", "Fe", "Co", "Pd"],       # common B2 formers
    "Pb": ["Ni", "Cu", "Au", "Pd", "Ag"],       # noble/late TM
}

# =====================================================================
# Lattice constants (Å) for initial POSCAR estimation
# FCC-equivalent a₀ for L1₂ Vegard estimation
# BCC a₀ for B2 Vegard estimation
# =====================================================================
ELEMENT_A0_FCC = {
    "Al": 4.050, "Ag": 4.085, "Au": 4.078, "Be": 3.190, "Co": 3.545,
    "Cr": 3.640, "Cu": 3.615, "Fe": 3.570, "Ge": 4.050, "Ir": 3.839,
    "Mn": 3.630, "Mo": 3.960, "Nb": 4.160, "Ni": 3.524, "Os": 3.850,
    "Pb": 4.950, "Pd": 3.890, "Pt": 3.924, "Re": 3.890, "Rh": 3.803,
    "Ru": 3.827, "Ta": 4.150, "Ti": 4.100, "V": 3.820, "W": 3.980,
    "Zr": 4.540,
}

ELEMENT_A0_BCC = {
    "Ag": 3.340, "Al": 3.240, "Au": 3.340, "Be": 2.540, "Co": 2.820,
    "Cr": 2.880, "Cu": 2.870, "Fe": 2.870, "Ge": 3.220, "Mn": 2.870,
    "Mo": 3.147, "Nb": 3.301, "Ni": 2.810, "Os": 3.050, "Pb": 3.950,
    "Pd": 3.090, "Pt": 3.120, "Re": 3.070, "Rh": 3.020, "Ru": 3.040,
    "Ta": 3.303, "Ti": 3.250, "V": 3.024, "W": 3.165, "Zr": 3.570,
}

# POTCAR pseudopotential recommendations (VASP 5.4+ PAW-PBE)
POTCAR_VARIANTS = {
    "Ag": "Ag",
    "Al": "Al",
    "Au": "Au",
    "Be": "Be",
    "Co": "Co",
    "Cr": "Cr_pv",
    "Cu": "Cu_pv",
    "Fe": "Fe_pv",
    "Ge": "Ge_d",
    "Ir": "Ir",
    "Mn": "Mn_pv",
    "Mo": "Mo_pv",
    "Nb": "Nb_pv",
    "Ni": "Ni_pv",
    "Os": "Os_pv",
    "Pb": "Pb_d",
    "Pd": "Pd",
    "Pt": "Pt",
    "Re": "Re_pv",
    "Rh": "Rh_pv",
    "Ru": "Ru_pv",
    "Ta": "Ta_pv",
    "Ti": "Ti_pv",
    "V":  "V_sv",
    "W":  "W_pv",
    "Zr": "Zr_sv",
}


# =====================================================================
# INCAR templates
# =====================================================================
INCAR_L12 = """SYSTEM = L12 structure optimization
# Electronic relaxation
ENCUT  = 320
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

# Spin polarization
ISPIN  = 2

# Output
LORBIT = 11
LWAVE  = .FALSE.
LCHARG = .FALSE.

# Performance
NCORE  = 4
"""

INCAR_B2 = """SYSTEM = B2 structure optimization
# Electronic relaxation
ENCUT  = 320
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

# Spin polarization
ISPIN  = 2

# Output
LORBIT = 11
LWAVE  = .FALSE.
LCHARG = .FALSE.

# Performance
NCORE  = 4
"""


def write_poscar_l12(dirpath, el_face, el_corner, a0):
    """
    Write POSCAR for L1₂ (Cu3Au-type, Pm-3m) structure.
    el_face: element on face centers (3 atoms, majority)
    el_corner: element on corner (1 atom, minority)
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


def write_poscar_b2(dirpath, el_corner, el_body, a0):
    """
    Write POSCAR for B2 (CsCl-type, Pm-3m) structure.
    el_corner: element at corner (0,0,0)
    el_body: element at body center (0.5,0.5,0.5)
    => Formula: el_corner el_body
    """
    content = f"""{el_corner}{el_body} B2 (CsCl-type, Pm-3m)
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


def write_kpoints(dirpath, kmesh=6):
    """Write KPOINTS file (Gamma-centered)."""
    content = f"""Automatic mesh
0
Gamma
  {kmesh} {kmesh} {kmesh}
  0 0 0
"""
    with open(os.path.join(dirpath, "KPOINTS"), "w") as f:
        f.write(content)


def estimate_l12_a0(el_face, el_corner):
    """Estimate initial L1₂ lattice parameter from Vegard's law."""
    a_face = ELEMENT_A0_FCC.get(el_face, 3.80)
    a_corner = ELEMENT_A0_FCC.get(el_corner, 3.80)
    return 0.75 * a_face + 0.25 * a_corner


def estimate_b2_a0(el_a, el_b):
    """Estimate initial B2 lattice parameter from Vegard's law."""
    a_a = ELEMENT_A0_BCC.get(el_a, 3.10)
    a_b = ELEMENT_A0_BCC.get(el_b, 3.10)
    return 0.5 * a_a + 0.5 * a_b


def main():
    base_dir = "table6_missing"
    os.makedirs(base_dir, exist_ok=True)

    all_calcs = []  # (subdir, el1, el2) for POTCAR script

    # =================================================================
    # Part 1: L1₂ calculations for Be, Mo, Os, Re, W
    # =================================================================
    l12_dir = os.path.join(base_dir, "L12")
    os.makedirs(l12_dir, exist_ok=True)

    print("=" * 60)
    print("Generating L1₂ VASP inputs for Table 6 missing elements")
    print("=" * 60)

    l12_count = 0
    for missing_el in L12_MISSING:
        partners = L12_PARTNERS[missing_el]
        print(f"\n  {missing_el} (L1₂ δ missing) — partners: {partners}")

        for partner in partners:
            # A₃B: missing element as majority (face centers)
            dirname_a3b = f"{missing_el}3{partner}"
            dirpath_a3b = os.path.join(l12_dir, dirname_a3b)
            os.makedirs(dirpath_a3b, exist_ok=True)
            a0_a3b = estimate_l12_a0(missing_el, partner)
            with open(os.path.join(dirpath_a3b, "INCAR"), "w") as f:
                f.write(INCAR_L12)
            write_poscar_l12(dirpath_a3b, missing_el, partner, a0_a3b)
            write_kpoints(dirpath_a3b)
            all_calcs.append((f"L12/{dirname_a3b}", missing_el, partner))
            l12_count += 1

            # B₃A: missing element as minority (corner)
            dirname_b3a = f"{partner}3{missing_el}"
            dirpath_b3a = os.path.join(l12_dir, dirname_b3a)
            os.makedirs(dirpath_b3a, exist_ok=True)
            a0_b3a = estimate_l12_a0(partner, missing_el)
            with open(os.path.join(dirpath_b3a, "INCAR"), "w") as f:
                f.write(INCAR_L12)
            write_poscar_l12(dirpath_b3a, partner, missing_el, a0_b3a)
            write_kpoints(dirpath_b3a)
            all_calcs.append((f"L12/{dirname_b3a}", partner, missing_el))
            l12_count += 1

    # =================================================================
    # Part 2: B2 calculations for Ge, Pb
    # =================================================================
    b2_dir = os.path.join(base_dir, "B2")
    os.makedirs(b2_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("Generating B2 VASP inputs for Table 6 missing elements")
    print("=" * 60)

    b2_count = 0
    for missing_el in B2_MISSING:
        partners = B2_PARTNERS[missing_el]
        print(f"\n  {missing_el} (B2 δ missing) — partners: {partners}")

        for partner in partners:
            # AB: missing element at corner
            dirname_ab = f"{missing_el}{partner}"
            dirpath_ab = os.path.join(b2_dir, dirname_ab)
            os.makedirs(dirpath_ab, exist_ok=True)
            a0_ab = estimate_b2_a0(missing_el, partner)
            with open(os.path.join(dirpath_ab, "INCAR"), "w") as f:
                f.write(INCAR_B2)
            write_poscar_b2(dirpath_ab, missing_el, partner, a0_ab)
            write_kpoints(dirpath_ab)
            all_calcs.append((f"B2/{dirname_ab}", missing_el, partner))
            b2_count += 1

            # BA: partner at corner, missing element at body center
            dirname_ba = f"{partner}{missing_el}"
            dirpath_ba = os.path.join(b2_dir, dirname_ba)
            os.makedirs(dirpath_ba, exist_ok=True)
            a0_ba = estimate_b2_a0(partner, missing_el)
            with open(os.path.join(dirpath_ba, "INCAR"), "w") as f:
                f.write(INCAR_B2)
            write_poscar_b2(dirpath_ba, partner, missing_el, a0_ba)
            write_kpoints(dirpath_ba)
            all_calcs.append((f"B2/{dirname_ba}", partner, missing_el))
            b2_count += 1

    # =================================================================
    # Generate POTCAR concatenation script
    # =================================================================
    potcar_script = os.path.join(base_dir, "make_potcar.sh")
    with open(potcar_script, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Auto-generated POTCAR creation script\n")
        f.write("# Usage: bash make_potcar.sh\n")
        f.write("# Requires: $VASPPOT pointing to VASP pseudopotential directory\n")
        f.write("#   e.g., export VASPPOT=/path/to/potpaw_PBE.54\n\n")
        f.write('if [ -z "$VASPPOT" ]; then\n')
        f.write('    echo "ERROR: Set VASPPOT to your VASP pseudopotential directory"\n')
        f.write('    echo "  e.g., export VASPPOT=/path/to/potpaw_PBE.54"\n')
        f.write("    exit 1\n")
        f.write("fi\n\n")

        for subdir, el1, el2 in all_calcs:
            pot1 = POTCAR_VARIANTS.get(el1, el1)
            pot2 = POTCAR_VARIANTS.get(el2, el2)
            f.write(f'echo "Creating POTCAR for {subdir}"\n')
            f.write(f'cat $VASPPOT/{pot1}/POTCAR $VASPPOT/{pot2}/POTCAR > {subdir}/POTCAR\n')
        f.write(f'\necho "Done. Created {len(all_calcs)} POTCAR files."\n')
    os.chmod(potcar_script, 0o755)

    # =================================================================
    # Generate batch submission script
    # =================================================================
    run_script = os.path.join(base_dir, "run_all.sh")
    with open(run_script, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Auto-generated batch submission script\n")
        f.write("# Modify the SBATCH/qsub commands for your cluster\n\n")
        f.write("# Option 1: Serial execution (testing)\n")
        f.write("# for dir in L12/* B2/*; do\n")
        f.write("#     cd $dir && mpirun -np 4 vasp_std > vasp.out 2>&1 && cd ../..\n")
        f.write("# done\n\n")
        f.write("# Option 2: SLURM array job\n")
        f.write("# Create job list\n")
        f.write("DIRS=(\n")
        for subdir, _, _ in all_calcs:
            f.write(f"  \"{subdir}\"\n")
        f.write(")\n\n")
        f.write("for dir in \"${DIRS[@]}\"; do\n")
        f.write("    echo \"Submitting $dir\"\n")
        f.write("    cd $dir\n")
        f.write("    # Uncomment one of the following:\n")
        f.write("    # sbatch job.sh                    # SLURM\n")
        f.write("    # qsub job.sh                     # PBS/Torque\n")
        f.write("    # mpirun -np 4 vasp_std > vasp.out 2>&1 &  # Local\n")
        f.write("    cd $(dirname $(dirname $(pwd)))/table6_missing\n")
        f.write("done\n\n")
        f.write(f"echo \"Submitted {len(all_calcs)} calculations.\"\n")
    os.chmod(run_script, 0o755)

    # =================================================================
    # Generate SLURM job template
    # =================================================================
    job_template = os.path.join(base_dir, "job_template.sh")
    with open(job_template, "w") as f:
        f.write("""#!/bin/bash
#SBATCH --job-name=tab6_fill
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=02:00:00
#SBATCH --partition=normal

# Load VASP module (adjust for your cluster)
# module load vasp/6.3.2

cd $SLURM_SUBMIT_DIR

mpirun vasp_std > vasp.out 2>&1

# Check convergence
if grep -q "reached required accuracy" vasp.out; then
    echo "CONVERGED" > STATUS
else
    echo "NOT_CONVERGED" > STATUS
fi
""")
    os.chmod(job_template, 0o755)

    # =================================================================
    # Generate extraction script (post-calculation)
    # =================================================================
    extract_script = os.path.join(base_dir, "extract_results.py")
    with open(extract_script, "w") as f:
        f.write('''#!/usr/bin/env python3
"""
Extract lattice constants from completed VASP calculations.
Run after all calculations have converged.

Usage:
    cd table6_missing
    python extract_results.py

Output:
    table6_new_data.csv — ready to import into generate_all_figures.py
"""

import os
import csv
import numpy as np

def read_contcar(filepath):
    """Read lattice constant from CONTCAR."""
    if not os.path.exists(filepath):
        return None
    with open(filepath) as f:
        lines = f.readlines()
    if len(lines) < 5:
        return None
    try:
        scale = float(lines[1].strip())
        a_vec = [float(x) for x in lines[2].split()]
        a = scale * np.sqrt(sum(x**2 for x in a_vec))
        return a
    except (ValueError, IndexError):
        return None

def main():
    results = []

    for struct_type in ["L12", "B2"]:
        struct_dir = struct_type
        if not os.path.isdir(struct_dir):
            continue
        for dirname in sorted(os.listdir(struct_dir)):
            dirpath = os.path.join(struct_dir, dirname)
            if not os.path.isdir(dirpath):
                continue

            contcar = os.path.join(dirpath, "CONTCAR")
            a = read_contcar(contcar)

            if a is None:
                print(f"  SKIP {struct_type}/{dirname}: no CONTCAR or parse error")
                continue

            # Parse elements from dirname
            # L12: e.g., "Be3Ni" -> face=Be(3), corner=Ni(1)
            # B2:  e.g., "GeNi" -> corner=Ge, body=Ni
            if struct_type == "L12":
                # Find the "3" separator
                idx3 = dirname.index("3")
                el_maj = dirname[:idx3]
                el_min = dirname[idx3+1:]
                count_A = 3
                count_B = 1
            else:  # B2
                # Two elements concatenated, each 1-2 chars
                # Try 2-char + rest, then 1-char + rest
                el_a, el_b = None, None
                for i in [2, 1]:
                    candidate = dirname[:i]
                    rest = dirname[i:]
                    if candidate.isalpha() and rest.isalpha():
                        if candidate[0].isupper() and (len(candidate) == 1 or candidate[1].islower()):
                            if rest[0].isupper():
                                el_a = candidate
                                el_b = rest
                                break
                if el_a is None:
                    print(f"  SKIP {struct_type}/{dirname}: cannot parse element names")
                    continue
                el_maj = el_a  # corner
                el_min = el_b  # body-center
                count_A = 1
                count_B = 1

            results.append({
                "structure": struct_type,
                "dirname": dirname,
                "element_A": el_maj,
                "element_B": el_min,
                "count_A": count_A,
                "count_B": count_B,
                "lattice_constant": f"{a:.6f}",
            })
            print(f"  OK {struct_type}/{dirname}: a = {a:.4f} Å")

    # Write CSV
    outfile = "table6_new_data.csv"
    with open(outfile, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "structure", "dirname", "element_A", "element_B",
            "count_A", "count_B", "lattice_constant"
        ])
        writer.writeheader()
        writer.writerows(results)

    print(f"\\nWrote {len(results)} results to {outfile}")
    print(f"\\nNext steps:")
    print(f"  1. Copy {outfile} to ../data/")
    print(f"  2. Add to generate_all_figures.py data loading")
    print(f"  3. Re-run additive decomposition to fill Table 6")

if __name__ == "__main__":
    main()
''')
    os.chmod(extract_script, 0o755)

    # =================================================================
    # Summary
    # =================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n  L1₂ calculations: {l12_count} ({l12_count//2} pairs × 2 directions)")
    print(f"  B2  calculations: {b2_count} ({b2_count//2} pairs × 2 directions)")
    print(f"  Total:            {l12_count + b2_count} VASP jobs")
    print(f"\n  Output directory: {base_dir}/")
    print(f"\n  Elements to fill:")
    print(f"    L1₂ δ: {', '.join(L12_MISSING)}")
    print(f"    B2  δ: {', '.join(B2_MISSING)}")
    print(f"\n  Estimated compute time: ~1-2 hours per job")
    print(f"  Total wall time: ~2 hours (with 30+ parallel cores)")
    print(f"\n  Steps:")
    print(f"    1. cd {base_dir}")
    print(f"    2. bash make_potcar.sh          # generate POTCARs")
    print(f"    3. cp job_template.sh L12/*/     # copy job script")
    print(f"       cp job_template.sh B2/*/")
    print(f"    4. bash run_all.sh              # submit all jobs")
    print(f"    5. python extract_results.py    # after convergence")


if __name__ == "__main__":
    main()
