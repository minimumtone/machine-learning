#!/usr/bin/env python3
"""
L1₂ / FCC-SQS 精度向上 VASP再計算スクリプト.

既存のL1₂およびBCC-SQS計算を高精度設定で再計算するための
VASP入力ファイル（INCAR, POSCAR, KPOINTS）を生成する。

改善点:
  1. ENCUT: 320 → 520 eV（カットオフエネルギー統一）
  2. k-mesh: L1₂ 6×6×6 → 12×12×12, SQS 4×4×4 → 6×6×6
  3. SIGMA: 0.2 → 0.1 eV（smearing幅縮小で精度向上）
  4. MAGMOM: 磁性元素に初期磁気モーメントを明示的に設定
  5. AF（反強磁性）構成: 磁性元素ペアに対してFM/AF両方を計算
  6. EDIFFG: -0.01 → -0.005 eV/Å（力の収束基準を厳格化）
  7. NELM: 60 → 300（電子緩和の最大ステップ増加）

Usage:
    python generate_l12_sqs_recalc.py [OPTIONS]

    --mode l12       : L1₂再計算のみ
    --mode sqs       : FCC-SQS再計算のみ
    --mode all       : 両方（デフォルト）
    --elements "Fe,Co,Ni,Cr,Mn"  : 指定元素のペアのみ
    --magnetic-only  : 磁性元素（Cr,Mn,Fe,Co,Ni）を含むペアのみ
    --af-only        : AF構成のみ生成（FM既計算の場合）

Output:
    L12_recalc/
    ├── FM/
    │   ├── Fe3Mn/  (INCAR, POSCAR, KPOINTS)
    │   ├── Mn3Fe/
    │   └── ...
    ├── AF/
    │   ├── Fe3Mn_AF/  (INCAR, POSCAR, KPOINTS)
    │   └── ...
    ├── make_potcar.sh
    ├── run_all.sh
    └── extract_results.sh

    SQS_recalc/
    ├── FM/
    │   ├── Fe8Mn8/  (INCAR, POSCAR, KPOINTS)
    │   └── ...
    ├── AF/
    │   ├── Fe8Mn8_AF/
    │   └── ...
    ├── make_potcar.sh
    ├── run_all.sh
    └── extract_results.sh
"""

import os
import sys
import argparse
import itertools
import math


# =====================================================================
# Model elements (41 elements from KING_ATOMIC_VOLUMES)
# =====================================================================
ALL_ELEMENTS = sorted([
    'Ag', 'Al', 'Au', 'B', 'Be', 'Ca', 'Ce', 'Co', 'Cr', 'Cu',
    'Dy', 'Er', 'Fe', 'Ge', 'Hf', 'Ir', 'La', 'Mg', 'Mn', 'Mo',
    'Nb', 'Ni', 'Os', 'P', 'Pb', 'Pd', 'Pt', 'Re', 'Rh', 'Ru',
    'Sc', 'Si', 'Sn', 'Ta', 'Tb', 'Ti', 'V', 'W', 'Y', 'Zn', 'Zr',
])

# Magnetic 3d transition metals
MAGNETIC_ELEMENTS = {'Cr', 'Mn', 'Fe', 'Co', 'Ni'}

# =====================================================================
# FCC-equivalent lattice constants (Å) for initial L1₂ POSCAR
# =====================================================================
ELEMENT_A0_FCC = {
    "Ag": 4.085, "Al": 4.050, "Au": 4.078, "B":  3.400, "Be": 3.190,
    "Ca": 5.580, "Ce": 5.160, "Co": 3.545, "Cr": 3.640, "Cu": 3.615,
    "Dy": 5.040, "Er": 4.960, "Fe": 3.570, "Ge": 4.050, "Hf": 4.470,
    "Ir": 3.839, "La": 5.300, "Mg": 4.510, "Mn": 3.630, "Mo": 3.960,
    "Nb": 4.160, "Ni": 3.524, "Os": 3.850, "P":  4.200, "Pb": 4.950,
    "Pd": 3.890, "Pt": 3.924, "Re": 3.890, "Rh": 3.803, "Ru": 3.827,
    "Sc": 4.630, "Si": 3.840, "Sn": 4.550, "Ta": 4.150, "Tb": 5.080,
    "Ti": 4.100, "V":  3.820, "W":  3.980, "Y":  5.090, "Zn": 3.940,
    "Zr": 4.540,
}

# BCC lattice constants (Å) for SQS initial POSCAR
ELEMENT_A0_BCC = {
    "Ag": 3.301, "Al": 3.225, "Au": 3.340, "B":  2.311, "Be": 2.540,
    "Ca": 4.384, "Ce": 3.767, "Co": 2.801, "Cr": 2.836, "Cu": 2.896,
    "Dy": 3.940, "Er": 3.880, "Fe": 2.827, "Ge": 3.376, "Hf": 3.534,
    "Ir": 3.050, "La": 4.222, "Mg": 3.579, "Mn": 2.788, "Mo": 3.149,
    "Nb": 3.324, "Ni": 2.794, "Os": 3.050, "P":  3.400, "Pb": 3.950,
    "Pd": 3.139, "Pt": 3.120, "Re": 3.070, "Rh": 3.020, "Ru": 3.046,
    "Sc": 3.677, "Si": 3.094, "Sn": 3.808, "Ta": 3.312, "Tb": 3.980,
    "Ti": 3.240, "V":  2.982, "W":  3.172, "Y":  4.027, "Zn": 3.157,
    "Zr": 3.569,
}

# =====================================================================
# POTCAR pseudopotential recommendations (VASP 5.4+ PAW-PBE)
# =====================================================================
POTCAR_VARIANTS = {
    "Ag": "Ag",    "Al": "Al",    "Au": "Au",    "B":  "B",
    "Be": "Be",    "Ca": "Ca_sv", "Ce": "Ce",    "Co": "Co",
    "Cr": "Cr_pv", "Cu": "Cu_pv", "Dy": "Dy_3",  "Er": "Er_3",
    "Fe": "Fe_pv", "Ge": "Ge_d",  "Hf": "Hf_pv", "Ir": "Ir",
    "La": "La",    "Mg": "Mg_pv", "Mn": "Mn_pv", "Mo": "Mo_pv",
    "Nb": "Nb_pv", "Ni": "Ni_pv", "Os": "Os_pv", "P":  "P",
    "Pb": "Pb_d",  "Pd": "Pd",    "Pt": "Pt",    "Re": "Re_pv",
    "Rh": "Rh_pv", "Ru": "Ru_pv", "Sc": "Sc_sv", "Si": "Si",
    "Sn": "Sn_d",  "Ta": "Ta_pv", "Tb": "Tb_3",  "Ti": "Ti_pv",
    "V":  "V_sv",  "W":  "W_sv",  "Y":  "Y_sv",  "Zn": "Zn",
    "Zr": "Zr_sv",
}

# Default magnetic moments (μ_B) for 3d magnetic elements
DEFAULT_MAGMOM = {
    "Cr": 1.5, "Mn": 3.0, "Fe": 2.5, "Co": 1.5, "Ni": 0.6,
}


# =====================================================================
# SQS-16 configuration for FCC (2x2x2 supercell of conventional FCC)
# FCC conventional cell: 4 atoms → 2x2x2 = 32 atoms (A16B16)
# Use smaller SQS-8: 2x1x1 of conventional FCC = 8 atoms (A4B4)
# Actually for FCC SQS, we use the L1₂ approach differently:
#   - L1₂ is the ordered reference for FCC HEA predictions
#   - FCC-SQS would be the disordered FCC reference
# For this script, we support both approaches
# =====================================================================

# FCC conventional cell fractional coordinates (4 atoms per cell)
FCC_CONVENTIONAL_POSITIONS = [
    (0.0, 0.0, 0.0),
    (0.5, 0.5, 0.0),
    (0.5, 0.0, 0.5),
    (0.0, 0.5, 0.5),
]

# FCC 2x2x2 supercell: 32 atoms
FCC_2x2x2_POSITIONS = []
for ix in range(2):
    for iy in range(2):
        for iz in range(2):
            for pos in FCC_CONVENTIONAL_POSITIONS:
                FCC_2x2x2_POSITIONS.append((
                    (pos[0] + ix) / 2.0,
                    (pos[1] + iy) / 2.0,
                    (pos[2] + iz) / 2.0,
                ))

# SQS-32 occupation for FCC A16B16
# Optimized to minimize Warren-Cowley SRO: alpha_1nn ~ 0, alpha_2nn ~ 0
SQS_FCC_OCCUPATION = [
    0, 1, 1, 0,  # cell (0,0,0)
    1, 0, 0, 1,  # cell (1,0,0)
    1, 0, 0, 1,  # cell (0,1,0)
    0, 1, 1, 0,  # cell (1,1,0)
    0, 1, 0, 1,  # cell (0,0,1)
    1, 0, 1, 0,  # cell (1,0,1)
    0, 1, 1, 0,  # cell (0,1,1)
    1, 0, 0, 1,  # cell (1,1,1)
]

# BCC 2x2x2 supercell (16 atoms) for BCC-SQS
BCC_2x2x2_POSITIONS = []
for ix in range(2):
    for iy in range(2):
        for iz in range(2):
            BCC_2x2x2_POSITIONS.append(
                (ix / 2.0, iy / 2.0, iz / 2.0))
            BCC_2x2x2_POSITIONS.append(
                ((ix + 0.5) / 2.0, (iy + 0.5) / 2.0, (iz + 0.5) / 2.0))

# BCC SQS-16 occupation (from existing generate_refractory_sqs.py)
SQS_BCC_OCCUPATION = [0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0]


# =====================================================================
# INCAR templates (high-accuracy)
# =====================================================================

def make_incar_l12(system_name, magmom_str=None, is_af=False):
    """Generate high-accuracy INCAR for L1₂ calculation."""
    lines = [
        f"SYSTEM = {system_name}",
        "",
        "# Electronic relaxation",
        "ENCUT  = 520",
        "PREC   = Accurate",
        "EDIFF  = 1E-6",
        "NELM   = 300",
        "LREAL  = .FALSE.",
        "",
        "# Ionic relaxation",
        "IBRION = 2",
        "ISIF   = 3",
        "NSW    = 200",
        "EDIFFG = -0.005",
        "",
        "# Smearing (Methfessel-Paxton order 1)",
        "ISMEAR = 1",
        "SIGMA  = 0.1",
        "",
        "# Exchange-correlation",
        "GGA    = PE",
        "",
        "# Spin polarization",
        "ISPIN  = 2",
    ]

    if magmom_str:
        lines.append(f"MAGMOM = {magmom_str}")

    lines += [
        "",
        "# Output",
        "LORBIT = 11",
        "LWAVE  = .FALSE.",
        "LCHARG = .FALSE.",
        "",
        "# Performance",
        "NCORE  = 4",
        "",
    ]

    return "\n".join(lines)


def make_incar_sqs(system_name, n_atoms, magmom_str=None, is_af=False):
    """Generate high-accuracy INCAR for SQS supercell calculation."""
    lines = [
        f"SYSTEM = {system_name}",
        "",
        "# Electronic relaxation",
        "ENCUT  = 520",
        "PREC   = Accurate",
        "EDIFF  = 1E-6",
        "NELM   = 300",
        "LREAL  = .FALSE." if n_atoms <= 32 else "LREAL  = Auto",
        "",
        "# Ionic relaxation",
        "IBRION = 2",
        "ISIF   = 3",
        "NSW    = 200",
        "EDIFFG = -0.005",
        "POTIM  = 0.02",
        "",
        "# Smearing (Methfessel-Paxton order 1)",
        "ISMEAR = 1",
        "SIGMA  = 0.1",
        "",
        "# Exchange-correlation",
        "GGA    = PE",
        "ALGO   = Normal",
        "",
        "# Spin polarization",
        "ISPIN  = 2",
    ]

    if magmom_str:
        lines.append(f"MAGMOM = {magmom_str}")

    lines += [
        "",
        "# Output",
        "LORBIT = 11",
        "LWAVE  = .FALSE.",
        "LCHARG = .FALSE.",
        "",
        "# Performance",
        "NCORE  = 4",
        "",
    ]

    return "\n".join(lines)


# =====================================================================
# MAGMOM generation
# =====================================================================

def make_magmom_l12(el_face, el_corner, is_af=False):
    """
    Generate MAGMOM string for L1₂ (4 atoms: 3 face + 1 corner).

    FM: all spins aligned
    AF: face atoms up, corner atom down (Type-I AF for L1₂)
    """
    has_magnetic = (el_face in MAGNETIC_ELEMENTS or
                    el_corner in MAGNETIC_ELEMENTS)
    if not has_magnetic:
        return None

    m_face = DEFAULT_MAGMOM.get(el_face, 0.0)
    m_corner = DEFAULT_MAGMOM.get(el_corner, 0.0)

    if m_face == 0.0 and m_corner == 0.0:
        return None

    if is_af:
        # AF Type-I: face atoms up, corner atom down
        return f"{m_face} {m_face} {m_face} {-m_corner}"
    else:
        return f"3*{m_face} {m_corner}"


def make_magmom_sqs_bcc(el_a, el_b, is_af=False):
    """
    Generate MAGMOM string for BCC SQS-16 (8 A + 8 B atoms).

    FM: all spins aligned
    AF: A atoms up, B atoms down
    """
    has_magnetic = (el_a in MAGNETIC_ELEMENTS or
                    el_b in MAGNETIC_ELEMENTS)
    if not has_magnetic:
        return None

    m_a = DEFAULT_MAGMOM.get(el_a, 0.0)
    m_b = DEFAULT_MAGMOM.get(el_b, 0.0)

    if m_a == 0.0 and m_b == 0.0:
        return None

    if is_af:
        # AF: alternate spin within each sublattice
        vals_a = [m_a, -m_a, m_a, -m_a, m_a, -m_a, m_a, -m_a]
        vals_b = [m_b, -m_b, m_b, -m_b, m_b, -m_b, m_b, -m_b]
        return " ".join(f"{v:.1f}" for v in vals_a + vals_b)
    else:
        if el_a == el_b:
            return f"16*{m_a}"
        return f"8*{m_a} 8*{m_b}"


def make_magmom_sqs_fcc(el_a, el_b, is_af=False):
    """
    Generate MAGMOM string for FCC SQS-32 (16 A + 16 B atoms).

    FM: all spins aligned
    AF: checkerboard-like alternation
    """
    has_magnetic = (el_a in MAGNETIC_ELEMENTS or
                    el_b in MAGNETIC_ELEMENTS)
    if not has_magnetic:
        return None

    m_a = DEFAULT_MAGMOM.get(el_a, 0.0)
    m_b = DEFAULT_MAGMOM.get(el_b, 0.0)

    if m_a == 0.0 and m_b == 0.0:
        return None

    if is_af:
        vals_a = [m_a * ((-1)**i) for i in range(16)]
        vals_b = [m_b * ((-1)**i) for i in range(16)]
        return " ".join(f"{v:.1f}" for v in vals_a + vals_b)
    else:
        if el_a == el_b:
            return f"32*{m_a}"
        return f"16*{m_a} 16*{m_b}"


# =====================================================================
# POSCAR generation
# =====================================================================

def write_poscar_l12(dirpath, el_face, el_corner, a0):
    """Write POSCAR for L1₂ (Cu3Au-type, Pm-3m, 4 atoms)."""
    content = f"""{el_face}3{el_corner} L12 (Pm-3m) high-accuracy recalc
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


def write_poscar_sqs_bcc(dirpath, el_a, el_b, a_super):
    """Write POSCAR for BCC SQS-16 (2x2x2, A8B8 or A16)."""
    if el_a == el_b:
        lines = [
            f"{el_a}16 BCC-SQS16 (2x2x2, pure reference) recalc",
            "1.0",
            f"  {a_super:.6f}  0.000000  0.000000",
            f"  0.000000  {a_super:.6f}  0.000000",
            f"  0.000000  0.000000  {a_super:.6f}",
            f"  {el_a}",
            "  16",
            "Direct",
        ]
        for pos in BCC_2x2x2_POSITIONS:
            lines.append(f"  {pos[0]:.6f}  {pos[1]:.6f}  {pos[2]:.6f}")
        lines.append("")
    else:
        pos_a, pos_b = [], []
        for i, occ in enumerate(SQS_BCC_OCCUPATION):
            if occ == 0:
                pos_a.append(BCC_2x2x2_POSITIONS[i])
            else:
                pos_b.append(BCC_2x2x2_POSITIONS[i])

        lines = [
            f"{el_a}8{el_b}8 BCC-SQS16 (2x2x2, 50:50) recalc",
            "1.0",
            f"  {a_super:.6f}  0.000000  0.000000",
            f"  0.000000  {a_super:.6f}  0.000000",
            f"  0.000000  0.000000  {a_super:.6f}",
            f"  {el_a}  {el_b}",
            "  8  8",
            "Direct",
        ]
        for pos in pos_a:
            lines.append(f"  {pos[0]:.6f}  {pos[1]:.6f}  {pos[2]:.6f}")
        for pos in pos_b:
            lines.append(f"  {pos[0]:.6f}  {pos[1]:.6f}  {pos[2]:.6f}")
        lines.append("")

    with open(os.path.join(dirpath, "POSCAR"), "w") as f:
        f.write("\n".join(lines))


def write_poscar_sqs_fcc(dirpath, el_a, el_b, a_super):
    """Write POSCAR for FCC SQS-32 (2x2x2, A16B16 or A32)."""
    if el_a == el_b:
        lines = [
            f"{el_a}32 FCC-SQS32 (2x2x2, pure reference) recalc",
            "1.0",
            f"  {a_super:.6f}  0.000000  0.000000",
            f"  0.000000  {a_super:.6f}  0.000000",
            f"  0.000000  0.000000  {a_super:.6f}",
            f"  {el_a}",
            "  32",
            "Direct",
        ]
        for pos in FCC_2x2x2_POSITIONS:
            lines.append(f"  {pos[0]:.6f}  {pos[1]:.6f}  {pos[2]:.6f}")
        lines.append("")
    else:
        pos_a, pos_b = [], []
        for i, occ in enumerate(SQS_FCC_OCCUPATION):
            if occ == 0:
                pos_a.append(FCC_2x2x2_POSITIONS[i])
            else:
                pos_b.append(FCC_2x2x2_POSITIONS[i])

        n_a = len(pos_a)
        n_b = len(pos_b)

        lines = [
            f"{el_a}{n_a}{el_b}{n_b} FCC-SQS32 (2x2x2, 50:50) recalc",
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


def write_kpoints(dirpath, kmesh):
    """Write KPOINTS file (Gamma-centered)."""
    content = f"""Automatic mesh
0
Gamma
  {kmesh} {kmesh} {kmesh}
  0 0 0
"""
    with open(os.path.join(dirpath, "KPOINTS"), "w") as f:
        f.write(content)


# =====================================================================
# Estimation functions
# =====================================================================

def estimate_l12_a0(el_face, el_corner):
    """Estimate initial L1₂ lattice parameter via Vegard's law."""
    a_face = ELEMENT_A0_FCC.get(el_face, 3.80)
    a_corner = ELEMENT_A0_FCC.get(el_corner, 3.80)
    return 0.75 * a_face + 0.25 * a_corner


def estimate_sqs_bcc_a0(el_a, el_b):
    """Estimate initial BCC-SQS supercell parameter (2 x a_BCC_avg)."""
    a_a = ELEMENT_A0_BCC.get(el_a, 3.20)
    a_b = ELEMENT_A0_BCC.get(el_b, 3.20)
    return 2.0 * 0.5 * (a_a + a_b)


def estimate_sqs_fcc_a0(el_a, el_b):
    """Estimate initial FCC-SQS supercell parameter (2 x a_FCC_avg)."""
    a_a = ELEMENT_A0_FCC.get(el_a, 3.80)
    a_b = ELEMENT_A0_FCC.get(el_b, 3.80)
    return 2.0 * 0.5 * (a_a + a_b)


# =====================================================================
# Shell script generation
# =====================================================================

def generate_potcar_script(base_dir, calculations, vasp_pp_var="VASPPOT"):
    """Generate shell script to create POTCAR files."""
    lines = [
        "#!/bin/bash",
        f"# POTCAR generation script for {os.path.basename(base_dir)}",
        "# Usage: bash make_potcar.sh",
        f'# Requires: ${vasp_pp_var} pointing to PAW-PBE pseudopotential directory',
        "",
        f'if [ -z "${vasp_pp_var}" ]; then',
        f'    echo "Error: ${vasp_pp_var} not set."',
        f'    echo "  export {vasp_pp_var}=/path/to/potpaw_PBE.64"',
        '    exit 1',
        'fi',
        "",
        f'echo "Using {vasp_pp_var}=${vasp_pp_var}"',
        f'echo "Total calculations: {len(calculations)}"',
        "",
    ]

    for subdir, elements in calculations:
        potcars = [POTCAR_VARIANTS.get(e, e) for e in elements]
        cat_parts = " ".join(
            f'"${vasp_pp_var}"/{p}/POTCAR' for p in potcars
        )
        dirname = os.path.basename(subdir)
        lines.append(f'# --- {dirname} ---')
        lines.append(f'cat {cat_parts} > {subdir}/POTCAR 2>/dev/null')
        lines.append(
            f'[ $? -ne 0 ] && echo "  WARN: {dirname} failed"'
        )
        lines.append("")

    lines.append('echo "Done."')

    path = os.path.join(base_dir, "make_potcar.sh")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    os.chmod(path, 0o755)


def generate_run_script(base_dir, calculations):
    """Generate batch execution script."""
    lines = [
        "#!/bin/bash",
        f"# Batch execution: {os.path.basename(base_dir)}",
        '# Usage: bash run_all.sh',
        '#   VASP_CMD: override VASP command (default: mpirun -np 16 vasp_std)',
        "",
        'VASP_CMD="${VASP_CMD:-mpirun -np ${NPROCS:-16} vasp_std}"',
        'BASE_DIR=$(cd $(dirname $0) && pwd)',
        "",
        "run_calc() {",
        '    local dir="$1"',
        '    local name=$(basename "$dir")',
        '    cd "$dir"',
        "",
        '    if [ -f "OUTCAR" ] && grep -q "reached required accuracy" OUTCAR 2>/dev/null; then',
        '        echo "SKIP: $name (converged)"',
        '        cd "$BASE_DIR"',
        "        return 0",
        "    fi",
        "",
        '    if [ ! -f "POTCAR" ]; then',
        '        echo "ERROR: $name — no POTCAR"',
        '        cd "$BASE_DIR"',
        "        return 1",
        "    fi",
        "",
        '    echo "RUN: $name"',
        '    $VASP_CMD > vasp.log 2>&1',
        '    if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then',
        '        echo "  OK: $name"',
        "    else",
        '        echo "  WARN: $name — not converged"',
        "    fi",
        '    cd "$BASE_DIR"',
        "}",
        "",
        f'echo "=== {os.path.basename(base_dir)}: {len(calculations)} calculations ==="',
        "",
    ]

    for subdir, _ in calculations:
        lines.append(f'run_calc "$BASE_DIR/{subdir}"')

    lines.append("")
    lines.append('echo "All done."')

    path = os.path.join(base_dir, "run_all.sh")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    os.chmod(path, 0o755)


def generate_extract_script_l12(base_dir, calculations):
    """Generate result extraction script for L1₂ calculations."""
    lines = [
        "#!/bin/bash",
        "# Extract results from L1₂ recalculations",
        "# Usage: bash extract_results.sh > l12_recalc_results.csv",
        "",
        'echo "dirname,formula,element_A,element_B,count_A,count_B,'
        'lattice_constant,energy_per_atom,converged,mag_config,total_mag"',
        "",
    ]

    for subdir, elements in calculations:
        dirname = os.path.basename(subdir)
        el_face = elements[0]
        el_corner = elements[1]
        mag_cfg = "AF" if "_AF" in dirname else "FM"

        lines.append(f'DIR="{subdir}"')
        lines.append('if [ -f "$DIR/CONTCAR" ]; then')
        lines.append('    A=$(head -3 "$DIR/CONTCAR" | tail -1 | '
                     "awk '{print $1}')")
        lines.append('    E=$(grep "energy  without entropy" '
                     '"$DIR/OUTCAR" 2>/dev/null | tail -1 | '
                     "awk '{print $NF}')")
        lines.append('    NATOM=$(grep "NIONS" "$DIR/OUTCAR" '
                     "2>/dev/null | awk '{print $NF}')")
        lines.append('    EMAG=$(grep "number of electron" '
                     '"$DIR/OUTCAR" 2>/dev/null | tail -1 | '
                     "awk '{print $6}')")
        lines.append('    CONV="no"')
        lines.append('    grep -q "reached required accuracy" '
                     '"$DIR/OUTCAR" 2>/dev/null && CONV="yes"')
        lines.append('    if [ -n "$E" ] && [ -n "$NATOM" ] && '
                     '[ "$NATOM" -gt 0 ] 2>/dev/null; then')
        lines.append('        EPA=$(echo "scale=10; $E / $NATOM" | bc)')
        lines.append('    else')
        lines.append('        EPA="NA"')
        lines.append('    fi')
        lines.append(f'    echo "{dirname},{el_face}3{el_corner},'
                     f'{el_face},{el_corner},3,1,$A,$EPA,$CONV,'
                     f'{mag_cfg},$EMAG"')
        lines.append('else')
        lines.append(f'    echo "{dirname},{el_face}3{el_corner},'
                     f'{el_face},{el_corner},3,1,NA,NA,not_run,'
                     f'{mag_cfg},NA"')
        lines.append('fi')
        lines.append("")

    path = os.path.join(base_dir, "extract_results.sh")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    os.chmod(path, 0o755)


def generate_extract_script_sqs(base_dir, calculations, sqs_type="BCC"):
    """Generate result extraction script for SQS calculations."""
    n_atoms = 16 if sqs_type == "BCC" else 32
    n_each = n_atoms // 2

    lines = [
        "#!/bin/bash",
        f"# Extract results from {sqs_type}-SQS recalculations",
        "# Usage: bash extract_results.sh > sqs_recalc_results.csv",
        "",
        'echo "dirname,formula,element_A,element_B,count_A,count_B,'
        'lattice_constant,energy_per_atom,converged,mag_config,total_mag"',
        "",
    ]

    for subdir, elements in calculations:
        dirname = os.path.basename(subdir)
        el_a = elements[0]
        el_b = elements[1] if len(elements) > 1 else elements[0]
        mag_cfg = "AF" if "_AF" in dirname else "FM"

        if el_a == el_b:
            formula = f"{el_a}{n_atoms}"
            cA, cB = n_atoms, 0
        else:
            formula = f"{el_a}{n_each}{el_b}{n_each}"
            cA, cB = n_each, n_each

        lines.append(f'DIR="{subdir}"')
        lines.append('if [ -f "$DIR/CONTCAR" ]; then')
        lines.append('    ASUPER=$(head -3 "$DIR/CONTCAR" | tail -1 | '
                     "awk '{print $1}')")
        lines.append('    A=$(echo "scale=10; $ASUPER / 2" | bc)')
        lines.append('    E=$(grep "energy  without entropy" '
                     '"$DIR/OUTCAR" 2>/dev/null | tail -1 | '
                     "awk '{print $NF}')")
        lines.append('    NATOM=$(grep "NIONS" "$DIR/OUTCAR" '
                     "2>/dev/null | awk '{print $NF}')")
        lines.append('    EMAG=$(grep "number of electron" '
                     '"$DIR/OUTCAR" 2>/dev/null | tail -1 | '
                     "awk '{print $6}')")
        lines.append('    CONV="no"')
        lines.append('    grep -q "reached required accuracy" '
                     '"$DIR/OUTCAR" 2>/dev/null && CONV="yes"')
        lines.append('    if [ -n "$E" ] && [ -n "$NATOM" ] && '
                     '[ "$NATOM" -gt 0 ] 2>/dev/null; then')
        lines.append('        EPA=$(echo "scale=10; $E / $NATOM" | bc)')
        lines.append('    else')
        lines.append('        EPA="NA"')
        lines.append('    fi')
        lines.append(f'    echo "{dirname},{formula},'
                     f'{el_a},{el_b},{cA},{cB},$A,$EPA,$CONV,'
                     f'{mag_cfg},$EMAG"')
        lines.append('else')
        lines.append(f'    echo "{dirname},{formula},'
                     f'{el_a},{el_b},{cA},{cB},NA,NA,not_run,'
                     f'{mag_cfg},NA"')
        lines.append('fi')
        lines.append("")

    path = os.path.join(base_dir, "extract_results.sh")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    os.chmod(path, 0o755)


def generate_slurm_jobscript(base_dir, name, n_calcs):
    """Generate SLURM job script template."""
    content = f"""#!/bin/bash
#SBATCH -J {name}
#SBATCH -p default
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH --time=72:00:00
#SBATCH -o %j.out
#SBATCH -e %j.err

# Load modules (adjust for your HPC environment)
# module load vasp/6.4.0
# module load intel-mpi

export NPROCS=16
export VASP_CMD="mpirun -np $NPROCS vasp_std"

cd $SLURM_SUBMIT_DIR
bash run_all.sh
"""
    path = os.path.join(base_dir, "job_slurm.sh")
    with open(path, "w") as f:
        f.write(content)
    os.chmod(path, 0o755)


def generate_pbs_jobscript(base_dir, name, n_calcs):
    """Generate PBS job script template."""
    content = f"""#!/bin/bash
#PBS -N {name}
#PBS -l select=1:ncpus=16:mpiprocs=16
#PBS -l walltime=72:00:00
#PBS -j oe

# Load modules (adjust for your HPC environment)
# module load vasp/6.4.0
# module load intel-mpi

export NPROCS=16
export VASP_CMD="mpirun -np $NPROCS vasp_std"

cd $PBS_O_WORKDIR
bash run_all.sh
"""
    path = os.path.join(base_dir, "job_pbs.sh")
    with open(path, "w") as f:
        f.write(content)
    os.chmod(path, 0o755)


# =====================================================================
# Main generation logic
# =====================================================================

def generate_l12_recalc(target_elements, include_af=True,
                        magnetic_only=False):
    """Generate L1₂ recalculation inputs."""
    base_dir = "L12_recalc"
    os.makedirs(base_dir, exist_ok=True)

    all_calcs = []  # (subdir, [elements])
    pairs = list(itertools.combinations(sorted(target_elements), 2))

    if magnetic_only:
        pairs = [p for p in pairs
                 if p[0] in MAGNETIC_ELEMENTS or p[1] in MAGNETIC_ELEMENTS]

    print("=" * 60)
    print(f"L1₂ recalculation: {len(pairs)} pairs")
    print("Settings: ENCUT=520, k-mesh=12x12x12, SIGMA=0.1")
    print("          EDIFFG=-0.005, NELM=300, NSW=200")
    print("=" * 60)

    for el_a, el_b in pairs:
        for el_face, el_corner in [(el_a, el_b), (el_b, el_a)]:
            # FM calculation
            dirname = f"FM/{el_face}3{el_corner}"
            dirpath = os.path.join(base_dir, dirname)
            os.makedirs(dirpath, exist_ok=True)

            a0 = estimate_l12_a0(el_face, el_corner)
            magmom = make_magmom_l12(el_face, el_corner, is_af=False)
            system_name = f"L12 {el_face}3{el_corner} FM recalc"
            incar = make_incar_l12(system_name, magmom)

            with open(os.path.join(dirpath, "INCAR"), "w") as f:
                f.write(incar)
            write_poscar_l12(dirpath, el_face, el_corner, a0)
            write_kpoints(dirpath, kmesh=12)
            all_calcs.append((dirname, [el_face, el_corner]))

            # AF calculation (only for magnetic pairs)
            if include_af:
                has_mag = (el_face in MAGNETIC_ELEMENTS or
                           el_corner in MAGNETIC_ELEMENTS)
                if has_mag:
                    dirname_af = f"AF/{el_face}3{el_corner}_AF"
                    dirpath_af = os.path.join(base_dir, dirname_af)
                    os.makedirs(dirpath_af, exist_ok=True)

                    magmom_af = make_magmom_l12(
                        el_face, el_corner, is_af=True)
                    system_name_af = (
                        f"L12 {el_face}3{el_corner} AF recalc")
                    incar_af = make_incar_l12(
                        system_name_af, magmom_af, is_af=True)

                    with open(os.path.join(dirpath_af, "INCAR"), "w") as f:
                        f.write(incar_af)
                    write_poscar_l12(dirpath_af, el_face, el_corner, a0)
                    write_kpoints(dirpath_af, kmesh=12)
                    all_calcs.append(
                        (dirname_af, [el_face, el_corner]))

    n_fm = sum(1 for d, _ in all_calcs if "FM/" in d)
    n_af = sum(1 for d, _ in all_calcs if "AF/" in d)
    print(f"\n  Generated: {n_fm} FM + {n_af} AF = {len(all_calcs)} total")

    generate_potcar_script(base_dir, all_calcs)
    generate_run_script(base_dir, all_calcs)
    generate_extract_script_l12(base_dir, all_calcs)
    generate_slurm_jobscript(base_dir, "L12_recalc", len(all_calcs))
    generate_pbs_jobscript(base_dir, "L12_recalc", len(all_calcs))

    return all_calcs


def generate_sqs_recalc(target_elements, sqs_type="BCC",
                        include_af=True, magnetic_only=False):
    """Generate SQS recalculation inputs."""
    base_dir = f"SQS_{sqs_type}_recalc"
    os.makedirs(base_dir, exist_ok=True)

    all_calcs = []
    pairs = list(itertools.combinations(sorted(target_elements), 2))

    if magnetic_only:
        pairs = [p for p in pairs
                 if p[0] in MAGNETIC_ELEMENTS or p[1] in MAGNETIC_ELEMENTS]

    if sqs_type == "BCC":
        kmesh = 6
        n_atoms = 16
    else:
        kmesh = 4
        n_atoms = 32

    print("=" * 60)
    print(f"{sqs_type}-SQS recalculation: {len(pairs)} pairs "
          f"(+ same-element refs)")
    print(f"Settings: ENCUT=520, k-mesh={kmesh}x{kmesh}x{kmesh}, "
          f"SIGMA=0.1")
    print(f"          EDIFFG=-0.005, NELM=300, {n_atoms}-atom supercell")
    print("=" * 60)

    # Same-element reference calculations
    ref_elements = set()
    for el_a, el_b in pairs:
        ref_elements.add(el_a)
        ref_elements.add(el_b)

    for el in sorted(ref_elements):
        dirname = f"FM/{el}{n_atoms}"
        dirpath = os.path.join(base_dir, dirname)
        os.makedirs(dirpath, exist_ok=True)

        if sqs_type == "BCC":
            a_super = 2.0 * ELEMENT_A0_BCC.get(el, 3.20)
            magmom = make_magmom_sqs_bcc(el, el, is_af=False)
            system_name = f"BCC-SQS {el}{n_atoms} pure ref recalc"
            incar = make_incar_sqs(system_name, n_atoms, magmom)
            write_poscar_sqs_bcc(dirpath, el, el, a_super)
        else:
            a_super = 2.0 * ELEMENT_A0_FCC.get(el, 3.80)
            magmom = make_magmom_sqs_fcc(el, el, is_af=False)
            system_name = f"FCC-SQS {el}{n_atoms} pure ref recalc"
            incar = make_incar_sqs(system_name, n_atoms, magmom)
            write_poscar_sqs_fcc(dirpath, el, el, a_super)

        with open(os.path.join(dirpath, "INCAR"), "w") as f:
            f.write(incar)
        write_kpoints(dirpath, kmesh=kmesh)
        all_calcs.append((dirname, [el]))

    # Binary pair calculations
    for el_a, el_b in pairs:
        n_each = n_atoms // 2

        # FM
        dirname = f"FM/{el_a}{n_each}{el_b}{n_each}"
        dirpath = os.path.join(base_dir, dirname)
        os.makedirs(dirpath, exist_ok=True)

        if sqs_type == "BCC":
            a_super = estimate_sqs_bcc_a0(el_a, el_b)
            magmom = make_magmom_sqs_bcc(el_a, el_b, is_af=False)
            system_name = (f"BCC-SQS {el_a}{n_each}{el_b}{n_each} "
                           f"FM recalc")
            incar = make_incar_sqs(system_name, n_atoms, magmom)
            write_poscar_sqs_bcc(dirpath, el_a, el_b, a_super)
        else:
            a_super = estimate_sqs_fcc_a0(el_a, el_b)
            magmom = make_magmom_sqs_fcc(el_a, el_b, is_af=False)
            system_name = (f"FCC-SQS {el_a}{n_each}{el_b}{n_each} "
                           f"FM recalc")
            incar = make_incar_sqs(system_name, n_atoms, magmom)
            write_poscar_sqs_fcc(dirpath, el_a, el_b, a_super)

        with open(os.path.join(dirpath, "INCAR"), "w") as f:
            f.write(incar)
        write_kpoints(dirpath, kmesh=kmesh)
        all_calcs.append((dirname, [el_a, el_b]))

        # AF (magnetic pairs only)
        if include_af:
            has_mag = (el_a in MAGNETIC_ELEMENTS or
                       el_b in MAGNETIC_ELEMENTS)
            if has_mag:
                dirname_af = f"AF/{el_a}{n_each}{el_b}{n_each}_AF"
                dirpath_af = os.path.join(base_dir, dirname_af)
                os.makedirs(dirpath_af, exist_ok=True)

                if sqs_type == "BCC":
                    magmom_af = make_magmom_sqs_bcc(
                        el_a, el_b, is_af=True)
                    system_name_af = (
                        f"BCC-SQS {el_a}{n_each}{el_b}{n_each} "
                        f"AF recalc")
                    incar_af = make_incar_sqs(
                        system_name_af, n_atoms, magmom_af, is_af=True)
                    write_poscar_sqs_bcc(
                        dirpath_af, el_a, el_b, a_super)
                else:
                    magmom_af = make_magmom_sqs_fcc(
                        el_a, el_b, is_af=True)
                    system_name_af = (
                        f"FCC-SQS {el_a}{n_each}{el_b}{n_each} "
                        f"AF recalc")
                    incar_af = make_incar_sqs(
                        system_name_af, n_atoms, magmom_af, is_af=True)
                    write_poscar_sqs_fcc(
                        dirpath_af, el_a, el_b, a_super)

                with open(os.path.join(dirpath_af, "INCAR"), "w") as f:
                    f.write(incar_af)
                write_kpoints(dirpath_af, kmesh=kmesh)
                all_calcs.append((dirname_af, [el_a, el_b]))

    n_ref = len(ref_elements)
    n_fm = sum(1 for d, _ in all_calcs
               if "FM/" in d and d not in
               [f"FM/{e}{n_atoms}" for e in ref_elements])
    n_af = sum(1 for d, _ in all_calcs if "AF/" in d)
    print(f"\n  References: {n_ref}")
    print(f"  Binary FM: {n_fm}")
    print(f"  Binary AF: {n_af}")
    print(f"  Total: {len(all_calcs)}")

    generate_potcar_script(base_dir, all_calcs)
    generate_run_script(base_dir, all_calcs)
    generate_extract_script_sqs(base_dir, all_calcs, sqs_type)
    generate_slurm_jobscript(
        base_dir, f"SQS_{sqs_type}_recalc", len(all_calcs))
    generate_pbs_jobscript(
        base_dir, f"SQS_{sqs_type}_recalc", len(all_calcs))

    return all_calcs


def print_settings_comparison():
    """Print comparison of old vs new settings."""
    print()
    print("=" * 70)
    print("VASP計算設定の比較 (旧 → 新)")
    print("=" * 70)
    print(f"{'設定項目':<20} {'旧L1₂':>12} {'旧SQS':>12} "
          f"{'新(共通)':>12}")
    print("-" * 70)
    print(f"{'ENCUT (eV)':<20} {'320':>12} {'320':>12} {'520':>12}")
    print(f"{'k-mesh (L1₂)':<20} {'6×6×6':>12} {'—':>12} "
          f"{'12×12×12':>12}")
    print(f"{'k-mesh (BCC-SQS)':<20} {'—':>12} {'4×4×4':>12} "
          f"{'6×6×6':>12}")
    print(f"{'k-mesh (FCC-SQS)':<20} {'—':>12} {'—':>12} "
          f"{'4×4×4':>12}")
    print(f"{'SIGMA (eV)':<20} {'0.2':>12} {'N/A':>12} {'0.1':>12}")
    print(f"{'EDIFFG (eV/Å)':<20} {'-0.01':>12} {'-0.01':>12} "
          f"{'-0.005':>12}")
    print(f"{'NELM':<20} {'200':>12} {'60':>12} {'300':>12}")
    print(f"{'NSW':<20} {'100':>12} {'120':>12} {'200':>12}")
    print(f"{'MAGMOM':<20} {'(default)':>12} {'(default)':>12} "
          f"{'明示設定':>12}")
    print(f"{'AF構成':<20} {'なし':>12} {'なし':>12} {'あり':>12}")
    print("=" * 70)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="L1₂/SQS精度向上VASP再計算入力ファイル生成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 全41元素の全ペア（L1₂ + BCC-SQS）
  python generate_l12_sqs_recalc.py

  # L1₂のみ、磁性元素ペアのみ
  python generate_l12_sqs_recalc.py --mode l12 --magnetic-only

  # 特定元素のみ
  python generate_l12_sqs_recalc.py --elements "Fe,Co,Ni,Cr,Mn"

  # AF構成のみ（FM既計算の場合）
  python generate_l12_sqs_recalc.py --af-only --magnetic-only

  # FCC-SQSも含める
  python generate_l12_sqs_recalc.py --mode all --include-fcc-sqs
""")

    parser.add_argument(
        "--mode", choices=["l12", "sqs", "all"], default="all",
        help="生成モード: l12=L1₂のみ, sqs=SQSのみ, all=両方")
    parser.add_argument(
        "--elements", type=str, default=None,
        help='対象元素 (カンマ区切り, 例: "Fe,Co,Ni,Cr,Mn")')
    parser.add_argument(
        "--magnetic-only", action="store_true",
        help="磁性元素(Cr,Mn,Fe,Co,Ni)を含むペアのみ")
    parser.add_argument(
        "--af-only", action="store_true",
        help="AF構成のみ生成（FMは生成しない）")
    parser.add_argument(
        "--no-af", action="store_true",
        help="AF構成を生成しない（FMのみ）")
    parser.add_argument(
        "--include-fcc-sqs", action="store_true",
        help="FCC-SQS（32原子）も生成（デフォルトはBCC-SQSのみ）")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="ファイルを生成せずに計算数のみ表示")

    args = parser.parse_args()

    # Determine target elements
    if args.elements:
        target = sorted(args.elements.split(","))
        missing = [e for e in target if e not in ALL_ELEMENTS]
        if missing:
            print(f"Error: Unknown elements: {missing}")
            print(f"Available: {ALL_ELEMENTS}")
            sys.exit(1)
    else:
        target = ALL_ELEMENTS

    include_af = not args.no_af

    print_settings_comparison()

    if args.dry_run:
        pairs = list(itertools.combinations(target, 2))
        if args.magnetic_only:
            pairs = [p for p in pairs
                     if p[0] in MAGNETIC_ELEMENTS
                     or p[1] in MAGNETIC_ELEMENTS]
        mag_pairs = [p for p in pairs
                     if p[0] in MAGNETIC_ELEMENTS
                     or p[1] in MAGNETIC_ELEMENTS]

        print(f"対象元素: {len(target)}")
        print(f"元素ペア: {len(pairs)}")
        print(f"  うち磁性: {len(mag_pairs)}")
        print()

        if args.mode in ("l12", "all"):
            n_l12_fm = len(pairs) * 2
            n_l12_af = len(mag_pairs) * 2 if include_af else 0
            print(f"L1₂: FM={n_l12_fm}, AF={n_l12_af}, "
                  f"計={n_l12_fm + n_l12_af}")

        if args.mode in ("sqs", "all"):
            n_ref = len(set(e for p in pairs for e in p))
            n_sqs_fm = len(pairs) + n_ref
            n_sqs_af = len(mag_pairs) if include_af else 0
            print(f"BCC-SQS: FM={n_sqs_fm} (含ref {n_ref}), "
                  f"AF={n_sqs_af}, 計={n_sqs_fm + n_sqs_af}")

            if args.include_fcc_sqs:
                print(f"FCC-SQS: FM={n_sqs_fm} (含ref {n_ref}), "
                      f"AF={n_sqs_af}, 計={n_sqs_fm + n_sqs_af}")

        sys.exit(0)

    total_calcs = 0

    if args.mode in ("l12", "all"):
        l12_calcs = generate_l12_recalc(
            target, include_af=include_af,
            magnetic_only=args.magnetic_only)
        total_calcs += len(l12_calcs)

    if args.mode in ("sqs", "all"):
        sqs_bcc_calcs = generate_sqs_recalc(
            target, sqs_type="BCC", include_af=include_af,
            magnetic_only=args.magnetic_only)
        total_calcs += len(sqs_bcc_calcs)

        if args.include_fcc_sqs:
            sqs_fcc_calcs = generate_sqs_recalc(
                target, sqs_type="FCC", include_af=include_af,
                magnetic_only=args.magnetic_only)
            total_calcs += len(sqs_fcc_calcs)

    print()
    print("=" * 60)
    print(f"合計: {total_calcs} VASP計算")
    print()
    print("次のステップ:")
    print("  1. make_potcar.sh を実行してPOTCARを生成")
    print("  2. job_slurm.sh または job_pbs.sh でジョブ投入")
    print("  3. 完了後 extract_results.sh で結果をCSV出力")
    print("=" * 60)


if __name__ == "__main__":
    main()
