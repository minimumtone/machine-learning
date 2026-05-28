#!/usr/bin/env python3
"""
不足11元素のB2構造VASP計算入力ファイル生成スクリプト。

不足元素: Au, Be, Dy, Er, Ir, Os, Pb, Pt, Re, Rh, Tb
これら11元素と全38元素との二元B2ペア（両方向）+ 同種参照を生成。

出力構造:
    BCC_B2/
    ├── Au1Ag1/  (INCAR, POSCAR, KPOINTS)
    ├── Ag1Au1/
    ├── Au1Au1/  (同種参照)
    └── ...

使い方:
    python generate_b2_missing.py [--output-dir BCC_B2] [--dry-run]

    # POTCAR生成:
    cd BCC_B2 && bash make_potcar.sh

    # 全計算実行:
    cd BCC_B2 && bash run_all.sh

環境変数:
    VASP_PP_PATH : PAW-PBEポテンシャルディレクトリへのパス
    VASPBIN      : VASP実行コマンド
"""

import os
import argparse

# =====================================================================
# 全38ターゲット元素
# =====================================================================
ALL_ELEMENTS = sorted([
    'Ag', 'Al', 'Au', 'Be', 'Ca', 'Co', 'Cr', 'Cu', 'Dy', 'Er',
    'Fe', 'Ge', 'Hf', 'Ir', 'La', 'Mg', 'Mn', 'Mo', 'Nb', 'Ni',
    'Os', 'Pb', 'Pd', 'Pt', 'Re', 'Rh', 'Ru', 'Sc', 'Si', 'Sn',
    'Ta', 'Tb', 'Ti', 'V',  'W',  'Y',  'Zn', 'Zr',
])

# 不足11元素
MISSING_ELEMENTS = sorted([
    'Au', 'Be', 'Dy', 'Er', 'Ir', 'Os', 'Pb', 'Pt', 'Re', 'Rh', 'Tb'
])

# 既存27元素
EXISTING_ELEMENTS = sorted(set(ALL_ELEMENTS) - set(MISSING_ELEMENTS))

# =====================================================================
# BCC格子定数 (Å) — 初期構造用
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
# VASP input writers
# =====================================================================
def write_incar(dirpath):
    """INCAR for B2 (2 atoms, ENCUT=520, 12x12x12 k-mesh)."""
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


def write_poscar(dirpath, el_corner, el_body, a0):
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


def write_kpoints(dirpath):
    """KPOINTS: 12x12x12 Gamma-centered."""
    content = """\
Automatic mesh
0
Gamma
  12 12 12
  0  0  0
"""
    with open(os.path.join(dirpath, "KPOINTS"), "w") as f:
        f.write(content)


def main():
    parser = argparse.ArgumentParser(
        description='不足11元素のB2構造VASP入力ファイル生成')
    parser.add_argument('--output-dir', '-o', default='BCC_B2',
                        help='出力ディレクトリ (default: BCC_B2)')
    parser.add_argument('--dry-run', action='store_true',
                        help='ディレクトリ一覧のみ表示')
    args = parser.parse_args()

    # Generate pair list:
    # 1. missing × missing (both directions + homo)
    # 2. missing × existing (both directions)
    pairs = []

    # Homo references for missing elements
    for el in MISSING_ELEMENTS:
        pairs.append((el, el))

    # Missing × all (both directions, excluding homo already added)
    for m in MISSING_ELEMENTS:
        for e in ALL_ELEMENTS:
            if m == e:
                continue
            pairs.append((m, e))  # missing as corner
            pairs.append((e, m))  # missing as body

    # Remove duplicates (e.g. Au-Be and Be-Au as corner/body are different,
    # but Be-Au appears twice if both are missing)
    seen = set()
    unique_pairs = []
    for a, b in pairs:
        key = (a, b)
        if key not in seen:
            seen.add(key)
            unique_pairs.append((a, b))
    pairs = unique_pairs

    n_homo = sum(1 for a, b in pairs if a == b)
    n_hetero = len(pairs) - n_homo

    print("=" * 60)
    print("B2不足元素 VASP入力ファイル生成")
    print("=" * 60)
    print(f"不足元素 ({len(MISSING_ELEMENTS)}): {', '.join(MISSING_ELEMENTS)}")
    print(f"生成ペア数: {len(pairs)} (ヘテロ: {n_hetero}, 同種: {n_homo})")
    print(f"出力先: {args.output_dir}/")
    print()

    if args.dry_run:
        print("--- Dry run ---")
        for a, b in sorted(pairs):
            dirname = f"{a}1{b}1"
            a0 = (ELEMENT_A0_BCC[a] + ELEMENT_A0_BCC[b]) / 2
            print(f"  {dirname}/  (a0={a0:.3f} Å)")
        print(f"\n合計: {len(pairs)} ディレクトリ")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # Generate input files
    created = 0
    skipped = 0
    for a, b in pairs:
        dirname = f"{a}1{b}1"
        dirpath = os.path.join(args.output_dir, dirname)

        if os.path.isdir(dirpath):
            # Check if already has results
            contcar = os.path.join(dirpath, "CONTCAR")
            if os.path.isfile(contcar) and os.path.getsize(contcar) > 0:
                skipped += 1
                continue

        os.makedirs(dirpath, exist_ok=True)

        a0 = (ELEMENT_A0_BCC[a] + ELEMENT_A0_BCC[b]) / 2
        write_incar(dirpath)
        write_poscar(dirpath, a, b, a0)
        write_kpoints(dirpath)
        created += 1

    print(f"作成: {created} ディレクトリ")
    if skipped:
        print(f"スキップ (結果あり): {skipped}")

    # Generate make_potcar.sh
    potcar_script = os.path.join(args.output_dir, "make_potcar.sh")
    with open(potcar_script, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# POTCAR生成スクリプト\n")
        f.write("# Usage: VASP_PP_PATH=/path/to/vasp_pp bash make_potcar.sh\n# POTCARパス: $VASP_PP_PATH/PBE/{element}/POTCAR\n\n")
        f.write('if [ -z "$VASP_PP_PATH" ]; then\n')
        f.write('    echo "ERROR: VASP_PP_PATH not set"\n')
        f.write('    exit 1\n')
        f.write('fi\n\n')
        f.write('SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"\n\n')

        for a, b in sorted(pairs):
            dirname = f"{a}1{b}1"
            va = POTCAR_VARIANTS[a]
            vb = POTCAR_VARIANTS[b]
            if a == b:
                f.write(f'cat "$VASP_PP_PATH/PBE/{va}/POTCAR" '
                        f'> "$SCRIPT_DIR/{dirname}/POTCAR"\n')
            else:
                f.write(f'cat "$VASP_PP_PATH/PBE/{va}/POTCAR" '
                        f'"$VASP_PP_PATH/PBE/{vb}/POTCAR" '
                        f'> "$SCRIPT_DIR/{dirname}/POTCAR"\n')

        f.write(f'\necho "POTCAR generated for {len(pairs)} directories"\n')
    os.chmod(potcar_script, 0o755)

    # Generate run_all.sh
    run_script = os.path.join(args.output_dir, "run_all.sh")
    with open(run_script, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# 全B2計算実行スクリプト\n")
        f.write("# Usage: VASPBIN=/path/to/vasp bash run_all.sh\n\n")
        f.write('if [ -z "$VASPBIN" ]; then\n')
        f.write('    echo "ERROR: VASPBIN not set"\n')
        f.write('    exit 1\n')
        f.write('fi\n\n')
        f.write('SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"\n')
        f.write(f'TOTAL={len(pairs)}\n')
        f.write('COUNT=0\n')
        f.write('FAILED=0\n\n')

        for a, b in sorted(pairs):
            dirname = f"{a}1{b}1"
            f.write(f'# --- {dirname} ---\n')
            f.write(f'COUNT=$((COUNT + 1))\n')
            f.write(f'echo "[$COUNT/$TOTAL] {dirname}"\n')
            f.write(f'cd "$SCRIPT_DIR/{dirname}"\n')
            f.write(f'if [ -s CONTCAR ]; then\n')
            f.write(f'    echo "  Skip (CONTCAR exists)"\n')
            f.write(f'else\n')
            f.write(f'    $VASPBIN > vasp.log 2>&1\n')
            f.write(f'    if [ $? -ne 0 ]; then\n')
            f.write(f'        echo "  FAILED"\n')
            f.write(f'        FAILED=$((FAILED + 1))\n')
            f.write(f'    fi\n')
            f.write(f'fi\n\n')

        f.write('echo ""\n')
        f.write('echo "========================================"\n')
        f.write(f'echo "Complete: $((COUNT - FAILED))/$TOTAL succeeded"\n')
        f.write('if [ $FAILED -gt 0 ]; then\n')
        f.write('    echo "Failed: $FAILED"\n')
        f.write('fi\n')
    os.chmod(run_script, 0o755)

    print(f"\nスクリプト生成:")
    print(f"  {potcar_script}")
    print(f"  {run_script}")

    print(f"""
実行手順:
  1. POTCAR生成:
       cd {args.output_dir} && VASP_PP_PATH=/path/to/potpaw_PBE bash make_potcar.sh

  2. 全計算実行:
       cd {args.output_dir} && VASPBIN=/path/to/vasp bash run_all.sh

  3. 結果抽出:
       python vasp_inputs/reanalyze_all.py /path/to/DATA_ROOT
""")


if __name__ == '__main__':
    main()
