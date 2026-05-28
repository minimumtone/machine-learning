#!/usr/bin/env python3
"""
全38元素BCC-SQS計算入力ファイル生成スクリプト（ISIF=7: 立方体制約）。

16原子BCC-SQS (2×2×2) スーパーセル、ISIF=7（体積のみ緩和、セル形状固定）。
立方晶を維持することで V = a³/16 の仮定が正確になる。

出力構造:
    BCC_SQS_ISIF7/
    ├── Ag8Al8/  (INCAR, POSCAR, KPOINTS)
    ├── Ag8Ag8/  (同種参照)
    └── ...

使い方:
    python generate_sqs_isif7.py [--output-dir BCC_SQS_ISIF7] [--dry-run]

    # POTCAR生成 + 計算実行:
    cd BCC_SQS_ISIF7
    bash make_potcar.sh   # requires $VASP_PP_PATH
    bash run_all.sh       # requires $VASPBIN

環境変数:
    VASP_PP_PATH : VASPポテンシャルルートディレクトリ (POTCAR: $VASP_PP_PATH/potpaw_PBE/{element}/POTCAR)
    VASPBIN      : VASP実行コマンド
"""

import os
import argparse
import itertools

# =====================================================================
# 全38ターゲット元素
# =====================================================================
ALL_ELEMENTS = sorted([
    'Ag', 'Al', 'Au', 'Be', 'Ca', 'Co', 'Cr', 'Cu', 'Dy', 'Er',
    'Fe', 'Ge', 'Hf', 'Ir', 'La', 'Mg', 'Mn', 'Mo', 'Nb', 'Ni',
    'Os', 'Pb', 'Pd', 'Pt', 'Re', 'Rh', 'Ru', 'Sc', 'Si', 'Sn',
    'Ta', 'Tb', 'Ti', 'V',  'W',  'Y',  'Zn', 'Zr',
])

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
# SQS-16 BCC配置 (最適化済み α_1nn ≈ α_2nn ≈ α_3nn ≈ 0)
# BCC 2×2×2 = 16原子, 0=元素A, 1=元素B
# =====================================================================
SQS_OCCUPATION = [0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0]

BCC_2x2x2_POSITIONS = []
for _ix in range(2):
    for _iy in range(2):
        for _iz in range(2):
            BCC_2x2x2_POSITIONS.append(
                (_ix / 2.0, _iy / 2.0, _iz / 2.0))
            BCC_2x2x2_POSITIONS.append(
                ((_ix + 0.5) / 2.0, (_iy + 0.5) / 2.0, (_iz + 0.5) / 2.0))


# =====================================================================
# VASP input writers
# =====================================================================
def write_incar(dirpath):
    """INCAR for BCC-SQS (16 atoms, ISIF=7: volume-only, cubic shape preserved)."""
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


def write_poscar(dirpath, el_a, el_b, a_super):
    """POSCAR for BCC SQS-16 (2×2×2, A8B8)."""
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
        lines.append("")
    else:
        pos_a = [BCC_2x2x2_POSITIONS[i] for i, o in enumerate(SQS_OCCUPATION) if o == 0]
        pos_b = [BCC_2x2x2_POSITIONS[i] for i, o in enumerate(SQS_OCCUPATION) if o == 1]
        n_a, n_b = len(pos_a), len(pos_b)

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


def write_kpoints(dirpath):
    """KPOINTS: 4x4x4 Gamma-centered (16原子セル用)."""
    content = """\
Automatic mesh
0
Gamma
  4 4 4
  0 0 0
"""
    with open(os.path.join(dirpath, "KPOINTS"), "w") as f:
        f.write(content)


def main():
    parser = argparse.ArgumentParser(
        description='全38元素BCC-SQS (ISIF=7) VASP入力ファイル生成')
    parser.add_argument('--output-dir', '-o', default='BCC_SQS_ISIF7',
                        help='出力ディレクトリ (default: BCC_SQS_ISIF7)')
    parser.add_argument('--dry-run', action='store_true',
                        help='生成内容を表示のみ')
    args = parser.parse_args()

    # 全ペア（対称: A8B8 = B8A8 なので片方のみ）+ 同種参照
    pairs = []
    for el_a, el_b in itertools.combinations(ALL_ELEMENTS, 2):
        pairs.append((el_a, el_b))
    for el in ALL_ELEMENTS:
        pairs.append((el, el))

    n_hetero = len(list(itertools.combinations(ALL_ELEMENTS, 2)))
    n_homo = len(ALL_ELEMENTS)
    n_total = len(pairs)

    print("=" * 60)
    print("BCC-SQS (ISIF=7) VASP入力ファイル生成")
    print("=" * 60)
    print(f"元素数: {len(ALL_ELEMENTS)}")
    print(f"生成ペア数: {n_total} (ヘテロ: {n_hetero}, 同種: {n_homo})")
    print(f"出力先: {args.output_dir}/")
    print(f"ISIF=7: 体積のみ緩和、立方晶維持")
    print(f"セルサイズ: 16原子 (BCC 2×2×2)")

    if args.dry_run:
        print(f"\n[DRY RUN] 生成予定:")
        for a, b in sorted(pairs):
            dirname = f"{a}8{b}8"
            a_super = 2.0 * 0.5 * (ELEMENT_A0_BCC[a] + ELEMENT_A0_BCC[b])
            skip = ""
            contcar = os.path.join(args.output_dir, dirname, "CONTCAR")
            if os.path.isfile(contcar) and os.path.getsize(contcar) > 0:
                skip = " [SKIP: CONTCAR exists]"
            print(f"  {dirname}  a_super={a_super:.4f}{skip}")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    created = 0
    skipped = 0
    for a, b in sorted(pairs):
        dirname = f"{a}8{b}8"
        dirpath = os.path.join(args.output_dir, dirname)

        # CONTCAR存在チェック（計算済みスキップ）
        contcar = os.path.join(dirpath, "CONTCAR")
        if os.path.isfile(contcar) and os.path.getsize(contcar) > 0:
            skipped += 1
            continue

        os.makedirs(dirpath, exist_ok=True)
        a_super = 2.0 * 0.5 * (ELEMENT_A0_BCC[a] + ELEMENT_A0_BCC[b])
        write_incar(dirpath)
        write_poscar(dirpath, a, b, a_super)
        write_kpoints(dirpath)
        created += 1

    print(f"\n作成: {created} ディレクトリ (スキップ: {skipped})")

    # =====================================================================
    # make_potcar.sh
    # =====================================================================
    potcar_script = os.path.join(args.output_dir, "make_potcar.sh")
    with open(potcar_script, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# POTCAR生成スクリプト (BCC_SQS_ISIF7)\n")
        f.write("# Usage: bash make_potcar.sh  (requires $VASP_PP_PATH)\n")
        f.write("# POTCARパス: $VASP_PP_PATH/potpaw_PBE/{element}/POTCAR\n\n")
        f.write('if [ -z "$VASP_PP_PATH" ]; then\n')
        f.write('    echo "ERROR: VASP_PP_PATH not set"\n')
        f.write('    exit 1\n')
        f.write('fi\n\n')
        f.write('SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"\n\n')

        for a, b in sorted(pairs):
            dirname = f"{a}8{b}8"
            va = POTCAR_VARIANTS[a]
            vb = POTCAR_VARIANTS[b]
            if a == b:
                f.write(f'cat "$VASP_PP_PATH/potpaw_PBE/{va}/POTCAR" '
                        f'> "$SCRIPT_DIR/{dirname}/POTCAR"\n')
            else:
                f.write(f'cat "$VASP_PP_PATH/potpaw_PBE/{va}/POTCAR" '
                        f'"$VASP_PP_PATH/potpaw_PBE/{vb}/POTCAR" '
                        f'> "$SCRIPT_DIR/{dirname}/POTCAR"\n')

        f.write(f'\necho "POTCAR generated for {n_total} directories"\n')
    os.chmod(potcar_script, 0o755)

    # =====================================================================
    # run_all.sh
    # =====================================================================
    run_script = os.path.join(args.output_dir, "run_all.sh")
    with open(run_script, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# BCC_SQS_ISIF7 全計算実行スクリプト\n")
        f.write("# Usage: VASPBIN=/path/to/vasp bash run_all.sh\n\n")
        f.write('if [ -z "$VASPBIN" ]; then\n')
        f.write('    echo "ERROR: VASPBIN not set"\n')
        f.write('    exit 1\n')
        f.write('fi\n\n')
        f.write('SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"\n')
        f.write(f'TOTAL={n_total}\n')
        f.write('COUNT=0\n')
        f.write('SKIP=0\n')
        f.write('FAIL=0\n\n')

        for a, b in sorted(pairs):
            dirname = f"{a}8{b}8"
            f.write(f'COUNT=$((COUNT+1))\n')
            f.write(f'echo "[$COUNT/$TOTAL] {dirname}"\n')
            f.write(f'cd "$SCRIPT_DIR/{dirname}"\n')
            f.write(f'if [ -s CONTCAR ]; then\n')
            f.write(f'    echo "  Skip (CONTCAR exists)"\n')
            f.write(f'    SKIP=$((SKIP+1))\n')
            f.write(f'else\n')
            f.write(f'    $VASPBIN > vasp.log 2>&1\n')
            f.write(f'    if [ $? -ne 0 ]; then\n')
            f.write(f'        echo "  FAILED"\n')
            f.write(f'        FAIL=$((FAIL+1))\n')
            f.write(f'    fi\n')
            f.write(f'fi\n\n')

        f.write('echo "========================================"\n')
        f.write('echo "Complete: $COUNT/$TOTAL"\n')
        f.write('echo "  Skipped: $SKIP"\n')
        f.write('echo "  Failed:  $FAIL"\n')
    os.chmod(run_script, 0o755)

    print(f"\nスクリプト生成:")
    print(f"  {potcar_script}")
    print(f"  {run_script}")

    print(f"""
実行手順:
  1. POTCAR生成:
       cd {args.output_dir} && bash make_potcar.sh  # requires $VASP_PP_PATH

  2. 全計算実行:
       cd {args.output_dir} && VASPBIN=/path/to/vasp bash run_all.sh

  3. 結果抽出:
       python vasp_inputs/reanalyze_all.py /path/to/DATA_ROOT --sqs-dir BCC_SQS_ISIF7
""")


if __name__ == '__main__':
    main()
