#!/usr/bin/env python3
"""
既存BCC_SQSディレクトリのINCARをISIF=7（立方体制約）に更新するスクリプト。

ISIF=3（全セル緩和）→ ISIF=7（体積のみ緩和、セル形状固定）に変更。
SQSスーパーセルは対称性が破れているため、ISIF=3では非立方晶に歪む。
HEAは巨視的に立方晶であるため、ISIF=7で立方晶を維持する方が物理的に妥当。

使い方:
    python update_sqs_incar.py /path/to/BCC_SQS
    python update_sqs_incar.py /path/to/FCC_SQS

    # CONTCAR/OSZICARを削除して再計算:
    python update_sqs_incar.py /path/to/BCC_SQS --clean
"""

import os
import argparse
import glob


INCAR_CONTENT = """\
SYSTEM = SQS structure optimization (cubic constraint)

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

CLEAN_FILES = [
    "CONTCAR", "OSZICAR", "OUTCAR", "EIGENVAL", "DOSCAR",
    "CHG", "CHGCAR", "PCDAT", "REPORT", "XDATCAR",
    "WAVECAR", "IBZKPT", "vasprun.xml", "vasp.log",
]


def main():
    parser = argparse.ArgumentParser(
        description='SQSディレクトリのINCARをISIF=7に更新')
    parser.add_argument('sqs_dir', help='BCC_SQS or FCC_SQS ディレクトリ')
    parser.add_argument('--clean', action='store_true',
                        help='CONTCAR/OSZICAR等の結果ファイルも削除（再計算用）')
    parser.add_argument('--dry-run', action='store_true',
                        help='変更内容を表示のみ')
    args = parser.parse_args()

    if not os.path.isdir(args.sqs_dir):
        print(f"ERROR: ディレクトリが見つかりません: {args.sqs_dir}")
        return

    subdirs = sorted([
        d for d in os.listdir(args.sqs_dir)
        if os.path.isdir(os.path.join(args.sqs_dir, d))
        and os.path.isfile(os.path.join(args.sqs_dir, d, "INCAR"))
    ])

    if not subdirs:
        print(f"INCARを含むサブディレクトリが見つかりません: {args.sqs_dir}")
        return

    print(f"対象: {len(subdirs)} ディレクトリ in {args.sqs_dir}")

    updated = 0
    already_isif7 = 0
    cleaned = 0

    for subdir in subdirs:
        dirpath = os.path.join(args.sqs_dir, subdir)
        incar_path = os.path.join(dirpath, "INCAR")

        with open(incar_path) as f:
            content = f.read()

        if "ISIF" in content and "ISIF   = 7" in content or "ISIF = 7" in content:
            already_isif7 += 1
            continue

        if args.dry_run:
            print(f"  [更新] {subdir}/INCAR")
        else:
            with open(incar_path, "w") as f:
                f.write(INCAR_CONTENT)
            updated += 1

        if args.clean:
            for fname in CLEAN_FILES:
                fpath = os.path.join(dirpath, fname)
                if os.path.isfile(fpath):
                    if args.dry_run:
                        print(f"         削除: {fname}")
                    else:
                        os.remove(fpath)
                        cleaned += 1

    print(f"\n結果:")
    print(f"  INCAR更新: {updated}")
    print(f"  既にISIF=7: {already_isif7}")
    if args.clean:
        print(f"  結果ファイル削除: {cleaned}")

    if not args.dry_run and updated > 0:
        print(f"\n再計算を実行してください:")
        print(f"  cd {args.sqs_dir} && bash run_all.sh")


if __name__ == '__main__':
    main()
