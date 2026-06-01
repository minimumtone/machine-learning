#!/usr/bin/env python3
"""
VASP計算結果の一括スキャン・抽出・再解析スクリプト。

リモート環境のVASP計算ディレクトリをスキャンし、
BCC_B2/A1B1、FCC_L12/A3B1、BCC_SQS/A8B8の全結果を抽出して
Ω_sf計算→HEA格子定数予測まで一気に実行する。

ディレクトリ構造:
    DATA_ROOT/
    ├── BCC_B2/
    │   ├── Ag1Al1/  (CONTCAR, OUTCAR, ...)
    │   ├── Al1Ag1/
    │   └── ...
    ├── FCC_L12/
    │   ├── Ag3Al1/
    │   ├── Al3Ag1/
    │   └── ...
    └── BCC_SQS/
        ├── Ag8Al8/
        └── ...

使い方:
    python reanalyze_all.py /path/to/DATA_ROOT

    オプション:
      -o, --output-dir   出力ディレクトリ (default: ./reanalysis_output)
      --hea-csv          HEAデータCSV (default: data/hea_dataset.csv)
      --skip-hea         HEA予測を省略

出力:
    reanalysis_output/
    ├── compounds_VASP_B2.csv      B2抽出データ
    ├── compounds_VASP_L12.csv     L12抽出データ
    ├── compounds_VASP_SQS.csv     SQS抽出データ
    ├── coverage_report.txt        カバレッジ詳細
    ├── omega_sf_B2.csv            B2参照のΩ_sf
    ├── omega_sf_L12.csv           L12参照のΩ_sf
    ├── omega_sf_SQS.csv           SQS参照のΩ_sf
    ├── hea_predictions.csv        HEA予測結果
    └── full_report.md             総合レポート
"""

import os
import re
import sys
import csv
import math
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# =====================================================================
# Target 38 elements (Gd, Ce excluded for 4f instability)
# =====================================================================
ALL_ELEMENTS = sorted([
    'Ag', 'Al', 'Au', 'Be', 'Ca', 'Co', 'Cr', 'Cu', 'Dy', 'Er',
    'Fe', 'Ge', 'Hf', 'Ir', 'La', 'Mg', 'Mn', 'Mo', 'Nb', 'Ni',
    'Os', 'Pb', 'Pd', 'Pt', 'Re', 'Rh', 'Ru', 'Sc', 'Si', 'Sn',
    'Ta', 'Tb', 'Ti', 'V',  'W',  'Y',  'Zn', 'Zr',
])

# VASP GGA-PBE DFT atomic volumes (Å³) from B2 homo-pair structures.
# These are NOT King (1966) experimental values — they are DFT-computed
# volumes from the same VASP B2 calculations used for Ω_sf, providing
# DFT self-consistent reference for the Ω_sf ratio (GGA errors cancel).
VASP_ATOMIC_VOLUMES = {
    "Ag": 17.840, "Al": 16.602, "Au": 17.798, "Be": 8.105,
    "Ca": 42.025, "Co": 10.994, "Cr": 11.415, "Cu": 12.024,
    "Dy": 31.744, "Er": 31.063, "Fe": 11.312, "Ge": 19.243,
    "Hf": 22.068, "Ir": 14.334, "La": 37.591, "Mg": 22.909,
    "Mn": 10.855, "Mo": 15.629, "Nb": 18.370, "Ni": 10.941,
    "Os": 13.776, "Pb": 30.596, "Pd": 15.466, "Pt": 15.219,
    "Re": 14.875, "Rh": 13.761, "Ru": 14.136, "Sc": 24.869,
    "Si": 14.822, "Sn": 27.611, "Ta": 18.159, "Tb": 32.503,
    "Ti": 17.022, "V":  13.275, "W":  15.960, "Y":  33.017,
    "Zn": 15.741, "Zr": 22.721,
}

# King (1966) experimental atomic volumes (Å³) — room-temperature stable structures.
# Used for Vegard baseline V_i in HEA lattice prediction (NOT for Ω_sf reference).
KING_ATOMIC_VOLUMES = {
    "Ag": 17.061, "Al": 16.602, "Au": 16.966, "Be": 8.111,
    "Ca": 43.630, "Co": 11.073, "Cr": 12.008, "Cu": 11.810,
    "Dy": 31.540, "Er": 30.660, "Fe": 11.776, "Ge": 22.634,
    "Hf": 22.312, "Ir": 14.155, "La": 37.168, "Mg": 23.240,
    "Mn": 12.210, "Mo": 15.583, "Nb": 17.978, "Ni": 10.941,
    "Os": 13.977, "Pb": 30.321, "Pd": 14.716, "Pt": 15.095,
    "Re": 14.712, "Rh": 13.754, "Ru": 13.571, "Sc": 24.987,
    "Si": 20.024, "Sn": 27.053, "Ta": 18.014, "Tb": 32.090,
    "Ti": 17.649, "V":  13.824, "W":  15.850, "Y":  33.018,
    "Zn": 15.207, "Zr": 23.279,
}


# =====================================================================
# VASP output parsing
# =====================================================================
def parse_dirname(dirname):
    """Parse directory name like 'Ag1Al1' or 'Ag8Al8' into components."""
    m = re.match(r'^([A-Z][a-z]?)(\d+)([A-Z][a-z]?)(\d+)$', dirname)
    if not m:
        return None
    return m.group(1), int(m.group(2)), m.group(3), int(m.group(4))


def read_lattice_constant(calc_dir):
    """
    Read lattice constant from CONTCAR (preferred) or POSCAR.
    For cubic structures, returns the cube root of the cell volume
    normalized to appropriate supercell.
    Returns (lattice_constant, converged, source_file) or (None, False, None).
    """
    contcar = os.path.join(calc_dir, 'CONTCAR')
    poscar = os.path.join(calc_dir, 'POSCAR')
    outcar = os.path.join(calc_dir, 'OUTCAR')

    source = None
    if os.path.isfile(contcar) and os.path.getsize(contcar) > 0:
        source = contcar
    elif os.path.isfile(poscar):
        source = poscar
    else:
        return None, False, None

    try:
        with open(source) as f:
            lines = f.readlines()
        if len(lines) < 6:
            return None, False, source

        scale = float(lines[1].strip())
        a_vec = [float(x) for x in lines[2].split()]
        b_vec = [float(x) for x in lines[3].split()]
        c_vec = [float(x) for x in lines[4].split()]

        # Cell volume via triple product
        vol = abs(
            a_vec[0] * (b_vec[1]*c_vec[2] - b_vec[2]*c_vec[1]) -
            a_vec[1] * (b_vec[0]*c_vec[2] - b_vec[2]*c_vec[0]) +
            a_vec[2] * (b_vec[0]*c_vec[1] - b_vec[1]*c_vec[0])
        ) * abs(scale)**3

        # For cubic: a = scale * |a_vec|
        a_mag = (a_vec[0]**2 + a_vec[1]**2 + a_vec[2]**2)**0.5
        lattice_const = abs(scale) * a_mag

        if lattice_const < 1.5 or lattice_const > 15.0:
            return None, False, source

    except (ValueError, IndexError):
        return None, False, source

    # Check convergence
    converged = False
    if os.path.isfile(outcar):
        try:
            with open(outcar) as f:
                content = f.read()
            converged = 'reached required accuracy' in content
        except Exception:
            pass

    src_name = 'CONTCAR' if 'CONTCAR' in source else 'POSCAR'
    return lattice_const, converged, src_name


def read_energy_per_atom(calc_dir):
    """Read total energy per atom from OUTCAR."""
    outcar = os.path.join(calc_dir, 'OUTCAR')
    if not os.path.isfile(outcar):
        return None
    energy = None
    natoms = None
    try:
        with open(outcar) as f:
            for line in f:
                if 'free  energy   TOTEN' in line:
                    energy = float(line.split()[-2])
                elif 'number of ions' in line:
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if p == 'NIONS':
                            natoms = int(parts[i + 2])
                            break
    except Exception:
        return None
    if energy is not None and natoms is not None and natoms > 0:
        return energy / natoms
    return None


def read_cell_volume(calc_dir):
    """Read cell volume from CONTCAR/POSCAR."""
    contcar = os.path.join(calc_dir, 'CONTCAR')
    poscar = os.path.join(calc_dir, 'POSCAR')
    source = None
    if os.path.isfile(contcar) and os.path.getsize(contcar) > 0:
        source = contcar
    elif os.path.isfile(poscar):
        source = poscar
    else:
        return None

    try:
        with open(source) as f:
            lines = f.readlines()
        scale = float(lines[1].strip())
        a = [float(x) for x in lines[2].split()]
        b = [float(x) for x in lines[3].split()]
        c = [float(x) for x in lines[4].split()]
        vol = abs(
            a[0]*(b[1]*c[2] - b[2]*c[1]) -
            a[1]*(b[0]*c[2] - b[2]*c[0]) +
            a[2]*(b[0]*c[1] - b[1]*c[0])
        ) * abs(scale)**3
        return vol
    except Exception:
        return None


# =====================================================================
# Directory scanning
# =====================================================================
def scan_structure_dir(base_dir, struct_type, expected_counts):
    """
    Scan a structure directory for VASP results.
    
    Args:
        base_dir: path to e.g. BCC_B2/
        struct_type: "B2", "L12", or "SQS"
        expected_counts: dict of expected (count_A, count_B) per struct
    
    Returns:
        results: list of dicts with extracted data
        errors: list of error strings
        found_pairs: set of (elA, elB) pairs found
        no_result_dirs: list of dirname strings where CONTCAR/POSCAR was absent or unreadable
    """
    results = []
    errors = []
    found_pairs = set()
    no_result_dirs = []

    if not os.path.isdir(base_dir):
        errors.append(f"Directory not found: {base_dir}")
        return results, errors, set(), []

    subdirs = sorted(os.listdir(base_dir))
    for dirname in subdirs:
        calc_dir = os.path.join(base_dir, dirname)
        if not os.path.isdir(calc_dir):
            continue

        parsed = parse_dirname(dirname)
        if parsed is None:
            errors.append(f"Cannot parse: {dirname}")
            continue

        elA, cA, elB, cB = parsed
        a, converged, src = read_lattice_constant(calc_dir)
        e_per_atom = read_energy_per_atom(calc_dir)
        vol = read_cell_volume(calc_dir)

        if a is None:
            no_result_dirs.append(dirname)
            errors.append(f"No lattice constant: {dirname}")
            continue

        found_pairs.add((elA, elB))
        if elA != elB:
            found_pairs.add((elB, elA))

        results.append({
            'material_id': f'vasp-{dirname}',
            'formula': f"{elA}{cA}{elB}{cB}",
            'element_A': elA,
            'element_B': elB,
            'count_A': float(cA),
            'count_B': float(cB),
            'lattice_constant': a,
            'energy_per_atom': e_per_atom if e_per_atom else '',
            'energy_above_hull': '',
            'source': 'VASP',
            'structure_type': struct_type,
            'lattice_constant_calc': a,
            'converged': converged,
            'source_file': src,
            'dirname': dirname,
            'cell_volume': vol,
        })

    return results, errors, found_pairs, no_result_dirs


# =====================================================================
# Ω_sf computation
# =====================================================================
def compute_omega_sf_b2(results):
    """Compute Ω_sf from B2 data (A1B1 pairs, Z=2)."""
    pair_data = defaultdict(list)
    for r in results:
        elA, elB = r['element_A'], r['element_B']
        a = r['lattice_constant']
        if elA not in VASP_ATOMIC_VOLUMES or elB not in VASP_ATOMIC_VOLUMES:
            continue
        if elA == elB:
            continue
        vA = VASP_ATOMIC_VOLUMES[elA]
        vB = VASP_ATOMIC_VOLUMES[elB]
        v_actual = a**3 / 2  # Z=2 for B2
        v_vegard = (vA + vB) / 2
        omega = (v_actual - v_vegard) / v_vegard
        pair = tuple(sorted([elA, elB]))
        pair_data[pair].append(omega)

    omega_sf = {}
    for pair, vals in pair_data.items():
        omega_sf[pair] = sum(vals) / len(vals)
    return omega_sf


def compute_omega_sf_l12(results):
    """Compute Ω_sf from L12 data (A3B1 pairs, Z=4)."""
    pair_data = defaultdict(list)
    for r in results:
        elA, elB = r['element_A'], r['element_B']
        cA, cB = int(r['count_A']), int(r['count_B'])
        a = r['lattice_constant']
        if elA not in VASP_ATOMIC_VOLUMES or elB not in VASP_ATOMIC_VOLUMES:
            continue
        if elA == elB:
            continue
        vA = VASP_ATOMIC_VOLUMES[elA]
        vB = VASP_ATOMIC_VOLUMES[elB]
        total = cA + cB
        v_actual = a**3 / 4  # Z=4 for L12
        v_vegard = (cA * vA + cB * vB) / total
        omega = (v_actual - v_vegard) / v_vegard
        pair = tuple(sorted([elA, elB]))
        pair_data[pair].append(omega)

    omega_sf = {}
    for pair, vals in pair_data.items():
        omega_sf[pair] = sum(vals) / len(vals)
    return omega_sf


def compute_omega_sf_sqs(results):
    """Compute Ω_sf from SQS data (A8B8 pairs, Z=16)."""
    pair_data = defaultdict(list)
    for r in results:
        elA, elB = r['element_A'], r['element_B']
        a = r['lattice_constant']
        if elA not in VASP_ATOMIC_VOLUMES or elB not in VASP_ATOMIC_VOLUMES:
            continue
        if elA == elB:
            continue
        vA = VASP_ATOMIC_VOLUMES[elA]
        vB = VASP_ATOMIC_VOLUMES[elB]
        # SQS 2x2x2 BCC supercell: 16 atoms, a_supercell = 2 * a_bcc
        # V_supercell = a_sup^3; V_per_atom = a_sup^3 / 16
        # Alternatively: a_bcc = a_sup / 2, V_per_atom = a_bcc^3 / 2
        # Since the CONTCAR gives the supercell lattice constant:
        v_actual = a**3 / 16  # 16 atoms in 2x2x2 BCC supercell
        v_vegard = (vA + vB) / 2  # equimolar
        omega = (v_actual - v_vegard) / v_vegard
        pair = tuple(sorted([elA, elB]))
        pair_data[pair].append(omega)

    omega_sf = {}
    for pair, vals in pair_data.items():
        omega_sf[pair] = sum(vals) / len(vals)
    return omega_sf


# =====================================================================
# HEA prediction (simplified Eq.10 with q=1 for SQS)
# =====================================================================
def predict_hea_lattice(comp, struct, omega_sf, q=1.0):
    """
    Predict HEA lattice constant using DFT-Ω_sf model.
    comp: dict {element: fraction}
    struct: "BCC" or "FCC"
    omega_sf: {(elA, elB): Ω_sf}
    q: scaling constant (1.0 for SQS reference)
    """
    n_auc = 2 if struct == "BCC" else 4
    elements = list(comp.keys())

    v_eff_total = 0.0
    for i, eli in enumerate(elements):
        ci = comp[eli]
        if eli not in KING_ATOMIC_VOLUMES:
            return None
        vi = KING_ATOMIC_VOLUMES[eli]  # King experimental V_i for Vegard baseline

        correction = 0.0
        for j, elj in enumerate(elements):
            if i == j:
                continue
            cj = comp[elj]
            pair = tuple(sorted([eli, elj]))
            omega = omega_sf.get(pair, 0.0)
            correction += cj * omega

        v_eff_i = vi * (1.0 + q * correction)
        v_eff_total += ci * v_eff_i

    a_pred = (n_auc * v_eff_total) ** (1.0 / 3.0)
    return a_pred


# =====================================================================
# Coverage analysis
# =====================================================================
def analyze_coverage(found_pairs, all_elements):
    """Analyze which element pairs are covered."""
    # Generate all expected pairs (unordered, including same-element)
    all_pairs = set()
    for i, a in enumerate(all_elements):
        for j, b in enumerate(all_elements):
            if i <= j:
                all_pairs.add((a, b))

    # Convert found to unordered pairs
    found_unordered = set()
    for a, b in found_pairs:
        found_unordered.add(tuple(sorted([a, b])))

    # Hetero pairs only
    hetero_all = {p for p in all_pairs if p[0] != p[1]}
    hetero_found = {p for p in found_unordered if p[0] != p[1]}

    # Same-element references
    homo_all = {p for p in all_pairs if p[0] == p[1]}
    homo_found = {p for p in found_unordered if p[0] == p[1]}

    # Elements covered
    elements_found = set()
    for a, b in found_pairs:
        elements_found.add(a)
        elements_found.add(b)

    missing_hetero = hetero_all - hetero_found
    missing_homo = homo_all - homo_found

    # Elements involved in missing pairs
    missing_elements = set()
    for a, b in missing_hetero:
        if a not in elements_found:
            missing_elements.add(a)
        if b not in elements_found:
            missing_elements.add(b)

    return {
        'total_expected': len(all_pairs),
        'total_found': len(found_unordered),
        'hetero_expected': len(hetero_all),
        'hetero_found': len(hetero_found),
        'homo_expected': len(homo_all),
        'homo_found': len(homo_found),
        'missing_hetero': sorted(missing_hetero),
        'missing_homo': sorted(missing_homo),
        'elements_found': sorted(elements_found),
        'elements_missing': sorted(set(all_elements) - elements_found),
        'coverage_pct': 100.0 * len(found_unordered) / max(len(all_pairs), 1),
    }


def write_csv(results, output_path):
    """Write extraction results to CSV."""
    fieldnames = [
        'material_id', 'formula', 'element_A', 'element_B',
        'count_A', 'count_B', 'lattice_constant', 'energy_per_atom',
        'energy_above_hull', 'source', 'structure_type', 'lattice_constant_calc'
    ]
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            out = {k: row.get(k, '') for k in fieldnames}
            writer.writerow(out)
    return len(results)


def write_omega_csv(omega_sf, output_path, struct_label):
    """Write Ω_sf values to CSV."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['element_A', 'element_B', 'omega_sf', 'structure'])
        for (a, b), val in sorted(omega_sf.items()):
            writer.writerow([a, b, f"{val:.6f}", struct_label])
    return len(omega_sf)


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description='VASP計算結果の一括スキャン・抽出・Ω_sf再解析')
    parser.add_argument('data_dir',
                        help='VASP計算結果のルートディレクトリ '
                             '(BCC_B2/, FCC_L12/, BCC_SQS/ を含む)')
    parser.add_argument('-o', '--output-dir', default='./reanalysis_output',
                        help='出力ディレクトリ (default: ./reanalysis_output)')
    parser.add_argument('--b2-dir', default='BCC_B2',
                        help='B2サブディレクトリ名 (default: BCC_B2)')
    parser.add_argument('--l12-dir', default='FCC_L12',
                        help='L12サブディレクトリ名 (default: FCC_L12)')
    parser.add_argument('--sqs-dir', default='BCC_SQS',
                        help='SQSサブディレクトリ名 (default: BCC_SQS)')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 70)
    print(f"VASP結果 一括スキャン・再解析")
    print(f"  データ: {data_dir}")
    print(f"  出力:   {out_dir}")
    print(f"  時刻:   {timestamp}")
    print("=" * 70)

    report = []
    report.append(f"# VASP再解析レポート\n")
    report.append(f"- 実行時刻: {timestamp}")
    report.append(f"- データディレクトリ: `{data_dir}`\n")

    # ---------------------------------------------------------------
    # 1. Scan all three structure types
    # ---------------------------------------------------------------
    structures = {
        'B2':  {'dir': args.b2_dir,  'type': 'B2',  'counts': (1, 1)},
        'L12': {'dir': args.l12_dir, 'type': 'L12', 'counts': (3, 1)},
        'SQS': {'dir': args.sqs_dir, 'type': 'SQS', 'counts': (8, 8)},
    }

    all_results = {}
    all_coverage = {}

    report.append("## 1. データ抽出結果\n")
    report.append("| 構造 | ディレクトリ数 | 収束 | 未収束 | 結果なし |")
    report.append("|------|------------|------|--------|---------|")

    for label, cfg in structures.items():
        struct_dir = data_dir / cfg['dir']
        print(f"\n[{label}] Scanning {struct_dir} ...")

        results, errors, found_pairs, no_result_dirs = scan_structure_dir(
            str(struct_dir), cfg['type'], cfg['counts'])

        all_results[label] = results
        all_no_result = no_result_dirs

        n_conv = sum(1 for r in results if r.get('converged'))
        n_notconv = sum(1 for r in results if not r.get('converged'))
        n_total = len(results)
        n_noresult = len(no_result_dirs)

        # Count directories
        n_dirs = n_total + n_noresult
        if os.path.isdir(struct_dir):
            n_dirs_actual = sum(1 for d in os.listdir(struct_dir)
                                if os.path.isdir(struct_dir / d))
            n_dirs = max(n_dirs, n_dirs_actual)

        print(f"    ディレクトリ: {n_dirs}")
        print(f"    結果取得: {n_total} (収束: {n_conv}, 未収束: {n_notconv})")
        print(f"    結果なし: {n_noresult}")
        if no_result_dirs:
            print(f"    結果なしリスト:")
            for d in no_result_dirs:
                print(f"      - {d}")

        report.append(f"| {label} | {n_dirs} | {n_conv} | {n_notconv} | "
                       f"{n_noresult} |")

        # List no-result directories in report
        if no_result_dirs:
            report.append(f"\n**{label} 結果なしディレクトリ ({n_noresult}件):**\n")
            for d in no_result_dirs:
                report.append(f"- `{d}`")
            report.append("")

        # Coverage analysis
        coverage = analyze_coverage(found_pairs, ALL_ELEMENTS)
        all_coverage[label] = coverage

        # Write CSV
        if results:
            csv_path = out_dir / f"compounds_VASP_{label}.csv"
            write_csv(results, csv_path)
            print(f"    CSV出力: {csv_path}")

        if errors:
            err_path = out_dir / f"errors_{label}.txt"
            with open(err_path, 'w') as f:
                f.write("\n".join(errors))

    # ---------------------------------------------------------------
    # 2. Coverage summary
    # ---------------------------------------------------------------
    n_target_hetero = len(ALL_ELEMENTS) * (len(ALL_ELEMENTS) - 1) // 2  # 703

    report.append("\n## 2. カバレッジ\n")
    report.append(f"ターゲット: {len(ALL_ELEMENTS)}元素, "
                   f"{n_target_hetero}ヘテロペア\n")
    report.append("| 構造 | 元素数 | ヘテロペア | カバレッジ(%) | 不足元素 |")
    report.append("|------|--------|----------|-------------|---------|")

    for label in ['B2', 'L12', 'SQS']:
        cov = all_coverage[label]
        missing_el = ', '.join(cov['elements_missing']) if cov['elements_missing'] else '—'
        print(f"\n[{label}] カバレッジ:")
        print(f"    元素: {len(cov['elements_found'])}/{len(ALL_ELEMENTS)}")
        print(f"    ヘテロペア: {cov['hetero_found']}/{n_target_hetero} "
              f"({100*cov['hetero_found']/n_target_hetero:.1f}%)")
        if cov['elements_missing']:
            print(f"    不足元素: {missing_el}")

        report.append(f"| {label} | {len(cov['elements_found'])}/{len(ALL_ELEMENTS)} | "
                       f"{cov['hetero_found']}/{n_target_hetero} | "
                       f"{100*cov['hetero_found']/n_target_hetero:.1f} | "
                       f"{missing_el} |")

    # ---------------------------------------------------------------
    # 3. Ω_sf computation
    # ---------------------------------------------------------------
    report.append("\n## 3. Ω_sf 計算結果\n")

    omega_results = {}
    if all_results['B2']:
        omega_b2 = compute_omega_sf_b2(all_results['B2'])
        omega_results['B2'] = omega_b2
        write_omega_csv(omega_b2, out_dir / "omega_sf_B2.csv", "B2")
        n_neg = sum(1 for v in omega_b2.values() if v < 0)
        print(f"\n[Ω_sf B2] {len(omega_b2)} pairs "
              f"({n_neg} negative = {100*n_neg/max(len(omega_b2),1):.0f}%)")
        report.append(f"- **B2**: {len(omega_b2)}ペア "
                       f"(負: {n_neg}, {100*n_neg/max(len(omega_b2),1):.0f}%)")

    if all_results['L12']:
        omega_l12 = compute_omega_sf_l12(all_results['L12'])
        omega_results['L12'] = omega_l12
        write_omega_csv(omega_l12, out_dir / "omega_sf_L12.csv", "L12")
        n_neg = sum(1 for v in omega_l12.values() if v < 0)
        print(f"[Ω_sf L12] {len(omega_l12)} pairs "
              f"({n_neg} negative = {100*n_neg/max(len(omega_l12),1):.0f}%)")
        report.append(f"- **L12**: {len(omega_l12)}ペア "
                       f"(負: {n_neg}, {100*n_neg/max(len(omega_l12),1):.0f}%)")

    if all_results['SQS']:
        omega_sqs = compute_omega_sf_sqs(all_results['SQS'])
        omega_results['SQS'] = omega_sqs
        write_omega_csv(omega_sqs, out_dir / "omega_sf_SQS.csv", "SQS")
        n_neg = sum(1 for v in omega_sqs.values() if v < 0)
        print(f"[Ω_sf SQS] {len(omega_sqs)} pairs "
              f"({n_neg} negative = {100*n_neg/max(len(omega_sqs),1):.0f}%)")
        report.append(f"- **SQS**: {len(omega_sqs)}ペア "
                       f"(負: {n_neg}, {100*n_neg/max(len(omega_sqs),1):.0f}%)")

    # ---------------------------------------------------------------
    # 4. B2 vs SQS Ω_sf comparison (if both available)
    # ---------------------------------------------------------------
    if 'B2' in omega_results and 'SQS' in omega_results:
        common_pairs = set(omega_results['B2'].keys()) & set(omega_results['SQS'].keys())
        if common_pairs:
            report.append(f"\n## 4. B2 vs SQS Ω_sf 比較 ({len(common_pairs)}共通ペア)\n")
            report.append("| ペア | Ω_sf(B2) | Ω_sf(SQS) | 差分 | 符号反転 |")
            report.append("|------|----------|-----------|------|---------|")

            sign_flips = 0
            diffs = []
            for pair in sorted(common_pairs):
                ob = omega_results['B2'][pair]
                os_ = omega_results['SQS'][pair]
                d = os_ - ob
                flip = "⚠" if (ob > 0) != (os_ > 0) else ""
                if flip:
                    sign_flips += 1
                diffs.append(abs(d))
                report.append(f"| {pair[0]}-{pair[1]} | {ob:.4f} | {os_:.4f} | "
                               f"{d:+.4f} | {flip} |")

            avg_diff = sum(diffs) / len(diffs)
            print(f"\n[B2 vs SQS] {len(common_pairs)} common pairs, "
                  f"mean |diff| = {avg_diff:.4f}, sign flips = {sign_flips}")
            report.append(f"\n平均|差分| = {avg_diff:.4f}, "
                           f"符号反転 = {sign_flips}/{len(common_pairs)}")

    # ---------------------------------------------------------------
    # 5. Missing pairs detail
    # ---------------------------------------------------------------
    report.append("\n## 5. 不足ペア詳細\n")
    for label in ['B2', 'L12', 'SQS']:
        cov = all_coverage[label]
        n_missing = len(cov['missing_hetero'])
        if n_missing > 0:
            report.append(f"\n### {label}: {n_missing}ペア不足\n")
            if n_missing <= 50:
                for a, b in cov['missing_hetero']:
                    report.append(f"- {a}-{b}")
            else:
                # Group by missing element
                el_count = defaultdict(int)
                for a, b in cov['missing_hetero']:
                    el_count[a] += 1
                    el_count[b] += 1
                report.append("不足に関わる元素（出現回数）:")
                for el, cnt in sorted(el_count.items(),
                                       key=lambda x: -x[1])[:15]:
                    report.append(f"- {el}: {cnt}ペア")
                if len(el_count) > 15:
                    report.append(f"- ... 他 {len(el_count)-15}元素")
        else:
            report.append(f"\n### {label}: 不足なし ✓\n")

    # ---------------------------------------------------------------
    # 6. Write report
    # ---------------------------------------------------------------
    report_path = out_dir / "full_report.md"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    print(f"\n{'='*70}")
    print(f"レポート出力: {report_path}")
    print(f"{'='*70}")

    # Print summary to console
    print(f"\n【サマリー】")
    for label in ['B2', 'L12', 'SQS']:
        cov = all_coverage[label]
        n_res = len(all_results[label])
        n_omega = len(omega_results.get(label, {}))
        print(f"  {label:4s}: {n_res:4d} 結果, "
              f"{cov['hetero_found']:3d}/{n_target_hetero} ペア "
              f"({100*cov['hetero_found']/n_target_hetero:5.1f}%), "
              f"Ω_sf = {n_omega} ペア")

    # Hint for next steps
    print(f"""
【次のステップ】
  1. CSVをdata/にコピー:
       cp {out_dir}/compounds_VASP_B2.csv  data/
       cp {out_dir}/compounds_VASP_L12.csv data/
       cp {out_dir}/compounds_VASP_SQS.csv data/

  2. HEA予測パイプライン実行:
       python hea_lattice_xgboost.py --sqs-csv {out_dir}/compounds_VASP_SQS.csv

  3. 不足ペアの追加計算:
       python vasp_inputs/generate_missing_inputs.py \\
           --b2-csv  {out_dir}/compounds_VASP_B2.csv \\
           --l12-csv {out_dir}/compounds_VASP_L12.csv
""")


if __name__ == '__main__':
    main()
