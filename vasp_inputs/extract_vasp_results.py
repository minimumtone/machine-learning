#!/usr/bin/env python3
"""
Extract lattice constants from completed VASP calculations.

Reads CONTCAR (or POSCAR if CONTCAR absent) from each subdirectory
under FCC_L12/ and BCC_B2/ to extract optimized lattice constants.
Outputs CSV files compatible with hea_lattice_xgboost.py's compound_df format.

Directory naming convention:
    FCC_L12/Ag3Al1/  → L1₂ Ag₃Al  (element_A=Ag, count_A=3, element_B=Al, count_B=1)
    BCC_B2/Ag1Al1/   → B2  AgAl   (element_A=Ag, count_A=1, element_B=Al, count_B=1)

Usage:
    python extract_vasp_results.py /path/to/DATA

    where /path/to/DATA contains FCC_L12/ and BCC_B2/ subdirectories.

Output:
    compounds_VASP_B2.csv   — B2 compounds (compatible with data/compounds_*.csv)
    compounds_VASP_L12.csv  — L1₂ compounds
    extraction_report.txt   — summary with convergence status
"""

import os
import re
import sys
import csv
import argparse
from pathlib import Path


def parse_dirname(dirname):
    """
    Parse directory name like 'Ag3Al1' into (element_A, count_A, element_B, count_B).
    Pattern: <Element><count><Element><count>
    """
    m = re.match(r'^([A-Z][a-z]?)(\d+)([A-Z][a-z]?)(\d+)$', dirname)
    if not m:
        return None
    return m.group(1), int(m.group(2)), m.group(3), int(m.group(4))


def read_lattice_constant(calc_dir):
    """
    Read the lattice constant from CONTCAR (preferred) or POSCAR.
    For cubic structures (B2, L1₂), the lattice constant is the
    first scaling factor line (line 2) times the a-vector magnitude.

    Also checks OUTCAR for convergence if available.

    Returns (lattice_constant, converged, source_file) or (None, False, None).
    """
    contcar = os.path.join(calc_dir, 'CONTCAR')
    poscar = os.path.join(calc_dir, 'POSCAR')
    outcar = os.path.join(calc_dir, 'OUTCAR')

    # Prefer CONTCAR (relaxed structure); fall back to POSCAR
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

        # Line 2: scaling factor
        scale = float(lines[1].strip())

        # Lines 3-5: lattice vectors
        a_vec = [float(x) for x in lines[2].split()]
        # For cubic: a = scale * |a_vec|
        a_mag = (a_vec[0]**2 + a_vec[1]**2 + a_vec[2]**2) ** 0.5
        lattice_const = abs(scale) * a_mag

        # Sanity check: lattice constant should be 2-8 Å for metals
        if lattice_const < 2.0 or lattice_const > 8.0:
            return None, False, source

    except (ValueError, IndexError):
        return None, False, source

    # Check convergence from OUTCAR
    converged = False
    if os.path.isfile(outcar):
        try:
            with open(outcar) as f:
                content = f.read()
            # VASP prints "reached required accuracy" when converged
            converged = 'reached required accuracy' in content
            # Also check for ionic convergence
            if not converged:
                converged = 'FORCES: max atom force' in content
        except Exception:
            pass

    src_name = 'CONTCAR' if os.path.basename(source) == 'CONTCAR' else 'POSCAR'
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
                    # Last occurrence is the final energy
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


def extract_structure_data(base_dir, struct_type):
    """
    Extract data from all subdirectories in base_dir.
    struct_type: "B2" or "L12"
    """
    results = []
    errors = []

    if not os.path.isdir(base_dir):
        print(f"  WARNING: Directory not found: {base_dir}")
        return results, errors

    subdirs = sorted(os.listdir(base_dir))
    for dirname in subdirs:
        calc_dir = os.path.join(base_dir, dirname)
        if not os.path.isdir(calc_dir):
            continue

        parsed = parse_dirname(dirname)
        if parsed is None:
            errors.append(f"  Cannot parse directory name: {dirname}")
            continue

        elA, cA, elB, cB = parsed
        a, converged, src = read_lattice_constant(calc_dir)
        e_per_atom = read_energy_per_atom(calc_dir)

        if a is None:
            errors.append(f"  No lattice constant found: {dirname}")
            continue

        formula = f"{elA}{cA}{elB}{cB}" if cA > 1 else f"{elA}{elB}"
        if struct_type == "L12" and cA == 3:
            formula = f"{elA}3{elB}"
        elif struct_type == "L12" and cB == 3:
            formula = f"{elB}3{elA}"

        results.append({
            'material_id': f'vasp-{dirname}',
            'formula': formula,
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
        })

    return results, errors


def write_csv(results, output_path, struct_type):
    """Write results in format compatible with compounds_MP_*.csv"""
    # Match existing CSV column order
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


def main():
    parser = argparse.ArgumentParser(
        description='Extract lattice constants from VASP B2/L12 calculations')
    parser.add_argument('data_dir',
                        help='Base directory containing FCC_L12/ and BCC_B2/')
    parser.add_argument('--output-dir', '-o', default='.',
                        help='Output directory for CSV files (default: current dir)')
    parser.add_argument('--l12-dirname', default='FCC_L12',
                        help='Name of L12 subdirectory (default: FCC_L12)')
    parser.add_argument('--b2-dirname', default='BCC_B2',
                        help='Name of B2 subdirectory (default: BCC_B2)')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("VASP Results Extraction")
    print(f"  Data directory: {data_dir}")
    print(f"  Output directory: {out_dir}")
    print("=" * 70)

    all_errors = []
    report_lines = []

    # --- B2 ---
    print(f"\n[1] Extracting B2 data from {args.b2_dirname}/...")
    b2_dir = data_dir / args.b2_dirname
    b2_results, b2_errors = extract_structure_data(str(b2_dir), "B2")
    all_errors.extend(b2_errors)

    n_b2_conv = sum(1 for r in b2_results if r['converged'])
    n_b2_total = len(b2_results)
    print(f"    Found: {n_b2_total} compounds ({n_b2_conv} converged)")

    if b2_results:
        b2_csv = out_dir / "compounds_VASP_B2.csv"
        write_csv(b2_results, b2_csv, "B2")
        print(f"    Saved: {b2_csv}")

        # Unique element pairs
        b2_pairs = set()
        for r in b2_results:
            b2_pairs.add(tuple(sorted([r['element_A'], r['element_B']])))
        print(f"    Unique pairs: {len(b2_pairs)}")

    report_lines.append(f"B2 compounds: {n_b2_total} ({n_b2_conv} converged)")

    # --- L12 ---
    print(f"\n[2] Extracting L12 data from {args.l12_dirname}/...")
    l12_dir = data_dir / args.l12_dirname
    l12_results, l12_errors = extract_structure_data(str(l12_dir), "L12")
    all_errors.extend(l12_errors)

    n_l12_conv = sum(1 for r in l12_results if r['converged'])
    n_l12_total = len(l12_results)
    print(f"    Found: {n_l12_total} compounds ({n_l12_conv} converged)")

    if l12_results:
        l12_csv = out_dir / "compounds_VASP_L12.csv"
        write_csv(l12_results, l12_csv, "L12")
        print(f"    Saved: {l12_csv}")

        l12_pairs = set()
        for r in l12_results:
            l12_pairs.add(tuple(sorted([r['element_A'], r['element_B']])))
        print(f"    Unique pairs: {len(l12_pairs)}")

    report_lines.append(f"L12 compounds: {n_l12_total} ({n_l12_conv} converged)")

    # --- Lattice constant summary ---
    print("\n[3] Lattice constant summary:")
    for label, results in [("B2", b2_results), ("L12", l12_results)]:
        if not results:
            continue
        a_vals = [r['lattice_constant'] for r in results]
        print(f"    {label}: a = {min(a_vals):.4f} – {max(a_vals):.4f} Å "
              f"(mean {sum(a_vals)/len(a_vals):.4f} Å)")

    # --- Report ---
    report_path = out_dir / "extraction_report.txt"
    with open(report_path, 'w') as f:
        f.write("VASP Results Extraction Report\n")
        f.write(f"Data directory: {data_dir}\n")
        f.write("=" * 50 + "\n\n")

        for line in report_lines:
            f.write(line + "\n")

        f.write(f"\nTotal: {n_b2_total + n_l12_total} compounds\n")
        f.write(f"Converged: {n_b2_conv + n_l12_conv}\n")

        if all_errors:
            f.write(f"\nErrors/Warnings ({len(all_errors)}):\n")
            for err in all_errors:
                f.write(err + "\n")

        # List all extracted compounds
        f.write("\n\n--- B2 Compounds ---\n")
        f.write(f"{'Dir':<15} {'Formula':<10} {'a (Å)':>8} {'Conv':>5} {'Source':>8}\n")
        f.write("-" * 50 + "\n")
        for r in b2_results:
            conv = 'Y' if r['converged'] else 'N'
            f.write(f"{r['dirname']:<15} {r['formula']:<10} "
                    f"{r['lattice_constant']:>8.4f} {conv:>5} {r['source_file']:>8}\n")

        f.write("\n\n--- L12 Compounds ---\n")
        f.write(f"{'Dir':<15} {'Formula':<10} {'a (Å)':>8} {'Conv':>5} {'Source':>8}\n")
        f.write("-" * 50 + "\n")
        for r in l12_results:
            conv = 'Y' if r['converged'] else 'N'
            f.write(f"{r['dirname']:<15} {r['formula']:<10} "
                    f"{r['lattice_constant']:>8.4f} {conv:>5} {r['source_file']:>8}\n")

    print(f"\n    Report saved: {report_path}")

    if all_errors:
        print(f"\n    Warnings: {len(all_errors)}")
        for err in all_errors[:10]:
            print(f"    {err}")
        if len(all_errors) > 10:
            print(f"    ... and {len(all_errors) - 10} more (see report)")

    print("\n" + "=" * 70)
    print(f"Total: {n_b2_total + n_l12_total} compounds extracted")
    print(f"  B2:  {n_b2_total} ({n_b2_conv} converged)")
    print(f"  L12: {n_l12_total} ({n_l12_conv} converged)")
    print("=" * 70)

    # Usage hint
    print(f"""
To use with hea_lattice_xgboost.py:
  1. Copy CSV files to data/ directory:
       cp {out_dir}/compounds_VASP_B2.csv  data/
       cp {out_dir}/compounds_VASP_L12.csv data/
  2. The script will automatically load VASP data alongside MP/OQMD data.

Or load manually in Python:
  import pandas as pd
  b2  = pd.read_csv('{out_dir}/compounds_VASP_B2.csv')
  l12 = pd.read_csv('{out_dir}/compounds_VASP_L12.csv')
""")


if __name__ == '__main__':
    main()
