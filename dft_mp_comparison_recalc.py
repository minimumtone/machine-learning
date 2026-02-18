"""
DFT vs Materials Project Comparison and Recalculation Script

This script compares user's DFT calculation results with Materials Project data
and identifies compounds that need recalculation based on lattice constant discrepancies.

Features:
1. Load Materials Project reference data (B2 and L12 structures)
2. Scan user's DFT calculation directories OR load from CSV files
3. Compare lattice constants between DFT and MP data
4. Identify compounds that need recalculation:
   - Compounds with large discrepancy from MP data
   - Compounds not in MP database (need verification)
5. Execute VASP recalculation for identified compounds

Usage:
    # From directories (scans OUTCAR/CONTCAR files)
    python dft_mp_comparison_recalc.py --b2-dir /path/to/BCC_B2 --l12-dir /path/to/FCC_L12 [options]
    
    # From CSV files (user's pre-extracted data)
    python dft_mp_comparison_recalc.py --b2-csv B2_result.csv --l12-csv L12_result.csv [options]

Options:
    --b2-dir        Path to B2 calculation directory
    --l12-dir       Path to L12 calculation directory
    --b2-csv        Path to B2 results CSV file (alternative to --b2-dir)
    --l12-csv       Path to L12 results CSV file (alternative to --l12-dir)
    --mp-b2         Path to MP B2 reference CSV (default: reference_data/mp_b2_compounds.csv)
    --mp-l12        Path to MP L12 reference CSV (default: reference_data/mp_l12_compounds.csv)
    --threshold     Lattice constant discrepancy threshold in Angstrom (default: 0.1)
    --recalc-missing  Also recalculate compounds not in MP database
    --np            Number of MPI processes for VASP (default: 24)
    --dry-run       Show what would be recalculated without running VASP
    --output        Output directory for reports (default: mp_comparison_output)
"""

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class DFTCompound:
    directory: str
    element_A: str
    element_B: str
    count_A: int
    count_B: int
    structure_type: str
    lattice_constant: float
    energy: float
    full_path: str


@dataclass
class MPCompound:
    material_id: str
    formula: str
    element_A: str
    element_B: str
    lattice_constant: float
    energy_per_atom: float
    energy_above_hull: float


class DFTDirectoryScanner:
    
    def __init__(self):
        self.element_pattern = re.compile(r'([A-Z][a-z]?)(\d+)')
    
    def parse_directory_name(self, dirname: str) -> Optional[Tuple[str, int, str, int]]:
        matches = self.element_pattern.findall(dirname)
        if len(matches) != 2:
            return None
        
        element_A, count_A = matches[0]
        element_B, count_B = matches[1]
        return element_A, int(count_A), element_B, int(count_B)
    
    def detect_structure_type(self, count_A: int, count_B: int) -> str:
        if count_A == 1 and count_B == 1:
            return "B2"
        elif (count_A == 3 and count_B == 1) or (count_A == 1 and count_B == 3):
            return "L12"
        return "unknown"
    
    def parse_outcar_energy(self, outcar_path: str) -> Optional[float]:
        try:
            with open(outcar_path, 'r') as f:
                content = f.read()
            
            matches = re.findall(r'free  energy   TOTEN\s*=\s*([-\d.]+)\s*eV', content)
            if matches:
                return float(matches[-1])
        except Exception:
            pass
        return None
    
    def parse_contcar_lattice(self, contcar_path: str) -> Optional[float]:
        try:
            with open(contcar_path, 'r') as f:
                lines = f.readlines()
            
            if len(lines) < 5:
                return None
            
            scale = float(lines[1].strip())
            a_vec = [float(x) for x in lines[2].split()]
            a = np.linalg.norm(a_vec) * scale
            return a
        except Exception:
            pass
        return None
    
    def scan_directory(self, base_dir: str, expected_structure: str = None) -> List[DFTCompound]:
        compounds = []
        base_path = Path(base_dir)
        
        if not base_path.exists():
            print(f"Warning: Directory {base_dir} does not exist")
            return compounds
        
        for subdir in sorted(base_path.iterdir()):
            if not subdir.is_dir():
                continue
            
            dirname = subdir.name
            parsed = self.parse_directory_name(dirname)
            if parsed is None:
                continue
            
            element_A, count_A, element_B, count_B = parsed
            structure_type = self.detect_structure_type(count_A, count_B)
            
            if expected_structure and structure_type != expected_structure:
                continue
            
            outcar_path = subdir / "OUTCAR"
            contcar_path = subdir / "CONTCAR"
            
            if not outcar_path.exists() or not contcar_path.exists():
                continue
            
            energy = self.parse_outcar_energy(str(outcar_path))
            lattice = self.parse_contcar_lattice(str(contcar_path))
            
            if energy is None or lattice is None:
                continue
            
            compounds.append(DFTCompound(
                directory=dirname,
                element_A=element_A,
                element_B=element_B,
                count_A=count_A,
                count_B=count_B,
                structure_type=structure_type,
                lattice_constant=lattice,
                energy=energy,
                full_path=str(subdir)
            ))
        
        return compounds


class DFTCSVLoader:
    """Load DFT data from user's CSV files (B2_result_etot_lattice.csv format)"""
    
    def __init__(self):
        self.element_pattern = re.compile(r'([A-Z][a-z]?)(\d+)')
    
    def parse_directory_name(self, dirname: str) -> Optional[Tuple[str, int, str, int]]:
        matches = self.element_pattern.findall(dirname)
        if len(matches) != 2:
            return None
        element_A, count_A = matches[0]
        element_B, count_B = matches[1]
        return element_A, int(count_A), element_B, int(count_B)
    
    def detect_structure_type(self, count_A: int, count_B: int) -> str:
        if count_A == 1 and count_B == 1:
            return "B2"
        elif (count_A == 3 and count_B == 1) or (count_A == 1 and count_B == 3):
            return "L12"
        return "unknown"
    
    def load_csv(self, csv_path: str, base_dir: str = None) -> List[DFTCompound]:
        """Load DFT data from CSV file.
        
        Expected CSV format:
        directory,composition,E_tot[eV],a[Ang],b[Ang],c[Ang],element_order
        """
        df = pd.read_csv(csv_path)
        compounds = []
        
        # Anomalous data to exclude
        excluded = ["Al1Ce1", "Ce1Al1", "Co1Mo1", "Mo1Co1"]
        
        for _, row in df.iterrows():
            dirname = row['directory']
            
            # Skip excluded compounds
            if dirname in excluded:
                continue
            
            parsed = self.parse_directory_name(dirname)
            if parsed is None:
                continue
            
            element_A, count_A, element_B, count_B = parsed
            structure_type = self.detect_structure_type(count_A, count_B)
            
            # Get lattice constant (use 'a' column)
            lattice = row['a[Ang]']
            energy = row['E_tot[eV]']
            
            # Skip anomalous energy values
            if abs(energy) > 1e6:
                continue
            
            # Determine full path
            if base_dir:
                full_path = os.path.join(base_dir, dirname)
            else:
                full_path = dirname
            
            compounds.append(DFTCompound(
                directory=dirname,
                element_A=element_A,
                element_B=element_B,
                count_A=count_A,
                count_B=count_B,
                structure_type=structure_type,
                lattice_constant=lattice,
                energy=energy,
                full_path=full_path
            ))
        
        return compounds


class MPDataLoader:
    
    def load_b2_data(self, csv_path: str) -> List[MPCompound]:
        df = pd.read_csv(csv_path)
        compounds = []
        
        for _, row in df.iterrows():
            compounds.append(MPCompound(
                material_id=row['material_id'],
                formula=row['formula'],
                element_A=row['element_A'],
                element_B=row['element_B'],
                lattice_constant=row['lattice_constant'],
                energy_per_atom=row['energy_per_atom'],
                energy_above_hull=row['energy_above_hull']
            ))
        
        return compounds
    
    def load_l12_data(self, csv_path: str) -> List[MPCompound]:
        df = pd.read_csv(csv_path)
        compounds = []
        
        for _, row in df.iterrows():
            compounds.append(MPCompound(
                material_id=row['material_id'],
                formula=row['formula'],
                element_A=row['element_A'],
                element_B=row['element_B'],
                lattice_constant=row['lattice_constant'],
                energy_per_atom=row['energy_per_atom'],
                energy_above_hull=row['energy_above_hull']
            ))
        
        return compounds


class OQMDDataLoader:
    """Load reference data from OQMD CSV files"""
    
    def load_b2_data(self, csv_path: str) -> List[MPCompound]:
        df = pd.read_csv(csv_path)
        compounds = []
        
        for _, row in df.iterrows():
            compounds.append(MPCompound(
                material_id=f"oqmd-{row['entry_id']}",
                formula=row['formula'],
                element_A=row['element_A'],
                element_B=row['element_B'],
                lattice_constant=row['lattice_constant'],
                energy_per_atom=row.get('delta_e', 0),
                energy_above_hull=row.get('stability', 0)
            ))
        
        return compounds
    
    def load_l12_data(self, csv_path: str) -> List[MPCompound]:
        df = pd.read_csv(csv_path)
        compounds = []
        
        for _, row in df.iterrows():
            compounds.append(MPCompound(
                material_id=f"oqmd-{row['entry_id']}",
                formula=row['formula'],
                element_A=row['element_A'],
                element_B=row['element_B'],
                lattice_constant=row['lattice_constant'],
                energy_per_atom=row.get('delta_e', 0),
                energy_above_hull=row.get('stability', 0)
            ))
        
        return compounds


def merge_reference_data(mp_compounds: List[MPCompound], oqmd_compounds: List[MPCompound]) -> List[MPCompound]:
    """Merge MP and OQMD data, preferring MP data when both exist for same compound"""
    merged = {}
    
    # Add OQMD data first
    for compound in oqmd_compounds:
        key = (compound.element_A, compound.element_B)
        key_rev = (compound.element_B, compound.element_A)
        if key not in merged and key_rev not in merged:
            merged[key] = compound
    
    # Add/override with MP data (preferred)
    for compound in mp_compounds:
        key = (compound.element_A, compound.element_B)
        key_rev = (compound.element_B, compound.element_A)
        # Override OQMD with MP data
        if key in merged:
            merged[key] = compound
        elif key_rev in merged:
            merged[key_rev] = compound
        else:
            merged[key] = compound
    
    return list(merged.values())


class ComparisonAnalyzer:
    
    def __init__(self, dft_compounds: List[DFTCompound], mp_compounds: List[MPCompound]):
        self.dft_compounds = dft_compounds
        self.mp_compounds = mp_compounds
        self.mp_lookup = self._build_mp_lookup()
    
    def _build_mp_lookup(self) -> Dict[Tuple[str, str], MPCompound]:
        lookup = {}
        for mp in self.mp_compounds:
            key1 = (mp.element_A, mp.element_B)
            key2 = (mp.element_B, mp.element_A)
            if key1 not in lookup:
                lookup[key1] = mp
            if key2 not in lookup:
                lookup[key2] = mp
        return lookup
    
    def compare(self) -> pd.DataFrame:
        results = []
        
        for dft in self.dft_compounds:
            key = (dft.element_A, dft.element_B)
            mp = self.mp_lookup.get(key)
            
            if mp:
                discrepancy = dft.lattice_constant - mp.lattice_constant
                rel_error = abs(discrepancy) / mp.lattice_constant * 100
                in_mp = True
            else:
                discrepancy = None
                rel_error = None
                in_mp = False
            
            results.append({
                'directory': dft.directory,
                'element_A': dft.element_A,
                'element_B': dft.element_B,
                'structure_type': dft.structure_type,
                'dft_lattice': dft.lattice_constant,
                'mp_lattice': mp.lattice_constant if mp else None,
                'discrepancy': discrepancy,
                'rel_error_pct': rel_error,
                'in_mp': in_mp,
                'mp_material_id': mp.material_id if mp else None,
                'mp_energy_above_hull': mp.energy_above_hull if mp else None,
                'full_path': dft.full_path
            })
        
        return pd.DataFrame(results)
    
    def identify_recalc_candidates(
        self,
        comparison_df: pd.DataFrame,
        threshold: float = 0.1,
        include_missing: bool = False
    ) -> pd.DataFrame:
        candidates = comparison_df[
            (comparison_df['in_mp'] == True) & 
            (comparison_df['discrepancy'].abs() > threshold)
        ].copy()
        
        if include_missing:
            missing = comparison_df[comparison_df['in_mp'] == False].copy()
            candidates = pd.concat([candidates, missing], ignore_index=True)
        
        return candidates


def run_vasp_recalculation(
    directories: List[str],
    np_cores: int = 24,
    dry_run: bool = False
) -> List[str]:
    recalculated = []
    
    for dir_path in directories:
        print(f"\n{'[DRY-RUN] ' if dry_run else ''}Recalculating: {dir_path}")
        
        if dry_run:
            recalculated.append(dir_path)
            continue
        
        try:
            cmd = f"mpirun -np {np_cores} $VASPBIN"
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=dir_path,
                capture_output=True,
                text=True,
                timeout=3600
            )
            
            if result.returncode == 0:
                print(f"  Success: {dir_path}")
                recalculated.append(dir_path)
            else:
                print(f"  Failed: {dir_path}")
                print(f"  Error: {result.stderr[:500]}")
        except subprocess.TimeoutExpired:
            print(f"  Timeout: {dir_path}")
        except Exception as e:
            print(f"  Error: {dir_path} - {e}")
    
    return recalculated


def generate_comparison_report(
    comparison_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    structure_type: str,
    output_dir: str
):
    os.makedirs(output_dir, exist_ok=True)
    
    in_mp = comparison_df[comparison_df['in_mp'] == True]
    not_in_mp = comparison_df[comparison_df['in_mp'] == False]
    
    if len(in_mp) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        ax = axes[0]
        ax.scatter(in_mp['mp_lattice'], in_mp['dft_lattice'], alpha=0.6, s=50, c='steelblue')
        min_val = min(in_mp['mp_lattice'].min(), in_mp['dft_lattice'].min())
        max_val = max(in_mp['mp_lattice'].max(), in_mp['dft_lattice'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x')
        ax.set_xlabel('Materials Project Lattice Constant (Å)', fontsize=14)
        ax.set_ylabel('DFT Lattice Constant (Å)', fontsize=14)
        ax.set_title(f'{structure_type}: DFT vs MP Lattice Constants\nn={len(in_mp)}', fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=12)
        
        rmse = np.sqrt(np.mean(in_mp['discrepancy'] ** 2))
        mae = np.mean(np.abs(in_mp['discrepancy']))
        stats_text = f'RMSE: {rmse:.4f} Å\nMAE: {mae:.4f} Å'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, ha='left', va='top',
                fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax = axes[1]
        ax.hist(in_mp['discrepancy'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Discrepancy (DFT - MP) (Å)', fontsize=14)
        ax.set_ylabel('Count', fontsize=14)
        ax.set_title(f'{structure_type}: Lattice Constant Discrepancy Distribution', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{structure_type.lower()}_mp_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    comparison_df.to_csv(os.path.join(output_dir, f'{structure_type.lower()}_comparison.csv'), index=False)
    candidates_df.to_csv(os.path.join(output_dir, f'{structure_type.lower()}_recalc_candidates.csv'), index=False)
    
    report = f"""# {structure_type} Structure: DFT vs Materials Project Comparison Report

## 1. Overview

- Total DFT compounds: {len(comparison_df)}
- Compounds in MP database: {len(in_mp)}
- Compounds NOT in MP database: {len(not_in_mp)}

## 2. Comparison Statistics (compounds in MP)

"""
    if len(in_mp) > 0:
        rmse = np.sqrt(np.mean(in_mp['discrepancy'] ** 2))
        mae = np.mean(np.abs(in_mp['discrepancy']))
        mean_rel_error = in_mp['rel_error_pct'].mean()
        
        report += f"""- RMSE: {rmse:.4f} Å
- MAE: {mae:.4f} Å
- Mean relative error: {mean_rel_error:.2f}%
- Max discrepancy: {in_mp['discrepancy'].abs().max():.4f} Å
- Min discrepancy: {in_mp['discrepancy'].abs().min():.4f} Å

"""
    
    report += f"""## 3. Recalculation Candidates

Total candidates: {len(candidates_df)}

### 3.1 Compounds with large discrepancy from MP

"""
    large_discrepancy = candidates_df[candidates_df['in_mp'] == True]
    if len(large_discrepancy) > 0:
        report += f"Count: {len(large_discrepancy)}\n\n"
        report += "| Directory | DFT (Å) | MP (Å) | Discrepancy (Å) | Rel. Error (%) |\n"
        report += "|-----------|---------|--------|-----------------|----------------|\n"
        for _, row in large_discrepancy.head(20).iterrows():
            report += f"| {row['directory']} | {row['dft_lattice']:.4f} | {row['mp_lattice']:.4f} | {row['discrepancy']:.4f} | {row['rel_error_pct']:.2f} |\n"
        if len(large_discrepancy) > 20:
            report += f"\n... and {len(large_discrepancy) - 20} more compounds\n"
    else:
        report += "None\n"
    
    report += """
### 3.2 Compounds NOT in MP database

"""
    missing = candidates_df[candidates_df['in_mp'] == False]
    if len(missing) > 0:
        report += f"Count: {len(missing)}\n\n"
        report += "| Directory | DFT Lattice (Å) |\n"
        report += "|-----------|----------------|\n"
        for _, row in missing.head(20).iterrows():
            report += f"| {row['directory']} | {row['dft_lattice']:.4f} |\n"
        if len(missing) > 20:
            report += f"\n... and {len(missing) - 20} more compounds\n"
    else:
        report += "None\n"
    
    report += """
## 4. Files Generated

- `{0}_comparison.csv`: Full comparison data
- `{0}_recalc_candidates.csv`: Recalculation candidates
- `{0}_mp_comparison.png`: Comparison plots

---

*Report generated automatically.*
""".format(structure_type.lower())
    
    with open(os.path.join(output_dir, f'{structure_type.lower()}_comparison_report.md'), 'w') as f:
        f.write(report)
    
    print(f"Saved {structure_type} comparison report to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare DFT calculations with Materials Project data and identify recalculation candidates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Compare and identify recalculation candidates (using directories)
  python dft_mp_comparison_recalc.py --b2-dir /path/to/BCC_B2 --l12-dir /path/to/FCC_L12
  
  # Compare using CSV files
  python dft_mp_comparison_recalc.py --b2-csv B2_result.csv --l12-csv L12_result.csv
  
  # Dry run to see what would be recalculated
  python dft_mp_comparison_recalc.py --b2-dir /path/to/BCC_B2 --recalc --dry-run
  
  # Recalculate compounds with discrepancy > 0.15 Å
  python dft_mp_comparison_recalc.py --b2-dir /path/to/BCC_B2 --recalc --threshold 0.15 --np 24
  
  # Also recalculate compounds not in MP database
  python dft_mp_comparison_recalc.py --b2-dir /path/to/BCC_B2 --recalc --recalc-missing
  
  # Use OQMD data instead of Materials Project
  python dft_mp_comparison_recalc.py --b2-dir /path/to/BCC_B2 --use-oqmd
  
  # Use both MP and OQMD data
  python dft_mp_comparison_recalc.py --b2-dir /path/to/BCC_B2 --use-both
        """
    )
    
    parser.add_argument('--b2-dir', help='Path to B2 calculation directory')
    parser.add_argument('--l12-dir', help='Path to L12 calculation directory')
    parser.add_argument('--b2-csv', help='Path to B2 results CSV file (alternative to --b2-dir)')
    parser.add_argument('--l12-csv', help='Path to L12 results CSV file (alternative to --l12-dir)')
    parser.add_argument('--mp-b2', default='reference_data/mp_b2_compounds.csv',
                        help='Path to MP B2 reference CSV')
    parser.add_argument('--mp-l12', default='reference_data/mp_l12_compounds.csv',
                        help='Path to MP L12 reference CSV')
    parser.add_argument('--oqmd-b2', default='reference_data/oqmd_b2_compounds.csv',
                        help='Path to OQMD B2 reference CSV')
    parser.add_argument('--oqmd-l12', default='reference_data/oqmd_l12_compounds.csv',
                        help='Path to OQMD L12 reference CSV')
    parser.add_argument('--use-oqmd', action='store_true',
                        help='Use OQMD data instead of Materials Project')
    parser.add_argument('--use-both', action='store_true',
                        help='Use both Materials Project and OQMD data (merged)')
    parser.add_argument('--threshold', '-t', type=float, default=0.1,
                        help='Lattice constant discrepancy threshold in Angstrom (default: 0.1)')
    parser.add_argument('--recalc', action='store_true',
                        help='Execute VASP recalculation for identified candidates')
    parser.add_argument('--recalc-missing', action='store_true',
                        help='Also recalculate compounds not in MP database')
    parser.add_argument('--np', type=int, default=24,
                        help='Number of MPI processes for VASP (default: 24)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be recalculated without running VASP')
    parser.add_argument('--output', '-o', default='mp_comparison_output',
                        help='Output directory for reports (default: mp_comparison_output)')
    
    args = parser.parse_args()
    
    if not args.b2_dir and not args.l12_dir and not args.b2_csv and not args.l12_csv:
        print("Error: Please specify at least one of --b2-dir, --l12-dir, --b2-csv, or --l12-csv")
        parser.print_help()
        sys.exit(1)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Resolve reference data paths
    if not os.path.isabs(args.mp_b2):
        args.mp_b2 = os.path.join(script_dir, args.mp_b2)
    if not os.path.isabs(args.mp_l12):
        args.mp_l12 = os.path.join(script_dir, args.mp_l12)
    if not os.path.isabs(args.oqmd_b2):
        args.oqmd_b2 = os.path.join(script_dir, args.oqmd_b2)
    if not os.path.isabs(args.oqmd_l12):
        args.oqmd_l12 = os.path.join(script_dir, args.oqmd_l12)
    
    # Determine data source
    data_source = "Materials Project"
    if args.use_oqmd:
        data_source = "OQMD"
    elif args.use_both:
        data_source = "Materials Project + OQMD"
    
    print("=" * 70)
    print(f"DFT vs {data_source} Comparison")
    print("=" * 70)
    
    scanner = DFTDirectoryScanner()
    csv_loader = DFTCSVLoader()
    mp_loader = MPDataLoader()
    oqmd_loader = OQMDDataLoader()
    
    all_candidates = []
    
    # Process B2 structures
    dft_b2 = []
    if args.b2_dir:
        print(f"\n--- Processing B2 structures ---")
        print(f"Directory: {args.b2_dir}")
        dft_b2 = scanner.scan_directory(args.b2_dir, expected_structure="B2")
        print(f"Found {len(dft_b2)} B2 compounds in DFT directory")
    elif args.b2_csv:
        print(f"\n--- Processing B2 structures ---")
        print(f"CSV file: {args.b2_csv}")
        base_dir = args.b2_dir if args.b2_dir else None
        dft_b2 = csv_loader.load_csv(args.b2_csv, base_dir=base_dir)
        dft_b2 = [c for c in dft_b2 if c.structure_type == "B2"]
        print(f"Loaded {len(dft_b2)} B2 compounds from CSV")
    
    if dft_b2:
        # Load reference data
        ref_b2 = []
        if args.use_oqmd:
            if os.path.exists(args.oqmd_b2):
                ref_b2 = oqmd_loader.load_b2_data(args.oqmd_b2)
                print(f"Loaded {len(ref_b2)} B2 compounds from OQMD")
        elif args.use_both:
            mp_b2 = []
            oqmd_b2 = []
            if os.path.exists(args.mp_b2):
                mp_b2 = mp_loader.load_b2_data(args.mp_b2)
                print(f"Loaded {len(mp_b2)} B2 compounds from Materials Project")
            if os.path.exists(args.oqmd_b2):
                oqmd_b2 = oqmd_loader.load_b2_data(args.oqmd_b2)
                print(f"Loaded {len(oqmd_b2)} B2 compounds from OQMD")
            ref_b2 = merge_reference_data(mp_b2, oqmd_b2)
            print(f"Merged: {len(ref_b2)} unique B2 compounds")
        else:
            if os.path.exists(args.mp_b2):
                ref_b2 = mp_loader.load_b2_data(args.mp_b2)
                print(f"Loaded {len(ref_b2)} B2 compounds from Materials Project")
        
        if ref_b2:
            analyzer = ComparisonAnalyzer(dft_b2, ref_b2)
            comparison_df = analyzer.compare()
            candidates_df = analyzer.identify_recalc_candidates(
                comparison_df, 
                threshold=args.threshold,
                include_missing=args.recalc_missing
            )
            
            print(f"Recalculation candidates: {len(candidates_df)}")
            
            generate_comparison_report(comparison_df, candidates_df, "B2", args.output)
            
            all_candidates.extend(candidates_df['full_path'].tolist())
        else:
            print(f"Warning: No B2 reference data found")
    
    # Process L12 structures
    dft_l12 = []
    if args.l12_dir:
        print(f"\n--- Processing L12 structures ---")
        print(f"Directory: {args.l12_dir}")
        dft_l12 = scanner.scan_directory(args.l12_dir, expected_structure="L12")
        print(f"Found {len(dft_l12)} L12 compounds in DFT directory")
    elif args.l12_csv:
        print(f"\n--- Processing L12 structures ---")
        print(f"CSV file: {args.l12_csv}")
        base_dir = args.l12_dir if args.l12_dir else None
        dft_l12 = csv_loader.load_csv(args.l12_csv, base_dir=base_dir)
        dft_l12 = [c for c in dft_l12 if c.structure_type == "L12"]
        print(f"Loaded {len(dft_l12)} L12 compounds from CSV")
    
    if dft_l12:
        # Load reference data
        ref_l12 = []
        if args.use_oqmd:
            if os.path.exists(args.oqmd_l12):
                ref_l12 = oqmd_loader.load_l12_data(args.oqmd_l12)
                print(f"Loaded {len(ref_l12)} L12 compounds from OQMD")
        elif args.use_both:
            mp_l12 = []
            oqmd_l12 = []
            if os.path.exists(args.mp_l12):
                mp_l12 = mp_loader.load_l12_data(args.mp_l12)
                print(f"Loaded {len(mp_l12)} L12 compounds from Materials Project")
            if os.path.exists(args.oqmd_l12):
                oqmd_l12 = oqmd_loader.load_l12_data(args.oqmd_l12)
                print(f"Loaded {len(oqmd_l12)} L12 compounds from OQMD")
            ref_l12 = merge_reference_data(mp_l12, oqmd_l12)
            print(f"Merged: {len(ref_l12)} unique L12 compounds")
        else:
            if os.path.exists(args.mp_l12):
                ref_l12 = mp_loader.load_l12_data(args.mp_l12)
                print(f"Loaded {len(ref_l12)} L12 compounds from Materials Project")
        
        if ref_l12:
            analyzer = ComparisonAnalyzer(dft_l12, ref_l12)
            comparison_df = analyzer.compare()
            candidates_df = analyzer.identify_recalc_candidates(
                comparison_df,
                threshold=args.threshold,
                include_missing=args.recalc_missing
            )
            
            print(f"Recalculation candidates: {len(candidates_df)}")
            
            generate_comparison_report(comparison_df, candidates_df, "L12", args.output)
            
            all_candidates.extend(candidates_df['full_path'].tolist())
        else:
            print(f"Warning: No L12 reference data found")
    
    if args.recalc and all_candidates:
        print(f"\n--- {'[DRY-RUN] ' if args.dry_run else ''}Recalculating {len(all_candidates)} compounds ---")
        recalculated = run_vasp_recalculation(
            all_candidates,
            np_cores=args.np,
            dry_run=args.dry_run
        )
        print(f"\n{'Would recalculate' if args.dry_run else 'Recalculated'}: {len(recalculated)} compounds")
    
    print("\n" + "=" * 70)
    print("Comparison completed!")
    print(f"Output directory: {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
