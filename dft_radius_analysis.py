"""
DFT Effective Radius Analysis Script

This script provides:
1. B2 structure analysis with lattice constant filtering (around 4.0 Å)
2. L12 structure accuracy investigation and diagnosis

Author: Devin AI
"""

import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares


PAULING_RADII = {
    "Li": 1.52, "Be": 1.12, "B": 0.98, "C": 0.77, "N": 0.75, "O": 0.73, "F": 0.72,
    "Na": 1.90, "Mg": 1.60, "Al": 1.43, "Si": 1.17, "P": 1.10, "S": 1.04, "Cl": 0.99,
    "K": 2.35, "Ca": 1.97, "Sc": 1.64, "Ti": 1.47, "V": 1.35, "Cr": 1.29, "Mn": 1.37,
    "Fe": 1.26, "Co": 1.25, "Ni": 1.25, "Cu": 1.28, "Zn": 1.37, "Ga": 1.53, "Ge": 1.22,
    "As": 1.21, "Se": 1.17, "Br": 1.14, "Rb": 2.48, "Sr": 2.15, "Y": 1.82, "Zr": 1.60,
    "Nb": 1.47, "Mo": 1.40, "Tc": 1.35, "Ru": 1.34, "Rh": 1.34, "Pd": 1.37, "Ag": 1.44,
    "Cd": 1.52, "In": 1.67, "Sn": 1.58, "Sb": 1.61, "Te": 1.43, "I": 1.33, "Cs": 2.67,
    "Ba": 2.22, "La": 1.87, "Ce": 1.83, "Pr": 1.82, "Nd": 1.81, "Pm": 1.80, "Sm": 1.80,
    "Eu": 2.04, "Gd": 1.80, "Tb": 1.78, "Dy": 1.77, "Ho": 1.76, "Er": 1.75, "Tm": 1.74,
    "Yb": 1.94, "Lu": 1.74, "Hf": 1.59, "Ta": 1.47, "W": 1.41, "Re": 1.37, "Os": 1.35,
    "Ir": 1.36, "Pt": 1.39, "Au": 1.44, "Hg": 1.55, "Tl": 1.71, "Pb": 1.75, "Bi": 1.82
}

HEA_ELEMENTS = [
    "Al", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Zr", "Nb", "Mo", "Hf", "Ta", "W", "Re", "Si", "Mg", "Sc",
    "Y", "Pd", "Pt", "Au", "Ag"
]


@dataclass
class DFTCompoundData:
    directory: str
    composition: str
    structure_type: str
    element_A: str
    element_B: str
    count_A: int
    count_B: int
    lattice_constant: float
    energy: float


class DFTDataLoader:
    def __init__(self, b2_csv_path: str = None, l12_csv_path: str = None):
        self.b2_csv_path = b2_csv_path
        self.l12_csv_path = l12_csv_path

    def _parse_directory_name(self, directory: str, structure_type: str) -> Tuple[str, str, int, int]:
        base_dir = directory.split('/')[0]
        if structure_type == "B2":
            match = re.match(r'([A-Z][a-z]?)1([A-Z][a-z]?)1', base_dir)
            if match:
                return match.group(1), match.group(2), 1, 1
        elif structure_type == "L12":
            match = re.match(r'([A-Z][a-z]?)3([A-Z][a-z]?)1', base_dir)
            if match:
                return match.group(1), match.group(2), 3, 1
        return None, None, 0, 0

    def _detect_structure_type(self, dirname: str) -> Optional[str]:
        if re.match(r'([A-Z][a-z]?)1([A-Z][a-z]?)1$', dirname):
            return "B2"
        elif re.match(r'([A-Z][a-z]?)3([A-Z][a-z]?)1$', dirname):
            return "L12"
        elif re.match(r'([A-Z][a-z]?)1([A-Z][a-z]?)3$', dirname):
            return "L12"
        return None

    def load_from_directory(self, base_dir: str) -> List[DFTCompoundData]:
        compounds = []
        
        for subdir in os.listdir(base_dir):
            subdir_path = os.path.join(base_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue
            
            structure_type = self._detect_structure_type(subdir)
            if structure_type is None:
                continue
            
            outcar_path = os.path.join(subdir_path, "OUTCAR")
            contcar_path = os.path.join(subdir_path, "CONTCAR")
            
            if not os.path.exists(outcar_path) or not os.path.exists(contcar_path):
                continue
            
            energy = self._parse_outcar_energy(outcar_path)
            lattice_constant = self._parse_contcar_lattice(contcar_path)
            
            if energy is None or lattice_constant is None:
                continue
            if abs(energy) > 1e9:
                continue
            
            el_A, el_B, count_A, count_B = self._parse_directory_name(subdir, structure_type)
            if el_A is None:
                continue
            
            composition = f"{el_A}{count_A}{el_B}{count_B}"
            compounds.append(DFTCompoundData(
                directory=subdir,
                composition=composition,
                structure_type=structure_type,
                element_A=el_A,
                element_B=el_B,
                count_A=count_A,
                count_B=count_B,
                lattice_constant=lattice_constant,
                energy=energy
            ))
        
        return compounds

    def _parse_outcar_energy(self, outcar_path: str) -> Optional[float]:
        try:
            with open(outcar_path, 'r') as f:
                lines = f.readlines()
            for line in reversed(lines):
                if "free  energy   TOTEN" in line:
                    parts = line.split()
                    return float(parts[4])
        except Exception:
            pass
        return None

    def _parse_contcar_lattice(self, contcar_path: str) -> Optional[float]:
        try:
            with open(contcar_path, 'r') as f:
                lines = f.readlines()
            scale = float(lines[1].strip())
            a_vec = [float(x) for x in lines[2].split()]
            a_length = np.sqrt(sum(x**2 for x in a_vec)) * scale
            return a_length
        except Exception:
            pass
        return None

    def load_data(self) -> List[DFTCompoundData]:
        compounds = []
        
        if self.b2_csv_path and os.path.exists(self.b2_csv_path):
            b2_df = pd.read_csv(self.b2_csv_path)
            for _, row in b2_df.iterrows():
                if pd.isna(row['E_tot[eV]']) or pd.isna(row['a[Ang]']):
                    continue
                energy = row['E_tot[eV]']
                if abs(energy) > 1e9:
                    continue
                el_A, el_B, count_A, count_B = self._parse_directory_name(row['directory'], "B2")
                if el_A is None:
                    continue
                if '/backup' in row['directory']:
                    continue
                compounds.append(DFTCompoundData(
                    directory=row['directory'],
                    composition=row['composition'],
                    structure_type="B2",
                    element_A=el_A,
                    element_B=el_B,
                    count_A=count_A,
                    count_B=count_B,
                    lattice_constant=row['a[Ang]'],
                    energy=energy
                ))

        if self.l12_csv_path and os.path.exists(self.l12_csv_path):
            l12_df = pd.read_csv(self.l12_csv_path)
            for _, row in l12_df.iterrows():
                if pd.isna(row['E_tot[eV]']) or pd.isna(row['a[Ang]']):
                    continue
                energy = row['E_tot[eV]']
                if abs(energy) > 1e9:
                    continue
                el_A, el_B, count_A, count_B = self._parse_directory_name(row['directory'], "L12")
                if el_A is None:
                    continue
                if '/backup' in row['directory']:
                    continue
                compounds.append(DFTCompoundData(
                    directory=row['directory'],
                    composition=row['composition'],
                    structure_type="L12",
                    element_A=el_A,
                    element_B=el_B,
                    count_A=count_A,
                    count_B=count_B,
                    lattice_constant=row['a[Ang]'],
                    energy=energy
                ))

        return compounds


class FilteredRadiusCalculator:
    def __init__(self, compounds: List[DFTCompoundData]):
        self.compounds = compounds

    def calculate_radii_trf(
        self,
        compounds: List[DFTCompoundData],
        structure_type: Optional[str] = None,
        initial_guess: Optional[Dict[str, float]] = None
    ) -> Tuple[Dict[str, float], Dict]:
        if structure_type:
            compounds = [c for c in compounds if c.structure_type == structure_type]

        if len(compounds) == 0:
            return {}, {"error": "No compounds found"}

        elements = set()
        for c in compounds:
            elements.add(c.element_A)
            elements.add(c.element_B)
        elements = sorted(list(elements))
        element_to_idx = {el: i for i, el in enumerate(elements)}
        n_elements = len(elements)

        if n_elements == 0:
            return {}, {"error": "No elements found"}

        if initial_guess:
            x0 = np.array([initial_guess.get(el, 1.4) for el in elements])
        else:
            x0 = np.array([PAULING_RADII.get(el, 1.4) for el in elements])

        def residuals(radii):
            res = []
            for c in compounds:
                a = c.lattice_constant
                idx_A = element_to_idx[c.element_A]
                idx_B = element_to_idx[c.element_B]

                if c.structure_type == "B2":
                    d_obs = (np.sqrt(3) / 2) * a
                    d_calc = radii[idx_A] + radii[idx_B]
                    res.append(d_calc - d_obs)
                elif c.structure_type == "L12":
                    r_A = radii[idx_A]
                    r_B = radii[idx_B]
                    if c.count_A > c.count_B:
                        r_major = r_A
                        r_minor = r_B
                    else:
                        r_major = r_B
                        r_minor = r_A
                    res.append(2 * r_major - a / np.sqrt(2))
                    res.append(r_major + r_minor - a / np.sqrt(2))

            return np.array(res)

        result = least_squares(
            residuals,
            x0,
            bounds=(0.5, 3.0),
            method="trf",
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10
        )

        radii = {el: result.x[element_to_idx[el]] for el in elements}

        residual_values = result.fun
        rmse = np.sqrt(np.mean(residual_values ** 2))
        mae = np.mean(np.abs(residual_values))

        stats = {
            "n_compounds": len(compounds),
            "n_elements": n_elements,
            "n_residuals": len(residual_values),
            "rmse": rmse,
            "mae": mae,
            "success": result.success,
            "cost": result.cost,
            "optimality": result.optimality
        }

        return radii, stats

    def compare_lattice_constants_corrected(
        self,
        radii: Dict[str, float],
        compounds: List[DFTCompoundData]
    ) -> pd.DataFrame:
        results = []

        for c in compounds:
            if c.element_A not in radii or c.element_B not in radii:
                continue

            r_A = radii[c.element_A]
            r_B = radii[c.element_B]

            if c.structure_type == "B2":
                a_calc = (2 / np.sqrt(3)) * (r_A + r_B)
            else:
                if c.count_A > c.count_B:
                    r_major = r_A
                    r_minor = r_B
                else:
                    r_major = r_B
                    r_minor = r_A
                a_AA = np.sqrt(2) * 2 * r_major
                a_AB = np.sqrt(2) * (r_major + r_minor)
                a_calc = (a_AA + a_AB) / 2

            a_dft = c.lattice_constant
            error = a_calc - a_dft
            rel_error = error / a_dft * 100

            results.append({
                "directory": c.directory,
                "composition": c.composition,
                "structure_type": c.structure_type,
                "element_A": c.element_A,
                "element_B": c.element_B,
                "a_DFT": a_dft,
                "a_calc": a_calc,
                "error": error,
                "rel_error_pct": rel_error
            })

        return pd.DataFrame(results)


def analyze_b2_around_4A(compounds: List[DFTCompoundData], output_dir: str):
    print("\n" + "=" * 70)
    print("B2 Structure Analysis: Lattice Constants around 4.0 Å")
    print("=" * 70)

    b2_compounds = [c for c in compounds if c.structure_type == "B2"]
    print(f"\nTotal B2 compounds: {len(b2_compounds)}")

    lattice_values = [c.lattice_constant for c in b2_compounds]
    print(f"Lattice constant range: {min(lattice_values):.3f} - {max(lattice_values):.3f} Å")
    print(f"Mean: {np.mean(lattice_values):.3f} Å, Std: {np.std(lattice_values):.3f} Å")

    ranges = [
        (3.8, 4.2, "3.8-4.2 Å (narrow)"),
        (3.7, 4.3, "3.7-4.3 Å (medium)"),
        (3.5, 4.5, "3.5-4.5 Å (wide)")
    ]

    results = {}
    calculator = FilteredRadiusCalculator(compounds)

    for a_min, a_max, label in ranges:
        filtered = [c for c in b2_compounds if a_min <= c.lattice_constant <= a_max]
        print(f"\n--- {label}: {len(filtered)} compounds ---")

        if len(filtered) < 10:
            print("  Too few compounds for reliable fitting")
            continue

        radii, stats = calculator.calculate_radii_trf(filtered, structure_type="B2")
        print(f"  Elements: {stats['n_elements']}")
        print(f"  RMSE: {stats['rmse']:.4f} Å, MAE: {stats['mae']:.4f} Å")

        comparison = calculator.compare_lattice_constants_corrected(radii, filtered)
        lattice_rmse = np.sqrt(np.mean(comparison['error'] ** 2))
        lattice_mae = np.mean(np.abs(comparison['error']))
        print(f"  Lattice RMSE: {lattice_rmse:.4f} Å, MAE: {lattice_mae:.4f} Å")

        results[label] = {
            "range": (a_min, a_max),
            "n_compounds": len(filtered),
            "radii": radii,
            "stats": stats,
            "comparison": comparison
        }

    if "3.8-4.2 Å (narrow)" in results:
        result = results["3.8-4.2 Å (narrow)"]
        comparison = result["comparison"]
        radii = result["radii"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        ax = axes[0]
        ax.scatter(comparison['a_DFT'], comparison['a_calc'], alpha=0.6, s=50, c='steelblue')
        min_val = min(comparison['a_DFT'].min(), comparison['a_calc'].min())
        max_val = max(comparison['a_DFT'].max(), comparison['a_calc'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x')
        ax.set_xlabel('DFT Lattice Constant (Å)', fontsize=12)
        ax.set_ylabel('Calculated Lattice Constant (Å)', fontsize=12)
        ax.set_title(f'B2 (a ∈ [3.8, 4.2] Å): Parity Plot\nn={len(comparison)}', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        rmse = np.sqrt(np.mean(comparison['error'] ** 2))
        mae = np.mean(np.abs(comparison['error']))
        stats_text = f'RMSE: {rmse:.4f} Å\nMAE: {mae:.4f} Å'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, ha='left', va='top',
                fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax = axes[1]
        elements = sorted(radii.keys())
        r_calc = [radii[el] for el in elements]
        r_pauling = [PAULING_RADII.get(el, np.nan) for el in elements]

        valid_idx = [i for i, r in enumerate(r_pauling) if not np.isnan(r)]
        elements_valid = [elements[i] for i in valid_idx]
        r_calc_valid = [r_calc[i] for i in valid_idx]
        r_pauling_valid = [r_pauling[i] for i in valid_idx]

        ax.scatter(r_pauling_valid, r_calc_valid, alpha=0.7, s=60, c='forestgreen')
        for i, el in enumerate(elements_valid):
            ax.annotate(el, (r_pauling_valid[i], r_calc_valid[i]), fontsize=8, alpha=0.8)

        min_r = min(min(r_pauling_valid), min(r_calc_valid))
        max_r = max(max(r_pauling_valid), max(r_calc_valid))
        ax.plot([min_r, max_r], [min_r, max_r], 'r--', linewidth=2, label='y=x')
        ax.set_xlabel('Pauling Radius (Å)', fontsize=12)
        ax.set_ylabel('Calculated Effective Radius (Å)', fontsize=12)
        ax.set_title('Calculated vs Pauling Radii\n(B2, a ∈ [3.8, 4.2] Å)', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = os.path.join(output_dir, "b2_around_4A_analysis.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nSaved B2 analysis plot to {filepath}")

        radii_df = pd.DataFrame([
            {"element": el, "r_calc": radii[el], "r_pauling": PAULING_RADII.get(el, np.nan)}
            for el in sorted(radii.keys())
        ])
        radii_filepath = os.path.join(output_dir, "b2_around_4A_radii.csv")
        radii_df.to_csv(radii_filepath, index=False)
        print(f"Saved B2 radii to {radii_filepath}")

    return results


def analyze_l12_accuracy(compounds: List[DFTCompoundData], output_dir: str):
    print("\n" + "=" * 70)
    print("L12 Structure Accuracy Investigation")
    print("=" * 70)

    l12_compounds = [c for c in compounds if c.structure_type == "L12"]
    print(f"\nTotal L12 compounds: {len(l12_compounds)}")

    lattice_values = [c.lattice_constant for c in l12_compounds]
    print(f"Lattice constant range: {min(lattice_values):.3f} - {max(lattice_values):.3f} Å")
    print(f"Mean: {np.mean(lattice_values):.3f} Å, Std: {np.std(lattice_values):.3f} Å")

    calculator = FilteredRadiusCalculator(compounds)

    print("\n--- Full L12 dataset ---")
    radii_full, stats_full = calculator.calculate_radii_trf(l12_compounds, structure_type="L12")
    print(f"Elements: {stats_full['n_elements']}")
    print(f"RMSE: {stats_full['rmse']:.4f} Å, MAE: {stats_full['mae']:.4f} Å")

    comparison_full = calculator.compare_lattice_constants_corrected(radii_full, l12_compounds)

    print("\n--- Error Analysis ---")
    print(f"Mean error: {comparison_full['error'].mean():.4f} Å")
    print(f"Std error: {comparison_full['error'].std():.4f} Å")
    print(f"Lattice RMSE: {np.sqrt(np.mean(comparison_full['error']**2)):.4f} Å")
    print(f"Lattice MAE: {np.mean(np.abs(comparison_full['error'])):.4f} Å")

    print("\n--- Outlier Analysis ---")
    outliers = comparison_full[np.abs(comparison_full['error']) > 0.5]
    print(f"Compounds with |error| > 0.5 Å: {len(outliers)} ({100*len(outliers)/len(comparison_full):.1f}%)")

    outlier_elements = set()
    for _, row in outliers.iterrows():
        outlier_elements.add(row['element_A'])
        outlier_elements.add(row['element_B'])
    print(f"Elements involved in outliers: {sorted(outlier_elements)}")

    lanthanides = {'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu'}
    lanthanide_outliers = outliers[
        outliers['element_A'].isin(lanthanides) | outliers['element_B'].isin(lanthanides)
    ]
    print(f"Outliers involving lanthanides: {len(lanthanide_outliers)} ({100*len(lanthanide_outliers)/len(outliers):.1f}% of outliers)")

    print("\n--- Filtered L12 Analysis (excluding large lattice constants) ---")
    thresholds = [5.5, 5.0, 4.5]

    results = {}
    for threshold in thresholds:
        filtered = [c for c in l12_compounds if c.lattice_constant <= threshold]
        print(f"\n  Threshold: a <= {threshold} Å ({len(filtered)} compounds)")

        if len(filtered) < 50:
            print("    Too few compounds")
            continue

        radii, stats = calculator.calculate_radii_trf(filtered, structure_type="L12")
        comparison = calculator.compare_lattice_constants_corrected(radii, filtered)

        lattice_rmse = np.sqrt(np.mean(comparison['error'] ** 2))
        lattice_mae = np.mean(np.abs(comparison['error']))
        print(f"    Elements: {stats['n_elements']}")
        print(f"    Fitting RMSE: {stats['rmse']:.4f} Å, MAE: {stats['mae']:.4f} Å")
        print(f"    Lattice RMSE: {lattice_rmse:.4f} Å, MAE: {lattice_mae:.4f} Å")

        results[threshold] = {
            "n_compounds": len(filtered),
            "radii": radii,
            "stats": stats,
            "comparison": comparison
        }

    print("\n--- Excluding Lanthanides ---")
    non_lanthanide = [c for c in l12_compounds 
                     if c.element_A not in lanthanides and c.element_B not in lanthanides]
    print(f"L12 compounds without lanthanides: {len(non_lanthanide)}")

    if len(non_lanthanide) >= 50:
        radii_nl, stats_nl = calculator.calculate_radii_trf(non_lanthanide, structure_type="L12")
        comparison_nl = calculator.compare_lattice_constants_corrected(radii_nl, non_lanthanide)

        lattice_rmse = np.sqrt(np.mean(comparison_nl['error'] ** 2))
        lattice_mae = np.mean(np.abs(comparison_nl['error']))
        print(f"  Elements: {stats_nl['n_elements']}")
        print(f"  Fitting RMSE: {stats_nl['rmse']:.4f} Å, MAE: {stats_nl['mae']:.4f} Å")
        print(f"  Lattice RMSE: {lattice_rmse:.4f} Å, MAE: {lattice_mae:.4f} Å")

        results["no_lanthanides"] = {
            "n_compounds": len(non_lanthanide),
            "radii": radii_nl,
            "stats": stats_nl,
            "comparison": comparison_nl
        }

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ax = axes[0, 0]
    ax.hist(comparison_full['a_DFT'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(x=5.0, color='red', linestyle='--', linewidth=2, label='a=5.0 Å threshold')
    ax.set_xlabel('DFT Lattice Constant (Å)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('L12 Lattice Constant Distribution', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.scatter(comparison_full['a_DFT'], comparison_full['error'], alpha=0.3, s=20, c='steelblue')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, label='±0.5 Å')
    ax.axhline(y=-0.5, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('DFT Lattice Constant (Å)', fontsize=12)
    ax.set_ylabel('Error (a_calc - a_DFT) (Å)', fontsize=12)
    ax.set_title('L12 Error vs Lattice Constant', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if 5.0 in results:
        comparison_filtered = results[5.0]["comparison"]
        ax = axes[1, 0]
        ax.scatter(comparison_filtered['a_DFT'], comparison_filtered['a_calc'], 
                  alpha=0.5, s=30, c='forestgreen')
        min_val = min(comparison_filtered['a_DFT'].min(), comparison_filtered['a_calc'].min())
        max_val = max(comparison_filtered['a_DFT'].max(), comparison_filtered['a_calc'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x')
        ax.set_xlabel('DFT Lattice Constant (Å)', fontsize=12)
        ax.set_ylabel('Calculated Lattice Constant (Å)', fontsize=12)
        ax.set_title(f'L12 (a ≤ 5.0 Å): Parity Plot\nn={len(comparison_filtered)}', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        rmse = np.sqrt(np.mean(comparison_filtered['error'] ** 2))
        mae = np.mean(np.abs(comparison_filtered['error']))
        stats_text = f'RMSE: {rmse:.4f} Å\nMAE: {mae:.4f} Å'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, ha='left', va='top',
                fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    if "no_lanthanides" in results:
        comparison_nl = results["no_lanthanides"]["comparison"]
        ax = axes[1, 1]
        ax.scatter(comparison_nl['a_DFT'], comparison_nl['a_calc'], 
                  alpha=0.5, s=30, c='darkorange')
        min_val = min(comparison_nl['a_DFT'].min(), comparison_nl['a_calc'].min())
        max_val = max(comparison_nl['a_DFT'].max(), comparison_nl['a_calc'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x')
        ax.set_xlabel('DFT Lattice Constant (Å)', fontsize=12)
        ax.set_ylabel('Calculated Lattice Constant (Å)', fontsize=12)
        ax.set_title(f'L12 (no lanthanides): Parity Plot\nn={len(comparison_nl)}', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        rmse = np.sqrt(np.mean(comparison_nl['error'] ** 2))
        mae = np.mean(np.abs(comparison_nl['error']))
        stats_text = f'RMSE: {rmse:.4f} Å\nMAE: {mae:.4f} Å'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, ha='left', va='top',
                fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    filepath = os.path.join(output_dir, "l12_accuracy_investigation.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved L12 investigation plot to {filepath}")

    return results


def generate_analysis_report(b2_results: Dict, l12_results: Dict, output_dir: str):
    report = """# DFT有効原子半径解析レポート

## 1. B2構造解析（格子定数 ≈ 4.0 Å）

### 1.1 フィルタリング結果

"""
    for label, result in b2_results.items():
        a_min, a_max = result["range"]
        report += f"#### {label}\n"
        report += f"- 化合物数: {result['n_compounds']}\n"
        report += f"- 元素数: {result['stats']['n_elements']}\n"
        report += f"- フィッティングRMSE: {result['stats']['rmse']:.4f} Å\n"
        report += f"- フィッティングMAE: {result['stats']['mae']:.4f} Å\n\n"

    if "3.8-4.2 Å (narrow)" in b2_results:
        result = b2_results["3.8-4.2 Å (narrow)"]
        report += "### 1.2 計算された有効原子半径（a ∈ [3.8, 4.2] Å）\n\n"
        report += "| 元素 | 計算半径 (Å) | Pauling半径 (Å) | 差 (Å) |\n"
        report += "|------|-------------|-----------------|--------|\n"
        for el in sorted(result["radii"].keys()):
            r_calc = result["radii"][el]
            r_pauling = PAULING_RADII.get(el, np.nan)
            diff = r_calc - r_pauling if not np.isnan(r_pauling) else np.nan
            report += f"| {el} | {r_calc:.4f} | {r_pauling:.4f} | {diff:+.4f} |\n"

    report += """
## 2. L12構造精度調査

### 2.1 問題の特定

L12構造の精度が低い主な原因:

1. **異常に大きな格子定数**: 一部のL12化合物は6-8 Åの格子定数を持ち、硬球モデルの予測（3-5 Å）と大きく乖離している。

2. **ランタノイド元素の影響**: 外れ値の多くがランタノイド元素（La, Ce, Gd, Dy, Ho, Er, Tm等）を含む化合物である。

3. **構造の妥当性**: これらの大きな格子定数を持つ化合物は、標準的なL12構造ではない可能性がある。

### 2.2 フィルタリングによる改善

"""
    for key, result in l12_results.items():
        if key == "no_lanthanides":
            report += f"#### ランタノイド除外\n"
        else:
            report += f"#### 格子定数 ≤ {key} Å\n"
        report += f"- 化合物数: {result['n_compounds']}\n"
        report += f"- 元素数: {result['stats']['n_elements']}\n"
        report += f"- フィッティングRMSE: {result['stats']['rmse']:.4f} Å\n"
        comparison = result["comparison"]
        lattice_rmse = np.sqrt(np.mean(comparison['error'] ** 2))
        report += f"- 格子定数RMSE: {lattice_rmse:.4f} Å\n\n"

    report += """
### 2.3 推奨事項

1. **データフィルタリング**: 格子定数が5.0 Å以下の化合物のみを使用することで、精度が大幅に向上する。

2. **ランタノイドの除外**: ランタノイド元素を含む化合物を除外することで、さらに精度が向上する可能性がある。

3. **構造検証**: 異常に大きな格子定数を持つ化合物について、DFT計算の妥当性を確認することを推奨する。

## 3. 可視化

- `b2_around_4A_analysis.png`: B2構造（a ≈ 4.0 Å）の解析結果
- `l12_accuracy_investigation.png`: L12構造の精度調査結果

---

*本レポートは自動生成されました。*
"""

    filepath = os.path.join(output_dir, "DFT_RADIUS_ANALYSIS_REPORT.md")
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\nSaved analysis report to {filepath}")


EXCLUDED_COMPOUNDS = ["Co1Mo1", "Mo1Co1"]


def filter_compounds_by_hea_elements(compounds: List[DFTCompoundData], hea_elements: List[str]) -> List[DFTCompoundData]:
    hea_set = set(hea_elements)
    filtered = [c for c in compounds if c.element_A in hea_set and c.element_B in hea_set]
    filtered = [c for c in filtered if c.directory not in EXCLUDED_COMPOUNDS]
    return filtered


def analyze_hea_elements(compounds: List[DFTCompoundData], output_dir: str):
    print("\n" + "=" * 70)
    print("HEA Elements Analysis (25 elements)")
    print("=" * 70)
    print(f"HEA elements: {', '.join(HEA_ELEMENTS)}")

    hea_compounds = filter_compounds_by_hea_elements(compounds, HEA_ELEMENTS)
    b2_hea = [c for c in hea_compounds if c.structure_type == "B2"]
    l12_hea = [c for c in hea_compounds if c.structure_type == "L12"]

    print(f"\nFiltered to HEA elements:")
    print(f"  Total: {len(hea_compounds)} compounds")
    print(f"  B2: {len(b2_hea)}")
    print(f"  L12: {len(l12_hea)}")

    calculator = FilteredRadiusCalculator(hea_compounds)
    results = {}

    print("\n--- B2 Structure (HEA elements only) ---")
    radii_b2, stats_b2 = calculator.calculate_radii_trf(b2_hea, structure_type="B2")
    comparison_b2 = calculator.compare_lattice_constants_corrected(radii_b2, b2_hea)
    lattice_rmse_b2 = np.sqrt(np.mean(comparison_b2['error'] ** 2))
    lattice_mae_b2 = np.mean(np.abs(comparison_b2['error']))
    print(f"  Compounds: {stats_b2['n_compounds']}")
    print(f"  Elements: {stats_b2['n_elements']}")
    print(f"  Fitting RMSE: {stats_b2['rmse']:.4f} Å, MAE: {stats_b2['mae']:.4f} Å")
    print(f"  Lattice RMSE: {lattice_rmse_b2:.4f} Å, MAE: {lattice_mae_b2:.4f} Å")
    results["B2"] = {
        "radii": radii_b2,
        "stats": stats_b2,
        "comparison": comparison_b2
    }

    print("\n--- L12 Structure (HEA elements only) ---")
    radii_l12, stats_l12 = calculator.calculate_radii_trf(l12_hea, structure_type="L12")
    comparison_l12 = calculator.compare_lattice_constants_corrected(radii_l12, l12_hea)
    lattice_rmse_l12 = np.sqrt(np.mean(comparison_l12['error'] ** 2))
    lattice_mae_l12 = np.mean(np.abs(comparison_l12['error']))
    print(f"  Compounds: {stats_l12['n_compounds']}")
    print(f"  Elements: {stats_l12['n_elements']}")
    print(f"  Fitting RMSE: {stats_l12['rmse']:.4f} Å, MAE: {stats_l12['mae']:.4f} Å")
    print(f"  Lattice RMSE: {lattice_rmse_l12:.4f} Å, MAE: {lattice_mae_l12:.4f} Å")
    results["L12"] = {
        "radii": radii_l12,
        "stats": stats_l12,
        "comparison": comparison_l12
    }

    print("\n--- Combined (HEA elements only) ---")
    radii_combined, stats_combined = calculator.calculate_radii_trf(hea_compounds, structure_type=None)
    comparison_combined = pd.concat([comparison_b2, comparison_l12], ignore_index=True)
    print(f"  Compounds: {stats_combined['n_compounds']}")
    print(f"  Elements: {stats_combined['n_elements']}")
    print(f"  Fitting RMSE: {stats_combined['rmse']:.4f} Å, MAE: {stats_combined['mae']:.4f} Å")
    results["Combined"] = {
        "radii": radii_combined,
        "stats": stats_combined,
        "comparison": comparison_combined
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ax = axes[0, 0]
    ax.scatter(comparison_b2['a_DFT'], comparison_b2['a_calc'], alpha=0.5, s=40, c='steelblue')
    min_val = min(comparison_b2['a_DFT'].min(), comparison_b2['a_calc'].min())
    max_val = max(comparison_b2['a_DFT'].max(), comparison_b2['a_calc'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x')
    ax.set_xlabel('DFT Lattice Constant (Å)', fontsize=12)
    ax.set_ylabel('Calculated Lattice Constant (Å)', fontsize=12)
    ax.set_title(f'B2 (HEA elements): Parity Plot\nn={len(comparison_b2)}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    stats_text = f'RMSE: {lattice_rmse_b2:.4f} Å\nMAE: {lattice_mae_b2:.4f} Å'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, ha='left', va='top',
            fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax = axes[0, 1]
    ax.scatter(comparison_l12['a_DFT'], comparison_l12['a_calc'], alpha=0.5, s=40, c='forestgreen')
    min_val = min(comparison_l12['a_DFT'].min(), comparison_l12['a_calc'].min())
    max_val = max(comparison_l12['a_DFT'].max(), comparison_l12['a_calc'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x')
    ax.set_xlabel('DFT Lattice Constant (Å)', fontsize=12)
    ax.set_ylabel('Calculated Lattice Constant (Å)', fontsize=12)
    ax.set_title(f'L12 (HEA elements): Parity Plot\nn={len(comparison_l12)}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    stats_text = f'RMSE: {lattice_rmse_l12:.4f} Å\nMAE: {lattice_mae_l12:.4f} Å'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, ha='left', va='top',
            fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax = axes[1, 0]
    elements = sorted(radii_b2.keys())
    r_b2_list = [radii_b2.get(el, np.nan) for el in elements]
    r_pauling_b2 = [PAULING_RADII.get(el, np.nan) for el in elements]
    valid_idx_b2 = [i for i, (rb, rp) in enumerate(zip(r_b2_list, r_pauling_b2)) if not np.isnan(rb) and not np.isnan(rp)]
    elements_b2 = [elements[i] for i in valid_idx_b2]
    r_b2_valid = [r_b2_list[i] for i in valid_idx_b2]
    r_pauling_b2_valid = [r_pauling_b2[i] for i in valid_idx_b2]

    ax.scatter(r_pauling_b2_valid, r_b2_valid, alpha=0.7, s=60, c='steelblue', label='B2')
    for i, el in enumerate(elements_b2):
        ax.annotate(el, (r_pauling_b2_valid[i], r_b2_valid[i]), fontsize=9, alpha=0.8)

    elements_l12 = sorted(radii_l12.keys())
    r_l12_list = [radii_l12.get(el, np.nan) for el in elements_l12]
    r_pauling_l12 = [PAULING_RADII.get(el, np.nan) for el in elements_l12]
    valid_idx_l12 = [i for i, (rl, rp) in enumerate(zip(r_l12_list, r_pauling_l12)) if not np.isnan(rl) and not np.isnan(rp)]
    elements_l12_valid = [elements_l12[i] for i in valid_idx_l12]
    r_l12_valid = [r_l12_list[i] for i in valid_idx_l12]
    r_pauling_l12_valid = [r_pauling_l12[i] for i in valid_idx_l12]

    ax.scatter(r_pauling_l12_valid, r_l12_valid, alpha=0.7, s=60, c='forestgreen', marker='s', label='L12')

    all_pauling = r_pauling_b2_valid + r_pauling_l12_valid
    all_calc = r_b2_valid + r_l12_valid
    min_r = min(min(all_pauling), min(all_calc))
    max_r = max(max(all_pauling), max(all_calc))
    ax.plot([min_r, max_r], [min_r, max_r], 'r--', linewidth=2, label='y=x')
    ax.set_xlabel('Pauling Radius (Å)', fontsize=12)
    ax.set_ylabel('Effective Radius (Å)', fontsize=12)
    ax.set_title('Effective Radius vs Pauling Radius (HEA elements)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    ax = axes[1, 1]
    x = np.arange(len(elements))
    width = 0.25
    r_pauling_bar = [PAULING_RADII.get(el, np.nan) for el in elements]
    r_b2 = [radii_b2.get(el, np.nan) for el in elements]
    r_l12 = [radii_l12.get(el, np.nan) for el in elements]

    bars0 = ax.bar(x - width, r_pauling_bar, width, label='Pauling', color='darkorange', alpha=0.8)
    bars1 = ax.bar(x, r_b2, width, label='B2', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width, r_l12, width, label='L12', color='forestgreen', alpha=0.8)
    ax.set_xlabel('Element', fontsize=12)
    ax.set_ylabel('Effective Radius (Å)', fontsize=12)
    ax.set_title('Effective Radii Comparison: Pauling vs B2 vs L12 (HEA elements)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(elements, rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    filepath = os.path.join(output_dir, "hea_elements_analysis.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved HEA elements analysis plot to {filepath}")

    radii_df = pd.DataFrame([
        {
            "element": el,
            "r_B2": radii_b2.get(el, np.nan),
            "r_L12": radii_l12.get(el, np.nan),
            "r_combined": radii_combined.get(el, np.nan),
            "r_pauling": PAULING_RADII.get(el, np.nan)
        }
        for el in sorted(set(radii_b2.keys()) | set(radii_l12.keys()) | set(radii_combined.keys()))
    ])
    radii_filepath = os.path.join(output_dir, "hea_elements_radii.csv")
    radii_df.to_csv(radii_filepath, index=False)
    print(f"Saved HEA radii to {radii_filepath}")

    return results


def generate_hea_report(results: Dict, output_dir: str):
    report = """# HEA元素に限定したDFT有効原子半径解析レポート

## 1. 概要

本レポートは、高エントロピー合金（HEA）で利用される25元素のみに限定して、
DFT計算結果から有効原子半径を計算した結果をまとめたものである。

### 1.1 対象元素（25元素）

Al, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, Zr, Nb, Mo, Hf, Ta, W, Re, Si, Mg, Sc, Y, Pd, Pt, Au, Ag

## 2. 計算結果サマリー

"""
    for struct_type in ["B2", "L12", "Combined"]:
        if struct_type in results:
            result = results[struct_type]
            comparison = result["comparison"]
            lattice_rmse = np.sqrt(np.mean(comparison['error'] ** 2))
            lattice_mae = np.mean(np.abs(comparison['error']))
            report += f"### {struct_type}構造\n"
            report += f"- 化合物数: {result['stats']['n_compounds']}\n"
            report += f"- 元素数: {result['stats']['n_elements']}\n"
            report += f"- フィッティングRMSE: {result['stats']['rmse']:.4f} Å\n"
            report += f"- 格子定数RMSE: {lattice_rmse:.4f} Å\n"
            report += f"- 格子定数MAE: {lattice_mae:.4f} Å\n\n"

    report += "## 3. 計算された有効原子半径\n\n"
    report += "| 元素 | B2半径 (Å) | L12半径 (Å) | Combined (Å) | Pauling (Å) |\n"
    report += "|------|-----------|------------|--------------|-------------|\n"

    all_elements = set()
    for struct_type in ["B2", "L12", "Combined"]:
        if struct_type in results:
            all_elements.update(results[struct_type]["radii"].keys())

    for el in sorted(all_elements):
        r_b2 = results["B2"]["radii"].get(el, np.nan) if "B2" in results else np.nan
        r_l12 = results["L12"]["radii"].get(el, np.nan) if "L12" in results else np.nan
        r_combined = results["Combined"]["radii"].get(el, np.nan) if "Combined" in results else np.nan
        r_pauling = PAULING_RADII.get(el, np.nan)

        r_b2_str = f"{r_b2:.4f}" if not np.isnan(r_b2) else "-"
        r_l12_str = f"{r_l12:.4f}" if not np.isnan(r_l12) else "-"
        r_combined_str = f"{r_combined:.4f}" if not np.isnan(r_combined) else "-"
        r_pauling_str = f"{r_pauling:.4f}" if not np.isnan(r_pauling) else "-"

        report += f"| {el} | {r_b2_str} | {r_l12_str} | {r_combined_str} | {r_pauling_str} |\n"

    report += """
## 4. 考察

### 4.1 精度の改善

HEA元素のみに限定することで、ランタノイド元素による外れ値の影響を排除し、
より信頼性の高い有効原子半径を得ることができた。

### 4.2 構造依存性

B2構造とL12構造で計算された有効原子半径には差異が見られる。
これは、各構造における原子間の接触条件の違いを反映している。

## 5. 可視化

- `hea_elements_analysis.png`: HEA元素の解析結果
- `hea_elements_radii.csv`: 計算された有効原子半径のCSVファイル

---

*本レポートは自動生成されました。*
"""

    filepath = os.path.join(output_dir, "HEA_ELEMENTS_RADIUS_REPORT.md")
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Saved HEA report to {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="DFT Effective Radius Analysis Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Scan directories for VASP output files (auto-detect B2/L12 from directory names)
  python dft_radius_analysis.py --dir /path/to/BCC_B2 --dir /path/to/FCC_L12
  
  # Or use CSV files directly
  python dft_radius_analysis.py --b2 B2_result_etot_lattice.csv --l12 L12_result_etot_lattice.csv
  
  # Specify output directory
  python dft_radius_analysis.py --dir /path/to/data --output results/
        """
    )
    parser.add_argument(
        "--dir", "-d",
        action="append",
        dest="dirs",
        help="Directory to scan for VASP output files (can be specified multiple times)"
    )
    parser.add_argument(
        "--b2", "-b",
        help="Path to B2 structure CSV file (B2_result_etot_lattice.csv)"
    )
    parser.add_argument(
        "--l12", "-l",
        help="Path to L12 structure CSV file (L12_result_etot_lattice.csv)"
    )
    parser.add_argument(
        "--output", "-o",
        default="dft_radius_output",
        help="Output directory (default: dft_radius_output)"
    )

    args = parser.parse_args()

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("DFT Effective Radius Analysis")
    print("=" * 70)

    compounds = []
    
    if args.dirs:
        loader = DFTDataLoader()
        for dir_path in args.dirs:
            if os.path.isdir(dir_path):
                print(f"\nScanning directory: {dir_path}")
                dir_compounds = loader.load_from_directory(dir_path)
                compounds.extend(dir_compounds)
                print(f"  Found {len(dir_compounds)} compounds")
    
    if args.b2 or args.l12:
        loader = DFTDataLoader(args.b2, args.l12)
        csv_compounds = loader.load_data()
        compounds.extend(csv_compounds)
    
    if not compounds:
        print("\nNo data found. Please specify --dir or --b2/--l12 options.")
        print("Use --help for usage information.")
        return

    print(f"\nLoaded {len(compounds)} compounds")
    print(f"  B2: {len([c for c in compounds if c.structure_type == 'B2'])}")
    print(f"  L12: {len([c for c in compounds if c.structure_type == 'L12'])}")

    b2_results = analyze_b2_around_4A(compounds, output_dir)
    l12_results = analyze_l12_accuracy(compounds, output_dir)

    generate_analysis_report(b2_results, l12_results, output_dir)

    hea_results = analyze_hea_elements(compounds, output_dir)
    generate_hea_report(hea_results, output_dir)

    print("\n" + "=" * 70)
    print("Analysis completed!")
    print(f"Output directory: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
