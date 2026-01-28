"""
DFT-based Effective Atomic Radius Calculation

This script calculates effective atomic radii from user's own DFT calculation results
(B2 and L12 structure lattice constant data) using Trust Region Reflective (TRF) optimization.

Based on the methodology from hea_radius_estimation.py, adapted for custom DFT data.

Geometric relationships:
- B2 structure (AB): r_A + r_B = (sqrt(3)/2) * a
- L12 structure (A3B): Two contact conditions
  - A-A contact (major-major): 2*r_major = a/sqrt(2)
  - A-B contact (major-minor): r_major + r_minor = a/sqrt(2)
"""

import os
import re
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

warnings.filterwarnings("ignore")

PAULING_RADII = {
    "H": 0.53, "Li": 1.55, "Be": 1.12, "B": 0.98, "C": 0.77, "N": 0.75, "O": 0.73,
    "Na": 1.90, "Mg": 1.60, "Al": 1.43, "Si": 1.17, "P": 1.10, "S": 1.04, "Cl": 0.99,
    "K": 2.35, "Ca": 1.97, "Sc": 1.64, "Ti": 1.47, "V": 1.35, "Cr": 1.29, "Mn": 1.37,
    "Fe": 1.26, "Co": 1.25, "Ni": 1.25, "Cu": 1.28, "Zn": 1.37, "Ga": 1.53, "Ge": 1.22,
    "As": 1.21, "Se": 1.17, "Br": 1.14, "Rb": 2.48, "Sr": 2.15, "Y": 1.82, "Zr": 1.60,
    "Nb": 1.47, "Mo": 1.40, "Tc": 1.35, "Ru": 1.34, "Rh": 1.34, "Pd": 1.37, "Ag": 1.44,
    "Cd": 1.52, "In": 1.67, "Sn": 1.58, "Sb": 1.61, "Te": 1.43, "I": 1.33, "Cs": 2.67,
    "Ba": 2.22, "La": 1.87, "Ce": 1.83, "Pr": 1.82, "Nd": 1.81, "Sm": 1.80, "Eu": 2.04,
    "Gd": 1.80, "Tb": 1.78, "Dy": 1.77, "Ho": 1.76, "Er": 1.75, "Tm": 1.74, "Yb": 1.93,
    "Lu": 1.74, "Hf": 1.59, "Ta": 1.47, "W": 1.41, "Re": 1.37, "Os": 1.35, "Ir": 1.36,
    "Pt": 1.39, "Au": 1.44, "Hg": 1.55, "Tl": 1.71, "Pb": 1.75, "Bi": 1.82, "Th": 1.80,
    "Pa": 1.63, "U": 1.54, "Pu": 1.64
}


@dataclass
class DFTCompoundData:
    """Data class for DFT compound information."""
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
    """Load and parse DFT calculation results from CSV files."""

    def __init__(self, b2_csv_path: str, l12_csv_path: str):
        self.b2_csv_path = b2_csv_path
        self.l12_csv_path = l12_csv_path

    def _parse_directory_name(self, directory: str, structure_type: str) -> Tuple[str, str, int, int]:
        """Parse directory name to extract element information."""
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

    def load_data(self) -> List[DFTCompoundData]:
        """Load and parse both B2 and L12 CSV files."""
        compounds = []

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

        print(f"Loaded {len(compounds)} compounds")
        print(f"  B2: {len([c for c in compounds if c.structure_type == 'B2'])}")
        print(f"  L12: {len([c for c in compounds if c.structure_type == 'L12'])}")

        return compounds


class EffectiveRadiusCalculator:
    """Calculate effective atomic radii using TRF optimization."""

    def __init__(self, compounds: List[DFTCompoundData]):
        self.compounds = compounds
        self.all_elements = self._get_all_elements()
        self.radii_b2: Dict[str, float] = {}
        self.radii_l12: Dict[str, float] = {}
        self.radii_combined: Dict[str, float] = {}
        self.stats_b2: Dict = {}
        self.stats_l12: Dict = {}
        self.stats_combined: Dict = {}

    def _get_all_elements(self) -> List[str]:
        """Get all unique elements from compounds."""
        elements = set()
        for c in self.compounds:
            elements.add(c.element_A)
            elements.add(c.element_B)
        return sorted(list(elements))

    def calculate_radii_trf(
        self,
        compounds: List[DFTCompoundData],
        structure_type: Optional[str] = None,
        initial_guess: Optional[Dict[str, float]] = None
    ) -> Tuple[Dict[str, float], Dict]:
        """
        Calculate effective atomic radii using TRF (Trust Region Reflective) method.

        Args:
            compounds: List of compound data
            structure_type: "B2", "L12", or None for combined
            initial_guess: Initial radius values for optimization

        Returns:
            Tuple of (radii dict, statistics dict)
        """
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

    def calculate_all_radii(self) -> None:
        """Calculate radii for B2, L12, and combined datasets."""
        print("\n=== Calculating B2 radii ===")
        self.radii_b2, self.stats_b2 = self.calculate_radii_trf(
            self.compounds, structure_type="B2"
        )
        print(f"B2: {self.stats_b2['n_compounds']} compounds, {self.stats_b2['n_elements']} elements")
        print(f"B2 RMSE: {self.stats_b2['rmse']:.4f} Å, MAE: {self.stats_b2['mae']:.4f} Å")

        print("\n=== Calculating L12 radii ===")
        self.radii_l12, self.stats_l12 = self.calculate_radii_trf(
            self.compounds, structure_type="L12"
        )
        print(f"L12: {self.stats_l12['n_compounds']} compounds, {self.stats_l12['n_elements']} elements")
        print(f"L12 RMSE: {self.stats_l12['rmse']:.4f} Å, MAE: {self.stats_l12['mae']:.4f} Å")

        print("\n=== Calculating combined radii ===")
        self.radii_combined, self.stats_combined = self.calculate_radii_trf(
            self.compounds, structure_type=None
        )
        print(f"Combined: {self.stats_combined['n_compounds']} compounds, "
              f"{self.stats_combined['n_elements']} elements")
        print(f"Combined RMSE: {self.stats_combined['rmse']:.4f} Å, MAE: {self.stats_combined['mae']:.4f} Å")

    def compare_lattice_constants(
        self,
        radii: Dict[str, float],
        compounds: List[DFTCompoundData]
    ) -> pd.DataFrame:
        """Compare calculated lattice constants with DFT values."""
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
                a_BB = 2 * r_minor
                a_calc = (a_AA + a_AB + a_BB) / 3

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


class ReportGenerator:
    """Generate comprehensive report and visualizations."""

    def __init__(self, output_dir: str = "dft_radius_output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_radii_table(
        self,
        radii_b2: Dict[str, float],
        radii_l12: Dict[str, float],
        radii_combined: Dict[str, float],
        filename: str = "effective_radii.csv"
    ) -> pd.DataFrame:
        """Save effective radii table to CSV."""
        all_elements = set(radii_b2.keys()) | set(radii_l12.keys()) | set(radii_combined.keys())

        data = []
        for el in sorted(all_elements):
            data.append({
                "element": el,
                "r_B2": radii_b2.get(el, np.nan),
                "r_L12": radii_l12.get(el, np.nan),
                "r_combined": radii_combined.get(el, np.nan),
                "r_pauling": PAULING_RADII.get(el, np.nan)
            })

        df = pd.DataFrame(data)
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Saved radii table to {filepath}")
        return df

    def plot_parity(
        self,
        comparison_df: pd.DataFrame,
        title: str,
        filename: str
    ) -> None:
        """Create parity plot comparing DFT and calculated lattice constants."""
        fig, ax = plt.subplots(figsize=(8, 8))

        b2_data = comparison_df[comparison_df['structure_type'] == 'B2']
        l12_data = comparison_df[comparison_df['structure_type'] == 'L12']

        if len(b2_data) > 0:
            ax.scatter(b2_data['a_DFT'], b2_data['a_calc'],
                      alpha=0.6, label=f'B2 (n={len(b2_data)})', s=30)
        if len(l12_data) > 0:
            ax.scatter(l12_data['a_DFT'], l12_data['a_calc'],
                      alpha=0.6, label=f'L12 (n={len(l12_data)})', s=30)

        all_vals = list(comparison_df['a_DFT']) + list(comparison_df['a_calc'])
        min_val, max_val = min(all_vals), max(all_vals)
        margin = (max_val - min_val) * 0.05
        ax.plot([min_val - margin, max_val + margin],
                [min_val - margin, max_val + margin],
                'k--', alpha=0.5, label='y=x')

        ax.set_xlabel('DFT Lattice Constant (Å)', fontsize=12)
        ax.set_ylabel('Calculated Lattice Constant (Å)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        filepath = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved parity plot to {filepath}")

    def plot_parity_separate(
        self,
        comparison_df: pd.DataFrame,
        structure_type: str,
        filename: str
    ) -> None:
        """Create separate parity plot for B2 or L12 structure."""
        data = comparison_df[comparison_df['structure_type'] == structure_type]

        if len(data) == 0:
            print(f"No data for {structure_type} structure")
            return

        fig, ax = plt.subplots(figsize=(10, 10))

        rmse = np.sqrt(np.mean(data['error'] ** 2))
        mae = np.mean(np.abs(data['error']))
        mean_rel_error = np.mean(np.abs(data['rel_error_pct']))

        ax.scatter(data['a_DFT'], data['a_calc'], alpha=0.5, s=40, c='steelblue', edgecolors='none')

        all_vals = list(data['a_DFT']) + list(data['a_calc'])
        min_val, max_val = min(all_vals), max(all_vals)
        margin = (max_val - min_val) * 0.05
        ax.plot([min_val - margin, max_val + margin],
                [min_val - margin, max_val + margin],
                'r--', linewidth=2, alpha=0.8, label='y=x (ideal)')

        ax.set_xlabel('DFT Lattice Constant (Å)', fontsize=14)
        ax.set_ylabel('Calculated Lattice Constant from Effective Radii (Å)', fontsize=14)
        ax.set_title(f'{structure_type} Structure: DFT vs Calculated Lattice Constants\n(n={len(data)} compounds)', fontsize=16)
        ax.legend(loc='upper left', fontsize=12)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        stats_text = f'RMSE: {rmse:.4f} Å\nMAE: {mae:.4f} Å\nMean |Rel. Error|: {mean_rel_error:.2f}%'
        ax.text(0.95, 0.05, stats_text, transform=ax.transAxes, ha='right', va='bottom',
                fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        filepath = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved {structure_type} parity plot to {filepath}")

    def plot_radius_comparison(
        self,
        radii_b2: Dict[str, float],
        radii_l12: Dict[str, float],
        filename: str = "radius_comparison.png"
    ) -> None:
        """Plot comparison of B2 and L12 radii with Pauling radii."""
        common_elements = sorted(set(radii_b2.keys()) & set(radii_l12.keys()))

        if len(common_elements) == 0:
            print("No common elements between B2 and L12")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        r_b2 = [radii_b2[el] for el in common_elements]
        r_l12 = [radii_l12[el] for el in common_elements]

        ax = axes[0]
        ax.scatter(r_b2, r_l12, alpha=0.6, s=50)
        for i, el in enumerate(common_elements):
            ax.annotate(el, (r_b2[i], r_l12[i]), fontsize=8, alpha=0.7)

        min_val = min(min(r_b2), min(r_l12))
        max_val = max(max(r_b2), max(r_l12))
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        ax.set_xlabel('B2 Effective Radius (Å)', fontsize=12)
        ax.set_ylabel('L12 Effective Radius (Å)', fontsize=12)
        ax.set_title('B2 vs L12 Effective Radii', fontsize=14)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        r_pauling = [PAULING_RADII.get(el, np.nan) for el in common_elements]
        r_combined = [(radii_b2[el] + radii_l12[el]) / 2 for el in common_elements]

        valid_idx = [i for i, r in enumerate(r_pauling) if not np.isnan(r)]
        if len(valid_idx) > 0:
            r_pauling_valid = [r_pauling[i] for i in valid_idx]
            r_combined_valid = [r_combined[i] for i in valid_idx]
            elements_valid = [common_elements[i] for i in valid_idx]

            ax.scatter(r_pauling_valid, r_combined_valid, alpha=0.6, s=50)
            for i, el in enumerate(elements_valid):
                ax.annotate(el, (r_pauling_valid[i], r_combined_valid[i]), fontsize=8, alpha=0.7)

            min_val = min(min(r_pauling_valid), min(r_combined_valid))
            max_val = max(max(r_pauling_valid), max(r_combined_valid))
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)

        ax.set_xlabel('Pauling Radius (Å)', fontsize=12)
        ax.set_ylabel('DFT Effective Radius (Å)', fontsize=12)
        ax.set_title('DFT vs Pauling Radii', fontsize=14)
        ax.grid(True, alpha=0.3)

        filepath = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved radius comparison plot to {filepath}")

    def plot_error_distribution(
        self,
        comparison_df: pd.DataFrame,
        filename: str = "error_distribution.png"
    ) -> None:
        """Plot error distribution histograms."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        for i, struct_type in enumerate(['B2', 'L12']):
            data = comparison_df[comparison_df['structure_type'] == struct_type]
            if len(data) == 0:
                continue

            ax = axes[0, i]
            ax.hist(data['error'], bins=30, alpha=0.7, edgecolor='black')
            ax.axvline(x=0, color='r', linestyle='--', alpha=0.7)
            ax.set_xlabel('Absolute Error (Å)', fontsize=11)
            ax.set_ylabel('Count', fontsize=11)
            ax.set_title(f'{struct_type} Absolute Error Distribution', fontsize=12)
            rmse = np.sqrt(np.mean(data['error'] ** 2))
            mae = np.mean(np.abs(data['error']))
            ax.text(0.95, 0.95, f'RMSE: {rmse:.4f} Å\nMAE: {mae:.4f} Å',
                   transform=ax.transAxes, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            ax = axes[1, i]
            ax.hist(data['rel_error_pct'], bins=30, alpha=0.7, edgecolor='black')
            ax.axvline(x=0, color='r', linestyle='--', alpha=0.7)
            ax.set_xlabel('Relative Error (%)', fontsize=11)
            ax.set_ylabel('Count', fontsize=11)
            ax.set_title(f'{struct_type} Relative Error Distribution', fontsize=12)
            mean_rel = np.mean(np.abs(data['rel_error_pct']))
            ax.text(0.95, 0.95, f'Mean |Rel. Error|: {mean_rel:.2f}%',
                   transform=ax.transAxes, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        filepath = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved error distribution plot to {filepath}")

    def generate_markdown_report(
        self,
        stats_b2: Dict,
        stats_l12: Dict,
        stats_combined: Dict,
        radii_df: pd.DataFrame,
        comparison_df: pd.DataFrame,
        filename: str = "DFT_EFFECTIVE_RADIUS_REPORT.md"
    ) -> str:
        """Generate comprehensive markdown report."""
        report = """# DFT計算に基づく有効原子半径推定レポート

## 1. 研究概要

本レポートは、ユーザー独自のDFT（密度汎関数理論）計算結果から、B2およびL12構造の二元系化合物の格子定数データを用いて、Trust Region Reflective (TRF) 法により有効原子半径を計算した結果をまとめたものである。

### 1.1 対象構造

| 構造 | 化学量論 | 空間群 | 幾何学的拘束条件 |
|-----|---------|-------|----------------|
| B2 (CsCl型) | AB | Pm-3m (221) | r_A + r_B = (√3/2) × a |
| L12 (Cu3Au型) | A3B | Pm-3m (221) | 2r_major = a/√2 (A-A接触), r_major + r_minor = a/√2 (A-B接触) |

### 1.2 最適化アルゴリズム

| パラメータ | 設定値 |
|-----------|-------|
| 手法 | Trust Region Reflective (TRF) |
| 目的関数 | 残差二乗和の最小化 |
| 半径下限 | 0.5 Å |
| 半径上限 | 3.0 Å |
| 初期値 | Pauling金属半径 |

---

## 2. データサマリー

"""
        report += f"""### 2.1 データ収集統計

| 項目 | B2構造 | L12構造 | 合計 |
|-----|--------|---------|------|
| 化合物数 | {stats_b2.get('n_compounds', 'N/A')} | {stats_l12.get('n_compounds', 'N/A')} | {stats_combined.get('n_compounds', 'N/A')} |
| 元素数 | {stats_b2.get('n_elements', 'N/A')} | {stats_l12.get('n_elements', 'N/A')} | {stats_combined.get('n_elements', 'N/A')} |

### 2.2 フィッティング精度

| 構造 | RMSE (Å) | MAE (Å) |
|-----|----------|---------|
| B2 | {stats_b2.get('rmse', 0):.4f} | {stats_b2.get('mae', 0):.4f} |
| L12 | {stats_l12.get('rmse', 0):.4f} | {stats_l12.get('mae', 0):.4f} |
| Combined | {stats_combined.get('rmse', 0):.4f} | {stats_combined.get('mae', 0):.4f} |

---

## 3. 有効原子半径結果

### 3.1 計算された有効原子半径

"""
        report += "| 元素 | B2半径 (Å) | L12半径 (Å) | Combined (Å) | Pauling (Å) | B2-Pauling差 (Å) |\n"
        report += "|-----|-----------|------------|--------------|-------------|------------------|\n"

        for _, row in radii_df.iterrows():
            el = row['element']
            r_b2 = row['r_B2']
            r_l12 = row['r_L12']
            r_comb = row['r_combined']
            r_paul = row['r_pauling']

            r_b2_str = f"{r_b2:.4f}" if not np.isnan(r_b2) else "-"
            r_l12_str = f"{r_l12:.4f}" if not np.isnan(r_l12) else "-"
            r_comb_str = f"{r_comb:.4f}" if not np.isnan(r_comb) else "-"
            r_paul_str = f"{r_paul:.4f}" if not np.isnan(r_paul) else "-"

            if not np.isnan(r_b2) and not np.isnan(r_paul):
                diff = r_b2 - r_paul
                diff_str = f"{diff:+.4f}"
            else:
                diff_str = "-"

            report += f"| {el} | {r_b2_str} | {r_l12_str} | {r_comb_str} | {r_paul_str} | {diff_str} |\n"

        report += """
---

## 4. 格子定数比較

### 4.1 パリティプロット

![パリティプロット](parity_plot_combined.png)

DFT計算による格子定数とTRF最適化有効原子半径から計算した格子定数の比較を示す。理想的な一致はy=x線上に位置する。

### 4.2 誤差分布

![誤差分布](error_distribution.png)

上段は絶対誤差、下段は相対誤差の分布を示す。

### 4.3 半径比較

![半径比較](radius_comparison.png)

左図はB2構造とL12構造で計算された有効原子半径の相関を示す。右図はDFT計算による有効原子半径とPauling半径の比較を示す。

---

## 5. 格子定数予測精度

"""
        b2_comp = comparison_df[comparison_df['structure_type'] == 'B2']
        l12_comp = comparison_df[comparison_df['structure_type'] == 'L12']

        if len(b2_comp) > 0:
            b2_rmse_lat = np.sqrt(np.mean(b2_comp['error'] ** 2))
            b2_mae_lat = np.mean(np.abs(b2_comp['error']))
            b2_mean_rel = np.mean(np.abs(b2_comp['rel_error_pct']))
        else:
            b2_rmse_lat = b2_mae_lat = b2_mean_rel = 0

        if len(l12_comp) > 0:
            l12_rmse_lat = np.sqrt(np.mean(l12_comp['error'] ** 2))
            l12_mae_lat = np.mean(np.abs(l12_comp['error']))
            l12_mean_rel = np.mean(np.abs(l12_comp['rel_error_pct']))
        else:
            l12_rmse_lat = l12_mae_lat = l12_mean_rel = 0

        report += f"""| 構造 | RMSE (Å) | MAE (Å) | 平均相対誤差 (%) |
|-----|----------|---------|-----------------|
| B2 | {b2_rmse_lat:.4f} | {b2_mae_lat:.4f} | {b2_mean_rel:.2f} |
| L12 | {l12_rmse_lat:.4f} | {l12_mae_lat:.4f} | {l12_mean_rel:.2f} |

---

## 6. 結論

### 6.1 主要な知見

1. **B2構造**: {stats_b2.get('n_compounds', 0)}個の化合物から{stats_b2.get('n_elements', 0)}種類の元素の有効原子半径を決定。RMSE = {stats_b2.get('rmse', 0):.4f} Åの精度を達成。

2. **L12構造**: {stats_l12.get('n_compounds', 0)}個の化合物から{stats_l12.get('n_elements', 0)}種類の元素の有効原子半径を決定。RMSE = {stats_l12.get('rmse', 0):.4f} Åの精度を達成。

3. **構造依存性**: B2構造とL12構造で計算された有効原子半径には相関があるが、完全な一致ではなく、構造依存性が存在する。

### 6.2 今後の展望

- 高エントロピー合金（HEA）の格子定数予測への応用
- 元素組み合わせごとの独立計算による精度向上の検討
- 他の結晶構造（FCC, BCC, HCP等）への拡張

---

*本レポートは自動生成されました。*
"""

        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Saved markdown report to {filepath}")

        return report


def main():
    """Main function to run the effective radius calculation."""
    b2_csv = "/home/ubuntu/attachments/89d9b86d-9a88-4da4-a5de-d30f4e1b60d3/B2_result_etot_lattice.csv"
    l12_csv = "/home/ubuntu/attachments/1c357809-e12f-4a25-9e25-1a26a18956ae/L12_result_etot_lattice.csv"

    output_dir = "docs/dft_radius_output"

    print("=" * 60)
    print("DFT-based Effective Atomic Radius Calculation")
    print("=" * 60)

    loader = DFTDataLoader(b2_csv, l12_csv)
    compounds = loader.load_data()

    calculator = EffectiveRadiusCalculator(compounds)
    calculator.calculate_all_radii()

    report_gen = ReportGenerator(output_dir)

    radii_df = report_gen.save_radii_table(
        calculator.radii_b2,
        calculator.radii_l12,
        calculator.radii_combined
    )

    b2_compounds = [c for c in compounds if c.structure_type == 'B2']
    l12_compounds = [c for c in compounds if c.structure_type == 'L12']

    comparison_b2 = calculator.compare_lattice_constants(calculator.radii_b2, b2_compounds)
    comparison_l12 = calculator.compare_lattice_constants(calculator.radii_l12, l12_compounds)
    comparison_combined = pd.concat([comparison_b2, comparison_l12], ignore_index=True)

    comparison_combined.to_csv(os.path.join(output_dir, "lattice_comparison.csv"), index=False)

    report_gen.plot_parity(
        comparison_combined,
        "DFT vs Calculated Lattice Constants",
        "parity_plot_combined.png"
    )

    report_gen.plot_parity_separate(
        comparison_combined,
        "B2",
        "parity_plot_B2.png"
    )

    report_gen.plot_parity_separate(
        comparison_combined,
        "L12",
        "parity_plot_L12.png"
    )

    report_gen.plot_radius_comparison(
        calculator.radii_b2,
        calculator.radii_l12
    )

    report_gen.plot_error_distribution(comparison_combined)

    report_gen.generate_markdown_report(
        calculator.stats_b2,
        calculator.stats_l12,
        calculator.stats_combined,
        radii_df,
        comparison_combined
    )

    print("\n" + "=" * 60)
    print("Calculation completed successfully!")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    return calculator, radii_df, comparison_combined


if __name__ == "__main__":
    main()
