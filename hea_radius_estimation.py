"""
HEA (High Entropy Alloy) Effective Atomic Radius Estimation

This script calculates effective atomic radii from B2 and L12 structure lattice constant data
using Trust Region Reflective (TRF) optimization method.

Key features:
1. Extract compound lattice constant data from Materials Project
2. Calculate effective atomic radii using TRF method
3. Support HEA 4-element and 5-element subset calculations
4. Exhaustive calculation for 25 major HEA elements
5. Compare with Pauling and Goldschmidt radii

Geometric relationships:
- B2 structure (AB): r_A + r_B = (sqrt(3)/2) * a
- L12 structure (A3B): r_A + r_B = a/sqrt(2)
"""

import itertools
import os
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

GOLDSCHMIDT_RADII = {
    "Li": 1.52, "Be": 1.12, "Na": 1.86, "Mg": 1.60, "Al": 1.43, "K": 2.27, "Ca": 1.97,
    "Sc": 1.61, "Ti": 1.45, "V": 1.32, "Cr": 1.25, "Mn": 1.12, "Fe": 1.24, "Co": 1.25,
    "Ni": 1.25, "Cu": 1.28, "Zn": 1.33, "Ga": 1.35, "Rb": 2.48, "Sr": 2.15, "Y": 1.81,
    "Zr": 1.60, "Nb": 1.43, "Mo": 1.36, "Ru": 1.34, "Rh": 1.34, "Pd": 1.37, "Ag": 1.44,
    "Cd": 1.49, "In": 1.63, "Sn": 1.41, "Cs": 2.65, "Ba": 2.17, "La": 1.87, "Ce": 1.82,
    "Pr": 1.83, "Nd": 1.82, "Sm": 1.81, "Eu": 2.04, "Gd": 1.79, "Tb": 1.77, "Dy": 1.77,
    "Ho": 1.76, "Er": 1.75, "Tm": 1.74, "Yb": 1.94, "Lu": 1.73, "Hf": 1.58, "Ta": 1.43,
    "W": 1.37, "Re": 1.37, "Os": 1.34, "Ir": 1.36, "Pt": 1.38, "Au": 1.44, "Hg": 1.50,
    "Tl": 1.71, "Pb": 1.75, "Bi": 1.55, "Th": 1.80, "U": 1.38
}

GOLDSCHMIDT_CN12_CORRECTION = 1.12 / 1.00

MAJOR_HEA_ELEMENTS = [
    "Al", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Zr", "Nb", "Mo", "Hf", "Ta", "W", "Re", "Si", "Mg", "Sc",
    "Y", "Pd", "Pt", "Au", "Ag"
]


@dataclass
class CompoundData:
    """Data class for binary compound information."""
    material_id: str
    formula: str
    structure_type: str
    element_A: str
    element_B: str
    count_A: float
    count_B: float
    lattice_constant: float
    energy_per_atom: float
    energy_above_hull: float


class MaterialsProjectDataExtractor:
    """Extract B2 and L12 compound data from Materials Project."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._mpr = None

    @property
    def mpr(self):
        if self._mpr is None:
            from mp_api.client import MPRester
            self._mpr = MPRester(self.api_key)
        return self._mpr

    def extract_compounds(
        self,
        energy_above_hull_max: float = 0.1
    ) -> List[CompoundData]:
        """Extract B2 and L12 compounds from Materials Project."""
        print("Extracting compounds from Materials Project...")

        docs = self.mpr.materials.summary.search(
            spacegroup_number=221,
            num_elements=2,
            energy_above_hull=(0, energy_above_hull_max),
            fields=[
                "material_id",
                "formula_pretty",
                "structure",
                "energy_per_atom",
                "energy_above_hull",
                "composition"
            ]
        )

        print(f"Found {len(docs)} binary cubic structures")

        compounds = []
        for doc in docs:
            try:
                structure = doc.structure
                lattice = structure.lattice

                if not self._is_cubic(lattice):
                    continue

                composition = doc.composition
                elements = list(composition.elements)

                if len(elements) != 2:
                    continue

                element_A = str(elements[0])
                element_B = str(elements[1])
                count_A = composition[elements[0]]
                count_B = composition[elements[1]]
                total_atoms = count_A + count_B

                ratio_A = count_A / total_atoms
                ratio_B = count_B / total_atoms

                structure_type = self._classify_structure(ratio_A, ratio_B)
                if structure_type is None:
                    continue

                compounds.append(CompoundData(
                    material_id=str(doc.material_id),
                    formula=doc.formula_pretty,
                    structure_type=structure_type,
                    element_A=element_A,
                    element_B=element_B,
                    count_A=count_A,
                    count_B=count_B,
                    lattice_constant=lattice.a,
                    energy_per_atom=doc.energy_per_atom,
                    energy_above_hull=doc.energy_above_hull
                ))

            except Exception as e:
                print(f"Error processing {doc.material_id}: {e}")
                continue

        print(f"Extracted {len(compounds)} valid B2/L12 compounds")
        return compounds

    def _is_cubic(self, lattice, tolerance: float = 0.01) -> bool:
        a, b, c = lattice.a, lattice.b, lattice.c
        avg = (a + b + c) / 3
        return (
            abs(a - avg) / avg < tolerance and
            abs(b - avg) / avg < tolerance and
            abs(c - avg) / avg < tolerance
        )

    def _classify_structure(
        self,
        ratio_A: float,
        ratio_B: float,
        tolerance: float = 0.05
    ) -> Optional[str]:
        if abs(ratio_A - 0.5) < tolerance and abs(ratio_B - 0.5) < tolerance:
            return "B2"
        elif abs(ratio_A - 0.75) < tolerance and abs(ratio_B - 0.25) < tolerance:
            return "L12"
        elif abs(ratio_A - 0.25) < tolerance and abs(ratio_B - 0.75) < tolerance:
            return "L12"
        return None


class HEARadiusCalculator:
    """Calculate effective atomic radii using TRF optimization for HEA applications."""

    def __init__(self, compounds: List[CompoundData]):
        self.compounds = compounds
        self.all_elements = self._get_all_elements()
        self.radii_b2: Dict[str, float] = {}
        self.radii_l12: Dict[str, float] = {}
        self.radii_combined: Dict[str, float] = {}

    def _get_all_elements(self) -> List[str]:
        elements = set()
        for c in self.compounds:
            elements.add(c.element_A)
            elements.add(c.element_B)
        return sorted(list(elements))

    def filter_compounds_by_elements(
        self,
        target_elements: List[str]
    ) -> List[CompoundData]:
        """Filter compounds to only include those with specified elements."""
        target_set = set(target_elements)
        filtered = [
            c for c in self.compounds
            if c.element_A in target_set and c.element_B in target_set
        ]
        return filtered

    def calculate_radii_trf(
        self,
        compounds: List[CompoundData],
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
                    d_obs = a / np.sqrt(2)
                    d_calc = radii[idx_A] + radii[idx_B]
                    res.append(d_calc - d_obs)

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
        self.radii_b2, stats_b2 = self.calculate_radii_trf(
            self.compounds, structure_type="B2"
        )
        print(f"B2: {stats_b2['n_compounds']} compounds, {stats_b2['n_elements']} elements")
        print(f"B2 RMSE: {stats_b2['rmse']:.4f} Å, MAE: {stats_b2['mae']:.4f} Å")

        print("\n=== Calculating L12 radii ===")
        self.radii_l12, stats_l12 = self.calculate_radii_trf(
            self.compounds, structure_type="L12"
        )
        print(f"L12: {stats_l12['n_compounds']} compounds, {stats_l12['n_elements']} elements")
        print(f"L12 RMSE: {stats_l12['rmse']:.4f} Å, MAE: {stats_l12['mae']:.4f} Å")

        print("\n=== Calculating combined radii ===")
        self.radii_combined, stats_combined = self.calculate_radii_trf(
            self.compounds, structure_type=None
        )
        print(f"Combined: {stats_combined['n_compounds']} compounds, "
              f"{stats_combined['n_elements']} elements")
        print(f"Combined RMSE: {stats_combined['rmse']:.4f} Å, MAE: {stats_combined['mae']:.4f} Å")

    def calculate_hea_subset_radii(
        self,
        target_elements: List[str],
        structure_type: Optional[str] = None
    ) -> Tuple[Dict[str, float], Dict, List[CompoundData]]:
        """
        Calculate radii using only compounds containing the specified HEA elements.

        Args:
            target_elements: List of elements in the HEA system
            structure_type: "B2", "L12", or None for combined

        Returns:
            Tuple of (radii dict, statistics dict, filtered compounds)
        """
        filtered = self.filter_compounds_by_elements(target_elements)

        if len(filtered) == 0:
            return {}, {"error": "No compounds found for specified elements"}, []

        radii, stats = self.calculate_radii_trf(
            filtered, structure_type=structure_type
        )

        return radii, stats, filtered

    def compare_lattice_constants(
        self,
        radii: Dict[str, float],
        compounds: List[CompoundData]
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
                a_calc = np.sqrt(2) * (r_A + r_B)

            a_dft = c.lattice_constant
            error = a_calc - a_dft
            rel_error = error / a_dft * 100

            results.append({
                "formula": c.formula,
                "structure_type": c.structure_type,
                "element_A": c.element_A,
                "element_B": c.element_B,
                "a_DFT": a_dft,
                "a_calc": a_calc,
                "error": error,
                "rel_error_pct": rel_error
            })

        return pd.DataFrame(results)


class HEAExhaustiveCalculator:
    """Perform exhaustive calculations for HEA element combinations."""

    def __init__(
        self,
        calculator: HEARadiusCalculator,
        major_elements: List[str] = None
    ):
        self.calculator = calculator
        self.major_elements = major_elements or MAJOR_HEA_ELEMENTS
        self.available_elements = [
            el for el in self.major_elements
            if el in calculator.all_elements
        ]
        print(f"Available major elements: {len(self.available_elements)}/{len(self.major_elements)}")
        print(f"Elements: {self.available_elements}")

    def calculate_n_element_combinations(
        self,
        n: int,
        structure_type: Optional[str] = None,
        min_compounds: int = 3
    ) -> pd.DataFrame:
        """
        Calculate radii for all n-element combinations.

        Args:
            n: Number of elements in combination (4 or 5 for HEA)
            structure_type: "B2", "L12", or None for combined
            min_compounds: Minimum number of compounds required

        Returns:
            DataFrame with results for each combination
        """
        combinations = list(itertools.combinations(self.available_elements, n))
        print(f"\nCalculating {len(combinations)} {n}-element combinations...")

        results = []
        for i, combo in enumerate(combinations):
            if (i + 1) % 100 == 0:
                print(f"Progress: {i + 1}/{len(combinations)}")

            elements = list(combo)
            radii, stats, filtered = self.calculator.calculate_hea_subset_radii(
                elements, structure_type=structure_type
            )

            if stats.get("error") or stats.get("n_compounds", 0) < min_compounds:
                continue

            comparison_df = self.calculator.compare_lattice_constants(radii, filtered)

            if len(comparison_df) == 0:
                continue

            rmse_lattice = np.sqrt(np.mean(comparison_df["error"] ** 2))
            mae_lattice = np.mean(np.abs(comparison_df["error"]))
            max_error = np.max(np.abs(comparison_df["error"]))
            mean_rel_error = np.mean(np.abs(comparison_df["rel_error_pct"]))

            results.append({
                "elements": "-".join(sorted(elements)),
                "n_elements": n,
                "n_compounds": stats["n_compounds"],
                "rmse_radius": stats["rmse"],
                "mae_radius": stats["mae"],
                "rmse_lattice": rmse_lattice,
                "mae_lattice": mae_lattice,
                "max_error_lattice": max_error,
                "mean_rel_error_pct": mean_rel_error,
                "radii": radii
            })

        results_df = pd.DataFrame(results)
        if len(results_df) > 0:
            results_df = results_df.sort_values("rmse_lattice")

        print(f"Completed: {len(results_df)} valid combinations")
        return results_df


class RadiusComparisonAnalyzer:
    """Analyze and compare effective radii with reference values."""

    def __init__(self, calculated_radii: Dict[str, float]):
        self.calculated_radii = calculated_radii

    def compare_with_pauling(self) -> pd.DataFrame:
        """Compare calculated radii with Pauling metallic radii."""
        results = []
        for el, r_calc in self.calculated_radii.items():
            r_pauling = PAULING_RADII.get(el)
            if r_pauling is not None:
                diff = r_calc - r_pauling
                rel_diff = diff / r_pauling * 100
                results.append({
                    "element": el,
                    "r_calculated": r_calc,
                    "r_pauling": r_pauling,
                    "difference": diff,
                    "rel_diff_pct": rel_diff
                })
        return pd.DataFrame(results)

    def compare_with_goldschmidt(self, apply_cn12_correction: bool = True) -> pd.DataFrame:
        """Compare calculated radii with Goldschmidt radii."""
        results = []
        for el, r_calc in self.calculated_radii.items():
            r_gold = GOLDSCHMIDT_RADII.get(el)
            if r_gold is not None:
                if apply_cn12_correction:
                    r_gold_corrected = r_gold * GOLDSCHMIDT_CN12_CORRECTION
                else:
                    r_gold_corrected = r_gold

                diff = r_calc - r_gold_corrected
                rel_diff = diff / r_gold_corrected * 100
                results.append({
                    "element": el,
                    "r_calculated": r_calc,
                    "r_goldschmidt": r_gold,
                    "r_goldschmidt_CN12": r_gold_corrected,
                    "difference": diff,
                    "rel_diff_pct": rel_diff
                })
        return pd.DataFrame(results)


class ReportGenerator:
    """Generate reports and visualizations."""

    def __init__(self, output_dir: str = "hea_radius_output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_radii_table(
        self,
        radii_b2: Dict[str, float],
        radii_l12: Dict[str, float],
        radii_combined: Dict[str, float],
        filename: str = "effective_radii.csv"
    ) -> None:
        """Save effective radii table to CSV."""
        all_elements = set(radii_b2.keys()) | set(radii_l12.keys()) | set(radii_combined.keys())

        data = []
        for el in sorted(all_elements):
            data.append({
                "element": el,
                "r_B2": radii_b2.get(el, np.nan),
                "r_L12": radii_l12.get(el, np.nan),
                "r_combined": radii_combined.get(el, np.nan),
                "r_pauling": PAULING_RADII.get(el, np.nan),
                "r_goldschmidt": GOLDSCHMIDT_RADII.get(el, np.nan)
            })

        df = pd.DataFrame(data)
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Saved radii table to {filepath}")

    def save_hea_results(
        self,
        results_df: pd.DataFrame,
        n_elements: int,
        structure_type: str,
        filename: str = None
    ) -> None:
        """Save HEA combination results to CSV."""
        if filename is None:
            filename = f"hea_{n_elements}element_{structure_type or 'combined'}_results.csv"

        results_to_save = results_df.drop(columns=["radii"], errors="ignore")
        filepath = os.path.join(self.output_dir, filename)
        results_to_save.to_csv(filepath, index=False)
        print(f"Saved HEA results to {filepath}")

    def plot_parity(
        self,
        comparison_df: pd.DataFrame,
        title: str,
        filename: str = "parity_plot.png"
    ) -> None:
        """Generate parity plot for lattice constant comparison."""
        fig, ax = plt.subplots(figsize=(8, 8))

        b2_data = comparison_df[comparison_df["structure_type"] == "B2"]
        l12_data = comparison_df[comparison_df["structure_type"] == "L12"]

        if len(b2_data) > 0:
            ax.scatter(
                b2_data["a_DFT"], b2_data["a_calc"],
                alpha=0.7, label=f"B2 (n={len(b2_data)})",
                edgecolors="k", linewidth=0.5
            )

        if len(l12_data) > 0:
            ax.scatter(
                l12_data["a_DFT"], l12_data["a_calc"],
                alpha=0.7, label=f"L12 (n={len(l12_data)})",
                marker="s", edgecolors="k", linewidth=0.5
            )

        all_vals = np.concatenate([comparison_df["a_DFT"], comparison_df["a_calc"]])
        min_val, max_val = all_vals.min(), all_vals.max()
        margin = (max_val - min_val) * 0.1
        ax.plot(
            [min_val - margin, max_val + margin],
            [min_val - margin, max_val + margin],
            "r--", linewidth=2, label="y = x"
        )

        rmse = np.sqrt(np.mean(comparison_df["error"] ** 2))
        mae = np.mean(np.abs(comparison_df["error"]))
        ax.text(
            0.05, 0.95,
            f"RMSE = {rmse:.4f} Å\nMAE = {mae:.4f} Å",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        )

        ax.set_xlabel("DFT Lattice Constant (Å)", fontsize=12)
        ax.set_ylabel("Calculated Lattice Constant (Å)", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc="lower right")
        ax.set_aspect("equal", adjustable="box")

        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved parity plot to {filepath}")

    def plot_radius_comparison(
        self,
        radii: Dict[str, float],
        title: str,
        filename: str = "radius_comparison.png"
    ) -> None:
        """Plot comparison of calculated radii with Pauling and Goldschmidt."""
        elements = sorted([el for el in radii.keys() if el in PAULING_RADII])

        if len(elements) == 0:
            return

        r_calc = [radii[el] for el in elements]
        r_pauling = [PAULING_RADII[el] for el in elements]
        r_gold = [GOLDSCHMIDT_RADII.get(el, np.nan) for el in elements]

        fig, ax = plt.subplots(figsize=(14, 6))

        x = np.arange(len(elements))
        width = 0.25

        ax.bar(x - width, r_calc, width, label="Calculated", alpha=0.8)
        ax.bar(x, r_pauling, width, label="Pauling", alpha=0.8)
        ax.bar(x + width, r_gold, width, label="Goldschmidt", alpha=0.8)

        ax.set_xlabel("Element", fontsize=12)
        ax.set_ylabel("Atomic Radius (Å)", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(elements, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved radius comparison plot to {filepath}")

    def plot_hea_error_distribution(
        self,
        results_df: pd.DataFrame,
        n_elements: int,
        filename: str = None
    ) -> None:
        """Plot error distribution for HEA combinations."""
        if len(results_df) == 0:
            return

        if filename is None:
            filename = f"hea_{n_elements}element_error_dist.png"

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].hist(results_df["rmse_lattice"], bins=30, edgecolor="black", alpha=0.7)
        axes[0].set_xlabel("RMSE (Å)", fontsize=12)
        axes[0].set_ylabel("Count", fontsize=12)
        axes[0].set_title(f"{n_elements}-Element Combinations: Lattice RMSE Distribution")
        axes[0].axvline(
            results_df["rmse_lattice"].median(),
            color="r", linestyle="--",
            label=f"Median: {results_df['rmse_lattice'].median():.4f}"
        )
        axes[0].legend()

        axes[1].hist(results_df["mean_rel_error_pct"], bins=30, edgecolor="black", alpha=0.7)
        axes[1].set_xlabel("Mean Relative Error (%)", fontsize=12)
        axes[1].set_ylabel("Count", fontsize=12)
        axes[1].set_title(f"{n_elements}-Element Combinations: Relative Error Distribution")
        axes[1].axvline(
            results_df["mean_rel_error_pct"].median(),
            color="r", linestyle="--",
            label=f"Median: {results_df['mean_rel_error_pct'].median():.2f}%"
        )
        axes[1].legend()

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved error distribution plot to {filepath}")


def main(api_key: str, output_dir: str = "hea_radius_output"):
    """Main function to run the complete HEA radius estimation pipeline."""
    print("=" * 70)
    print("HEA Effective Atomic Radius Estimation")
    print("=" * 70)

    print("\n=== Step 1: Extracting compound data from Materials Project ===")
    extractor = MaterialsProjectDataExtractor(api_key)
    compounds = extractor.extract_compounds(energy_above_hull_max=0.1)

    if len(compounds) == 0:
        print("No compounds extracted. Please check API key.")
        return

    compounds_df = pd.DataFrame([
        {
            "material_id": c.material_id,
            "formula": c.formula,
            "structure_type": c.structure_type,
            "element_A": c.element_A,
            "element_B": c.element_B,
            "lattice_constant": c.lattice_constant,
            "energy_above_hull": c.energy_above_hull
        }
        for c in compounds
    ])
    os.makedirs(output_dir, exist_ok=True)
    compounds_df.to_csv(os.path.join(output_dir, "compounds_data.csv"), index=False)
    print(f"Saved compound data to {output_dir}/compounds_data.csv")

    print("\n=== Step 2: Calculating effective atomic radii ===")
    calculator = HEARadiusCalculator(compounds)
    calculator.calculate_all_radii()

    print("\n=== Step 3: Comparing lattice constants ===")
    comparison_df = calculator.compare_lattice_constants(
        calculator.radii_combined, compounds
    )
    print(f"Lattice constant comparison: {len(comparison_df)} compounds")

    report = ReportGenerator(output_dir)

    report.save_radii_table(
        calculator.radii_b2,
        calculator.radii_l12,
        calculator.radii_combined
    )

    report.plot_parity(
        comparison_df,
        "Lattice Constant: DFT vs Calculated (Combined Radii)",
        "parity_combined.png"
    )

    report.plot_radius_comparison(
        calculator.radii_combined,
        "Effective Atomic Radii Comparison",
        "radius_comparison.png"
    )

    print("\n=== Step 4: Comparing with Pauling and Goldschmidt radii ===")
    analyzer = RadiusComparisonAnalyzer(calculator.radii_combined)

    pauling_comparison = analyzer.compare_with_pauling()
    pauling_comparison.to_csv(
        os.path.join(output_dir, "pauling_comparison.csv"), index=False
    )
    print(f"Pauling comparison: Mean diff = {pauling_comparison['difference'].mean():.4f} Å")

    goldschmidt_comparison = analyzer.compare_with_goldschmidt()
    goldschmidt_comparison.to_csv(
        os.path.join(output_dir, "goldschmidt_comparison.csv"), index=False
    )
    print(f"Goldschmidt comparison: Mean diff = {goldschmidt_comparison['difference'].mean():.4f} Å")

    print("\n=== Step 5: Exhaustive HEA calculations ===")
    exhaustive = HEAExhaustiveCalculator(calculator)

    print("\n--- 5-element combinations (combined B2+L12) ---")
    results_5elem = exhaustive.calculate_n_element_combinations(
        n=5, structure_type=None, min_compounds=3
    )
    if len(results_5elem) > 0:
        report.save_hea_results(results_5elem, 5, "combined")
        report.plot_hea_error_distribution(results_5elem, 5)

        print("\nTop 10 best 5-element combinations (lowest RMSE):")
        print(results_5elem[["elements", "n_compounds", "rmse_lattice", "mean_rel_error_pct"]].head(10).to_string())

        print("\nTop 10 worst 5-element combinations (highest RMSE):")
        print(results_5elem[["elements", "n_compounds", "rmse_lattice", "mean_rel_error_pct"]].tail(10).to_string())

    print("\n--- 4-element combinations (combined B2+L12) ---")
    results_4elem = exhaustive.calculate_n_element_combinations(
        n=4, structure_type=None, min_compounds=3
    )
    if len(results_4elem) > 0:
        report.save_hea_results(results_4elem, 4, "combined")
        report.plot_hea_error_distribution(results_4elem, 4)

        print("\nTop 10 best 4-element combinations (lowest RMSE):")
        print(results_4elem[["elements", "n_compounds", "rmse_lattice", "mean_rel_error_pct"]].head(10).to_string())

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print(f"Output files saved to: {output_dir}")
    print("=" * 70)

    return {
        "compounds": compounds,
        "calculator": calculator,
        "comparison_df": comparison_df,
        "results_5elem": results_5elem if len(results_5elem) > 0 else None,
        "results_4elem": results_4elem if len(results_4elem) > 0 else None
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        api_key = sys.argv[1]
    else:
        api_key = os.environ.get("MP_API_KEY", "")

    if not api_key:
        print("Please provide Materials Project API key as argument or set MP_API_KEY")
        sys.exit(1)

    output_dir = sys.argv[2] if len(sys.argv) > 2 else "hea_radius_output"
    main(api_key, output_dir)
