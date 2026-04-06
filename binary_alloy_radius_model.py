"""
Binary Alloy (B2/L1_2 Structure) Data Extraction and Radius Estimation Model

This script extracts B2 and L1_2 structure binary compound data from Materials Project,
calculates effective bonding radii from lattice constants using least squares optimization,
and builds machine learning models to predict radii and total energy.

Based on geometric relationships:
- B2 structure (AB): r_A + r_B = (sqrt(3)/2) * a
- L1_2 structure (A3B): a = max(2*sqrt(2)*r_major, sqrt(2)*(r_major+r_minor), 2*r_minor)
"""

import os
import warnings
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


class MaterialsProjectExtractor:
    """Extract B2 and L1_2 structure data from Materials Project database."""

    def __init__(self, api_key: str):
        """
        Initialize the extractor with Materials Project API key.

        Args:
            api_key: Materials Project API key
        """
        self.api_key = api_key
        self._mpr = None

    @property
    def mpr(self):
        """Lazy initialization of MPRester."""
        if self._mpr is None:
            from mp_api.client import MPRester
            self._mpr = MPRester(self.api_key)
        return self._mpr

    def extract_binary_cubic_structures(
        self,
        energy_above_hull_max: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Extract binary cubic structures (space group 221) from Materials Project.

        Args:
            energy_above_hull_max: Maximum energy above hull (eV/atom) for filtering.
                If None, no filtering is applied (all compounds are included).

        Returns:
            DataFrame with extracted structure data
        """
        print("Extracting binary cubic structures from Materials Project...")
        if energy_above_hull_max is not None:
            print(f"  Filtering: energy_above_hull <= {energy_above_hull_max} eV/atom")
        else:
            print("  No energy_above_hull filter applied (including unstable compounds)")

        search_kwargs = {
            "spacegroup_number": 221,
            "num_elements": 2,
            "fields": [
                "material_id",
                "formula_pretty",
                "structure",
                "energy_per_atom",
                "energy_above_hull",
                "composition",
                "symmetry"
            ]
        }
        if energy_above_hull_max is not None:
            search_kwargs["energy_above_hull"] = (0, energy_above_hull_max)

        docs = self.mpr.materials.summary.search(**search_kwargs)

        print(f"Found {len(docs)} binary cubic structures")

        data = []
        for doc in docs:
            try:
                structure = doc.structure
                lattice = structure.lattice

                if not self._is_cubic(lattice):
                    continue

                formula = doc.formula_pretty
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

                data.append({
                    "material_id": str(doc.material_id),
                    "formula": formula,
                    "structure_type": structure_type,
                    "element_A": element_A,
                    "element_B": element_B,
                    "count_A": count_A,
                    "count_B": count_B,
                    "lattice_constant_a": lattice.a,
                    "lattice_constant_b": lattice.b,
                    "lattice_constant_c": lattice.c,
                    "energy_per_atom": doc.energy_per_atom,
                    "energy_above_hull": doc.energy_above_hull,
                    "space_group": doc.symmetry.symbol if doc.symmetry else "Pm-3m"
                })

            except Exception as e:
                print(f"Error processing {doc.material_id}: {e}")
                continue

        df = pd.DataFrame(data)
        print(f"Extracted {len(df)} valid structures")

        return df

    def _is_cubic(self, lattice, tolerance: float = 0.01) -> bool:
        """Check if lattice is cubic (a = b = c)."""
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
        """
        Classify structure as B2 or L1_2 based on composition ratio.

        B2: AB (1:1 ratio)
        L1_2: A3B or AB3 (3:1 or 1:3 ratio)
        """
        if abs(ratio_A - 0.5) < tolerance and abs(ratio_B - 0.5) < tolerance:
            return "B2"
        elif abs(ratio_A - 0.75) < tolerance and abs(ratio_B - 0.25) < tolerance:
            return "L1_2"
        elif abs(ratio_A - 0.25) < tolerance and abs(ratio_B - 0.75) < tolerance:
            return "L1_2"
        return None


class RadiusCalculator:
    """Calculate effective bonding radii using least squares optimization."""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with structure data.

        Args:
            df: DataFrame with structure data
        """
        self.df = df
        self.element_radii_b2: Dict[str, float] = {}
        self.element_radii_l12: Dict[str, float] = {}

    def calculate_b2_radii(self) -> Dict[str, float]:
        """
        Calculate effective radii for B2 structures.

        For B2 (AB): r_A + r_B = (sqrt(3)/2) * a
        """
        b2_data = self.df[self.df["structure_type"] == "B2"].copy()

        if len(b2_data) == 0:
            print("No B2 structures found")
            return {}

        elements = set()
        for _, row in b2_data.iterrows():
            elements.add(row["element_A"])
            elements.add(row["element_B"])

        elements = sorted(list(elements))
        element_to_idx = {el: i for i, el in enumerate(elements)}
        n_elements = len(elements)

        print(f"B2 structures: {len(b2_data)} compounds, {n_elements} elements")

        def residuals(radii):
            res = []
            for _, row in b2_data.iterrows():
                a = row["lattice_constant_a"]
                d_obs = (np.sqrt(3) / 2) * a

                idx_A = element_to_idx[row["element_A"]]
                idx_B = element_to_idx[row["element_B"]]

                d_calc = radii[idx_A] + radii[idx_B]
                res.append(d_calc - d_obs)
            return np.array(res)

        initial_radii = np.ones(n_elements) * 1.5

        result = least_squares(
            residuals,
            initial_radii,
            bounds=(0.5, 3.0),
            method="trf"
        )

        self.element_radii_b2 = {
            el: result.x[element_to_idx[el]] for el in elements
        }

        rmse = np.sqrt(np.mean(result.fun ** 2))
        print(f"B2 radius optimization RMSE: {rmse:.4f} Angstrom")

        return self.element_radii_b2

    def calculate_l12_radii(self) -> Dict[str, float]:
        """
        Calculate effective radii for L1_2 structures.

        For L1_2 (A3B), three contact conditions:
        - A-A contact: a = 2*sqrt(2)*r_major
        - A-B contact: a = sqrt(2)*(r_major + r_minor)
        - B-B contact: a = 2*r_minor
        Lattice constant is determined by max of the three.
        """
        l12_data = self.df[self.df["structure_type"] == "L1_2"].copy()

        if len(l12_data) == 0:
            print("No L1_2 structures found")
            return {}

        elements = set()
        for _, row in l12_data.iterrows():
            elements.add(row["element_A"])
            elements.add(row["element_B"])

        elements = sorted(list(elements))
        element_to_idx = {el: i for i, el in enumerate(elements)}
        n_elements = len(elements)

        print(f"L1_2 structures: {len(l12_data)} compounds, {n_elements} elements")

        def residuals(radii):
            res = []
            for _, row in l12_data.iterrows():
                a = row["lattice_constant_a"]
                d_obs = a / np.sqrt(2)

                idx_A = element_to_idx[row["element_A"]]
                idx_B = element_to_idx[row["element_B"]]

                count_A = row["count_A"]
                count_B = row["count_B"]

                if count_A > count_B:
                    major_idx = idx_A
                    minor_idx = idx_B
                else:
                    major_idx = idx_B
                    minor_idx = idx_A

                r_major = radii[major_idx]
                r_minor = radii[minor_idx]
                # Three contact conditions — use max
                a_AA = 2 * np.sqrt(2) * r_major
                a_AB = np.sqrt(2) * (r_major + r_minor)
                a_BB = 2 * r_minor
                a_calc = max(a_AA, a_AB, a_BB)
                res.append(a_calc - a)

            return np.array(res)

        initial_radii = np.ones(n_elements) * 1.5

        result = least_squares(
            residuals,
            initial_radii,
            bounds=(0.5, 3.0),
            method="trf"
        )

        self.element_radii_l12 = {
            el: result.x[element_to_idx[el]] for el in elements
        }

        rmse = np.sqrt(np.mean(result.fun ** 2))
        print(f"L1_2 radius optimization RMSE: {rmse:.4f} Angstrom")

        return self.element_radii_l12

    def get_radius_table(self) -> pd.DataFrame:
        """Generate a table of calculated radii for all elements."""
        all_elements = set(self.element_radii_b2.keys()) | set(
            self.element_radii_l12.keys()
        )

        data = []
        for el in sorted(all_elements):
            data.append({
                "element": el,
                "radius_B2": self.element_radii_b2.get(el, np.nan),
                "radius_L1_2": self.element_radii_l12.get(el, np.nan)
            })

        return pd.DataFrame(data)


class ElementFeatureExtractor:
    """Extract atomic features for machine learning."""

    def __init__(self):
        """Initialize the feature extractor."""
        self._element_data = None

    @property
    def element_data(self) -> Dict:
        """Lazy load element data from mendeleev."""
        if self._element_data is None:
            from mendeleev import element as get_element
            self._element_data = {}

            for z in range(1, 104):
                try:
                    el = get_element(z)
                    self._element_data[el.symbol] = {
                        "atomic_number": el.atomic_number,
                        "period": el.period,
                        "group_id": el.group_id if el.group_id else 0,
                        "atomic_weight": el.atomic_weight,
                        "electronegativity": (
                            el.electronegativity(scale="pauling")
                            if el.electronegativity(scale="pauling")
                            else 0
                        ),
                        "electron_affinity": el.electron_affinity if el.electron_affinity else 0,
                        "atomic_radius": el.atomic_radius if el.atomic_radius else 0,
                        "covalent_radius": el.covalent_radius_pyykko if el.covalent_radius_pyykko else 0,
                        "vdw_radius": el.vdw_radius if el.vdw_radius else 0,
                        "ionization_energy": el.ionenergies.get(1, 0) if el.ionenergies else 0,
                    }
                except Exception:
                    continue

        return self._element_data

    def get_element_features(self, element: str) -> np.ndarray:
        """Get feature vector for a single element."""
        if element not in self.element_data:
            return np.zeros(10)

        data = self.element_data[element]
        return np.array([
            data["atomic_number"],
            data["period"],
            data["group_id"],
            data["atomic_weight"],
            data["electronegativity"],
            data["electron_affinity"],
            data["atomic_radius"],
            data["covalent_radius"],
            data["vdw_radius"],
            data["ionization_energy"],
        ])

    def get_compound_features(self, element_A: str, element_B: str) -> np.ndarray:
        """Get combined feature vector for a binary compound."""
        feat_A = self.get_element_features(element_A)
        feat_B = self.get_element_features(element_B)

        combined = np.concatenate([
            feat_A,
            feat_B,
            feat_A - feat_B,
            (feat_A + feat_B) / 2,
        ])

        return combined


class MLModelBuilder:
    """Build machine learning models for radius and energy prediction."""

    def __init__(
        self,
        df: pd.DataFrame,
        radius_calculator: RadiusCalculator,
        feature_extractor: ElementFeatureExtractor
    ):
        """
        Initialize the model builder.

        Args:
            df: DataFrame with structure data
            radius_calculator: RadiusCalculator with computed radii
            feature_extractor: ElementFeatureExtractor for atomic features
        """
        self.df = df
        self.radius_calculator = radius_calculator
        self.feature_extractor = feature_extractor
        self.models: Dict = {}
        self.scalers: Dict = {}
        self.results: Dict = {}

    def prepare_radius_data(self, structure_type: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare data for radius prediction model."""
        if structure_type == "B2":
            radii = self.radius_calculator.element_radii_b2
        else:
            radii = self.radius_calculator.element_radii_l12

        elements = list(radii.keys())
        X = np.array([
            self.feature_extractor.get_element_features(el) for el in elements
        ])
        y = np.array([radii[el] for el in elements])

        return X, y, elements

    def prepare_energy_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare data for energy prediction model with enhanced features."""
        X_list = []
        y_list = []
        formulas = []

        for _, row in self.df.iterrows():
            element_features = self.feature_extractor.get_compound_features(
                row["element_A"], row["element_B"]
            )

            structure_type = row["structure_type"]
            element_A = row["element_A"]
            element_B = row["element_B"]

            if structure_type == "B2":
                radii = self.radius_calculator.element_radii_b2
            else:
                radii = self.radius_calculator.element_radii_l12

            radius_A = radii.get(element_A, 1.5)
            radius_B = radii.get(element_B, 1.5)

            lattice_a = row["lattice_constant_a"]

            structure_features = np.array([
                lattice_a,
                radius_A,
                radius_B,
                radius_A + radius_B,
                abs(radius_A - radius_B),
                radius_A / radius_B if radius_B > 0 else 1.0,
                1.0 if structure_type == "B2" else 0.0,
                row["count_A"] / (row["count_A"] + row["count_B"]),
            ])

            features = np.concatenate([element_features, structure_features])
            X_list.append(features)
            y_list.append(row["energy_above_hull"])
            formulas.append(row["formula"])

        return np.array(X_list), np.array(y_list), formulas

    def train_radius_model(
        self,
        structure_type: str,
        model_type: str = "rf"
    ) -> Dict:
        """
        Train a model to predict atomic radii.

        Args:
            structure_type: "B2" or "L1_2"
            model_type: "rf" (Random Forest), "gb" (Gradient Boosting), or "gpr" (Gaussian Process)

        Returns:
            Dictionary with model performance metrics
        """
        X, y, elements = self.prepare_radius_data(structure_type)

        if len(X) < 5:
            print(f"Not enough data for {structure_type} radius model ({len(X)} samples)")
            return {}

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if model_type == "rf":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == "gb":
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        elif model_type == "gpr":
            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
            model = GaussianProcessRegressor(kernel=kernel, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        n_splits = min(5, len(X))
        cv_scores = cross_val_score(model, X_scaled, y, cv=n_splits, scoring="r2")
        y_pred = cross_val_predict(model, X_scaled, y, cv=n_splits)

        model.fit(X_scaled, y)

        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        r2 = np.mean(cv_scores)

        model_key = f"radius_{structure_type}_{model_type}"
        self.models[model_key] = model
        self.scalers[model_key] = scaler
        self.results[model_key] = {
            "rmse": rmse,
            "r2": r2,
            "y_true": y,
            "y_pred": y_pred,
            "elements": elements,
            "cv_scores": cv_scores
        }

        print(f"{structure_type} Radius Model ({model_type}): RMSE={rmse:.4f}, R2={r2:.4f}")

        return self.results[model_key]

    def train_energy_model(self, model_type: str = "rf") -> Dict:
        """
        Train a model to predict energy above hull (thermodynamic stability).

        Args:
            model_type: "rf" (Random Forest), "gb" (Gradient Boosting), or "gpr" (Gaussian Process)

        Returns:
            Dictionary with model performance metrics
        """
        X, y, formulas = self.prepare_energy_data()

        if len(X) < 5:
            print(f"Not enough data for energy model ({len(X)} samples)")
            return {}

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if model_type == "rf":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == "gb":
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        elif model_type == "gpr":
            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
            model = GaussianProcessRegressor(kernel=kernel, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        n_splits = min(5, len(X))
        cv_scores = cross_val_score(model, X_scaled, y, cv=n_splits, scoring="r2")
        y_pred = cross_val_predict(model, X_scaled, y, cv=n_splits)

        model.fit(X_scaled, y)

        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        r2 = np.mean(cv_scores)

        model_key = f"energy_{model_type}"
        self.models[model_key] = model
        self.scalers[model_key] = scaler
        self.results[model_key] = {
            "rmse": rmse,
            "r2": r2,
            "y_true": y,
            "y_pred": y_pred,
            "formulas": formulas,
            "cv_scores": cv_scores
        }

        print(f"Energy Model ({model_type}): RMSE={rmse:.4f}, R2={r2:.4f}")

        return self.results[model_key]


class ReportGenerator:
    """Generate evaluation reports and parity plots."""

    def __init__(self, output_dir: str = "."):
        """
        Initialize the report generator.

        Args:
            output_dir: Directory to save output files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_data_csv(self, df: pd.DataFrame, filename: str = "binary_alloy_data.csv"):
        """Save structure data to CSV."""
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Saved data to {filepath}")

    def save_radius_table(
        self,
        radius_table: pd.DataFrame,
        filename: str = "atomic_radius_table.csv"
    ):
        """Save atomic radius table to CSV."""
        filepath = os.path.join(self.output_dir, filename)
        radius_table.to_csv(filepath, index=False)
        print(f"Saved radius table to {filepath}")

    def generate_parity_plot(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str,
        xlabel: str,
        ylabel: str,
        labels: Optional[List[str]] = None,
        filename: str = "parity_plot.png"
    ):
        """Generate and save a parity plot."""
        fig, ax = plt.subplots(figsize=(8, 8))

        ax.scatter(y_true, y_pred, alpha=0.7, edgecolors="k", linewidth=0.5)

        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        margin = (max_val - min_val) * 0.1
        ax.plot(
            [min_val - margin, max_val + margin],
            [min_val - margin, max_val + margin],
            "r--",
            linewidth=2,
            label="y = x"
        )

        if labels is not None and len(labels) <= 30:
            for i, label in enumerate(labels):
                ax.annotate(
                    label,
                    (y_true[i], y_pred[i]),
                    fontsize=8,
                    alpha=0.7
                )

        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

        ax.text(
            0.05, 0.95,
            f"RMSE = {rmse:.4f}\nR$^2$ = {r2:.4f}",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        )

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc="lower right")
        ax.set_aspect("equal", adjustable="box")

        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved parity plot to {filepath}")

    def generate_evaluation_report(
        self,
        model_builder: MLModelBuilder,
        filename: str = "evaluation_report.txt"
    ):
        """Generate a text evaluation report."""
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("Binary Alloy Radius Model - Evaluation Report\n")
            f.write("=" * 60 + "\n\n")

            for model_key, results in model_builder.results.items():
                f.write(f"Model: {model_key}\n")
                f.write("-" * 40 + "\n")
                f.write(f"RMSE: {results['rmse']:.4f}\n")
                f.write(f"R2 Score: {results['r2']:.4f}\n")
                f.write(f"CV Scores: {results['cv_scores']}\n")
                f.write(f"CV Mean: {np.mean(results['cv_scores']):.4f}\n")
                f.write(f"CV Std: {np.std(results['cv_scores']):.4f}\n")
                f.write("\n")

        print(f"Saved evaluation report to {filepath}")


def remove_duplicate_compositions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate compositions, keeping the lowest energy structure.

    Args:
        df: DataFrame with structure data

    Returns:
        DataFrame with duplicates removed
    """
    df_sorted = df.sort_values("energy_per_atom")
    df_unique = df_sorted.drop_duplicates(
        subset=["element_A", "element_B", "structure_type"],
        keep="first"
    )
    print(f"Removed {len(df) - len(df_unique)} duplicate compositions")
    return df_unique


def main(api_key: str, output_dir: str = "binary_alloy_output"):
    """
    Main function to run the complete analysis pipeline.

    Args:
        api_key: Materials Project API key
        output_dir: Directory to save output files
    """
    print("=" * 60)
    print("Binary Alloy (B2/L1_2) Data Extraction and Radius Model")
    print("=" * 60)

    extractor = MaterialsProjectExtractor(api_key)
    df = extractor.extract_binary_cubic_structures(energy_above_hull_max=None)

    if len(df) == 0:
        print("No data extracted. Please check API key and connection.")
        return

    df = remove_duplicate_compositions(df)

    print("\n" + "=" * 60)
    print("Step 2: Calculating Effective Bonding Radii")
    print("=" * 60)

    radius_calc = RadiusCalculator(df)
    radius_calc.calculate_b2_radii()
    radius_calc.calculate_l12_radii()
    radius_table = radius_calc.get_radius_table()

    print("\nCalculated Radii:")
    print(radius_table.to_string(index=False))

    print("\n" + "=" * 60)
    print("Step 3: Building Machine Learning Models")
    print("=" * 60)

    feature_extractor = ElementFeatureExtractor()
    model_builder = MLModelBuilder(df, radius_calc, feature_extractor)

    if len(radius_calc.element_radii_b2) >= 5:
        model_builder.train_radius_model("B2", "rf")
        model_builder.train_radius_model("B2", "gb")

    if len(radius_calc.element_radii_l12) >= 5:
        model_builder.train_radius_model("L1_2", "rf")
        model_builder.train_radius_model("L1_2", "gb")

    if len(df) >= 5:
        model_builder.train_energy_model("rf")
        model_builder.train_energy_model("gb")

    print("\n" + "=" * 60)
    print("Step 4: Generating Reports and Plots")
    print("=" * 60)

    report_gen = ReportGenerator(output_dir)

    report_gen.save_data_csv(df)
    report_gen.save_radius_table(radius_table)

    for model_key, results in model_builder.results.items():
        if "radius" in model_key:
            report_gen.generate_parity_plot(
                results["y_true"],
                results["y_pred"],
                f"Parity Plot: {model_key}",
                "Calculated Radius (Angstrom)",
                "Predicted Radius (Angstrom)",
                results.get("elements"),
                f"parity_{model_key}.png"
            )
        else:
            report_gen.generate_parity_plot(
                results["y_true"],
                results["y_pred"],
                f"Parity Plot: {model_key}",
                "Energy Above Hull (eV/atom)",
                "Predicted Energy Above Hull (eV/atom)",
                None,
                f"parity_{model_key}.png"
            )

    report_gen.generate_evaluation_report(model_builder)

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print(f"Output files saved to: {output_dir}")
    print("=" * 60)

    return df, radius_calc, model_builder


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        api_key = sys.argv[1]
    else:
        api_key = os.environ.get("MP_API_KEY", "")

    if not api_key:
        print("Please provide Materials Project API key as argument or set MP_API_KEY environment variable")
        sys.exit(1)

    main(api_key)
