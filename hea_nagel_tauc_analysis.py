#!/usr/bin/env python3
"""
HEA Formation Analysis with Mizutani Parameters and Nagel-Tauc Theory
HEA形成能解析：Mizutaniパラメータとナーゲル・タウク理論

This module provides tools for analyzing High-Entropy Alloy (HEA) formation ability
based on Mizutani's electron concentration parameters and Nagel-Tauc theory.

The Nagel-Tauc theory (in the context of amorphous metals) suggests that:
- Electronic stability is achieved when the Fermi level (E_F) falls in a pseudo-gap
- The pseudo-gap arises from Fermi surface-Brillouin zone boundary interactions
- Electron concentration (e/a) determines E_F position relative to the pseudo-gap

Workflow:
1. Descriptor Generation: Calculate VEC, e/a, electronegativity, atomic radius, etc.
2. Statistical Analysis: Classification model for HEA formation, feature importance
3. First-Principles Workflow: Generate SQS structures, VASP inputs for DOS calculation
4. DOS Analysis: Extract pseudo-gap indicators and correlate with e/a

References:
    - Mizutani, U. (2010). Hume-Rothery Rules for Structurally Complex Alloy Phases
    - Nagel, S.R. & Tauc, J. (1975). Nearly-Free-Electron Approach to the Theory of 
      Metallic Glass Alloys. Phys. Rev. Lett. 35, 380
    - Guo, S. et al. (2011). Effect of valence electron concentration on stability of 
      fcc or bcc phase in high entropy alloys. J. Appl. Phys. 109, 103505

Author: Devin AI for minimumtone
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import json
import warnings
from enum import Enum

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. Statistical analysis features disabled.")

try:
    from ase import Atoms
    from ase.build import bulk
    from ase.io import write as ase_write
    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False
    warnings.warn("ASE not available. Structure generation features disabled.")


class CrystalStructure(Enum):
    """Crystal structure types for HEA"""
    FCC = "fcc"
    BCC = "bcc"
    HCP = "hcp"
    AMORPHOUS = "amorphous"
    MULTI_PHASE = "multi_phase"


@dataclass
class ElementData:
    """
    Element data for HEA descriptor calculation
    元素データ：HEA記述子計算用
    
    Data sources:
    - VEC: Valence Electron Concentration (Mizutani definition)
    - Pauling electronegativity
    - Metallic radius (Å)
    - Melting point (K)
    """
    symbol: str
    atomic_number: int
    vec: float  # Valence Electron Concentration
    electronegativity: float  # Pauling scale
    atomic_radius: float  # Metallic radius in Å
    melting_point: float  # K
    
    # Mizutani's e/a values for different phases
    # These may differ from simple VEC
    e_a_fcc: Optional[float] = None
    e_a_bcc: Optional[float] = None


class ElementDatabase:
    """
    Database of element properties for HEA analysis
    HEA解析用元素特性データベース
    
    VEC values follow Mizutani's convention where:
    - Transition metals: d-electrons + s-electrons
    - Main group: s + p electrons
    """
    
    def __init__(self):
        self._elements: Dict[str, ElementData] = {}
        self._load_default_elements()
    
    def _load_default_elements(self):
        """Load default element data"""
        # Common HEA constituent elements
        # VEC values based on Mizutani's definition
        elements_data = [
            # 3d transition metals
            ("Ti", 22, 4, 1.54, 1.47, 1941),
            ("V", 23, 5, 1.63, 1.34, 2183),
            ("Cr", 24, 6, 1.66, 1.28, 2180),
            ("Mn", 25, 7, 1.55, 1.27, 1519),
            ("Fe", 26, 8, 1.83, 1.26, 1811),
            ("Co", 27, 9, 1.88, 1.25, 1768),
            ("Ni", 28, 10, 1.91, 1.24, 1728),
            ("Cu", 29, 11, 1.90, 1.28, 1358),
            ("Zn", 30, 12, 1.65, 1.34, 693),
            # 4d transition metals
            ("Zr", 40, 4, 1.33, 1.60, 2128),
            ("Nb", 41, 5, 1.60, 1.46, 2750),
            ("Mo", 42, 6, 2.16, 1.39, 2896),
            ("Pd", 46, 10, 2.20, 1.37, 1828),
            ("Ag", 47, 11, 1.93, 1.44, 1235),
            # 5d transition metals
            ("Hf", 72, 4, 1.30, 1.59, 2506),
            ("Ta", 73, 5, 1.50, 1.46, 3290),
            ("W", 74, 6, 2.36, 1.39, 3695),
            ("Pt", 78, 10, 2.28, 1.39, 2041),
            ("Au", 79, 11, 2.54, 1.44, 1337),
            # Main group elements
            ("Al", 13, 3, 1.61, 1.43, 933),
            ("Si", 14, 4, 1.90, 1.18, 1687),
            ("Ge", 32, 4, 2.01, 1.22, 1211),
            ("Sn", 50, 4, 1.96, 1.40, 505),
            # Rare earth
            ("Y", 39, 3, 1.22, 1.80, 1799),
            ("La", 57, 3, 1.10, 1.87, 1193),
            ("Ce", 58, 3, 1.12, 1.82, 1068),
            # Others
            ("Be", 4, 2, 1.57, 1.12, 1560),
            ("Mg", 12, 2, 1.31, 1.60, 923),
            ("Ca", 20, 2, 1.00, 1.97, 1115),
            ("B", 5, 3, 2.04, 0.87, 2349),
            ("P", 15, 5, 2.19, 1.10, 317),
        ]
        
        for symbol, z, vec, en, r, tm in elements_data:
            self._elements[symbol] = ElementData(
                symbol=symbol,
                atomic_number=z,
                vec=vec,
                electronegativity=en,
                atomic_radius=r,
                melting_point=tm
            )
    
    def get_element(self, symbol: str) -> Optional[ElementData]:
        """Get element data by symbol"""
        return self._elements.get(symbol)
    
    def add_element(self, element: ElementData):
        """Add or update element data"""
        self._elements[element.symbol] = element
    
    def list_elements(self) -> List[str]:
        """List all available element symbols"""
        return list(self._elements.keys())


@dataclass
class HEAComposition:
    """
    HEA composition representation
    HEA組成の表現
    """
    elements: Dict[str, float]  # Element symbol -> atomic fraction
    name: Optional[str] = None
    is_hea: Optional[bool] = None  # Experimental label
    structure: Optional[CrystalStructure] = None
    mizutani_e_a: Optional[float] = None  # User-provided Mizutani parameter
    
    def __post_init__(self):
        # Normalize composition
        total = sum(self.elements.values())
        if abs(total - 1.0) > 1e-6:
            self.elements = {k: v/total for k, v in self.elements.items()}
    
    @classmethod
    def from_formula(cls, formula: str, is_hea: Optional[bool] = None) -> 'HEAComposition':
        """
        Parse composition from formula string
        例: "CoCrFeMnNi" -> equiatomic, "Co20Cr20Fe20Mn20Ni20" -> with percentages
        """
        import re
        
        # Pattern to match element and optional number
        pattern = r'([A-Z][a-z]?)(\d*\.?\d*)'
        matches = re.findall(pattern, formula)
        
        elements = {}
        for element, amount in matches:
            if element:
                if amount:
                    elements[element] = float(amount)
                else:
                    elements[element] = 1.0
        
        return cls(elements=elements, name=formula, is_hea=is_hea)
    
    def get_formula_string(self) -> str:
        """Generate formula string from composition"""
        parts = []
        for elem, frac in sorted(self.elements.items()):
            if abs(frac - round(frac)) < 0.01:
                parts.append(f"{elem}{int(round(frac*100))}")
            else:
                parts.append(f"{elem}{frac*100:.1f}")
        return "".join(parts)


@dataclass
class HEADescriptors:
    """
    Calculated descriptors for HEA
    HEA用計算記述子
    """
    composition: HEAComposition
    
    # Electron concentration parameters
    vec: float = 0.0  # Average Valence Electron Concentration
    e_a: float = 0.0  # Mizutani's electron concentration (if provided)
    
    # Electronegativity parameters
    chi_avg: float = 0.0  # Average electronegativity
    delta_chi: float = 0.0  # Electronegativity difference (std dev)
    
    # Atomic size parameters
    r_avg: float = 0.0  # Average atomic radius
    delta_r: float = 0.0  # Atomic size difference (%)
    
    # Thermodynamic parameters
    delta_H_mix: float = 0.0  # Mixing enthalpy (estimated)
    delta_S_mix: float = 0.0  # Mixing entropy
    omega: float = 0.0  # Ω parameter (T_m * ΔS_mix / |ΔH_mix|)
    
    # Melting point
    T_m_avg: float = 0.0  # Average melting point
    
    # Phase prediction parameters
    vec_fcc_bcc: str = ""  # FCC/BCC prediction based on VEC
    
    # Nagel-Tauc related
    k_f_estimate: float = 0.0  # Estimated Fermi wave vector
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame"""
        return {
            'formula': self.composition.get_formula_string(),
            'is_hea': self.composition.is_hea,
            'structure': self.composition.structure.value if self.composition.structure else None,
            'VEC': self.vec,
            'e_a': self.e_a if self.e_a > 0 else self.vec,
            'chi_avg': self.chi_avg,
            'delta_chi': self.delta_chi,
            'r_avg': self.r_avg,
            'delta_r': self.delta_r,
            'delta_H_mix': self.delta_H_mix,
            'delta_S_mix': self.delta_S_mix,
            'omega': self.omega,
            'T_m_avg': self.T_m_avg,
            'vec_fcc_bcc': self.vec_fcc_bcc,
            'k_f_estimate': self.k_f_estimate,
        }


class HEADescriptorCalculator:
    """
    Calculator for HEA descriptors
    HEA記述子計算器
    
    Implements:
    - VEC (Valence Electron Concentration) calculation
    - Mizutani e/a parameter handling
    - Electronegativity difference
    - Atomic size difference (δ)
    - Mixing enthalpy estimation (Miedema model approximation)
    - Mixing entropy
    - Ω parameter
    - Fermi wave vector estimation for Nagel-Tauc analysis
    """
    
    def __init__(self, element_db: Optional[ElementDatabase] = None):
        self.element_db = element_db or ElementDatabase()
        
        # Miedema mixing enthalpy parameters (simplified)
        # ΔH_mix = 4 * Σ_i Σ_j>i c_i * c_j * ΔH_ij
        # Using tabulated binary mixing enthalpies
        self._load_mixing_enthalpies()
    
    def _load_mixing_enthalpies(self):
        """Load binary mixing enthalpy data (kJ/mol)"""
        # Simplified Miedema-based values for common pairs
        # Negative = exothermic (favorable mixing)
        self.delta_H_binary: Dict[Tuple[str, str], float] = {
            # 3d-3d pairs
            ("Co", "Cr"): -4, ("Co", "Fe"): -1, ("Co", "Mn"): -5,
            ("Co", "Ni"): 0, ("Cr", "Fe"): -1, ("Cr", "Mn"): 2,
            ("Cr", "Ni"): -7, ("Fe", "Mn"): 0, ("Fe", "Ni"): -2,
            ("Mn", "Ni"): -8, ("Ti", "V"): -2, ("Ti", "Cr"): -7,
            ("Ti", "Fe"): -17, ("Ti", "Ni"): -35, ("Ti", "Co"): -28,
            ("V", "Cr"): -2, ("V", "Fe"): -7, ("V", "Ni"): -18,
            ("Cu", "Ni"): 4, ("Cu", "Co"): 6, ("Cu", "Fe"): 13,
            ("Cu", "Mn"): 4, ("Cu", "Cr"): 12,
            # Al pairs
            ("Al", "Co"): -19, ("Al", "Cr"): -10, ("Al", "Fe"): -11,
            ("Al", "Mn"): -19, ("Al", "Ni"): -22, ("Al", "Ti"): -30,
            ("Al", "Cu"): -1, ("Al", "Zr"): -44,
            # Zr pairs
            ("Zr", "Ti"): 0, ("Zr", "Ni"): -49, ("Zr", "Cu"): -23,
            ("Zr", "Co"): -41, ("Zr", "Fe"): -25,
            # Nb, Mo, Ta, W pairs
            ("Nb", "Ti"): 2, ("Nb", "Zr"): 4, ("Mo", "Ti"): -4,
            ("Ta", "Ti"): 1, ("W", "Ti"): -6,
        }
    
    def _get_binary_enthalpy(self, elem1: str, elem2: str) -> float:
        """Get binary mixing enthalpy for element pair"""
        if elem1 == elem2:
            return 0.0
        key1 = (elem1, elem2)
        key2 = (elem2, elem1)
        if key1 in self.delta_H_binary:
            return self.delta_H_binary[key1]
        elif key2 in self.delta_H_binary:
            return self.delta_H_binary[key2]
        else:
            # Estimate from electronegativity difference
            e1 = self.element_db.get_element(elem1)
            e2 = self.element_db.get_element(elem2)
            if e1 and e2:
                delta_chi = abs(e1.electronegativity - e2.electronegativity)
                return -10 * delta_chi  # Rough estimate
            return 0.0
    
    def calculate(self, composition: HEAComposition) -> HEADescriptors:
        """
        Calculate all descriptors for a composition
        組成に対する全記述子を計算
        """
        elements = composition.elements
        n_elements = len(elements)
        
        # Initialize accumulators
        vec_sum = 0.0
        chi_sum = 0.0
        r_sum = 0.0
        tm_sum = 0.0
        
        vec_list = []
        chi_list = []
        r_list = []
        c_list = []
        
        for elem, c in elements.items():
            elem_data = self.element_db.get_element(elem)
            if elem_data is None:
                warnings.warn(f"Element {elem} not in database, using defaults")
                elem_data = ElementData(elem, 0, 8, 1.5, 1.3, 1500)
            
            vec_sum += c * elem_data.vec
            chi_sum += c * elem_data.electronegativity
            r_sum += c * elem_data.atomic_radius
            tm_sum += c * elem_data.melting_point
            
            vec_list.append(elem_data.vec)
            chi_list.append(elem_data.electronegativity)
            r_list.append(elem_data.atomic_radius)
            c_list.append(c)
        
        c_arr = np.array(c_list)
        vec_arr = np.array(vec_list)
        chi_arr = np.array(chi_list)
        r_arr = np.array(r_list)
        
        # Average values
        vec_avg = vec_sum
        chi_avg = chi_sum
        r_avg = r_sum
        tm_avg = tm_sum
        
        # Electronegativity difference (weighted std dev)
        delta_chi = np.sqrt(np.sum(c_arr * (chi_arr - chi_avg)**2))
        
        # Atomic size difference δ (%)
        delta_r = 100 * np.sqrt(np.sum(c_arr * (1 - r_arr/r_avg)**2))
        
        # Mixing entropy (ideal)
        R = 8.314  # J/(mol·K)
        delta_S_mix = -R * np.sum(c_arr * np.log(c_arr + 1e-10))
        
        # Mixing enthalpy (Miedema approximation)
        delta_H_mix = 0.0
        elem_list = list(elements.keys())
        for i, elem1 in enumerate(elem_list):
            for j, elem2 in enumerate(elem_list):
                if i < j:
                    c1 = elements[elem1]
                    c2 = elements[elem2]
                    delta_H_ij = self._get_binary_enthalpy(elem1, elem2)
                    delta_H_mix += 4 * c1 * c2 * delta_H_ij
        
        # Ω parameter
        if abs(delta_H_mix) > 0.1:
            omega = tm_avg * delta_S_mix / (abs(delta_H_mix) * 1000)  # Convert kJ to J
        else:
            omega = float('inf')
        
        # VEC-based FCC/BCC prediction (Guo et al. 2011)
        if vec_avg >= 8.0:
            vec_fcc_bcc = "FCC"
        elif vec_avg <= 6.87:
            vec_fcc_bcc = "BCC"
        else:
            vec_fcc_bcc = "FCC+BCC"
        
        # Mizutani e/a (use provided value or VEC)
        e_a = composition.mizutani_e_a if composition.mizutani_e_a else vec_avg
        
        # Fermi wave vector estimate (free electron model)
        # k_F = (3π²n)^(1/3) where n = e/a / V_atom
        # Simplified: k_F ∝ (e/a)^(1/3) / r_avg
        # In units of 1/Å
        k_f_estimate = (3 * np.pi**2 * e_a)**(1/3) / (r_avg * 2)  # Approximate
        
        return HEADescriptors(
            composition=composition,
            vec=vec_avg,
            e_a=e_a,
            chi_avg=chi_avg,
            delta_chi=delta_chi,
            r_avg=r_avg,
            delta_r=delta_r,
            delta_H_mix=delta_H_mix,
            delta_S_mix=delta_S_mix,
            omega=omega,
            T_m_avg=tm_avg,
            vec_fcc_bcc=vec_fcc_bcc,
            k_f_estimate=k_f_estimate,
        )
    
    def calculate_batch(self, compositions: List[HEAComposition]) -> pd.DataFrame:
        """Calculate descriptors for multiple compositions"""
        results = []
        for comp in compositions:
            desc = self.calculate(comp)
            results.append(desc.to_dict())
        return pd.DataFrame(results)


class HEAFormationClassifier:
    """
    Statistical classifier for HEA formation prediction
    HEA形成予測用統計分類器
    
    Uses machine learning to:
    1. Predict HEA formation from descriptors
    2. Identify important features (especially e/a dependence)
    3. Estimate e/a threshold for HEA formation
    """
    
    def __init__(self, model_type: str = "random_forest"):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for classification")
        
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.feature_importances_: Optional[np.ndarray] = None
        
        self._init_model()
    
    def _init_model(self):
        """Initialize the classification model"""
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
        elif self.model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
        elif self.model_type == "logistic":
            self.model = LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, df: pd.DataFrame, target_col: str = 'is_hea',
            feature_cols: Optional[List[str]] = None):
        """
        Train the classifier
        
        Args:
            df: DataFrame with descriptors
            target_col: Column name for target variable
            feature_cols: List of feature column names (default: all numeric)
        """
        if feature_cols is None:
            feature_cols = ['VEC', 'e_a', 'chi_avg', 'delta_chi', 
                          'r_avg', 'delta_r', 'delta_H_mix', 
                          'delta_S_mix', 'omega', 'T_m_avg']
        
        # Filter to available columns
        feature_cols = [c for c in feature_cols if c in df.columns]
        self.feature_names = feature_cols
        
        X = df[feature_cols].values
        y = df[target_col].values.astype(int)
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0, posinf=100.0, neginf=-100.0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        
        # Store feature importances
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances_ = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            self.feature_importances_ = np.abs(self.model.coef_[0])
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict HEA formation"""
        X = df[self.feature_names].values
        X = np.nan_to_num(X, nan=0.0, posinf=100.0, neginf=-100.0)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict HEA formation probability"""
        X = df[self.feature_names].values
        X = np.nan_to_num(X, nan=0.0, posinf=100.0, neginf=-100.0)
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def cross_validate(self, df: pd.DataFrame, target_col: str = 'is_hea',
                      n_splits: int = 5) -> Dict[str, float]:
        """Perform cross-validation"""
        feature_cols = self.feature_names if self.feature_names else \
            ['VEC', 'e_a', 'chi_avg', 'delta_chi', 'r_avg', 'delta_r',
             'delta_H_mix', 'delta_S_mix', 'omega', 'T_m_avg']
        feature_cols = [c for c in feature_cols if c in df.columns]
        
        X = df[feature_cols].values
        y = df[target_col].values.astype(int)
        X = np.nan_to_num(X, nan=0.0, posinf=100.0, neginf=-100.0)
        
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        
        return {
            'mean_accuracy': scores.mean(),
            'std_accuracy': scores.std(),
            'scores': scores.tolist()
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance ranking"""
        if self.feature_importances_ is None:
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.feature_importances_
        })
        return importance_df.sort_values('importance', ascending=False)
    
    def estimate_e_a_threshold(self, df: pd.DataFrame, 
                               target_col: str = 'is_hea') -> Dict[str, Any]:
        """
        Estimate e/a threshold for HEA formation
        e/a閾値の推定
        
        Analyzes the distribution of e/a values for HEA vs non-HEA
        """
        hea_mask = df[target_col] == True
        
        e_a_hea = df.loc[hea_mask, 'e_a'].values
        e_a_non_hea = df.loc[~hea_mask, 'e_a'].values
        
        if len(e_a_hea) == 0 or len(e_a_non_hea) == 0:
            return {'threshold': None, 'message': 'Insufficient data'}
        
        # Simple threshold estimation
        mean_hea = np.mean(e_a_hea)
        mean_non_hea = np.mean(e_a_non_hea)
        threshold = (mean_hea + mean_non_hea) / 2
        
        # Optimal threshold search
        e_a_all = df['e_a'].values
        y = df[target_col].values.astype(int)
        
        best_threshold = threshold
        best_accuracy = 0
        
        for t in np.linspace(e_a_all.min(), e_a_all.max(), 100):
            pred = (e_a_all >= t).astype(int)
            acc = np.mean(pred == y)
            if acc > best_accuracy:
                best_accuracy = acc
                best_threshold = t
        
        return {
            'threshold': best_threshold,
            'accuracy_at_threshold': best_accuracy,
            'mean_e_a_hea': mean_hea,
            'mean_e_a_non_hea': mean_non_hea,
            'std_e_a_hea': np.std(e_a_hea),
            'std_e_a_non_hea': np.std(e_a_non_hea),
        }


class SQSGenerator:
    """
    Special Quasirandom Structure (SQS) generator for HEA
    HEA用特殊準乱数構造(SQS)生成器
    
    Generates structures that approximate random solid solutions
    for first-principles calculations.
    """
    
    def __init__(self, structure_type: str = "fcc", supercell_size: Tuple[int, int, int] = (3, 3, 3)):
        if not ASE_AVAILABLE:
            raise ImportError("ASE required for structure generation")
        
        self.structure_type = structure_type
        self.supercell_size = supercell_size
    
    def generate(self, composition: HEAComposition, 
                 lattice_constant: float = 3.6,
                 n_structures: int = 1) -> List[Atoms]:
        """
        Generate SQS-like structures
        
        Args:
            composition: HEA composition
            lattice_constant: Lattice constant in Å
            n_structures: Number of structures to generate
        
        Returns:
            List of ASE Atoms objects
        """
        # Create base structure
        if self.structure_type == "fcc":
            base = bulk("Cu", crystalstructure="fcc", a=lattice_constant, cubic=True)
        elif self.structure_type == "bcc":
            base = bulk("Fe", crystalstructure="bcc", a=lattice_constant, cubic=True)
        else:
            raise ValueError(f"Unsupported structure type: {self.structure_type}")
        
        # Create supercell
        supercell = base * self.supercell_size
        n_atoms = len(supercell)
        
        structures = []
        for _ in range(n_structures):
            # Assign elements randomly according to composition
            symbols = self._assign_elements(composition, n_atoms)
            
            # Create new structure with assigned elements
            new_structure = supercell.copy()
            new_structure.set_chemical_symbols(symbols)
            
            structures.append(new_structure)
        
        return structures
    
    def _assign_elements(self, composition: HEAComposition, n_atoms: int) -> List[str]:
        """Assign elements to atomic sites according to composition"""
        elements = list(composition.elements.keys())
        fractions = list(composition.elements.values())
        
        # Calculate number of atoms for each element
        n_per_element = [int(round(f * n_atoms)) for f in fractions]
        
        # Adjust for rounding errors
        diff = n_atoms - sum(n_per_element)
        if diff > 0:
            # Add to most abundant element
            idx = np.argmax(fractions)
            n_per_element[idx] += diff
        elif diff < 0:
            # Remove from most abundant element
            idx = np.argmax(fractions)
            n_per_element[idx] += diff
        
        # Create symbol list
        symbols = []
        for elem, n in zip(elements, n_per_element):
            symbols.extend([elem] * n)
        
        # Shuffle for randomness
        np.random.shuffle(symbols)
        
        return symbols


class VASPInputGenerator:
    """
    VASP input file generator for HEA DOS calculations
    HEA DOS計算用VASP入力ファイル生成器
    """
    
    def __init__(self, output_dir: str = "./vasp_inputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_inputs(self, structure: Atoms, calc_name: str,
                       encut: float = 400.0, kpoints: Tuple[int, int, int] = (4, 4, 4),
                       ismear: int = 1, sigma: float = 0.2) -> Path:
        """
        Generate VASP input files for DOS calculation
        
        Args:
            structure: ASE Atoms object
            calc_name: Name for the calculation directory
            encut: Plane wave cutoff energy (eV)
            kpoints: K-point mesh
            ismear: Smearing method (1 = Methfessel-Paxton)
            sigma: Smearing width (eV)
        
        Returns:
            Path to calculation directory
        """
        calc_dir = self.output_dir / calc_name
        calc_dir.mkdir(parents=True, exist_ok=True)
        
        # Write POSCAR
        ase_write(str(calc_dir / "POSCAR"), structure, format="vasp")
        
        # Write INCAR for relaxation
        self._write_incar_relax(calc_dir, encut, ismear, sigma)
        
        # Write INCAR for DOS
        self._write_incar_dos(calc_dir, encut, ismear, sigma)
        
        # Write KPOINTS
        self._write_kpoints(calc_dir, kpoints)
        
        # Write job script template
        self._write_job_script(calc_dir, calc_name)
        
        # Write metadata
        self._write_metadata(calc_dir, structure, calc_name)
        
        return calc_dir
    
    def _write_incar_relax(self, calc_dir: Path, encut: float, 
                          ismear: int, sigma: float):
        """Write INCAR for structure relaxation"""
        incar_content = f"""# INCAR for HEA structure relaxation
# Generated by HEA Nagel-Tauc Analysis Tool

# General settings
SYSTEM = HEA_relax
PREC = Accurate
ENCUT = {encut}
EDIFF = 1E-6
EDIFFG = -0.01

# Electronic relaxation
ISMEAR = {ismear}
SIGMA = {sigma}
LREAL = Auto

# Ionic relaxation
IBRION = 2
NSW = 100
ISIF = 3

# Output
LWAVE = .TRUE.
LCHARG = .TRUE.
LORBIT = 11

# Parallelization
NCORE = 4
"""
        with open(calc_dir / "INCAR_relax", 'w') as f:
            f.write(incar_content)
    
    def _write_incar_dos(self, calc_dir: Path, encut: float,
                        ismear: int, sigma: float):
        """Write INCAR for DOS calculation"""
        incar_content = f"""# INCAR for HEA DOS calculation
# Generated by HEA Nagel-Tauc Analysis Tool

# General settings
SYSTEM = HEA_DOS
PREC = Accurate
ENCUT = {encut}
EDIFF = 1E-6

# Electronic settings
ISMEAR = -5
SIGMA = {sigma}
LREAL = .FALSE.

# DOS settings
NEDOS = 3001
EMIN = -15
EMAX = 15

# No ionic relaxation
IBRION = -1
NSW = 0

# Output
LWAVE = .FALSE.
LCHARG = .FALSE.
LORBIT = 11

# Parallelization
NCORE = 4
"""
        with open(calc_dir / "INCAR_dos", 'w') as f:
            f.write(incar_content)
    
    def _write_kpoints(self, calc_dir: Path, kpoints: Tuple[int, int, int]):
        """Write KPOINTS file"""
        kpoints_content = f"""Automatic mesh
0
Gamma
{kpoints[0]} {kpoints[1]} {kpoints[2]}
0 0 0
"""
        with open(calc_dir / "KPOINTS", 'w') as f:
            f.write(kpoints_content)
    
    def _write_job_script(self, calc_dir: Path, calc_name: str):
        """Write job submission script template"""
        script_content = f"""#!/bin/bash
#SBATCH --job-name={calc_name}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=24:00:00

# Load VASP module (adjust for your system)
# module load vasp/6.3.0

# Step 1: Structure relaxation
cp INCAR_relax INCAR
mpirun -np $SLURM_NTASKS vasp_std > vasp_relax.out

# Step 2: DOS calculation
cp CONTCAR POSCAR
cp INCAR_dos INCAR
mpirun -np $SLURM_NTASKS vasp_std > vasp_dos.out

echo "Calculation completed"
"""
        with open(calc_dir / "run_vasp.sh", 'w') as f:
            f.write(script_content)
    
    def _write_metadata(self, calc_dir: Path, structure: Atoms, calc_name: str):
        """Write calculation metadata"""
        symbols = structure.get_chemical_symbols()
        unique_elements = list(set(symbols))
        composition = {elem: symbols.count(elem)/len(symbols) for elem in unique_elements}
        
        metadata = {
            'calc_name': calc_name,
            'n_atoms': len(structure),
            'elements': unique_elements,
            'composition': composition,
            'cell': structure.get_cell().tolist(),
        }
        
        with open(calc_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)


@dataclass
class PseudoGapIndicator:
    """
    Pseudo-gap indicator from DOS analysis
    DOS解析からの擬ギャップ指標
    """
    has_pseudogap: bool
    gap_width: float  # eV
    gap_depth: float  # Ratio of min DOS to max DOS
    gap_position: float  # Energy relative to E_F
    dos_at_fermi: float
    e_a: float  # Electron concentration
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'has_pseudogap': self.has_pseudogap,
            'gap_width': self.gap_width,
            'gap_depth': self.gap_depth,
            'gap_position': self.gap_position,
            'dos_at_fermi': self.dos_at_fermi,
            'e_a': self.e_a,
        }


class NagelTaucAnalyzer:
    """
    Nagel-Tauc theory analyzer for HEA
    HEA用ナーゲル・タウク理論解析器
    
    Analyzes the relationship between:
    - Electron concentration (e/a)
    - Fermi level position
    - Pseudo-gap in DOS
    - HEA formation ability
    
    Based on Nagel-Tauc theory:
    - Pseudo-gap forms when 2k_F ≈ K_p (reciprocal lattice vector)
    - Stability is enhanced when E_F falls in the pseudo-gap
    - e/a determines k_F through free electron model
    """
    
    def __init__(self):
        self.results: List[Dict[str, Any]] = []
    
    def analyze_e_a_pseudogap_correlation(self, 
                                          descriptors_df: pd.DataFrame,
                                          pseudogap_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze correlation between e/a and pseudo-gap indicators
        e/aと擬ギャップ指標の相関解析
        
        Args:
            descriptors_df: DataFrame with HEA descriptors (including e/a)
            pseudogap_df: DataFrame with pseudo-gap analysis results
        
        Returns:
            Merged DataFrame with correlation analysis
        """
        # Merge on formula or material_id
        if 'formula' in descriptors_df.columns and 'material_id' in pseudogap_df.columns:
            merged = pd.merge(
                descriptors_df, pseudogap_df,
                left_on='formula', right_on='material_id',
                how='inner'
            )
        else:
            merged = pd.concat([descriptors_df, pseudogap_df], axis=1)
        
        return merged
    
    def calculate_nagel_tauc_criterion(self, e_a: float, 
                                       lattice_constant: float,
                                       structure: str = "fcc") -> Dict[str, float]:
        """
        Calculate Nagel-Tauc criterion for given e/a
        与えられたe/aに対するナーゲル・タウク基準を計算
        
        The criterion is: 2k_F ≈ K_p
        where k_F = (3π²n)^(1/3) and K_p is the first reciprocal lattice vector
        
        Args:
            e_a: Electron concentration (electrons per atom)
            lattice_constant: Lattice constant in Å
            structure: Crystal structure ("fcc" or "bcc")
        
        Returns:
            Dictionary with k_F, K_p, and 2k_F/K_p ratio
        """
        # Atomic volume (Å³)
        if structure == "fcc":
            V_atom = lattice_constant**3 / 4  # 4 atoms per FCC unit cell
            # First reciprocal lattice vector for FCC: (111) type
            K_p = 2 * np.pi / lattice_constant * np.sqrt(3)
        elif structure == "bcc":
            V_atom = lattice_constant**3 / 2  # 2 atoms per BCC unit cell
            # First reciprocal lattice vector for BCC: (110) type
            K_p = 2 * np.pi / lattice_constant * np.sqrt(2)
        else:
            V_atom = lattice_constant**3
            K_p = 2 * np.pi / lattice_constant
        
        # Electron density (electrons per Å³)
        n = e_a / V_atom
        
        # Fermi wave vector (1/Å)
        k_F = (3 * np.pi**2 * n)**(1/3)
        
        # Nagel-Tauc ratio
        ratio_2kF_Kp = 2 * k_F / K_p
        
        return {
            'k_F': k_F,
            'K_p': K_p,
            '2k_F': 2 * k_F,
            '2k_F/K_p': ratio_2kF_Kp,
            'optimal_e_a': self._calculate_optimal_e_a(K_p, V_atom),
        }
    
    def _calculate_optimal_e_a(self, K_p: float, V_atom: float) -> float:
        """Calculate optimal e/a for Nagel-Tauc criterion (2k_F = K_p)"""
        # From 2k_F = K_p and k_F = (3π²n)^(1/3)
        # n = (K_p/2)³ / (3π²)
        # e/a = n * V_atom
        n_optimal = (K_p / 2)**3 / (3 * np.pi**2)
        return n_optimal * V_atom
    
    def generate_e_a_scan_report(self, 
                                 e_a_range: Tuple[float, float] = (4.0, 12.0),
                                 lattice_constant: float = 3.6,
                                 structure: str = "fcc",
                                 n_points: int = 50) -> pd.DataFrame:
        """
        Generate report scanning e/a values
        e/a値をスキャンしたレポートを生成
        """
        e_a_values = np.linspace(e_a_range[0], e_a_range[1], n_points)
        results = []
        
        for e_a in e_a_values:
            nt_result = self.calculate_nagel_tauc_criterion(e_a, lattice_constant, structure)
            results.append({
                'e_a': e_a,
                **nt_result
            })
        
        return pd.DataFrame(results)


class HEAExperimentalWorkflow:
    """
    Main workflow class for HEA formation analysis
    HEA形成解析のメインワークフロークラス
    
    Implements the complete experimental procedure:
    1. Load composition data and experimental labels
    2. Calculate descriptors (VEC, e/a, etc.)
    3. Train classification model
    4. Identify important features and e/a threshold
    5. Generate first-principles calculation inputs
    6. Analyze DOS results and correlate with e/a
    """
    
    def __init__(self, output_dir: str = "./hea_analysis_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.element_db = ElementDatabase()
        self.descriptor_calculator = HEADescriptorCalculator(self.element_db)
        self.classifier: Optional[HEAFormationClassifier] = None
        self.nagel_tauc_analyzer = NagelTaucAnalyzer()
        
        self.compositions: List[HEAComposition] = []
        self.descriptors_df: Optional[pd.DataFrame] = None
    
    def load_compositions_from_csv(self, csv_path: str,
                                   formula_col: str = 'formula',
                                   is_hea_col: str = 'is_hea',
                                   mizutani_col: Optional[str] = None) -> int:
        """
        Load compositions from CSV file
        CSVファイルから組成を読み込む
        
        Args:
            csv_path: Path to CSV file
            formula_col: Column name for formula
            is_hea_col: Column name for HEA label
            mizutani_col: Column name for Mizutani parameter (optional)
        
        Returns:
            Number of compositions loaded
        """
        df = pd.read_csv(csv_path)
        
        self.compositions = []
        for _, row in df.iterrows():
            formula = row[formula_col]
            is_hea = row.get(is_hea_col, None)
            mizutani_e_a = row.get(mizutani_col, None) if mizutani_col else None
            
            comp = HEAComposition.from_formula(formula, is_hea=is_hea)
            if mizutani_e_a is not None and not pd.isna(mizutani_e_a):
                comp.mizutani_e_a = float(mizutani_e_a)
            
            self.compositions.append(comp)
        
        return len(self.compositions)
    
    def add_composition(self, formula: str, is_hea: Optional[bool] = None,
                       mizutani_e_a: Optional[float] = None):
        """Add a single composition"""
        comp = HEAComposition.from_formula(formula, is_hea=is_hea)
        if mizutani_e_a is not None:
            comp.mizutani_e_a = mizutani_e_a
        self.compositions.append(comp)
    
    def calculate_all_descriptors(self) -> pd.DataFrame:
        """Calculate descriptors for all loaded compositions"""
        self.descriptors_df = self.descriptor_calculator.calculate_batch(self.compositions)
        return self.descriptors_df
    
    def train_classifier(self, model_type: str = "random_forest") -> Dict[str, Any]:
        """
        Train HEA formation classifier
        HEA形成分類器を訓練
        """
        if self.descriptors_df is None:
            self.calculate_all_descriptors()
        
        # Filter to compositions with labels
        labeled_df = self.descriptors_df[self.descriptors_df['is_hea'].notna()].copy()
        
        if len(labeled_df) < 10:
            return {'error': 'Insufficient labeled data (need at least 10 samples)'}
        
        self.classifier = HEAFormationClassifier(model_type=model_type)
        self.classifier.fit(labeled_df)
        
        # Cross-validation
        cv_results = self.classifier.cross_validate(labeled_df)
        
        # Feature importance
        importance_df = self.classifier.get_feature_importance()
        
        # e/a threshold estimation
        threshold_results = self.classifier.estimate_e_a_threshold(labeled_df)
        
        return {
            'cv_results': cv_results,
            'feature_importance': importance_df.to_dict('records'),
            'e_a_threshold': threshold_results,
        }
    
    def generate_vasp_inputs(self, 
                            compositions: Optional[List[HEAComposition]] = None,
                            structure_type: str = "fcc",
                            lattice_constant: float = 3.6,
                            supercell_size: Tuple[int, int, int] = (3, 3, 3),
                            n_structures_per_comp: int = 3) -> List[Path]:
        """
        Generate VASP input files for first-principles calculations
        第一原理計算用VASP入力ファイルを生成
        """
        if compositions is None:
            compositions = self.compositions
        
        sqs_generator = SQSGenerator(structure_type, supercell_size)
        vasp_generator = VASPInputGenerator(str(self.output_dir / "vasp_calculations"))
        
        calc_dirs = []
        for comp in compositions:
            structures = sqs_generator.generate(comp, lattice_constant, n_structures_per_comp)
            
            for i, structure in enumerate(structures):
                calc_name = f"{comp.get_formula_string()}_config{i+1}"
                calc_dir = vasp_generator.generate_inputs(structure, calc_name)
                calc_dirs.append(calc_dir)
        
        return calc_dirs
    
    def analyze_nagel_tauc(self, lattice_constant: float = 3.6,
                          structure: str = "fcc") -> pd.DataFrame:
        """
        Perform Nagel-Tauc analysis for all compositions
        全組成に対してナーゲル・タウク解析を実行
        """
        if self.descriptors_df is None:
            self.calculate_all_descriptors()
        
        results = []
        for _, row in self.descriptors_df.iterrows():
            e_a = row['e_a']
            nt_result = self.nagel_tauc_analyzer.calculate_nagel_tauc_criterion(
                e_a, lattice_constant, structure
            )
            results.append({
                'formula': row['formula'],
                'e_a': e_a,
                'is_hea': row['is_hea'],
                **nt_result
            })
        
        return pd.DataFrame(results)
    
    def generate_report(self) -> str:
        """
        Generate comprehensive analysis report
        包括的な解析レポートを生成
        """
        report_lines = [
            "=" * 80,
            "HEA Formation Analysis Report",
            "HEA形成能解析レポート",
            "=" * 80,
            "",
            f"Total compositions analyzed: {len(self.compositions)}",
            "",
        ]
        
        if self.descriptors_df is not None:
            report_lines.extend([
                "Descriptor Statistics:",
                "-" * 40,
                self.descriptors_df.describe().to_string(),
                "",
            ])
        
        if self.classifier is not None:
            importance_df = self.classifier.get_feature_importance()
            report_lines.extend([
                "Feature Importance:",
                "-" * 40,
                importance_df.to_string(),
                "",
            ])
        
        report_lines.extend([
            "=" * 80,
            "Nagel-Tauc Theory Notes:",
            "-" * 40,
            "The Nagel-Tauc criterion suggests that metallic glass (and HEA) stability",
            "is enhanced when 2k_F ≈ K_p, where:",
            "  - k_F: Fermi wave vector, determined by electron concentration (e/a)",
            "  - K_p: First reciprocal lattice vector",
            "",
            "When this condition is met, a pseudo-gap forms at the Fermi level,",
            "lowering the electronic energy and stabilizing the structure.",
            "",
            "For FCC structure with a ≈ 3.6 Å, optimal e/a ≈ 1.5-2.0",
            "For BCC structure with a ≈ 2.9 Å, optimal e/a ≈ 1.4-1.8",
            "",
            "=" * 80,
        ])
        
        report = "\n".join(report_lines)
        
        # Save report
        report_path = self.output_dir / "analysis_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        return report
    
    def save_results(self):
        """Save all results to files"""
        if self.descriptors_df is not None:
            self.descriptors_df.to_csv(self.output_dir / "descriptors.csv", index=False)
        
        # Save Nagel-Tauc scan
        nt_scan = self.nagel_tauc_analyzer.generate_e_a_scan_report()
        nt_scan.to_csv(self.output_dir / "nagel_tauc_scan.csv", index=False)
        
        # Generate and save report
        self.generate_report()


# Streamlit app integration
def create_streamlit_app():
    """Create Streamlit app for interactive HEA analysis"""
    try:
        import streamlit as st
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:
        print("Streamlit and/or Plotly not available")
        return
    
    st.set_page_config(
        page_title="HEA Nagel-Tauc Analysis",
        page_icon="🔬",
        layout="wide"
    )
    
    st.title("🔬 HEA Formation Analysis with Nagel-Tauc Theory")
    st.markdown("""
    This application analyzes High-Entropy Alloy (HEA) formation ability based on:
    - **Mizutani's electron concentration (e/a) parameter**
    - **Nagel-Tauc theory** for electronic stability
    
    ナーゲル・タウク理論に基づくHEA形成能解析ツール
    """)
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    # Initialize workflow
    if 'workflow' not in st.session_state:
        st.session_state.workflow = HEAExperimentalWorkflow()
    
    workflow = st.session_state.workflow
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Input",
        "🧮 Descriptors",
        "🤖 Classification",
        "⚛️ First-Principles",
        "📈 Nagel-Tauc Analysis"
    ])
    
    with tab1:
        st.header("Data Input")
        
        st.subheader("Add Composition Manually")
        col1, col2, col3 = st.columns(3)
        with col1:
            formula = st.text_input("Formula (e.g., CoCrFeMnNi)", "CoCrFeMnNi")
        with col2:
            is_hea = st.selectbox("Is HEA?", [None, True, False])
        with col3:
            mizutani_e_a = st.number_input("Mizutani e/a (optional)", value=0.0, step=0.1)
        
        if st.button("Add Composition"):
            workflow.add_composition(
                formula, 
                is_hea=is_hea if is_hea is not None else None,
                mizutani_e_a=mizutani_e_a if mizutani_e_a > 0 else None
            )
            st.success(f"Added: {formula}")
        
        st.subheader("Or Upload CSV")
        uploaded_file = st.file_uploader("Upload CSV with compositions", type=['csv'])
        if uploaded_file is not None:
            # Save temporarily and load
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            n_loaded = workflow.load_compositions_from_csv(tmp_path)
            st.success(f"Loaded {n_loaded} compositions")
        
        st.subheader("Current Compositions")
        if workflow.compositions:
            comp_data = [{'formula': c.get_formula_string(), 'is_hea': c.is_hea, 
                         'mizutani_e_a': c.mizutani_e_a} for c in workflow.compositions]
            st.dataframe(pd.DataFrame(comp_data))
    
    with tab2:
        st.header("Descriptor Calculation")
        
        if st.button("Calculate Descriptors"):
            if workflow.compositions:
                df = workflow.calculate_all_descriptors()
                st.session_state.descriptors_df = df
                st.success("Descriptors calculated!")
        
        if 'descriptors_df' in st.session_state:
            st.dataframe(st.session_state.descriptors_df)
            
            # Visualizations
            st.subheader("Descriptor Distributions")
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(st.session_state.descriptors_df, x='VEC', 
                                  color='is_hea', title='VEC Distribution')
                st.plotly_chart(fig)
            
            with col2:
                fig = px.histogram(st.session_state.descriptors_df, x='delta_r',
                                  color='is_hea', title='Atomic Size Difference (δ)')
                st.plotly_chart(fig)
    
    with tab3:
        st.header("HEA Formation Classification")
        
        model_type = st.selectbox("Model Type", 
                                 ["random_forest", "gradient_boosting", "logistic"])
        
        if st.button("Train Classifier"):
            if workflow.descriptors_df is not None:
                results = workflow.train_classifier(model_type)
                st.session_state.classifier_results = results
                
                if 'error' in results:
                    st.error(results['error'])
                else:
                    st.success("Classifier trained!")
                    
                    st.subheader("Cross-Validation Results")
                    st.write(f"Mean Accuracy: {results['cv_results']['mean_accuracy']:.3f}")
                    st.write(f"Std: {results['cv_results']['std_accuracy']:.3f}")
                    
                    st.subheader("Feature Importance")
                    importance_df = pd.DataFrame(results['feature_importance'])
                    fig = px.bar(importance_df, x='feature', y='importance',
                               title='Feature Importance')
                    st.plotly_chart(fig)
                    
                    st.subheader("e/a Threshold Analysis")
                    threshold = results['e_a_threshold']
                    st.write(f"Estimated threshold: {threshold['threshold']:.2f}")
                    st.write(f"Accuracy at threshold: {threshold['accuracy_at_threshold']:.3f}")
            else:
                st.warning("Please calculate descriptors first")
    
    with tab4:
        st.header("First-Principles Workflow")
        
        st.markdown("""
        Generate VASP input files for DOS calculations to verify Nagel-Tauc theory.
        
        DOS計算用VASP入力ファイルを生成し、ナーゲル・タウク理論を検証します。
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            structure_type = st.selectbox("Structure Type", ["fcc", "bcc"])
            lattice_constant = st.number_input("Lattice Constant (Å)", value=3.6, step=0.1)
        with col2:
            supercell_x = st.number_input("Supercell X", value=3, min_value=1, max_value=5)
            supercell_y = st.number_input("Supercell Y", value=3, min_value=1, max_value=5)
            supercell_z = st.number_input("Supercell Z", value=3, min_value=1, max_value=5)
        
        n_structures = st.number_input("Structures per composition", value=3, min_value=1, max_value=10)
        
        if st.button("Generate VASP Inputs"):
            if workflow.compositions:
                try:
                    calc_dirs = workflow.generate_vasp_inputs(
                        structure_type=structure_type,
                        lattice_constant=lattice_constant,
                        supercell_size=(supercell_x, supercell_y, supercell_z),
                        n_structures_per_comp=n_structures
                    )
                    st.success(f"Generated {len(calc_dirs)} calculation directories")
                    st.write("Output directory:", str(workflow.output_dir / "vasp_calculations"))
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("Please add compositions first")
    
    with tab5:
        st.header("Nagel-Tauc Analysis")
        
        st.markdown("""
        ### Theory Background
        
        The Nagel-Tauc criterion states that electronic stability is achieved when:
        
        **2k_F ≈ K_p**
        
        where:
        - k_F = (3π²n)^(1/3) is the Fermi wave vector
        - K_p is the first reciprocal lattice vector
        - n = e/a / V_atom is the electron density
        
        When this condition is met, a pseudo-gap forms at the Fermi level.
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            nt_lattice = st.number_input("Lattice Constant for N-T (Å)", value=3.6, step=0.1)
        with col2:
            nt_structure = st.selectbox("Structure for N-T", ["fcc", "bcc"])
        
        if st.button("Run Nagel-Tauc Analysis"):
            if workflow.descriptors_df is not None:
                nt_df = workflow.analyze_nagel_tauc(nt_lattice, nt_structure)
                st.session_state.nt_df = nt_df
                
                st.subheader("Results")
                st.dataframe(nt_df)
                
                # Plot 2k_F/K_p vs e/a
                fig = go.Figure()
                
                # Add scatter for compositions
                colors = ['green' if h else 'red' for h in nt_df['is_hea']]
                fig.add_trace(go.Scatter(
                    x=nt_df['e_a'],
                    y=nt_df['2k_F/K_p'],
                    mode='markers',
                    marker=dict(color=colors, size=10),
                    text=nt_df['formula'],
                    name='Compositions'
                ))
                
                # Add horizontal line at 2k_F/K_p = 1
                fig.add_hline(y=1.0, line_dash="dash", 
                            annotation_text="Nagel-Tauc criterion (2k_F = K_p)")
                
                fig.update_layout(
                    title="Nagel-Tauc Analysis: 2k_F/K_p vs e/a",
                    xaxis_title="Electron Concentration (e/a)",
                    yaxis_title="2k_F/K_p Ratio",
                )
                st.plotly_chart(fig)
                
                # e/a scan
                st.subheader("e/a Scan")
                scan_df = workflow.nagel_tauc_analyzer.generate_e_a_scan_report(
                    lattice_constant=nt_lattice, structure=nt_structure
                )
                
                fig2 = px.line(scan_df, x='e_a', y='2k_F/K_p',
                              title='2k_F/K_p vs e/a (theoretical)')
                fig2.add_hline(y=1.0, line_dash="dash")
                st.plotly_chart(fig2)
                
                optimal_e_a = scan_df.loc[
                    (scan_df['2k_F/K_p'] - 1.0).abs().idxmin(), 'e_a'
                ]
                st.info(f"Optimal e/a for Nagel-Tauc criterion: {optimal_e_a:.2f}")
            else:
                st.warning("Please calculate descriptors first")
        
        st.subheader("Generate Report")
        if st.button("Generate Full Report"):
            report = workflow.generate_report()
            workflow.save_results()
            st.text_area("Report", report, height=400)
            st.success(f"Results saved to {workflow.output_dir}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--streamlit":
        create_streamlit_app()
    else:
        # Demo usage
        print("HEA Nagel-Tauc Analysis Tool")
        print("=" * 50)
        
        # Create workflow
        workflow = HEAExperimentalWorkflow(output_dir="./hea_analysis_demo")
        
        # Add some example compositions
        example_compositions = [
            ("CoCrFeMnNi", True),  # Cantor alloy - HEA
            ("CoCrFeNi", True),    # 4-component HEA
            ("AlCoCrFeNi", True),  # Al-containing HEA
            ("TiVCrMnFe", True),   # Refractory HEA
            ("CuNi", False),       # Binary - not HEA
            ("FeNi", False),       # Binary - not HEA
            ("AlTiVCrMo", True),   # Refractory HEA
            ("NbMoTaW", True),     # Refractory HEA
        ]
        
        for formula, is_hea in example_compositions:
            workflow.add_composition(formula, is_hea=is_hea)
        
        # Calculate descriptors
        print("\nCalculating descriptors...")
        df = workflow.calculate_all_descriptors()
        print(df[['formula', 'VEC', 'e_a', 'delta_r', 'delta_H_mix', 'vec_fcc_bcc']].to_string())
        
        # Nagel-Tauc analysis
        print("\nNagel-Tauc Analysis:")
        nt_df = workflow.analyze_nagel_tauc(lattice_constant=3.6, structure="fcc")
        print(nt_df[['formula', 'e_a', '2k_F/K_p', 'optimal_e_a']].to_string())
        
        # Generate report
        print("\nGenerating report...")
        report = workflow.generate_report()
        print(report)
        
        print(f"\nResults saved to: {workflow.output_dir}")
        print("\nTo run the Streamlit app:")
        print("  streamlit run hea_nagel_tauc_analysis.py -- --streamlit")
