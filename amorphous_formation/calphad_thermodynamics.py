"""
CALPHAD Thermodynamics Module for Amorphous Formation
CALPHAD熱力学モジュール：非晶質形成用

This module calculates the Gibbs energy difference between liquid and solid phases
(driving force for crystallization) using CALPHAD methodology.

Key equations:
    ΔG_m = G^L - G^S (Gibbs energy difference)
    ΔS_f = ΔH_f / T_m (Entropy of fusion)
    
The driving force increases as temperature decreases below T_m.

References:
    - Turnbull, D. (1950). J. Appl. Phys. 21, 1022
    - Thompson, C.V. & Spaepen, F. (1979). Acta Metall. 27, 1855
"""

import numpy as np
from typing import Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class ThermodynamicResult:
    """Container for thermodynamic calculation results."""
    temperature: np.ndarray
    delta_G: np.ndarray
    delta_S: np.ndarray
    delta_H: np.ndarray
    driving_force_normalized: np.ndarray


class CALPHADThermodynamics:
    """
    CALPHAD-based thermodynamic calculations for glass formation.
    ガラス形成のためのCALPHAD熱力学計算
    
    This class calculates the thermodynamic driving force for crystallization
    using various approximations for the Gibbs energy difference between
    liquid and solid phases.
    
    Attributes:
        T_m: Melting temperature [K]
        delta_H_f: Heat of fusion [J/mol]
        delta_S_f: Entropy of fusion [J/(mol·K)]
        delta_Cp: Heat capacity difference (liquid - solid) [J/(mol·K)]
    """
    
    R = 8.314  # Gas constant [J/(mol·K)]
    
    def __init__(
        self,
        T_m: float,
        delta_H_f: float,
        delta_S_f: Optional[float] = None,
        delta_Cp: float = 0.0,
        T_g: Optional[float] = None
    ):
        """
        Initialize CALPHAD thermodynamics calculator.
        
        Args:
            T_m: Melting temperature [K]
            delta_H_f: Heat of fusion [J/mol]
            delta_S_f: Entropy of fusion [J/(mol·K)], calculated from delta_H_f/T_m if None
            delta_Cp: Heat capacity difference [J/(mol·K)], default 0 (Turnbull approx.)
            T_g: Glass transition temperature [K], optional
        """
        self.T_m = T_m
        self.delta_H_f = delta_H_f
        self.delta_S_f = delta_S_f if delta_S_f is not None else delta_H_f / T_m
        self.delta_Cp = delta_Cp
        self.T_g = T_g if T_g is not None else 0.5 * T_m
        
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate input parameters."""
        if self.T_m <= 0:
            raise ValueError("Melting temperature must be positive")
        if self.delta_H_f <= 0:
            raise ValueError("Heat of fusion must be positive")
        if self.T_g >= self.T_m:
            raise ValueError("Glass transition temperature must be below melting point")
    
    def delta_G_turnbull(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate Gibbs energy difference using Turnbull approximation.
        Turnbull近似によるギブスエネルギー差の計算
        
        ΔG = ΔH_f · (T_m - T) / T_m
        
        This is the simplest approximation assuming ΔCp = 0.
        
        Args:
            T: Temperature [K]
            
        Returns:
            ΔG [J/mol] - Gibbs energy difference (G^L - G^S)
        """
        T = np.asarray(T)
        delta_T = self.T_m - T
        return self.delta_H_f * delta_T / self.T_m
    
    def delta_G_thompson_spaepen(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate Gibbs energy difference using Thompson-Spaepen approximation.
        Thompson-Spaepen近似によるギブスエネルギー差の計算
        
        ΔG = ΔH_f · (T_m - T) / T_m · (2T / (T_m + T))
        
        This accounts for the temperature dependence of ΔCp.
        
        Args:
            T: Temperature [K]
            
        Returns:
            ΔG [J/mol]
        """
        T = np.asarray(T)
        delta_T = self.T_m - T
        return self.delta_H_f * delta_T / self.T_m * (2 * T / (self.T_m + T))
    
    def delta_G_hoffman(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate Gibbs energy difference using Hoffman approximation.
        Hoffman近似によるギブスエネルギー差の計算
        
        ΔG = ΔH_f · (T_m - T) · T / T_m²
        
        Args:
            T: Temperature [K]
            
        Returns:
            ΔG [J/mol]
        """
        T = np.asarray(T)
        delta_T = self.T_m - T
        return self.delta_H_f * delta_T * T / (self.T_m ** 2)
    
    def delta_G_full(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate Gibbs energy difference with heat capacity correction.
        熱容量補正を含むギブスエネルギー差の計算
        
        ΔG = ΔH_f · (1 - T/T_m) - ΔCp · [(T_m - T) - T · ln(T_m/T)]
        
        Args:
            T: Temperature [K]
            
        Returns:
            ΔG [J/mol]
        """
        T = np.asarray(T)
        term1 = self.delta_H_f * (1 - T / self.T_m)
        term2 = self.delta_Cp * ((self.T_m - T) - T * np.log(self.T_m / T))
        return term1 - term2
    
    def delta_G(
        self, 
        T: Union[float, np.ndarray], 
        method: str = "thompson_spaepen"
    ) -> Union[float, np.ndarray]:
        """
        Calculate Gibbs energy difference using specified method.
        指定された方法でギブスエネルギー差を計算
        
        Args:
            T: Temperature [K]
            method: Calculation method
                - "turnbull": Simple linear approximation
                - "thompson_spaepen": Improved approximation (default)
                - "hoffman": Hoffman approximation
                - "full": Full calculation with ΔCp
                
        Returns:
            ΔG [J/mol]
        """
        methods = {
            "turnbull": self.delta_G_turnbull,
            "thompson_spaepen": self.delta_G_thompson_spaepen,
            "hoffman": self.delta_G_hoffman,
            "full": self.delta_G_full
        }
        
        if method not in methods:
            raise ValueError(f"Unknown method: {method}. Available: {list(methods.keys())}")
        
        return methods[method](T)
    
    def delta_S(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate entropy difference between liquid and solid.
        液体と固体のエントロピー差を計算
        
        ΔS = ΔS_f - ΔCp · ln(T_m/T)
        
        Args:
            T: Temperature [K]
            
        Returns:
            ΔS [J/(mol·K)]
        """
        T = np.asarray(T)
        return self.delta_S_f - self.delta_Cp * np.log(self.T_m / T)
    
    def delta_H(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate enthalpy difference between liquid and solid.
        液体と固体のエンタルピー差を計算
        
        ΔH = ΔH_f - ΔCp · (T_m - T)
        
        Args:
            T: Temperature [K]
            
        Returns:
            ΔH [J/mol]
        """
        T = np.asarray(T)
        return self.delta_H_f - self.delta_Cp * (self.T_m - T)
    
    def driving_force_normalized(
        self, 
        T: Union[float, np.ndarray],
        method: str = "thompson_spaepen"
    ) -> Union[float, np.ndarray]:
        """
        Calculate normalized driving force ΔG/(R·T).
        正規化された駆動力 ΔG/(R·T) を計算
        
        This dimensionless quantity is useful for comparing different materials.
        
        Args:
            T: Temperature [K]
            method: Calculation method for ΔG
            
        Returns:
            ΔG/(R·T) [dimensionless]
        """
        T = np.asarray(T)
        delta_G = self.delta_G(T, method)
        return delta_G / (self.R * T)
    
    def calculate_all(
        self,
        T_min: Optional[float] = None,
        T_max: Optional[float] = None,
        n_points: int = 100,
        method: str = "thompson_spaepen"
    ) -> ThermodynamicResult:
        """
        Calculate all thermodynamic quantities over a temperature range.
        温度範囲にわたってすべての熱力学量を計算
        
        Args:
            T_min: Minimum temperature [K], default T_g
            T_max: Maximum temperature [K], default T_m
            n_points: Number of temperature points
            method: Calculation method for ΔG
            
        Returns:
            ThermodynamicResult containing all calculated quantities
        """
        if T_min is None:
            T_min = self.T_g
        if T_max is None:
            T_max = self.T_m
        
        T = np.linspace(T_min, T_max, n_points)
        
        return ThermodynamicResult(
            temperature=T,
            delta_G=self.delta_G(T, method),
            delta_S=self.delta_S(T),
            delta_H=self.delta_H(T),
            driving_force_normalized=self.driving_force_normalized(T, method)
        )
    
    def verify_at_melting_point(self, tol: float = 1e-6) -> bool:
        """
        Verify that ΔG = 0 at T = T_m (validation check).
        T = T_m で ΔG = 0 であることを検証（妥当性チェック）
        
        Args:
            tol: Tolerance for zero check
            
        Returns:
            True if verification passes
        """
        delta_G_at_Tm = self.delta_G(self.T_m)
        return abs(delta_G_at_Tm) < tol
    
    def get_entropy_of_fusion(self) -> float:
        """
        Get entropy of fusion ΔS_f = ΔH_f / T_m.
        融解エントロピーを取得
        
        Returns:
            ΔS_f [J/(mol·K)]
        """
        return self.delta_S_f
    
    def richard_rule_check(self) -> Tuple[float, str]:
        """
        Check Richard's rule: ΔS_f ≈ R for metals.
        Richardの法則をチェック：金属では ΔS_f ≈ R
        
        Returns:
            Tuple of (ΔS_f/R ratio, classification string)
        """
        ratio = self.delta_S_f / self.R
        
        if 0.8 <= ratio <= 1.2:
            classification = "Typical metal (Richard's rule satisfied)"
        elif ratio < 0.8:
            classification = "Low entropy of fusion (possible covalent bonding)"
        else:
            classification = "High entropy of fusion (complex structure)"
        
        return ratio, classification
    
    def get_validation_report(self, method: str = "thompson_spaepen") -> str:
        """
        Generate a validation report for the thermodynamic model.
        熱力学モデルの検証レポートを生成
        
        Returns:
            Formatted validation report string
        """
        lines = [
            "=" * 60,
            "CALPHAD Thermodynamics Validation Report",
            "CALPHAD熱力学検証レポート",
            "=" * 60,
            "",
            "Input Parameters / 入力パラメータ:",
            f"  T_m (融点) = {self.T_m:.1f} K",
            f"  T_g (ガラス転移点) = {self.T_g:.1f} K",
            f"  ΔH_f (融解熱) = {self.delta_H_f:.1f} J/mol",
            f"  ΔS_f (融解エントロピー) = {self.delta_S_f:.2f} J/(mol·K)",
            f"  ΔCp (熱容量差) = {self.delta_Cp:.2f} J/(mol·K)",
            "",
            "Validation Checks / 検証チェック:",
        ]
        
        delta_G_at_Tm = self.delta_G(self.T_m, method)
        check1 = "PASS" if abs(delta_G_at_Tm) < 1e-6 else "FAIL"
        lines.append(f"  1. ΔG(T_m) = 0: {check1} (value = {delta_G_at_Tm:.2e} J/mol)")
        
        delta_G_at_Tg = self.delta_G(self.T_g, method)
        check2 = "PASS" if delta_G_at_Tg > 0 else "FAIL"
        lines.append(f"  2. ΔG(T_g) > 0: {check2} (value = {delta_G_at_Tg:.1f} J/mol)")
        
        T_test = np.linspace(self.T_g, self.T_m, 10)
        delta_G_test = self.delta_G(T_test, method)
        monotonic = np.all(np.diff(delta_G_test) <= 0)
        check3 = "PASS" if monotonic else "FAIL"
        lines.append(f"  3. ΔG monotonically decreasing with T: {check3}")
        
        ratio, classification = self.richard_rule_check()
        lines.append(f"  4. Richard's rule (ΔS_f/R = {ratio:.2f}): {classification}")
        
        lines.extend([
            "",
            "Calculated Values at Key Temperatures / 主要温度での計算値:",
            f"  At T_m ({self.T_m:.1f} K):",
            f"    ΔG = {delta_G_at_Tm:.2e} J/mol",
            f"  At T_g ({self.T_g:.1f} K):",
            f"    ΔG = {delta_G_at_Tg:.1f} J/mol",
            f"    ΔG/(R·T) = {self.driving_force_normalized(self.T_g, method):.3f}",
            f"  At 0.75·T_m ({0.75*self.T_m:.1f} K):",
            f"    ΔG = {self.delta_G(0.75*self.T_m, method):.1f} J/mol",
            "",
            "=" * 60
        ])
        
        return "\n".join(lines)


def create_from_material(material) -> CALPHADThermodynamics:
    """
    Create CALPHADThermodynamics instance from a Material object.
    Materialオブジェクトから CALPHADThermodynamics インスタンスを作成
    
    Args:
        material: Material object from materials_database
        
    Returns:
        CALPHADThermodynamics instance
    """
    return CALPHADThermodynamics(
        T_m=material.T_m,
        delta_H_f=material.delta_H_f,
        delta_S_f=material.delta_S_f,
        T_g=material.T_g
    )


if __name__ == "__main__":
    print("Testing CALPHAD Thermodynamics Module")
    print("=" * 50)
    
    thermo = CALPHADThermodynamics(
        T_m=937.0,
        delta_H_f=8200.0,
        T_g=625.0
    )
    
    print(thermo.get_validation_report())
    
    print("\nComparison of different approximations:")
    T_test = 750.0
    print(f"At T = {T_test} K:")
    print(f"  Turnbull: ΔG = {thermo.delta_G_turnbull(T_test):.1f} J/mol")
    print(f"  Thompson-Spaepen: ΔG = {thermo.delta_G_thompson_spaepen(T_test):.1f} J/mol")
    print(f"  Hoffman: ΔG = {thermo.delta_G_hoffman(T_test):.1f} J/mol")
