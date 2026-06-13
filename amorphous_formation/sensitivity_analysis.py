"""
Sensitivity Analysis Module for Amorphous Formation
感度解析モジュール：非晶質形成用

This module performs parameter sensitivity analysis to understand how
uncertainties in material parameters (especially interface energy σ)
affect the predicted critical cooling rate and TTT curve.

Key analysis:
    - Effect of σ variation on TTT nose position
    - Effect of σ variation on critical cooling rate R_c
    - Identification of most sensitive parameters

References:
    - Uhlmann, D.R. (1972). J. Non-Cryst. Solids 7, 337
    - Lu, Z.P. & Liu, C.T. (2002). Acta Mater. 50, 3501
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings

from .davis_uhlmann_model import DavisUhlmannModel


@dataclass
class SensitivityResult:
    """Container for sensitivity analysis results."""
    parameter_name: str
    base_value: float
    variations: np.ndarray
    values: np.ndarray
    R_c_values: np.ndarray
    t_nose_values: np.ndarray
    T_nose_values: np.ndarray
    log_R_c_change: np.ndarray


class SensitivityAnalysis:
    """
    Sensitivity analysis for amorphous formation model parameters.
    非晶質形成モデルパラメータの感度解析
    
    This class analyzes how variations in model parameters affect
    the predicted critical cooling rate and TTT curve characteristics.
    
    The interface energy σ is particularly important because:
    - It appears cubed in the nucleation barrier (ΔG* ∝ σ³)
    - Small changes in σ can cause orders of magnitude change in R_c
    
    Attributes:
        base_model: Reference DavisUhlmannModel instance
    """
    
    def __init__(
        self,
        T_m: float,
        T_g: float,
        delta_H_f: float,
        sigma: float,
        V_m: float = 1.0e-5,
        eta_0: float = 1.0e-5,
        D_star: float = 20.0,
        T_0: Optional[float] = None
    ):
        """
        Initialize sensitivity analysis with base parameters.
        
        Args:
            T_m: Melting temperature [K]
            T_g: Glass transition temperature [K]
            delta_H_f: Heat of fusion [J/mol]
            sigma: Base interface energy [J/m²]
            V_m: Molar volume [m³/mol]
            eta_0: Pre-exponential viscosity factor [Pa·s]
            D_star: Fragility parameter
            T_0: VFT temperature [K]
        """
        self.T_m = T_m
        self.T_g = T_g
        self.delta_H_f = delta_H_f
        self.sigma_base = sigma
        self.V_m = V_m
        self.eta_0 = eta_0
        self.D_star = D_star
        self.T_0 = T_0 if T_0 is not None else T_g - 50.0
        
        self.base_model = DavisUhlmannModel(
            T_m=T_m, T_g=T_g, delta_H_f=delta_H_f, sigma=sigma,
            V_m=V_m, eta_0=eta_0, D_star=D_star, T_0=self.T_0
        )
        
        self.base_R_c = self.base_model.critical_cooling_rate()
        T_nose, t_nose = self.base_model.find_nose()
        self.base_T_nose = T_nose
        self.base_t_nose = t_nose
    
    def analyze_sigma_sensitivity(
        self,
        variation_percent: float = 10.0,
        n_points: int = 21
    ) -> SensitivityResult:
        """
        Analyze sensitivity to interface energy σ.
        界面エネルギー σ に対する感度を解析
        
        Args:
            variation_percent: Percent variation (e.g., 10 for ±10%)
            n_points: Number of points in variation range
            
        Returns:
            SensitivityResult containing analysis data
        """
        factor_min = 1 - variation_percent / 100
        factor_max = 1 + variation_percent / 100
        factors = np.linspace(factor_min, factor_max, n_points)
        
        sigma_values = self.sigma_base * factors
        R_c_values = np.zeros(n_points)
        t_nose_values = np.zeros(n_points)
        T_nose_values = np.zeros(n_points)
        
        for i, sigma in enumerate(sigma_values):
            try:
                model = DavisUhlmannModel(
                    T_m=self.T_m, T_g=self.T_g, delta_H_f=self.delta_H_f,
                    sigma=sigma, V_m=self.V_m, eta_0=self.eta_0,
                    D_star=self.D_star, T_0=self.T_0
                )
                R_c_values[i] = model.critical_cooling_rate()
                T_nose, t_nose = model.find_nose()
                T_nose_values[i] = T_nose
                t_nose_values[i] = t_nose
            except Exception:
                R_c_values[i] = np.nan
                T_nose_values[i] = np.nan
                t_nose_values[i] = np.nan
        
        log_R_c_change = np.log10(R_c_values / self.base_R_c)
        
        return SensitivityResult(
            parameter_name="σ (interface energy)",
            base_value=self.sigma_base,
            variations=(factors - 1) * 100,
            values=sigma_values,
            R_c_values=R_c_values,
            t_nose_values=t_nose_values,
            T_nose_values=T_nose_values,
            log_R_c_change=log_R_c_change
        )
    
    def analyze_D_star_sensitivity(
        self,
        variation_percent: float = 20.0,
        n_points: int = 21
    ) -> SensitivityResult:
        """
        Analyze sensitivity to fragility parameter D*.
        脆弱性パラメータ D* に対する感度を解析
        
        Args:
            variation_percent: Percent variation
            n_points: Number of points
            
        Returns:
            SensitivityResult
        """
        factor_min = 1 - variation_percent / 100
        factor_max = 1 + variation_percent / 100
        factors = np.linspace(factor_min, factor_max, n_points)
        
        D_star_values = self.D_star * factors
        R_c_values = np.zeros(n_points)
        t_nose_values = np.zeros(n_points)
        T_nose_values = np.zeros(n_points)
        
        for i, D_star in enumerate(D_star_values):
            try:
                model = DavisUhlmannModel(
                    T_m=self.T_m, T_g=self.T_g, delta_H_f=self.delta_H_f,
                    sigma=self.sigma_base, V_m=self.V_m, eta_0=self.eta_0,
                    D_star=D_star, T_0=self.T_0
                )
                R_c_values[i] = model.critical_cooling_rate()
                T_nose, t_nose = model.find_nose()
                T_nose_values[i] = T_nose
                t_nose_values[i] = t_nose
            except Exception:
                R_c_values[i] = np.nan
                T_nose_values[i] = np.nan
                t_nose_values[i] = np.nan
        
        log_R_c_change = np.log10(R_c_values / self.base_R_c)
        
        return SensitivityResult(
            parameter_name="D* (fragility)",
            base_value=self.D_star,
            variations=(factors - 1) * 100,
            values=D_star_values,
            R_c_values=R_c_values,
            t_nose_values=t_nose_values,
            T_nose_values=T_nose_values,
            log_R_c_change=log_R_c_change
        )
    
    def analyze_delta_H_f_sensitivity(
        self,
        variation_percent: float = 10.0,
        n_points: int = 21
    ) -> SensitivityResult:
        """
        Analyze sensitivity to heat of fusion ΔH_f.
        融解熱 ΔH_f に対する感度を解析
        
        Args:
            variation_percent: Percent variation
            n_points: Number of points
            
        Returns:
            SensitivityResult
        """
        factor_min = 1 - variation_percent / 100
        factor_max = 1 + variation_percent / 100
        factors = np.linspace(factor_min, factor_max, n_points)
        
        delta_H_f_values = self.delta_H_f * factors
        R_c_values = np.zeros(n_points)
        t_nose_values = np.zeros(n_points)
        T_nose_values = np.zeros(n_points)
        
        for i, delta_H_f in enumerate(delta_H_f_values):
            try:
                model = DavisUhlmannModel(
                    T_m=self.T_m, T_g=self.T_g, delta_H_f=delta_H_f,
                    sigma=self.sigma_base, V_m=self.V_m, eta_0=self.eta_0,
                    D_star=self.D_star, T_0=self.T_0
                )
                R_c_values[i] = model.critical_cooling_rate()
                T_nose, t_nose = model.find_nose()
                T_nose_values[i] = T_nose
                t_nose_values[i] = t_nose
            except Exception:
                R_c_values[i] = np.nan
                T_nose_values[i] = np.nan
                t_nose_values[i] = np.nan
        
        log_R_c_change = np.log10(R_c_values / self.base_R_c)
        
        return SensitivityResult(
            parameter_name="ΔH_f (heat of fusion)",
            base_value=self.delta_H_f,
            variations=(factors - 1) * 100,
            values=delta_H_f_values,
            R_c_values=R_c_values,
            t_nose_values=t_nose_values,
            T_nose_values=T_nose_values,
            log_R_c_change=log_R_c_change
        )
    
    def full_sensitivity_analysis(
        self,
        variation_percent: float = 10.0,
        n_points: int = 21
    ) -> Dict[str, SensitivityResult]:
        """
        Perform full sensitivity analysis on all key parameters.
        すべての主要パラメータに対する完全な感度解析を実行
        
        Args:
            variation_percent: Percent variation for each parameter
            n_points: Number of points per parameter
            
        Returns:
            Dictionary of SensitivityResult for each parameter
        """
        results = {}
        
        results["sigma"] = self.analyze_sigma_sensitivity(variation_percent, n_points)
        results["D_star"] = self.analyze_D_star_sensitivity(variation_percent * 2, n_points)
        results["delta_H_f"] = self.analyze_delta_H_f_sensitivity(variation_percent, n_points)
        
        return results
    
    def calculate_sensitivity_coefficients(
        self,
        delta_percent: float = 1.0
    ) -> Dict[str, float]:
        """
        Calculate local sensitivity coefficients (∂ln(R_c)/∂ln(p)).
        局所感度係数を計算
        
        The sensitivity coefficient S_p = ∂ln(R_c)/∂ln(p) indicates
        how many percent R_c changes for 1% change in parameter p.
        
        Args:
            delta_percent: Small perturbation for numerical derivative
            
        Returns:
            Dictionary of sensitivity coefficients
        """
        coefficients = {}
        delta = delta_percent / 100
        
        model_plus = DavisUhlmannModel(
            T_m=self.T_m, T_g=self.T_g, delta_H_f=self.delta_H_f,
            sigma=self.sigma_base * (1 + delta), V_m=self.V_m,
            eta_0=self.eta_0, D_star=self.D_star, T_0=self.T_0
        )
        model_minus = DavisUhlmannModel(
            T_m=self.T_m, T_g=self.T_g, delta_H_f=self.delta_H_f,
            sigma=self.sigma_base * (1 - delta), V_m=self.V_m,
            eta_0=self.eta_0, D_star=self.D_star, T_0=self.T_0
        )
        R_c_plus = model_plus.critical_cooling_rate()
        R_c_minus = model_minus.critical_cooling_rate()
        S_sigma = (np.log(R_c_plus) - np.log(R_c_minus)) / (2 * delta)
        coefficients["σ"] = S_sigma
        
        model_plus = DavisUhlmannModel(
            T_m=self.T_m, T_g=self.T_g, delta_H_f=self.delta_H_f,
            sigma=self.sigma_base, V_m=self.V_m,
            eta_0=self.eta_0, D_star=self.D_star * (1 + delta), T_0=self.T_0
        )
        model_minus = DavisUhlmannModel(
            T_m=self.T_m, T_g=self.T_g, delta_H_f=self.delta_H_f,
            sigma=self.sigma_base, V_m=self.V_m,
            eta_0=self.eta_0, D_star=self.D_star * (1 - delta), T_0=self.T_0
        )
        R_c_plus = model_plus.critical_cooling_rate()
        R_c_minus = model_minus.critical_cooling_rate()
        S_D_star = (np.log(R_c_plus) - np.log(R_c_minus)) / (2 * delta)
        coefficients["D*"] = S_D_star
        
        model_plus = DavisUhlmannModel(
            T_m=self.T_m, T_g=self.T_g, delta_H_f=self.delta_H_f * (1 + delta),
            sigma=self.sigma_base, V_m=self.V_m,
            eta_0=self.eta_0, D_star=self.D_star, T_0=self.T_0
        )
        model_minus = DavisUhlmannModel(
            T_m=self.T_m, T_g=self.T_g, delta_H_f=self.delta_H_f * (1 - delta),
            sigma=self.sigma_base, V_m=self.V_m,
            eta_0=self.eta_0, D_star=self.D_star, T_0=self.T_0
        )
        R_c_plus = model_plus.critical_cooling_rate()
        R_c_minus = model_minus.critical_cooling_rate()
        S_delta_H_f = (np.log(R_c_plus) - np.log(R_c_minus)) / (2 * delta)
        coefficients["ΔH_f"] = S_delta_H_f
        
        return coefficients
    
    def get_sensitivity_table(self, variation_percent: float = 10.0) -> str:
        """
        Generate a formatted sensitivity analysis table.
        感度解析テーブルを生成
        
        Args:
            variation_percent: Percent variation for analysis
            
        Returns:
            Formatted table string
        """
        sigma_result = self.analyze_sigma_sensitivity(variation_percent, n_points=5)
        
        lines = [
            "=" * 70,
            "Sensitivity Analysis Results / 感度解析結果",
            "=" * 70,
            "",
            "Base Parameters / 基準パラメータ:",
            f"  σ (base) = {self.sigma_base:.4e} J/m²",
            f"  R_c (base) = {self.base_R_c:.2e} K/s",
            f"  T_nose (base) = {self.base_T_nose:.1f} K",
            f"  t_nose (base) = {self.base_t_nose:.2e} s",
            "",
            f"Effect of ±{variation_percent}% change in σ:",
            "-" * 70,
            f"{'σ variation':>15} {'σ [J/m²]':>15} {'R_c [K/s]':>15} {'log₁₀(R_c/R_c₀)':>15}",
            "-" * 70,
        ]
        
        for i in range(len(sigma_result.variations)):
            var = sigma_result.variations[i]
            sigma = sigma_result.values[i]
            R_c = sigma_result.R_c_values[i]
            log_change = sigma_result.log_R_c_change[i]
            lines.append(f"{var:>+14.1f}% {sigma:>15.4e} {R_c:>15.2e} {log_change:>+15.2f}")
        
        lines.extend([
            "-" * 70,
            "",
            "Sensitivity Coefficients / 感度係数:",
        ])
        
        coeffs = self.calculate_sensitivity_coefficients()
        for param, coeff in coeffs.items():
            lines.append(f"  S_{param} = {coeff:.2f} (1% change in {param} → {coeff:.2f}% change in R_c)")
        
        lines.extend([
            "",
            "Key Observation / 重要な観察:",
            f"  σ is the most sensitive parameter (S_σ = {coeffs['σ']:.1f})",
            "  Small uncertainties in σ cause large uncertainties in R_c",
            "",
            "=" * 70
        ])
        
        return "\n".join(lines)
    
    def compare_with_experiment(
        self,
        R_c_exp: float,
        material_name: str = "Unknown"
    ) -> str:
        """
        Compare calculated R_c with experimental value.
        計算された R_c を実験値と比較
        
        Args:
            R_c_exp: Experimental critical cooling rate [K/s]
            material_name: Name of the material
            
        Returns:
            Comparison report string
        """
        log_diff = np.log10(self.base_R_c / R_c_exp)
        
        lines = [
            "=" * 60,
            f"Comparison with Experiment: {material_name}",
            f"実験値との比較: {material_name}",
            "=" * 60,
            "",
            f"Calculated R_c = {self.base_R_c:.2e} K/s",
            f"Experimental R_c = {R_c_exp:.2e} K/s",
            f"Ratio (calc/exp) = {self.base_R_c/R_c_exp:.2f}",
            f"Log difference = {log_diff:+.2f} orders of magnitude",
            "",
        ]
        
        if abs(log_diff) <= 1:
            lines.append("Assessment: GOOD agreement (within 1 order of magnitude)")
            lines.append("評価: 良好な一致（1桁以内）")
        elif abs(log_diff) <= 2:
            lines.append("Assessment: ACCEPTABLE agreement (within 2 orders of magnitude)")
            lines.append("評価: 許容範囲の一致（2桁以内）")
        else:
            lines.append("Assessment: POOR agreement (more than 2 orders of magnitude)")
            lines.append("評価: 不十分な一致（2桁以上のずれ）")
            
            if log_diff > 0:
                lines.extend([
                    "",
                    "Possible causes for overestimation:",
                    "  - σ may be too low (increase σ to reduce R_c)",
                    "  - D* may be too high (decrease D* to reduce R_c)"
                ])
            else:
                lines.extend([
                    "",
                    "Possible causes for underestimation:",
                    "  - σ may be too high (decrease σ to increase R_c)",
                    "  - D* may be too low (increase D* to increase R_c)"
                ])
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)


def create_from_material(material) -> SensitivityAnalysis:
    """
    Create SensitivityAnalysis instance from a Material object.
    Materialオブジェクトから SensitivityAnalysis インスタンスを作成
    
    Args:
        material: Material object from materials_database
        
    Returns:
        SensitivityAnalysis instance
    """
    return SensitivityAnalysis(
        T_m=material.T_m,
        T_g=material.T_g,
        delta_H_f=material.delta_H_f,
        sigma=material.sigma,
        V_m=material.V_m,
        eta_0=material.eta_0,
        D_star=material.D_star,
        T_0=material.T_0
    )


if __name__ == "__main__":
    print("Testing Sensitivity Analysis Module")
    print("=" * 50)
    
    analysis = SensitivityAnalysis(
        T_m=937.0,
        T_g=625.0,
        delta_H_f=8200.0,
        sigma=0.08,
        V_m=1.1e-5,
        eta_0=1e-5,
        D_star=18.5
    )
    
    print(analysis.get_sensitivity_table(variation_percent=10.0))
    
    print("\n")
    print(analysis.compare_with_experiment(R_c_exp=1.0, material_name="Vitreloy 1"))
