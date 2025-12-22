"""
Davis-Uhlmann Model for TTT Curve Generation
Davis-Uhlmannモデル：TTT曲線生成用

This module implements the Davis-Uhlmann model for calculating Time-Temperature-
Transformation (TTT) curves, which predict the time required for crystallization
at different temperatures.

Key equations:
    Nucleation rate: I = I₀ · exp(-ΔG*/kT) · exp(-Q_D/kT)
    Growth rate: U = f · a · ν · exp(-ΔG_a/kT) · [1 - exp(-ΔG_m/kT)]
    TTT time: t = [π·I·U³/3·ln(1/(1-X))]^(-1/4)

The TTT curve has a characteristic "C" shape (nose) due to the competition
between thermodynamic driving force and kinetic mobility.

References:
    - Uhlmann, D.R. (1972). J. Non-Cryst. Solids 7, 337
    - Davies, H.A. (1976). Phys. Chem. Glasses 17, 159
    - Turnbull, D. (1969). Contemp. Phys. 10, 473
"""

import numpy as np
from typing import Tuple, Optional, Union, Dict
from dataclasses import dataclass
import warnings


@dataclass
class TTTResult:
    """Container for TTT curve calculation results."""
    temperature: np.ndarray
    time: np.ndarray
    log_time: np.ndarray
    nucleation_rate: np.ndarray
    growth_rate: np.ndarray
    nose_temperature: float
    nose_time: float
    critical_cooling_rate: float


class DavisUhlmannModel:
    """
    Davis-Uhlmann model for TTT curve calculation.
    TTT曲線計算のためのDavis-Uhlmannモデル
    
    This class calculates the time required for a detectable amount of
    crystallization (typically 10⁻⁶ volume fraction) as a function of
    temperature, producing the characteristic C-shaped TTT curve.
    
    The model combines:
    1. Classical nucleation theory for nucleation rate
    2. Wilson-Frenkel model for crystal growth rate
    3. Johnson-Mehl-Avrami-Kolmogorov (JMAK) kinetics
    
    Attributes:
        T_m: Melting temperature [K]
        T_g: Glass transition temperature [K]
        delta_H_f: Heat of fusion [J/mol]
        sigma: Solid-liquid interface energy [J/m²]
        V_m: Molar volume [m³/mol]
        eta_0: Pre-exponential viscosity factor [Pa·s]
        D_star: Fragility parameter
        T_0: VFT temperature [K]
    """
    
    R = 8.314
    k_B = 1.38e-23
    N_A = 6.022e23
    
    def __init__(
        self,
        T_m: float,
        T_g: float,
        delta_H_f: float,
        sigma: float,
        V_m: float = 1.0e-5,
        eta_0: float = 1.0e-5,
        D_star: float = 20.0,
        T_0: Optional[float] = None,
        a: float = 3.0e-10
    ):
        """
        Initialize Davis-Uhlmann model.
        
        Args:
            T_m: Melting temperature [K]
            T_g: Glass transition temperature [K]
            delta_H_f: Heat of fusion [J/mol]
            sigma: Solid-liquid interface energy [J/m²]
            V_m: Molar volume [m³/mol]
            eta_0: Pre-exponential viscosity factor [Pa·s]
            D_star: Fragility parameter
            T_0: VFT temperature [K]
            a: Atomic jump distance [m]
        """
        self.T_m = T_m
        self.T_g = T_g
        self.delta_H_f = delta_H_f
        self.sigma = sigma
        self.V_m = V_m
        self.eta_0 = eta_0
        self.D_star = D_star
        self.T_0 = T_0 if T_0 is not None else T_g - 50.0
        self.a = a
        
        self.v_m = V_m / self.N_A
        
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate input parameters."""
        if self.T_m <= 0 or self.T_g <= 0:
            raise ValueError("Temperatures must be positive")
        if self.T_g >= self.T_m:
            raise ValueError("T_g must be less than T_m")
        if self.sigma <= 0:
            raise ValueError("Interface energy must be positive")
        if self.delta_H_f <= 0:
            raise ValueError("Heat of fusion must be positive")
    
    def delta_G_v(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate volumetric Gibbs energy difference (driving force).
        体積ギブスエネルギー差（駆動力）を計算
        
        ΔG_v = ΔH_f · (T_m - T) / T_m / V_m
        
        Args:
            T: Temperature [K]
            
        Returns:
            ΔG_v [J/m³]
        """
        T = np.asarray(T)
        delta_T = self.T_m - T
        delta_G_mol = self.delta_H_f * delta_T / self.T_m
        return delta_G_mol / self.V_m
    
    def delta_G_star(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate critical nucleation barrier.
        臨界核生成障壁を計算
        
        ΔG* = 16π·σ³ / (3·ΔG_v²)
        
        Args:
            T: Temperature [K]
            
        Returns:
            ΔG* [J]
        """
        T = np.asarray(T)
        delta_G_v = self.delta_G_v(T)
        
        delta_G_v = np.where(np.abs(delta_G_v) < 1e-10, 1e-10, delta_G_v)
        
        return 16 * np.pi * self.sigma**3 / (3 * delta_G_v**2)
    
    def critical_radius(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate critical nucleus radius.
        臨界核半径を計算
        
        r* = 2σ / ΔG_v
        
        Args:
            T: Temperature [K]
            
        Returns:
            r* [m]
        """
        T = np.asarray(T)
        delta_G_v = self.delta_G_v(T)
        delta_G_v = np.where(np.abs(delta_G_v) < 1e-10, 1e-10, delta_G_v)
        return 2 * self.sigma / np.abs(delta_G_v)
    
    def viscosity(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate viscosity using VFT equation.
        VFT方程式を用いて粘度を計算
        
        Args:
            T: Temperature [K]
            
        Returns:
            η [Pa·s]
        """
        T = np.asarray(T)
        T_safe = np.maximum(T, self.T_0 + 1.0)
        exponent = self.D_star * self.T_0 / (T_safe - self.T_0)
        exponent = np.minimum(exponent, 100.0)
        return self.eta_0 * np.exp(exponent)
    
    def diffusivity(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate diffusivity from Stokes-Einstein relation.
        Stokes-Einstein関係から拡散係数を計算
        
        D = k_B · T / (3π · η · a)
        
        Args:
            T: Temperature [K]
            
        Returns:
            D [m²/s]
        """
        T = np.asarray(T)
        eta = self.viscosity(T)
        return self.k_B * T / (3 * np.pi * eta * self.a)
    
    def nucleation_rate(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate steady-state nucleation rate.
        定常核生成速度を計算
        
        I = I₀ · exp(-ΔG*/kT) · D/a²
        
        where I₀ ≈ N_v · ν ≈ 10³⁹ m⁻³s⁻¹
        
        Args:
            T: Temperature [K]
            
        Returns:
            I [m⁻³s⁻¹]
        """
        T = np.asarray(T)
        
        N_v = self.N_A / self.V_m
        nu = self.k_B * T / (3 * np.pi * self.a**3 * self.viscosity(T))
        I_0 = N_v * nu
        
        delta_G_star = self.delta_G_star(T)
        
        exponent = -delta_G_star / (self.k_B * T)
        exponent = np.maximum(exponent, -300.0)
        
        I = I_0 * np.exp(exponent)
        
        return I
    
    def growth_rate(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate crystal growth rate using Wilson-Frenkel model.
        Wilson-Frenkelモデルを用いて結晶成長速度を計算
        
        U = (D/a) · [1 - exp(-ΔG_m/RT)]
        
        Args:
            T: Temperature [K]
            
        Returns:
            U [m/s]
        """
        T = np.asarray(T)
        
        D = self.diffusivity(T)
        
        delta_G_m = self.delta_H_f * (self.T_m - T) / self.T_m
        
        driving = 1 - np.exp(-delta_G_m / (self.R * T))
        driving = np.maximum(driving, 0.0)
        
        U = (D / self.a) * driving
        
        return U
    
    def time_for_crystallization(
        self, 
        T: Union[float, np.ndarray],
        X: float = 1e-6
    ) -> Union[float, np.ndarray]:
        """
        Calculate time for specified volume fraction of crystallization.
        指定された体積分率の結晶化に必要な時間を計算
        
        Using JMAK kinetics:
        X = 1 - exp(-π·I·U³·t⁴/3)
        
        Solving for t:
        t = [3·ln(1/(1-X)) / (π·I·U³)]^(1/4)
        
        Args:
            T: Temperature [K]
            X: Volume fraction crystallized (default 10⁻⁶)
            
        Returns:
            t [s]
        """
        T = np.asarray(T)
        
        I = self.nucleation_rate(T)
        U = self.growth_rate(T)
        
        I = np.maximum(I, 1e-100)
        U = np.maximum(U, 1e-100)
        
        numerator = 3 * np.log(1 / (1 - X))
        denominator = np.pi * I * U**3
        
        t = (numerator / denominator) ** 0.25
        
        return t
    
    def calculate_ttt_curve(
        self,
        T_min: Optional[float] = None,
        T_max: Optional[float] = None,
        n_points: int = 100,
        X: float = 1e-6
    ) -> TTTResult:
        """
        Calculate complete TTT curve.
        完全なTTT曲線を計算
        
        Args:
            T_min: Minimum temperature [K]
            T_max: Maximum temperature [K]
            n_points: Number of temperature points
            X: Volume fraction for detection
            
        Returns:
            TTTResult containing curve data and nose parameters
        """
        if T_min is None:
            T_min = self.T_g
        if T_max is None:
            T_max = self.T_m - 1.0
        
        T = np.linspace(T_min, T_max, n_points)
        
        t = self.time_for_crystallization(T, X)
        I = self.nucleation_rate(T)
        U = self.growth_rate(T)
        
        valid_mask = (t > 0) & (t < 1e30) & np.isfinite(t)
        
        if np.any(valid_mask):
            t_valid = t[valid_mask]
            T_valid = T[valid_mask]
            nose_idx = np.argmin(t_valid)
            T_nose = T_valid[nose_idx]
            t_nose = t_valid[nose_idx]
        else:
            T_nose = 0.75 * self.T_m
            t_nose = 1.0
            warnings.warn("Could not find valid TTT nose")
        
        R_c = (self.T_m - T_nose) / t_nose
        
        return TTTResult(
            temperature=T,
            time=t,
            log_time=np.log10(np.maximum(t, 1e-100)),
            nucleation_rate=I,
            growth_rate=U,
            nose_temperature=T_nose,
            nose_time=t_nose,
            critical_cooling_rate=R_c
        )
    
    def find_nose(self, n_points: int = 200) -> Tuple[float, float]:
        """
        Find the nose (minimum time) of the TTT curve.
        TTT曲線のノーズ（最小時間）を見つける
        
        Returns:
            Tuple of (T_nose [K], t_nose [s])
        """
        T = np.linspace(self.T_g, self.T_m - 1.0, n_points)
        t = self.time_for_crystallization(T)
        
        valid_mask = (t > 0) & (t < 1e30) & np.isfinite(t)
        
        if not np.any(valid_mask):
            return 0.75 * self.T_m, 1.0
        
        t_valid = t[valid_mask]
        T_valid = T[valid_mask]
        
        nose_idx = np.argmin(t_valid)
        return T_valid[nose_idx], t_valid[nose_idx]
    
    def critical_cooling_rate(self) -> float:
        """
        Calculate critical cooling rate to avoid crystallization.
        結晶化を回避するための臨界冷却速度を計算
        
        R_c = (T_m - T_n) / t_n
        
        Returns:
            R_c [K/s]
        """
        T_nose, t_nose = self.find_nose()
        return (self.T_m - T_nose) / t_nose
    
    def verify_nose_position(self) -> Tuple[bool, str]:
        """
        Verify that nose temperature is in expected range.
        ノーズ温度が期待範囲内にあることを検証
        
        Expected: T_n ≈ 0.7-0.8 · T_m
        
        Returns:
            Tuple of (pass/fail, message)
        """
        T_nose, _ = self.find_nose()
        T_n_reduced = T_nose / self.T_m
        
        if 0.65 <= T_n_reduced <= 0.85:
            return True, f"PASS: T_n/T_m = {T_n_reduced:.3f} (expected 0.7-0.8)"
        elif T_n_reduced < 0.65:
            return False, f"FAIL: T_n/T_m = {T_n_reduced:.3f} too low (check σ - may be too high)"
        else:
            return False, f"FAIL: T_n/T_m = {T_n_reduced:.3f} too high (check σ - may be too low)"
    
    def get_validation_report(self) -> str:
        """
        Generate a validation report for the Davis-Uhlmann model.
        Davis-Uhlmannモデルの検証レポートを生成
        
        Returns:
            Formatted validation report string
        """
        T_nose, t_nose = self.find_nose()
        R_c = self.critical_cooling_rate()
        
        lines = [
            "=" * 60,
            "Davis-Uhlmann Model Validation Report",
            "Davis-Uhlmannモデル検証レポート",
            "=" * 60,
            "",
            f"Input Parameters / 入力パラメータ:",
            f"  T_m (融点) = {self.T_m:.1f} K",
            f"  T_g (ガラス転移点) = {self.T_g:.1f} K",
            f"  ΔH_f (融解熱) = {self.delta_H_f:.1f} J/mol",
            f"  σ (界面エネルギー) = {self.sigma:.4e} J/m²",
            f"  V_m (モル体積) = {self.V_m:.2e} m³/mol",
            "",
            f"TTT Curve Results / TTT曲線結果:",
            f"  Nose temperature T_n = {T_nose:.1f} K",
            f"  Nose time t_n = {t_nose:.2e} s",
            f"  T_n/T_m = {T_nose/self.T_m:.3f}",
            f"  Critical cooling rate R_c = {R_c:.2e} K/s",
            "",
            f"Validation Checks / 検証チェック:",
        ]
        
        pass_nose, msg_nose = self.verify_nose_position()
        check1 = "PASS" if pass_nose else "FAIL"
        lines.append(f"  1. Nose position: {check1}")
        lines.append(f"     {msg_nose}")
        
        ttt = self.calculate_ttt_curve()
        has_c_shape = (ttt.time[0] > ttt.nose_time) and (ttt.time[-1] > ttt.nose_time)
        check2 = "PASS" if has_c_shape else "FAIL"
        lines.append(f"  2. C-shape curve: {check2}")
        
        I_nose = self.nucleation_rate(T_nose)
        U_nose = self.growth_rate(T_nose)
        lines.extend([
            "",
            f"Kinetic Parameters at Nose / ノーズでの動力学パラメータ:",
            f"  Nucleation rate I = {I_nose:.2e} m⁻³s⁻¹",
            f"  Growth rate U = {U_nose:.2e} m/s",
            f"  Critical radius r* = {self.critical_radius(T_nose):.2e} m",
            f"  Nucleation barrier ΔG* = {self.delta_G_star(T_nose):.2e} J",
            "",
            "=" * 60
        ])
        
        return "\n".join(lines)


def create_from_material(material) -> DavisUhlmannModel:
    """
    Create DavisUhlmannModel instance from a Material object.
    Materialオブジェクトから DavisUhlmannModel インスタンスを作成
    
    Args:
        material: Material object from materials_database
        
    Returns:
        DavisUhlmannModel instance
    """
    return DavisUhlmannModel(
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
    print("Testing Davis-Uhlmann Model")
    print("=" * 50)
    
    model = DavisUhlmannModel(
        T_m=937.0,
        T_g=625.0,
        delta_H_f=8200.0,
        sigma=0.08,
        V_m=1.1e-5,
        eta_0=1e-5,
        D_star=18.5
    )
    
    print(model.get_validation_report())
    
    print("\nTTT curve data (selected points):")
    ttt = model.calculate_ttt_curve(n_points=10)
    for i in range(len(ttt.temperature)):
        print(f"  T = {ttt.temperature[i]:.1f} K, t = {ttt.time[i]:.2e} s")
