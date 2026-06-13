"""
Doolittle Viscosity Model for Amorphous Formation
Doolittle粘度モデル：非晶質形成用

This module implements temperature-dependent viscosity models including:
- Doolittle equation (free volume theory)
- Vogel-Fulcher-Tammann (VFT) equation
- Angell fragility analysis

Key equations:
    Doolittle: η = A·exp(B·v₀/(v-v₀))
    VFT: η = η₀·exp(D*·T₀/(T-T₀))
    
At T_g, viscosity reaches ~10¹² - 10¹³ Pa·s.

References:
    - Doolittle, A.K. (1951). J. Appl. Phys. 22, 1471
    - Angell, C.A. (1995). Science 267, 1924
    - Debenedetti, P.G. & Stillinger, F.H. (2001). Nature 410, 259
"""

import numpy as np
from typing import Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class ViscosityResult:
    """Container for viscosity calculation results."""
    temperature: np.ndarray
    viscosity: np.ndarray
    log_viscosity: np.ndarray
    T_g_over_T: np.ndarray
    fragility_index: float


class DoolittleViscosity:
    """
    Doolittle/VFT viscosity model for glass-forming liquids.
    ガラス形成液体のDoolittle/VFT粘度モデル
    
    This class calculates temperature-dependent viscosity using the
    Vogel-Fulcher-Tammann (VFT) equation, which is equivalent to the
    Doolittle free volume equation.
    
    The VFT equation:
        η(T) = η₀ · exp(D* · T₀ / (T - T₀))
    
    where:
        η₀: Pre-exponential factor [Pa·s]
        D*: Fragility parameter (strength parameter)
        T₀: VFT temperature [K] (ideal glass transition)
    
    Attributes:
        T_m: Melting temperature [K]
        T_g: Glass transition temperature [K]
        eta_0: Pre-exponential viscosity factor [Pa·s]
        D_star: Fragility parameter
        T_0: VFT temperature [K]
    """
    
    ETA_G = 1e12
    ETA_LIQUID = 1e-3
    
    def __init__(
        self,
        T_m: float,
        T_g: float,
        eta_0: float = 1e-5,
        D_star: Optional[float] = None,
        T_0: Optional[float] = None,
        B: Optional[float] = None
    ):
        """
        Initialize Doolittle viscosity model.
        
        Args:
            T_m: Melting temperature [K]
            T_g: Glass transition temperature [K]
            eta_0: Pre-exponential factor [Pa·s]
            D_star: Fragility parameter (if None, calculated from B or default)
            T_0: VFT temperature [K] (if None, estimated from T_g)
            B: Doolittle B parameter [K] (alternative to D_star)
        """
        self.T_m = T_m
        self.T_g = T_g
        self.eta_0 = eta_0
        
        if T_0 is None:
            self.T_0 = self._estimate_T0()
        else:
            self.T_0 = T_0
        
        if D_star is not None:
            self.D_star = D_star
        elif B is not None:
            self.D_star = B / self.T_0
        else:
            self.D_star = self._calculate_D_star_from_Tg()
        
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate input parameters."""
        if self.T_m <= 0 or self.T_g <= 0:
            raise ValueError("Temperatures must be positive")
        if self.T_g >= self.T_m:
            raise ValueError("T_g must be less than T_m")
        if self.T_0 >= self.T_g:
            raise ValueError("T_0 must be less than T_g")
        if self.D_star <= 0:
            raise ValueError("D_star must be positive")
    
    def _estimate_T0(self) -> float:
        """
        Estimate VFT temperature T_0 from T_g.
        T_g から VFT温度 T_0 を推定
        
        Typically T_0 ≈ T_g - 50K for fragile liquids
        """
        return self.T_g - 50.0
    
    def _calculate_D_star_from_Tg(self) -> float:
        """
        Calculate D* such that η(T_g) = 10¹² Pa·s.
        η(T_g) = 10¹² Pa·s となるように D* を計算
        """
        log_eta_g = np.log(self.ETA_G)
        log_eta_0 = np.log(self.eta_0)
        D_star = (log_eta_g - log_eta_0) * (self.T_g - self.T_0) / self.T_0
        return max(D_star, 1.0)
    
    def viscosity(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate viscosity using VFT equation.
        VFT方程式を用いて粘度を計算
        
        η(T) = η₀ · exp(D* · T₀ / (T - T₀))
        
        Args:
            T: Temperature [K]
            
        Returns:
            Viscosity η [Pa·s]
        """
        T = np.asarray(T)
        
        T_safe = np.maximum(T, self.T_0 + 1.0)
        
        exponent = self.D_star * self.T_0 / (T_safe - self.T_0)
        exponent = np.minimum(exponent, 100.0)
        
        eta = self.eta_0 * np.exp(exponent)
        
        return eta
    
    def log_viscosity(self, T: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate log₁₀ of viscosity.
        粘度の常用対数を計算
        
        Args:
            T: Temperature [K]
            
        Returns:
            log₁₀(η) [Pa·s]
        """
        return np.log10(self.viscosity(T))
    
    def viscosity_arrhenius(
        self, 
        T: Union[float, np.ndarray],
        E_a: float,
        eta_inf: float = 1e-5
    ) -> Union[float, np.ndarray]:
        """
        Calculate viscosity using Arrhenius equation (for comparison).
        アレニウス方程式を用いて粘度を計算（比較用）
        
        η(T) = η_∞ · exp(E_a / (R·T))
        
        Args:
            T: Temperature [K]
            E_a: Activation energy [J/mol]
            eta_inf: Pre-exponential factor [Pa·s]
            
        Returns:
            Viscosity η [Pa·s]
        """
        R = 8.314
        T = np.asarray(T)
        return eta_inf * np.exp(E_a / (R * T))
    
    def fragility_index_m(self) -> float:
        """
        Calculate kinetic fragility index m.
        動的脆弱性指数 m を計算
        
        m = d(log η) / d(T_g/T) |_{T=T_g}
        
        For strong liquids: m ≈ 16-20 (SiO₂)
        For fragile liquids: m ≈ 100-200 (o-terphenyl)
        
        Returns:
            Fragility index m
        """
        m = self.D_star * self.T_0 * self.T_g / ((self.T_g - self.T_0) ** 2 * np.log(10))
        return m
    
    def classify_fragility(self) -> str:
        """
        Classify liquid as strong or fragile based on fragility index.
        脆弱性指数に基づいて液体を強い/弱いに分類
        
        Returns:
            Classification string
        """
        m = self.fragility_index_m()
        
        if m < 30:
            return f"Strong liquid (m = {m:.1f})"
        elif m < 60:
            return f"Intermediate liquid (m = {m:.1f})"
        else:
            return f"Fragile liquid (m = {m:.1f})"
    
    def angell_plot_data(
        self,
        T_min: Optional[float] = None,
        T_max: Optional[float] = None,
        n_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate data for Angell plot (log η vs T_g/T).
        Angellプロット用データを生成（log η vs T_g/T）
        
        Args:
            T_min: Minimum temperature [K]
            T_max: Maximum temperature [K]
            n_points: Number of data points
            
        Returns:
            Tuple of (T_g/T array, log₁₀(η) array)
        """
        if T_min is None:
            T_min = self.T_g
        if T_max is None:
            T_max = self.T_m * 1.2
        
        T = np.linspace(T_min, T_max, n_points)
        T_g_over_T = self.T_g / T
        log_eta = self.log_viscosity(T)
        
        return T_g_over_T, log_eta
    
    def calculate_all(
        self,
        T_min: Optional[float] = None,
        T_max: Optional[float] = None,
        n_points: int = 100
    ) -> ViscosityResult:
        """
        Calculate all viscosity quantities over a temperature range.
        温度範囲にわたってすべての粘度量を計算
        
        Args:
            T_min: Minimum temperature [K]
            T_max: Maximum temperature [K]
            n_points: Number of temperature points
            
        Returns:
            ViscosityResult containing all calculated quantities
        """
        if T_min is None:
            T_min = self.T_g
        if T_max is None:
            T_max = self.T_m * 1.2
        
        T = np.linspace(T_min, T_max, n_points)
        eta = self.viscosity(T)
        
        return ViscosityResult(
            temperature=T,
            viscosity=eta,
            log_viscosity=np.log10(eta),
            T_g_over_T=self.T_g / T,
            fragility_index=self.fragility_index_m()
        )
    
    def verify_at_Tg(self, tol_log: float = 1.0) -> bool:
        """
        Verify that η(T_g) ≈ 10¹² Pa·s (validation check).
        η(T_g) ≈ 10¹² Pa·s であることを検証
        
        Args:
            tol_log: Tolerance in log₁₀ units
            
        Returns:
            True if verification passes
        """
        log_eta_at_Tg = self.log_viscosity(self.T_g)
        target_log = np.log10(self.ETA_G)
        return abs(log_eta_at_Tg - target_log) < tol_log
    
    def verify_at_Tm(self, tol_log: float = 2.0) -> bool:
        """
        Verify that η(T_m) is in liquid range (10⁻³ - 10⁰ Pa·s).
        η(T_m) が液体範囲にあることを検証
        
        Args:
            tol_log: Tolerance in log₁₀ units
            
        Returns:
            True if verification passes
        """
        log_eta_at_Tm = self.log_viscosity(self.T_m)
        return -3 - tol_log <= log_eta_at_Tm <= 0 + tol_log
    
    def diffusivity(self, T: Union[float, np.ndarray], a: float = 3e-10) -> Union[float, np.ndarray]:
        """
        Estimate diffusivity from viscosity using Stokes-Einstein relation.
        Stokes-Einstein関係を用いて粘度から拡散係数を推定
        
        D = k_B · T / (6π · η · a)
        
        Args:
            T: Temperature [K]
            a: Atomic radius [m], default 3 Å
            
        Returns:
            Diffusivity D [m²/s]
        """
        k_B = 1.38e-23
        T = np.asarray(T)
        eta = self.viscosity(T)
        return k_B * T / (6 * np.pi * eta * a)
    
    def get_validation_report(self) -> str:
        """
        Generate a validation report for the viscosity model.
        粘度モデルの検証レポートを生成
        
        Returns:
            Formatted validation report string
        """
        lines = [
            "=" * 60,
            "Doolittle/VFT Viscosity Model Validation Report",
            "Doolittle/VFT粘度モデル検証レポート",
            "=" * 60,
            "",
            "Input Parameters / 入力パラメータ:",
            f"  T_m (融点) = {self.T_m:.1f} K",
            f"  T_g (ガラス転移点) = {self.T_g:.1f} K",
            f"  T_0 (VFT温度) = {self.T_0:.1f} K",
            f"  η₀ (前指数因子) = {self.eta_0:.2e} Pa·s",
            f"  D* (脆弱性パラメータ) = {self.D_star:.2f}",
            "",
            "Derived Parameters / 導出パラメータ:",
            f"  T_rg = T_g/T_m = {self.T_g/self.T_m:.3f}",
            f"  Fragility index m = {self.fragility_index_m():.1f}",
            f"  Classification: {self.classify_fragility()}",
            "",
            "Validation Checks / 検証チェック:",
        ]
        
        log_eta_Tg = self.log_viscosity(self.T_g)
        check1 = "PASS" if self.verify_at_Tg() else "FAIL"
        lines.append(f"  1. η(T_g) ≈ 10¹² Pa·s: {check1}")
        lines.append(f"     (log₁₀η = {log_eta_Tg:.1f}, target = 12)")
        
        log_eta_Tm = self.log_viscosity(self.T_m)
        check2 = "PASS" if self.verify_at_Tm() else "FAIL"
        lines.append(f"  2. η(T_m) in liquid range: {check2}")
        lines.append(f"     (log₁₀η = {log_eta_Tm:.1f}, target = -3 to 0)")
        
        T_test = np.linspace(self.T_g, self.T_m, 10)
        eta_test = self.viscosity(T_test)
        monotonic = np.all(np.diff(eta_test) <= 0)
        check3 = "PASS" if monotonic else "FAIL"
        lines.append(f"  3. η monotonically decreasing with T: {check3}")
        
        lines.extend([
            "",
            "Viscosity at Key Temperatures / 主要温度での粘度:",
            f"  At T_g ({self.T_g:.1f} K): η = {self.viscosity(self.T_g):.2e} Pa·s",
            f"  At T_m ({self.T_m:.1f} K): η = {self.viscosity(self.T_m):.2e} Pa·s",
            f"  At 1.2·T_m ({1.2*self.T_m:.1f} K): η = {self.viscosity(1.2*self.T_m):.2e} Pa·s",
            "",
            "=" * 60
        ])
        
        return "\n".join(lines)


def create_from_material(material) -> DoolittleViscosity:
    """
    Create DoolittleViscosity instance from a Material object.
    Materialオブジェクトから DoolittleViscosity インスタンスを作成
    
    Args:
        material: Material object from materials_database
        
    Returns:
        DoolittleViscosity instance
    """
    return DoolittleViscosity(
        T_m=material.T_m,
        T_g=material.T_g,
        eta_0=material.eta_0,
        D_star=material.D_star,
        T_0=material.T_0
    )


def generate_strong_liquid_reference(T_g: float, n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate reference Angell plot data for a strong liquid (SiO₂-like).
    強い液体（SiO₂様）の参照Angellプロットデータを生成
    
    Args:
        T_g: Glass transition temperature for scaling
        n_points: Number of data points
        
    Returns:
        Tuple of (T_g/T array, log₁₀(η) array)
    """
    T_g_over_T = np.linspace(0.4, 1.0, n_points)
    log_eta = -5 + 17 * T_g_over_T
    return T_g_over_T, log_eta


def generate_fragile_liquid_reference(T_g: float, n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate reference Angell plot data for a fragile liquid (o-terphenyl-like).
    弱い液体（o-terphenyl様）の参照Angellプロットデータを生成
    
    Args:
        T_g: Glass transition temperature for scaling
        n_points: Number of data points
        
    Returns:
        Tuple of (T_g/T array, log₁₀(η) array)
    """
    T_g_over_T = np.linspace(0.4, 1.0, n_points)
    log_eta = -5 + 17 * (T_g_over_T ** 3)
    return T_g_over_T, log_eta


if __name__ == "__main__":
    print("Testing Doolittle Viscosity Module")
    print("=" * 50)
    
    visc = DoolittleViscosity(
        T_m=937.0,
        T_g=625.0,
        eta_0=1e-5,
        D_star=18.5
    )
    
    print(visc.get_validation_report())
    
    print("\nAngell plot data (first 5 points):")
    Tg_T, log_eta = visc.angell_plot_data(n_points=10)
    for i in range(5):
        print(f"  T_g/T = {Tg_T[i]:.3f}, log₁₀(η) = {log_eta[i]:.2f}")
