"""
Materials Database for Amorphous Formation Validation
非晶質形成検証用材料データベース

This module contains experimentally validated parameters for various
glass-forming alloys used to validate the amorphous formation model.

References:
    - Turnbull, D. (1969). Under what conditions can a glass be formed?
    - Inoue, A. (2000). Stabilization of metallic supercooled liquid and bulk amorphous alloys.
    - Johnson, W.L. (1999). Bulk glass-forming metallic alloys: Science and technology.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List
import numpy as np


@dataclass
class Material:
    """
    Material class containing thermodynamic and kinetic parameters.
    材料クラス：熱力学・動力学パラメータを含む
    
    Attributes:
        name: Material name / 材料名
        composition: Chemical composition / 化学組成
        T_m: Melting temperature [K] / 融点
        T_g: Glass transition temperature [K] / ガラス転移点
        delta_H_f: Heat of fusion [J/mol] / 融解熱
        delta_S_f: Entropy of fusion [J/(mol·K)] / 融解エントロピー
        sigma: Solid-liquid interface energy [J/m²] / 固液界面エネルギー
        V_m: Molar volume [m³/mol] / モル体積
        eta_0: Pre-exponential viscosity factor [Pa·s] / 粘度前指数因子
        B: Doolittle parameter (fragility) / Doolittleパラメータ
        T_0: VFT temperature [K] / VFT温度
        D_star: Fragility parameter / 脆弱性パラメータ
        R_c_exp: Experimental critical cooling rate [K/s] / 実験臨界冷却速度
        reference: Literature reference / 文献参照
    """
    name: str
    composition: str
    T_m: float
    T_g: float
    delta_H_f: float
    delta_S_f: Optional[float] = None
    sigma: Optional[float] = None
    V_m: float = 1.0e-5
    eta_0: float = 1.0e-5
    B: float = 5000.0
    T_0: float = 0.0
    D_star: float = 20.0
    R_c_exp: Optional[float] = None
    reference: str = ""
    
    def __post_init__(self):
        if self.delta_S_f is None:
            self.delta_S_f = self.delta_H_f / self.T_m
        if self.sigma is None:
            self.sigma = self.estimate_interface_energy()
        if self.T_0 == 0.0:
            self.T_0 = self.T_g - 50.0
    
    def estimate_interface_energy(self) -> float:
        """
        Estimate solid-liquid interface energy using Turnbull's relation.
        Turnbullの関係式を用いて固液界面エネルギーを推定
        
        σ ≈ α · ΔH_f / (N_A^(1/3) · V_m^(2/3))
        where α ≈ 0.45 for metals
        """
        N_A = 6.022e23
        alpha = 0.45
        sigma = alpha * self.delta_H_f / (N_A**(1/3) * self.V_m**(2/3))
        return sigma
    
    def get_reduced_temperature(self, T: float) -> float:
        """Calculate reduced temperature T_r = (T_m - T) / (T_m - T_g)"""
        return (self.T_m - T) / (self.T_m - self.T_g)
    
    def get_T_rg(self) -> float:
        """Calculate reduced glass transition temperature T_rg = T_g / T_m"""
        return self.T_g / self.T_m


class MaterialsDatabase:
    """
    Database of glass-forming materials with validated parameters.
    検証済みパラメータを持つガラス形成材料のデータベース
    """
    
    def __init__(self):
        self._materials: Dict[str, Material] = {}
        self._load_default_materials()
    
    def _load_default_materials(self):
        """Load default materials with literature values."""
        
        self._materials["Pd82Si18"] = Material(
            name="Pd82Si18",
            composition="Pd₈₂Si₁₈",
            T_m=1071.0,
            T_g=633.0,
            delta_H_f=16700.0,
            V_m=9.2e-6,
            eta_0=4.0e-5,
            B=4500.0,
            D_star=15.0,
            R_c_exp=1.0e3,
            reference="Chen, H.S. (1974) Acta Metall. 22, 1505"
        )
        
        self._materials["Pd40Ni40P20"] = Material(
            name="Pd40Ni40P20",
            composition="Pd₄₀Ni₄₀P₂₀",
            T_m=884.0,
            T_g=580.0,
            delta_H_f=8500.0,
            V_m=8.5e-6,
            eta_0=1.0e-5,
            B=3800.0,
            D_star=12.0,
            R_c_exp=1.0e0,
            reference="Inoue, A. (1995) Mater. Trans. JIM 36, 866"
        )
        
        self._materials["Zr41Ti14Cu12Ni10Be23"] = Material(
            name="Zr41Ti14Cu12Ni10Be23",
            composition="Zr₄₁.₂Ti₁₃.₈Cu₁₂.₅Ni₁₀Be₂₂.₅ (Vitreloy 1)",
            T_m=937.0,
            T_g=625.0,
            delta_H_f=8200.0,
            V_m=1.1e-5,
            eta_0=1.0e-5,
            B=4200.0,
            D_star=18.5,
            R_c_exp=1.0e0,
            reference="Peker, A. & Johnson, W.L. (1993) Appl. Phys. Lett. 63, 2342"
        )
        
        self._materials["Zr55Cu30Al10Ni5"] = Material(
            name="Zr55Cu30Al10Ni5",
            composition="Zr₅₅Cu₃₀Al₁₀Ni₅",
            T_m=1100.0,
            T_g=683.0,
            delta_H_f=10500.0,
            V_m=1.15e-5,
            eta_0=2.0e-5,
            B=4000.0,
            D_star=16.0,
            R_c_exp=1.0e1,
            reference="Inoue, A. (1998) Acta Mater. 46, 4551"
        )
        
        self._materials["Cu47Ti34Zr11Ni8"] = Material(
            name="Cu47Ti34Zr11Ni8",
            composition="Cu₄₇Ti₃₄Zr₁₁Ni₈",
            T_m=1150.0,
            T_g=698.0,
            delta_H_f=11000.0,
            V_m=9.8e-6,
            eta_0=3.0e-5,
            B=4100.0,
            D_star=14.0,
            R_c_exp=2.5e2,
            reference="Lin, X.H. & Johnson, W.L. (1995) J. Appl. Phys. 78, 6514"
        )
        
        self._materials["Fe80B20"] = Material(
            name="Fe80B20",
            composition="Fe₈₀B₂₀",
            T_m=1448.0,
            T_g=720.0,
            delta_H_f=13800.0,
            V_m=7.1e-6,
            eta_0=5.0e-5,
            B=5500.0,
            D_star=25.0,
            R_c_exp=1.0e5,
            reference="Luborsky, F.E. (1983) Amorphous Metallic Alloys"
        )
        
        self._materials["Au77Ge14Si9"] = Material(
            name="Au77Ge14Si9",
            composition="Au₇₇Ge₁₄Si₉",
            T_m=629.0,
            T_g=295.0,
            delta_H_f=5800.0,
            V_m=1.05e-5,
            eta_0=2.0e-5,
            B=3500.0,
            D_star=10.0,
            R_c_exp=1.0e6,
            reference="Klement, W. et al. (1960) Nature 187, 869"
        )
        
        self._materials["Mg65Cu25Y10"] = Material(
            name="Mg65Cu25Y10",
            composition="Mg₆₅Cu₂₅Y₁₀",
            T_m=730.0,
            T_g=420.0,
            delta_H_f=7500.0,
            V_m=1.4e-5,
            eta_0=1.5e-5,
            B=3600.0,
            D_star=13.0,
            R_c_exp=5.0e1,
            reference="Inoue, A. et al. (1991) Mater. Trans. JIM 32, 609"
        )
        
        self._materials["SiO2"] = Material(
            name="SiO2",
            composition="SiO₂ (Silica glass)",
            T_m=1996.0,
            T_g=1473.0,
            delta_H_f=9600.0,
            V_m=2.7e-5,
            eta_0=1.0e-7,
            B=8000.0,
            D_star=100.0,
            R_c_exp=1.0e-4,
            reference="Angell, C.A. (1995) Science 267, 1924"
        )
        
        self._materials["B2O3"] = Material(
            name="B2O3",
            composition="B₂O₃ (Boric oxide)",
            T_m=723.0,
            T_g=520.0,
            delta_H_f=24500.0,
            V_m=3.8e-5,
            eta_0=1.0e-6,
            B=6000.0,
            D_star=8.0,
            R_c_exp=1.0e-2,
            reference="Angell, C.A. (1991) J. Non-Cryst. Solids 131-133, 13"
        )
    
    def get_material(self, name: str) -> Optional[Material]:
        """Get material by name."""
        return self._materials.get(name)
    
    def list_materials(self) -> List[str]:
        """List all available material names."""
        return list(self._materials.keys())
    
    def add_material(self, material: Material):
        """Add a new material to the database."""
        self._materials[material.name] = material
    
    def get_all_materials(self) -> Dict[str, Material]:
        """Get all materials in the database."""
        return self._materials.copy()
    
    def get_metallic_glasses(self) -> List[Material]:
        """Get only metallic glass materials."""
        metallic = ["Pd82Si18", "Pd40Ni40P20", "Zr41Ti14Cu12Ni10Be23", 
                   "Zr55Cu30Al10Ni5", "Cu47Ti34Zr11Ni8", "Fe80B20", 
                   "Au77Ge14Si9", "Mg65Cu25Y10"]
        return [self._materials[name] for name in metallic if name in self._materials]
    
    def get_oxide_glasses(self) -> List[Material]:
        """Get only oxide glass materials."""
        oxides = ["SiO2", "B2O3"]
        return [self._materials[name] for name in oxides if name in self._materials]
    
    def get_materials_by_R_c_range(self, R_c_min: float, R_c_max: float) -> List[Material]:
        """Get materials within a critical cooling rate range."""
        return [
            mat for mat in self._materials.values()
            if mat.R_c_exp is not None and R_c_min <= mat.R_c_exp <= R_c_max
        ]
    
    def get_summary_table(self) -> str:
        """Generate a summary table of all materials."""
        header = f"{'Material':<25} {'T_m [K]':>10} {'T_g [K]':>10} {'T_rg':>8} {'R_c [K/s]':>12}"
        separator = "-" * 70
        lines = [header, separator]
        
        for mat in self._materials.values():
            T_rg = mat.get_T_rg()
            R_c_str = f"{mat.R_c_exp:.1e}" if mat.R_c_exp else "N/A"
            line = f"{mat.composition:<25} {mat.T_m:>10.1f} {mat.T_g:>10.1f} {T_rg:>8.3f} {R_c_str:>12}"
            lines.append(line)
        
        return "\n".join(lines)


if __name__ == "__main__":
    db = MaterialsDatabase()
    print("Available materials in database:")
    print(db.get_summary_table())
    
    print("\n\nDetailed info for Vitreloy 1:")
    vit1 = db.get_material("Zr41Ti14Cu12Ni10Be23")
    if vit1:
        print(f"  Composition: {vit1.composition}")
        print(f"  T_m = {vit1.T_m} K")
        print(f"  T_g = {vit1.T_g} K")
        print(f"  T_rg = {vit1.get_T_rg():.3f}")
        print(f"  ΔH_f = {vit1.delta_H_f} J/mol")
        print(f"  σ (estimated) = {vit1.sigma:.4e} J/m²")
        print(f"  R_c (exp) = {vit1.R_c_exp} K/s")
