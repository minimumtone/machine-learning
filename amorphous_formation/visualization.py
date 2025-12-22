"""
Visualization Module for Amorphous Formation
可視化モジュール：非晶質形成用

This module provides plotting functions for all required diagrams:
1. CALPHAD driving force vs temperature
2. Viscosity vs temperature (Doolittle plot)
3. Angell plot (log η vs T_g/T)
4. TTT curve with nose marker
5. Sensitivity analysis plots

All plots follow the validation requirements specified in the instruction document.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Optional, Tuple, List, Dict, Any
import warnings

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class AmorphousVisualization:
    """
    Visualization class for amorphous formation model validation.
    非晶質形成モデル検証用の可視化クラス
    
    This class provides methods to create all required plots for
    validating the amorphous formation model.
    """
    
    COLORS = {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e',
        'tertiary': '#2ca02c',
        'quaternary': '#d62728',
        'reference': '#7f7f7f',
        'nose': '#e377c2'
    }
    
    def __init__(self, figsize: Tuple[float, float] = (10, 6), dpi: int = 100):
        """
        Initialize visualization settings.
        
        Args:
            figsize: Default figure size (width, height) in inches
            dpi: Figure resolution
        """
        self.figsize = figsize
        self.dpi = dpi
        
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['legend.fontsize'] = 11
        plt.rcParams['xtick.labelsize'] = 11
        plt.rcParams['ytick.labelsize'] = 11
    
    def plot_calphad_driving_force(
        self,
        thermo,
        methods: Optional[List[str]] = None,
        ax: Optional[Axes] = None,
        show_validation: bool = True
    ) -> Tuple[Figure, Axes]:
        """
        Plot CALPHAD driving force (ΔG) vs temperature.
        CALPHAD駆動力（ΔG）対温度をプロット
        
        Validation criteria:
        - ΔG = 0 at T = T_m
        - ΔG increases as T decreases
        
        Args:
            thermo: CALPHADThermodynamics instance
            methods: List of methods to compare (default: all)
            ax: Matplotlib axes (creates new if None)
            show_validation: Show validation markers
            
        Returns:
            Tuple of (Figure, Axes)
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        else:
            fig = ax.get_figure()
        
        if methods is None:
            methods = ["turnbull", "thompson_spaepen", "hoffman"]
        
        T = np.linspace(thermo.T_g, thermo.T_m, 100)
        
        method_labels = {
            "turnbull": "Turnbull",
            "thompson_spaepen": "Thompson-Spaepen",
            "hoffman": "Hoffman",
            "full": "Full (with ΔCp)"
        }
        
        colors = [self.COLORS['primary'], self.COLORS['secondary'], 
                  self.COLORS['tertiary'], self.COLORS['quaternary']]
        
        for i, method in enumerate(methods):
            delta_G = thermo.delta_G(T, method) / 1000
            ax.plot(T, delta_G, label=method_labels.get(method, method),
                   color=colors[i % len(colors)], linewidth=2)
        
        if show_validation:
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.axvline(x=thermo.T_m, color='red', linestyle=':', alpha=0.7,
                      label=f'$T_m$ = {thermo.T_m:.0f} K')
            ax.axvline(x=thermo.T_g, color='blue', linestyle=':', alpha=0.7,
                      label=f'$T_g$ = {thermo.T_g:.0f} K')
            
            ax.plot(thermo.T_m, 0, 'ro', markersize=10, zorder=5)
            ax.annotate('ΔG = 0 at $T_m$', xy=(thermo.T_m, 0),
                       xytext=(thermo.T_m - 50, 1),
                       fontsize=10, ha='right')
        
        ax.set_xlabel('Temperature T [K]')
        ax.set_ylabel('Gibbs Energy Difference ΔG [kJ/mol]')
        ax.set_title('CALPHAD Driving Force for Crystallization\nCALPHAD結晶化駆動力')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(thermo.T_g - 20, thermo.T_m + 20)
        
        fig.tight_layout()
        return fig, ax
    
    def plot_viscosity_temperature(
        self,
        visc,
        ax: Optional[Axes] = None,
        show_validation: bool = True
    ) -> Tuple[Figure, Axes]:
        """
        Plot viscosity vs temperature (Doolittle plot).
        粘度対温度をプロット（Doolittleプロット）
        
        Validation criteria:
        - η(T_g) ≈ 10¹² - 10¹³ Pa·s
        - η(T_m) ≈ 10⁻³ - 10⁰ Pa·s
        
        Args:
            visc: DoolittleViscosity instance
            ax: Matplotlib axes
            show_validation: Show validation markers
            
        Returns:
            Tuple of (Figure, Axes)
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        else:
            fig = ax.get_figure()
        
        T = np.linspace(visc.T_g, visc.T_m * 1.2, 100)
        log_eta = visc.log_viscosity(T)
        
        ax.plot(T, log_eta, color=self.COLORS['primary'], linewidth=2.5,
               label='VFT model')
        
        if show_validation:
            ax.axhline(y=12, color='green', linestyle='--', alpha=0.7,
                      label='$η_{glass}$ = 10¹² Pa·s')
            ax.axhline(y=-3, color='orange', linestyle='--', alpha=0.7,
                      label='$η_{liquid}$ = 10⁻³ Pa·s')
            
            ax.axvline(x=visc.T_g, color='blue', linestyle=':', alpha=0.7)
            ax.axvline(x=visc.T_m, color='red', linestyle=':', alpha=0.7)
            
            eta_Tg = visc.log_viscosity(visc.T_g)
            eta_Tm = visc.log_viscosity(visc.T_m)
            ax.plot(visc.T_g, eta_Tg, 'bo', markersize=10, zorder=5)
            ax.plot(visc.T_m, eta_Tm, 'ro', markersize=10, zorder=5)
            
            ax.annotate(f'$T_g$ = {visc.T_g:.0f} K\nlog η = {eta_Tg:.1f}',
                       xy=(visc.T_g, eta_Tg), xytext=(visc.T_g + 30, eta_Tg - 2),
                       fontsize=10, arrowprops=dict(arrowstyle='->', color='blue'))
            ax.annotate(f'$T_m$ = {visc.T_m:.0f} K\nlog η = {eta_Tm:.1f}',
                       xy=(visc.T_m, eta_Tm), xytext=(visc.T_m - 80, eta_Tm + 2),
                       fontsize=10, arrowprops=dict(arrowstyle='->', color='red'))
        
        ax.set_xlabel('Temperature T [K]')
        ax.set_ylabel('log₁₀(η) [Pa·s]')
        ax.set_title('Viscosity vs Temperature (Doolittle/VFT Model)\n粘度の温度依存性')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        return fig, ax
    
    def plot_angell(
        self,
        visc,
        ax: Optional[Axes] = None,
        show_references: bool = True
    ) -> Tuple[Figure, Axes]:
        """
        Plot Angell plot (log η vs T_g/T).
        Angellプロットを作成（log η vs T_g/T）
        
        This plot distinguishes "strong" vs "fragile" liquids.
        
        Args:
            visc: DoolittleViscosity instance
            ax: Matplotlib axes
            show_references: Show strong/fragile reference lines
            
        Returns:
            Tuple of (Figure, Axes)
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        else:
            fig = ax.get_figure()
        
        Tg_T, log_eta = visc.angell_plot_data(n_points=100)
        
        ax.plot(Tg_T, log_eta, color=self.COLORS['primary'], linewidth=2.5,
               label=f'Material (m = {visc.fragility_index_m():.1f})')
        
        if show_references:
            Tg_T_ref = np.linspace(0.4, 1.0, 50)
            
            log_eta_strong = -5 + 17 * Tg_T_ref
            ax.plot(Tg_T_ref, log_eta_strong, '--', color=self.COLORS['reference'],
                   linewidth=1.5, alpha=0.7, label='Strong (SiO₂-like)')
            
            log_eta_fragile = -5 + 17 * (Tg_T_ref ** 3)
            ax.plot(Tg_T_ref, log_eta_fragile, ':', color=self.COLORS['reference'],
                   linewidth=1.5, alpha=0.7, label='Fragile (o-terphenyl-like)')
        
        ax.axhline(y=12, color='green', linestyle='--', alpha=0.5)
        ax.axvline(x=1.0, color='gray', linestyle=':', alpha=0.5)
        
        ax.plot(1.0, 12, 'ko', markersize=8, zorder=5)
        ax.annotate('$T_g$ definition\n(η = 10¹² Pa·s)',
                   xy=(1.0, 12), xytext=(0.85, 14),
                   fontsize=10, ha='center')
        
        ax.set_xlabel('$T_g$ / T')
        ax.set_ylabel('log₁₀(η) [Pa·s]')
        ax.set_title('Angell Plot: Strong vs Fragile Classification\nAngellプロット：強い液体 vs 弱い液体')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.4, 1.05)
        ax.set_ylim(-6, 16)
        
        classification = visc.classify_fragility()
        ax.text(0.95, 0.05, classification, transform=ax.transAxes,
               fontsize=11, ha='right', va='bottom',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        fig.tight_layout()
        return fig, ax
    
    def plot_ttt_curve(
        self,
        model,
        ax: Optional[Axes] = None,
        show_nose: bool = True,
        X: float = 1e-6
    ) -> Tuple[Figure, Axes]:
        """
        Plot TTT (Time-Temperature-Transformation) curve.
        TTT曲線をプロット
        
        Validation criteria:
        - C-shaped curve with nose
        - Nose at T_n ≈ 0.7-0.8 T_m
        
        Args:
            model: DavisUhlmannModel instance
            ax: Matplotlib axes
            show_nose: Mark the nose position
            X: Volume fraction for crystallization detection
            
        Returns:
            Tuple of (Figure, Axes)
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        else:
            fig = ax.get_figure()
        
        ttt = model.calculate_ttt_curve(n_points=200, X=X)
        
        valid_mask = (ttt.time > 0) & (ttt.time < 1e30) & np.isfinite(ttt.time)
        T_valid = ttt.temperature[valid_mask]
        log_t_valid = ttt.log_time[valid_mask]
        
        ax.plot(log_t_valid, T_valid, color=self.COLORS['primary'], linewidth=2.5,
               label=f'TTT curve (X = {X:.0e})')
        
        if show_nose:
            ax.plot(np.log10(ttt.nose_time), ttt.nose_temperature, 
                   marker='*', markersize=15, color=self.COLORS['nose'],
                   markeredgecolor='black', markeredgewidth=1,
                   label=f'Nose: $T_n$ = {ttt.nose_temperature:.0f} K, $t_n$ = {ttt.nose_time:.2e} s',
                   zorder=5)
            
            ax.annotate(f'$T_n$/$T_m$ = {ttt.nose_temperature/model.T_m:.3f}',
                       xy=(np.log10(ttt.nose_time), ttt.nose_temperature),
                       xytext=(np.log10(ttt.nose_time) + 1, ttt.nose_temperature + 30),
                       fontsize=10, arrowprops=dict(arrowstyle='->', color='black'))
        
        ax.axhline(y=model.T_m, color='red', linestyle=':', alpha=0.7,
                  label=f'$T_m$ = {model.T_m:.0f} K')
        ax.axhline(y=model.T_g, color='blue', linestyle=':', alpha=0.7,
                  label=f'$T_g$ = {model.T_g:.0f} K')
        
        ax.fill_between([log_t_valid.min(), log_t_valid.max()],
                       model.T_g, model.T_m,
                       alpha=0.1, color='gray',
                       label='Supercooled liquid region')
        
        ax.set_xlabel('log₁₀(time) [s]')
        ax.set_ylabel('Temperature T [K]')
        ax.set_title('TTT Curve (Time-Temperature-Transformation)\nTTT曲線')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        R_c = ttt.critical_cooling_rate
        ax.text(0.02, 0.02, f'Critical cooling rate: $R_c$ = {R_c:.2e} K/s',
               transform=ax.transAxes, fontsize=11,
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        fig.tight_layout()
        return fig, ax
    
    def plot_sensitivity_analysis(
        self,
        sensitivity_result,
        ax: Optional[Axes] = None
    ) -> Tuple[Figure, Axes]:
        """
        Plot sensitivity analysis results.
        感度解析結果をプロット
        
        Args:
            sensitivity_result: SensitivityResult from sensitivity analysis
            ax: Matplotlib axes
            
        Returns:
            Tuple of (Figure, Axes)
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        else:
            fig = ax.get_figure()
        
        valid_mask = np.isfinite(sensitivity_result.log_R_c_change)
        variations = sensitivity_result.variations[valid_mask]
        log_changes = sensitivity_result.log_R_c_change[valid_mask]
        
        ax.plot(variations, log_changes, 'o-', color=self.COLORS['primary'],
               linewidth=2, markersize=8, label=sensitivity_result.parameter_name)
        
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        ax.axhline(y=1, color='green', linestyle=':', alpha=0.5, label='1 order of magnitude')
        ax.axhline(y=-1, color='green', linestyle=':', alpha=0.5)
        
        ax.set_xlabel(f'{sensitivity_result.parameter_name} variation [%]')
        ax.set_ylabel('log₁₀($R_c$ / $R_{c,base}$)')
        ax.set_title(f'Sensitivity Analysis: Effect of {sensitivity_result.parameter_name}\n感度解析')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        if len(variations) > 2:
            slope = (log_changes[-1] - log_changes[0]) / (variations[-1] - variations[0])
            ax.text(0.95, 0.95, f'Slope ≈ {slope:.3f} per %',
                   transform=ax.transAxes, fontsize=11, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        fig.tight_layout()
        return fig, ax
    
    def plot_complete_validation(
        self,
        thermo,
        visc,
        model,
        sensitivity_result=None,
        material_name: str = "Material"
    ) -> Figure:
        """
        Create complete validation figure with all required plots.
        すべての必要なプロットを含む完全な検証図を作成
        
        Args:
            thermo: CALPHADThermodynamics instance
            visc: DoolittleViscosity instance
            model: DavisUhlmannModel instance
            sensitivity_result: Optional SensitivityResult
            material_name: Name for the title
            
        Returns:
            Figure with all subplots
        """
        if sensitivity_result is not None:
            fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=self.dpi)
        else:
            fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=self.dpi)
        
        self.plot_calphad_driving_force(thermo, ax=axes[0, 0])
        axes[0, 0].set_title('1. CALPHAD Driving Force\nCALPHAD駆動力')
        
        self.plot_viscosity_temperature(visc, ax=axes[0, 1])
        axes[0, 1].set_title('2. Viscosity (Doolittle)\n粘度（Doolittle）')
        
        self.plot_ttt_curve(model, ax=axes[1, 0])
        axes[1, 0].set_title('3. TTT Curve\nTTT曲線')
        
        if sensitivity_result is not None:
            self.plot_sensitivity_analysis(sensitivity_result, ax=axes[1, 1])
            axes[1, 1].set_title('4. Sensitivity Analysis\n感度解析')
        else:
            self.plot_angell(visc, ax=axes[1, 1])
            axes[1, 1].set_title('4. Angell Plot\nAngellプロット')
        
        fig.suptitle(f'Amorphous Formation Model Validation: {material_name}\n'
                    f'非晶質形成モデル検証', fontsize=16, fontweight='bold')
        
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig
    
    def plot_material_comparison(
        self,
        materials_data: List[Dict[str, Any]],
        plot_type: str = "ttt"
    ) -> Figure:
        """
        Compare multiple materials on the same plot.
        複数の材料を同じプロット上で比較
        
        Args:
            materials_data: List of dicts with 'name', 'model' keys
            plot_type: Type of plot ('ttt', 'angell', 'driving_force')
            
        Returns:
            Figure with comparison
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(materials_data)))
        
        for i, mat_data in enumerate(materials_data):
            name = mat_data['name']
            
            if plot_type == "ttt" and 'model' in mat_data:
                model = mat_data['model']
                ttt = model.calculate_ttt_curve(n_points=100)
                valid_mask = (ttt.time > 0) & (ttt.time < 1e30) & np.isfinite(ttt.time)
                ax.plot(ttt.log_time[valid_mask], ttt.temperature[valid_mask],
                       color=colors[i], linewidth=2, label=name)
                ax.plot(np.log10(ttt.nose_time), ttt.nose_temperature,
                       '*', color=colors[i], markersize=12)
            
            elif plot_type == "angell" and 'visc' in mat_data:
                visc = mat_data['visc']
                Tg_T, log_eta = visc.angell_plot_data(n_points=100)
                ax.plot(Tg_T, log_eta, color=colors[i], linewidth=2,
                       label=f"{name} (m={visc.fragility_index_m():.0f})")
        
        if plot_type == "ttt":
            ax.set_xlabel('log₁₀(time) [s]')
            ax.set_ylabel('Temperature T [K]')
            ax.set_title('TTT Curve Comparison\nTTT曲線比較')
        elif plot_type == "angell":
            ax.set_xlabel('$T_g$ / T')
            ax.set_ylabel('log₁₀(η) [Pa·s]')
            ax.set_title('Angell Plot Comparison\nAngellプロット比較')
        
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        
        return fig
    
    def save_all_figures(
        self,
        thermo,
        visc,
        model,
        sensitivity_result,
        output_dir: str = ".",
        prefix: str = "amorphous",
        format: str = "png"
    ) -> List[str]:
        """
        Save all validation figures to files.
        すべての検証図をファイルに保存
        
        Args:
            thermo: CALPHADThermodynamics instance
            visc: DoolittleViscosity instance
            model: DavisUhlmannModel instance
            sensitivity_result: SensitivityResult
            output_dir: Output directory
            prefix: Filename prefix
            format: Image format (png, pdf, svg)
            
        Returns:
            List of saved file paths
        """
        import os
        saved_files = []
        
        fig1, _ = self.plot_calphad_driving_force(thermo)
        path1 = os.path.join(output_dir, f"{prefix}_calphad_driving_force.{format}")
        fig1.savefig(path1, dpi=self.dpi, bbox_inches='tight')
        saved_files.append(path1)
        plt.close(fig1)
        
        fig2, _ = self.plot_viscosity_temperature(visc)
        path2 = os.path.join(output_dir, f"{prefix}_viscosity.{format}")
        fig2.savefig(path2, dpi=self.dpi, bbox_inches='tight')
        saved_files.append(path2)
        plt.close(fig2)
        
        fig3, _ = self.plot_angell(visc)
        path3 = os.path.join(output_dir, f"{prefix}_angell_plot.{format}")
        fig3.savefig(path3, dpi=self.dpi, bbox_inches='tight')
        saved_files.append(path3)
        plt.close(fig3)
        
        fig4, _ = self.plot_ttt_curve(model)
        path4 = os.path.join(output_dir, f"{prefix}_ttt_curve.{format}")
        fig4.savefig(path4, dpi=self.dpi, bbox_inches='tight')
        saved_files.append(path4)
        plt.close(fig4)
        
        fig5, _ = self.plot_sensitivity_analysis(sensitivity_result)
        path5 = os.path.join(output_dir, f"{prefix}_sensitivity.{format}")
        fig5.savefig(path5, dpi=self.dpi, bbox_inches='tight')
        saved_files.append(path5)
        plt.close(fig5)
        
        fig6 = self.plot_complete_validation(thermo, visc, model, sensitivity_result)
        path6 = os.path.join(output_dir, f"{prefix}_complete_validation.{format}")
        fig6.savefig(path6, dpi=self.dpi, bbox_inches='tight')
        saved_files.append(path6)
        plt.close(fig6)
        
        return saved_files


if __name__ == "__main__":
    print("Testing Visualization Module")
    print("=" * 50)
    
    from .calphad_thermodynamics import CALPHADThermodynamics
    from .doolittle_viscosity import DoolittleViscosity
    from .davis_uhlmann_model import DavisUhlmannModel
    from .sensitivity_analysis import SensitivityAnalysis
    
    thermo = CALPHADThermodynamics(T_m=937.0, delta_H_f=8200.0, T_g=625.0)
    visc = DoolittleViscosity(T_m=937.0, T_g=625.0, eta_0=1e-5, D_star=18.5)
    model = DavisUhlmannModel(T_m=937.0, T_g=625.0, delta_H_f=8200.0,
                              sigma=0.08, V_m=1.1e-5, eta_0=1e-5, D_star=18.5)
    analysis = SensitivityAnalysis(T_m=937.0, T_g=625.0, delta_H_f=8200.0,
                                   sigma=0.08, V_m=1.1e-5, eta_0=1e-5, D_star=18.5)
    
    viz = AmorphousVisualization()
    
    fig = viz.plot_complete_validation(thermo, visc, model,
                                       analysis.analyze_sigma_sensitivity(),
                                       material_name="Vitreloy 1")
    plt.show()
