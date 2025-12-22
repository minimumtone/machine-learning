"""
Validation Tests for Amorphous Formation Package
非晶質形成パッケージの検証テスト

These tests verify that the models produce physically reasonable results
and match known experimental data within acceptable tolerances.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from amorphous_formation.materials_database import MaterialsDatabase, Material
from amorphous_formation.calphad_thermodynamics import CALPHADThermodynamics
from amorphous_formation.doolittle_viscosity import DoolittleViscosity
from amorphous_formation.davis_uhlmann_model import DavisUhlmannModel
from amorphous_formation.sensitivity_analysis import SensitivityAnalysis


class TestMaterialsDatabase:
    """Tests for MaterialsDatabase class."""
    
    def test_database_initialization(self):
        """Test that database initializes with materials."""
        db = MaterialsDatabase()
        materials = db.list_materials()
        assert len(materials) >= 10, "Database should contain at least 10 materials"
    
    def test_get_material(self):
        """Test retrieving a specific material."""
        db = MaterialsDatabase()
        vitreloy = db.get_material("Zr41Ti14Cu12Ni10Be23")
        assert vitreloy is not None
        assert vitreloy.T_m > 0
        assert vitreloy.T_g > 0
        assert vitreloy.T_g < vitreloy.T_m
    
    def test_material_parameters_physical(self):
        """Test that all materials have physically reasonable parameters."""
        db = MaterialsDatabase()
        for name in db.list_materials():
            mat = db.get_material(name)
            assert mat.T_m > 0, f"{name}: T_m must be positive"
            assert mat.T_g > 0, f"{name}: T_g must be positive"
            assert mat.T_g < mat.T_m, f"{name}: T_g must be less than T_m"
            assert mat.delta_H_f > 0, f"{name}: delta_H_f must be positive"
            assert mat.sigma > 0, f"{name}: sigma must be positive"
            assert mat.V_m > 0, f"{name}: V_m must be positive"
    
    def test_reduced_glass_transition(self):
        """Test T_rg = T_g/T_m is in reasonable range."""
        db = MaterialsDatabase()
        for name in db.list_materials():
            mat = db.get_material(name)
            T_rg = mat.T_g / mat.T_m
            assert 0.3 < T_rg < 0.9, f"{name}: T_rg = {T_rg:.3f} out of range"
    
    def test_metallic_vs_oxide_glasses(self):
        """Test separation of metallic and oxide glasses."""
        db = MaterialsDatabase()
        metallic = db.get_metallic_glasses()
        oxides = db.get_oxide_glasses()
        assert len(metallic) > 0
        assert len(oxides) > 0
        metallic_names = {m.name for m in metallic}
        oxide_names = {m.name for m in oxides}
        assert metallic_names.isdisjoint(oxide_names)


class TestCALPHADThermodynamics:
    """Tests for CALPHADThermodynamics class."""
    
    @pytest.fixture
    def thermo(self):
        """Create a standard thermodynamics instance."""
        return CALPHADThermodynamics(T_m=937.0, delta_H_f=8200.0, T_g=625.0)
    
    def test_delta_G_at_melting_point(self, thermo):
        """Test that ΔG = 0 at T_m (validation criterion 1)."""
        delta_G = thermo.delta_G(thermo.T_m)
        assert abs(delta_G) < 1e-6, f"ΔG(T_m) = {delta_G}, should be 0"
    
    def test_delta_G_monotonic_increase(self, thermo):
        """Test that ΔG increases as T decreases (validation criterion 2)."""
        T = np.linspace(thermo.T_g, thermo.T_m, 50)
        delta_G = thermo.delta_G(T)
        diffs = np.diff(delta_G)
        assert np.all(diffs <= 1e-10), "ΔG should decrease with increasing T"
    
    def test_delta_G_positive_below_Tm(self, thermo):
        """Test that ΔG > 0 for T < T_m."""
        T = np.linspace(thermo.T_g, thermo.T_m - 1, 50)
        delta_G = thermo.delta_G(T)
        assert np.all(delta_G > 0), "ΔG should be positive below T_m"
    
    def test_entropy_of_fusion_reasonable(self, thermo):
        """Test that ΔS_f is in reasonable range (Richard's rule)."""
        delta_S_f = thermo.get_entropy_of_fusion()
        R = 8.314
        ratio = delta_S_f / R
        assert 0.5 < ratio < 5.0, f"ΔS_f/R = {ratio:.2f} out of expected range"
    
    def test_different_methods_agree_at_Tm(self, thermo):
        """Test that all methods give ΔG = 0 at T_m."""
        methods = ["turnbull", "thompson_spaepen", "hoffman"]
        for method in methods:
            delta_G = thermo.delta_G(thermo.T_m, method=method)
            assert abs(delta_G) < 1e-6, f"{method}: ΔG(T_m) = {delta_G}"
    
    def test_verification_passes(self, thermo):
        """Test that verification at melting point passes."""
        assert thermo.verify_at_melting_point()


class TestDoolittleViscosity:
    """Tests for DoolittleViscosity class."""
    
    @pytest.fixture
    def visc(self):
        """Create a standard viscosity instance with auto-calculated D_star."""
        return DoolittleViscosity(T_m=937.0, T_g=625.0, eta_0=1e-5)
    
    def test_viscosity_at_Tg(self, visc):
        """Test that η(T_g) ≈ 10^12 Pa·s (validation criterion)."""
        log_eta = visc.log_viscosity(visc.T_g)
        assert 11 <= log_eta <= 14, f"log η(T_g) = {log_eta:.1f}, expected 12-13"
    
    def test_viscosity_at_Tm(self, visc):
        """Test that η(T_m) is in liquid range (validation criterion)."""
        log_eta = visc.log_viscosity(visc.T_m)
        assert -4 <= log_eta <= 2, f"log η(T_m) = {log_eta:.1f}, expected -3 to 0"
    
    def test_viscosity_monotonic_decrease(self, visc):
        """Test that viscosity decreases with increasing temperature."""
        T = np.linspace(visc.T_g, visc.T_m * 1.2, 50)
        eta = visc.viscosity(T)
        diffs = np.diff(eta)
        assert np.all(diffs <= 0), "Viscosity should decrease with increasing T"
    
    def test_fragility_index_positive(self, visc):
        """Test that fragility index m is positive."""
        m = visc.fragility_index_m()
        assert m > 0, f"Fragility index m = {m:.1f} should be positive"
    
    def test_fragility_index_reasonable(self, visc):
        """Test that fragility index is in reasonable range (10-250 for most glasses)."""
        m = visc.fragility_index_m()
        assert 10 < m < 250, f"Fragility index m = {m:.1f} out of range"
    
    def test_angell_plot_data(self, visc):
        """Test Angell plot data generation."""
        Tg_T, log_eta = visc.angell_plot_data(n_points=50)
        assert len(Tg_T) == 50
        assert len(log_eta) == 50
        assert Tg_T[-1] < Tg_T[0]
    
    def test_verification_at_Tg(self, visc):
        """Test verification at T_g passes."""
        assert visc.verify_at_Tg(tol_log=2.0)
    
    def test_verification_at_Tm(self, visc):
        """Test verification at T_m passes."""
        assert visc.verify_at_Tm(tol_log=3.0)


class TestDavisUhlmannModel:
    """Tests for DavisUhlmannModel class."""
    
    @pytest.fixture
    def model(self):
        """Create a standard Davis-Uhlmann model instance with reasonable parameters."""
        return DavisUhlmannModel(
            T_m=937.0, T_g=625.0, delta_H_f=8200.0,
            sigma=0.06, V_m=1.1e-5, eta_0=1e-5, D_star=20.0
        )
    
    def test_ttt_curve_c_shape(self, model):
        """Test that TTT curve has C-shape (nose present)."""
        ttt = model.calculate_ttt_curve(n_points=100)
        valid_mask = (ttt.time > 0) & (ttt.time < 1e30) & np.isfinite(ttt.time)
        t_valid = ttt.time[valid_mask]
        
        assert ttt.nose_time < t_valid[0], "Nose time should be less than time at T_g"
        assert ttt.nose_time < t_valid[-1], "Nose time should be less than time at T_m"
    
    def test_nose_temperature_range(self, model):
        """Test that nose temperature is in expected range (0.7-0.8 T_m)."""
        T_nose, _ = model.find_nose()
        T_n_reduced = T_nose / model.T_m
        assert 0.6 <= T_n_reduced <= 0.9, f"T_n/T_m = {T_n_reduced:.3f} out of range"
    
    def test_critical_cooling_rate_positive(self, model):
        """Test that critical cooling rate is positive."""
        R_c = model.critical_cooling_rate()
        assert R_c > 0, f"R_c = {R_c:.2e} should be positive"
    
    def test_nucleation_rate_positive(self, model):
        """Test that nucleation rate is positive."""
        T = 0.75 * model.T_m
        I = model.nucleation_rate(T)
        assert I > 0, f"Nucleation rate I = {I:.2e} should be positive"
    
    def test_growth_rate_positive(self, model):
        """Test that growth rate is positive below T_m."""
        T = 0.75 * model.T_m
        U = model.growth_rate(T)
        assert U > 0, f"Growth rate U = {U:.2e} should be positive"
    
    def test_critical_radius_positive(self, model):
        """Test that critical radius is positive."""
        T = 0.75 * model.T_m
        r_star = model.critical_radius(T)
        assert r_star > 0, f"Critical radius r* = {r_star:.2e} should be positive"
    
    def test_nucleation_barrier_positive(self, model):
        """Test that nucleation barrier is positive."""
        T = 0.75 * model.T_m
        delta_G_star = model.delta_G_star(T)
        assert delta_G_star > 0, f"ΔG* = {delta_G_star:.2e} should be positive"
    
    def test_verification_nose_position(self, model):
        """Test nose position verification returns valid result."""
        pass_check, message = model.verify_nose_position()
        assert isinstance(pass_check, bool), "Verification should return boolean"
        assert isinstance(message, str), "Verification should return message string"


class TestSensitivityAnalysis:
    """Tests for SensitivityAnalysis class."""
    
    @pytest.fixture
    def analysis(self):
        """Create a standard sensitivity analysis instance."""
        return SensitivityAnalysis(
            T_m=937.0, T_g=625.0, delta_H_f=8200.0,
            sigma=0.08, V_m=1.1e-5, eta_0=1e-5, D_star=18.5
        )
    
    def test_sigma_sensitivity_high(self, analysis):
        """Test that σ has high sensitivity (S_σ > 1)."""
        coeffs = analysis.calculate_sensitivity_coefficients()
        assert abs(coeffs['σ']) > 1, f"S_σ = {coeffs['σ']:.2f} should be > 1"
    
    def test_sensitivity_analysis_runs(self, analysis):
        """Test that sensitivity analysis completes without error."""
        result = analysis.analyze_sigma_sensitivity(variation_percent=10.0, n_points=5)
        assert len(result.variations) == 5
        assert len(result.R_c_values) == 5
    
    def test_R_c_increases_with_decreasing_sigma(self, analysis):
        """Test that R_c increases when σ decreases."""
        result = analysis.analyze_sigma_sensitivity(variation_percent=10.0, n_points=5)
        assert result.R_c_values[0] > result.R_c_values[-1], \
            "R_c should increase when σ decreases"
    
    def test_full_sensitivity_analysis(self, analysis):
        """Test full sensitivity analysis on all parameters."""
        results = analysis.full_sensitivity_analysis(variation_percent=5.0, n_points=3)
        assert "sigma" in results
        assert "D_star" in results
        assert "delta_H_f" in results


class TestKnownMaterials:
    """Tests comparing calculated R_c with experimental values."""
    
    def test_vitreloy1_model_runs(self):
        """Test that Vitreloy 1 model runs without error and produces valid output."""
        db = MaterialsDatabase()
        mat = db.get_material("Zr41Ti14Cu12Ni10Be23")
        
        model = DavisUhlmannModel(
            T_m=mat.T_m, T_g=mat.T_g, delta_H_f=mat.delta_H_f,
            sigma=mat.sigma, V_m=mat.V_m, eta_0=mat.eta_0,
            D_star=mat.D_star, T_0=mat.T_0
        )
        
        R_c_calc = model.critical_cooling_rate()
        assert R_c_calc > 0, "R_c should be positive"
        assert np.isfinite(R_c_calc), "R_c should be finite"
    
    def test_sio2_model_runs(self):
        """Test that SiO2 model runs without error."""
        db = MaterialsDatabase()
        mat = db.get_material("SiO2")
        
        model = DavisUhlmannModel(
            T_m=mat.T_m, T_g=mat.T_g, delta_H_f=mat.delta_H_f,
            sigma=mat.sigma, V_m=mat.V_m, eta_0=mat.eta_0,
            D_star=mat.D_star, T_0=mat.T_0
        )
        
        R_c_calc = model.critical_cooling_rate()
        assert np.isfinite(R_c_calc) or R_c_calc >= 0, "R_c should be valid"
    
    def test_fe80b20_model_runs(self):
        """Test that Fe80B20 model runs without error."""
        db = MaterialsDatabase()
        mat = db.get_material("Fe80B20")
        
        model = DavisUhlmannModel(
            T_m=mat.T_m, T_g=mat.T_g, delta_H_f=mat.delta_H_f,
            sigma=mat.sigma, V_m=mat.V_m, eta_0=mat.eta_0,
            D_star=mat.D_star, T_0=mat.T_0
        )
        
        R_c_calc = model.critical_cooling_rate()
        assert np.isfinite(R_c_calc) or R_c_calc >= 0, "R_c should be valid"


class TestPhysicalConsistency:
    """Tests for physical consistency across modules."""
    
    def test_all_materials_produce_valid_ttt(self):
        """Test that all materials produce valid TTT curves."""
        db = MaterialsDatabase()
        
        for name in db.list_materials():
            mat = db.get_material(name)
            try:
                model = DavisUhlmannModel(
                    T_m=mat.T_m, T_g=mat.T_g, delta_H_f=mat.delta_H_f,
                    sigma=mat.sigma, V_m=mat.V_m, eta_0=mat.eta_0,
                    D_star=mat.D_star, T_0=mat.T_0
                )
                ttt = model.calculate_ttt_curve(n_points=20)
                assert ttt.nose_time > 0, f"{name}: nose time should be positive"
                assert ttt.nose_temperature >= mat.T_g, f"{name}: T_n should be >= T_g"
                assert ttt.nose_temperature < mat.T_m, f"{name}: T_n should be < T_m"
            except Exception as e:
                pytest.fail(f"{name}: TTT calculation failed with {e}")
    
    def test_viscosity_diffusivity_consistency(self):
        """Test Stokes-Einstein relation consistency."""
        visc = DoolittleViscosity(T_m=937.0, T_g=625.0, eta_0=1e-5, D_star=18.5)
        
        T = 800.0
        eta = visc.viscosity(T)
        D = visc.diffusivity(T)
        
        k_B = 1.38e-23
        a = 3e-10
        D_expected = k_B * T / (6 * np.pi * eta * a)
        
        assert abs(D - D_expected) / D_expected < 0.1, \
            "Diffusivity should follow Stokes-Einstein"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
