"""Tests for t2vasp.calculator — energy, structure, and crystal-field analysis."""

import numpy as np
import pytest

from t2vasp.calculator import (
    CalculationResult,
    CrystalFieldResult,
    EnergyResult,
    JahnTellerResult,
    StructureResult,
    analyse,
    compute_cfse,
    compute_crystal_field,
    compute_delta_energy,
    compute_energy,
    compute_jahn_teller_energy,
    compute_structure_metrics,
)
from t2vasp.parser import DosData, OutcarData, StructureData


# ── Helpers ──────────────────────────────────────────────────────────

def _make_outcar(**overrides) -> OutcarData:
    defaults = dict(total_energy=-21.568, num_atoms=4, converged=True)
    defaults.update(overrides)
    return OutcarData(**defaults)


def _make_structure(**overrides) -> StructureData:
    defaults = dict(
        lattice=np.diag([3.524, 3.524, 3.524]),
        species=["Ni"] * 4,
        positions=np.array([[0, 0, 0], [0.5, 0.5, 0],
                            [0.5, 0, 0.5], [0, 0.5, 0.5]]),
    )
    defaults.update(overrides)
    return StructureData(**defaults)


# ── Energy tests ─────────────────────────────────────────────────────

class TestComputeEnergy:
    def test_energy_per_atom(self) -> None:
        oc = _make_outcar()
        er = compute_energy(oc)
        assert pytest.approx(er.energy_per_atom, abs=0.01) == -21.568 / 4

    def test_formation_energy_with_refs(self) -> None:
        oc = _make_outcar(total_energy=-21.0, num_atoms=4)
        refs = {"Ni": -5.0}
        er = compute_energy(oc, reference_energies=refs, species=["Ni"] * 4)
        # formation = (-21 - 4*(-5)) / 4 = (-21 + 20) / 4 = -0.25
        assert er.formation_energy is not None
        assert pytest.approx(er.formation_energy, abs=0.01) == -0.25

    def test_missing_energy_raises(self) -> None:
        oc = OutcarData()
        with pytest.raises(ValueError):
            compute_energy(oc)


# ── Structure tests ──────────────────────────────────────────────────

class TestComputeStructureMetrics:
    def test_lattice_constant(self) -> None:
        sd = _make_structure()
        sr = compute_structure_metrics(sd)
        assert pytest.approx(sr.lattice_constant, abs=0.001) == 3.524

    def test_volume(self) -> None:
        sd = _make_structure()
        sr = compute_structure_metrics(sd)
        assert pytest.approx(sr.volume, abs=0.01) == 3.524 ** 3

    def test_c_over_a_cubic(self) -> None:
        sd = _make_structure()
        sr = compute_structure_metrics(sd)
        assert pytest.approx(sr.c_over_a, abs=0.001) == 1.0

    def test_max_force_from_outcar(self) -> None:
        oc = _make_outcar(forces=np.array([[0.1, 0.0, 0.0],
                                            [0.0, 0.2, 0.0],
                                            [0.0, 0.0, 0.05],
                                            [0.0, 0.0, 0.0]]))
        sd = _make_structure()
        sr = compute_structure_metrics(sd, oc)
        assert sr.max_force is not None
        assert pytest.approx(sr.max_force, abs=0.01) == 0.2

    def test_pressure_from_stress(self) -> None:
        oc = _make_outcar(stress_tensor=np.array([-3.0, -3.0, -3.0, 0, 0, 0]))
        sd = _make_structure()
        sr = compute_structure_metrics(sd, oc)
        assert sr.pressure is not None
        assert pytest.approx(sr.pressure, abs=0.01) == -3.0


# ── Crystal-field tests ─────────────────────────────────────────────

class TestComputeCrystalField:
    def test_empty_without_projected(self) -> None:
        dos = DosData(energies=np.linspace(-5, 5, 100),
                      total_dos=np.ones(100), fermi_energy=0.0)
        cf = compute_crystal_field(dos)
        assert cf.splitting is None

    def test_splitting_computed(self) -> None:
        e = np.linspace(-6, 4, 200)
        t2g = np.exp(-((e + 1) ** 2) / 0.5)
        eg = np.exp(-((e - 2) ** 2) / 0.5)
        projected = {"Ni_dxy": t2g, "Ni_dz2": eg}
        dos = DosData(energies=e, total_dos=t2g + eg, fermi_energy=0.0,
                      projected_dos=projected)
        cf = compute_crystal_field(dos)
        assert cf.t2g_center is not None
        assert cf.eg_center is not None
        assert cf.splitting is not None
        # eg centre should be above t2g centre
        assert cf.splitting > 0

    def test_cfse_and_jt_with_d_electrons(self) -> None:
        """CFSE and JT indicators when n_d_electrons is supplied."""
        e = np.linspace(-6, 4, 200)
        t2g = np.exp(-((e + 1) ** 2) / 0.5)
        eg = np.exp(-((e - 2) ** 2) / 0.5)
        projected = {"Cu_dxy": t2g, "Cu_dz2": eg}
        dos = DosData(energies=e, total_dos=t2g + eg, fermi_energy=0.0,
                      projected_dos=projected)
        # Cu2+ is d9 — strong Jahn-Teller active
        cf = compute_crystal_field(dos, n_d_electrons=9, c_over_a=1.05)
        assert cf.cfse is not None
        assert cf.jt_active is True
        assert cf.jt_strength == "strong"
        assert cf.tetragonality is not None
        assert pytest.approx(cf.tetragonality, abs=0.001) == 0.05

    def test_eg_splitting_from_dos(self) -> None:
        """eg splitting extracted when dz2 and dx2 have different centres."""
        e = np.linspace(-6, 4, 200)
        dz2 = np.exp(-((e - 1.5) ** 2) / 0.3)
        dx2 = np.exp(-((e - 3.0) ** 2) / 0.3)
        t2g = np.exp(-((e + 1) ** 2) / 0.5)
        projected = {"Mn_dxy": t2g, "Mn_dz2": dz2, "Mn_dx2": dx2}
        dos = DosData(energies=e, total_dos=t2g + dz2 + dx2,
                      fermi_energy=0.0, projected_dos=projected)
        cf = compute_crystal_field(dos)
        assert cf.eg_splitting is not None
        assert cf.eg_splitting > 1.0  # dz2 at ~1.5, dx2 at ~3.0


# ── ΔE ranking tests ────────────────────────────────────────────────

class TestDeltaEnergy:
    def test_relative_to_min(self) -> None:
        r1 = CalculationResult(label="A", energy=EnergyResult(-5.0, -5.0))
        r2 = CalculationResult(label="B", energy=EnergyResult(-4.5, -4.5))
        delta = compute_delta_energy([r1, r2])
        assert delta["A"] == 0.0
        assert pytest.approx(delta["B"], abs=0.01) == 0.5

    def test_explicit_reference(self) -> None:
        r1 = CalculationResult(label="A", energy=EnergyResult(-5.0, -5.0))
        r2 = CalculationResult(label="B", energy=EnergyResult(-4.5, -4.5))
        delta = compute_delta_energy([r1, r2], reference_label="B")
        assert pytest.approx(delta["A"], abs=0.01) == -0.5
        assert delta["B"] == 0.0


# ── CFSE tests ──────────────────────────────────────────────────────

class TestComputeCFSE:
    def test_d0_zero_cfse(self) -> None:
        cfse, cfse_delta, jt = compute_cfse(0, delta_oct=2.0)
        assert cfse == 0.0
        assert cfse_delta == 0.0
        assert jt is None

    def test_d3_hs_stabilized(self) -> None:
        """d3 (e.g. Cr3+) has CFSE = -1.2Δ (t2g^3, half-filled)."""
        cfse, cfse_delta, jt = compute_cfse(3, delta_oct=2.0)
        assert pytest.approx(cfse_delta) == -1.2
        assert pytest.approx(cfse) == -2.4
        assert jt is None  # not JT-active

    def test_d4_hs_strong_jt(self) -> None:
        """d4 high-spin (e.g. Cr2+) — strong Jahn-Teller."""
        cfse, cfse_delta, jt = compute_cfse(4, delta_oct=1.5)
        assert pytest.approx(cfse_delta) == 3 * (-0.4) + 1 * 0.6  # -0.6
        assert pytest.approx(cfse) == -0.6 * 1.5
        assert jt == "strong"

    def test_d9_strong_jt(self) -> None:
        """d9 (e.g. Cu2+) — canonical strong Jahn-Teller case."""
        cfse, cfse_delta, jt = compute_cfse(9, delta_oct=1.0)
        assert pytest.approx(cfse_delta) == 6 * (-0.4) + 3 * 0.6  # -0.6
        assert jt == "strong"

    def test_d7_ls_strong_jt(self) -> None:
        """d7 low-spin (e.g. Ni3+ in NaNiO2) — strong JT."""
        cfse, cfse_delta, jt = compute_cfse(7, delta_oct=2.0, low_spin=True)
        assert pytest.approx(cfse_delta) == 6 * (-0.4) + 1 * 0.6  # -1.8
        assert jt == "strong"

    def test_pairing_energy_contribution(self) -> None:
        """Pairing energy adds n_pairs × P to CFSE."""
        cfse_no_p, _, _ = compute_cfse(9, delta_oct=2.0, pairing_energy=0.0)
        cfse_with_p, _, _ = compute_cfse(9, delta_oct=2.0, pairing_energy=0.5)
        # d9 HS has 4 extra pairs
        assert pytest.approx(cfse_with_p - cfse_no_p) == 4 * 0.5

    def test_invalid_d_count_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_cfse(11, delta_oct=1.0)
        with pytest.raises(ValueError):
            compute_cfse(-1, delta_oct=1.0)


# ── Jahn-Teller stabilisation energy tests ──────────────────────────

class TestComputeJahnTellerEnergy:
    def test_jtse_positive_for_distortion(self) -> None:
        """Distorted structure lower in energy → positive JTSE."""
        und = CalculationResult(
            label="cubic",
            energy=EnergyResult(-20.0, -5.0),
            structure=StructureResult(3.5, 42.875, 10.71875, 1.0),
        )
        dis = CalculationResult(
            label="tetragonal",
            energy=EnergyResult(-20.3, -5.075),
            structure=StructureResult(3.5, 42.875, 10.71875, 1.05),
        )
        jt = compute_jahn_teller_energy(und, dis)
        assert jt.jtse > 0
        assert pytest.approx(jt.jtse, abs=0.001) == 0.3
        assert pytest.approx(jt.delta_c_over_a, abs=0.001) == 0.05

    def test_jtse_per_atom(self) -> None:
        und = CalculationResult(
            label="cubic",
            energy=EnergyResult(-20.0, -5.0),  # 4 atoms
        )
        dis = CalculationResult(
            label="tetragonal",
            energy=EnergyResult(-20.4, -5.1),  # 4 atoms
        )
        jt = compute_jahn_teller_energy(und, dis)
        assert pytest.approx(jt.jtse_per_atom, abs=0.001) == 0.1

    def test_missing_energy_raises(self) -> None:
        und = CalculationResult(label="A")
        dis = CalculationResult(label="B", energy=EnergyResult(-10, -5))
        with pytest.raises(ValueError):
            compute_jahn_teller_energy(und, dis)


# ── analyse() integration test ──────────────────────────────────────

class TestAnalyse:
    def test_full_analysis(self) -> None:
        parsed = {
            "path": "/test/calc",
            "outcar": _make_outcar(),
            "structure": _make_structure(),
        }
        cr = analyse(parsed)
        assert cr.energy is not None
        assert cr.structure is not None
        assert cr.converged is True
