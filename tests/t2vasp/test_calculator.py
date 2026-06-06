"""Tests for t2vasp.calculator — energy, structure, and crystal-field analysis."""

import numpy as np
import pytest

from t2vasp.calculator import (
    CalculationResult,
    CrystalFieldResult,
    EnergyResult,
    StructureResult,
    analyse,
    compute_crystal_field,
    compute_delta_energy,
    compute_energy,
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
