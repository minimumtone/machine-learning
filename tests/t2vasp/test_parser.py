"""Tests for t2vasp.parser — VASP file parsing."""

import textwrap
from pathlib import Path

import numpy as np
import pytest

from t2vasp.parser import (
    DosData,
    OutcarData,
    StructureData,
    parse_doscar,
    parse_outcar,
    parse_poscar,
    parse_calc_dir,
)


# ── Fixtures: create minimal VASP files in a tmp directory ───────────

@pytest.fixture()
def poscar_path(tmp_path: Path) -> Path:
    content = textwrap.dedent("""\
        FCC Ni
        3.524
        1.0  0.0  0.0
        0.0  1.0  0.0
        0.0  0.0  1.0
        Ni
        4
        Direct
        0.0  0.0  0.0
        0.5  0.5  0.0
        0.5  0.0  0.5
        0.0  0.5  0.5
    """)
    p = tmp_path / "POSCAR"
    p.write_text(content)
    return p


@pytest.fixture()
def outcar_path(tmp_path: Path) -> Path:
    content = textwrap.dedent("""\
        some preamble
        NIONS =      4
        free  energy   TOTEN  =      -21.56789012 eV
        energy  without entropy=     -21.50000000
        E-fermi :   5.12340
        number of electron     32.0000000 magnetization       0.00010
        reached required accuracy
        FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)
        FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)
        in kB    -1.23   -1.23   -1.23    0.00    0.00    0.00
        Elapsed time (sec):    42.5
    """)
    p = tmp_path / "OUTCAR"
    p.write_text(content)
    return p


@pytest.fixture()
def doscar_path(tmp_path: Path) -> Path:
    lines = [
        "   4   4   1   0",
        "  0.40635010E+02  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00",
        " -0.50000000E+01  0.50000000E+01  0.00000000E+00  0.00000000E+00  0.00000000E+00",
        "  0.34000000E+02",
        "  0.00000000E+00",
        "  10.0000  -5.0000   5   5.1234   1  1",
    ]
    # 5 DOS points
    for i in range(5):
        e = -5.0 + 2.5 * i
        dos_val = max(0, 2.0 - abs(e))
        lines.append(f"  {e:.4f}   {dos_val:.4f}   0.0000")
    p = tmp_path / "DOSCAR"
    p.write_text("\n".join(lines) + "\n")
    return p


# ── POSCAR tests ─────────────────────────────────────────────────────

class TestParsePoscar:
    def test_lattice_shape(self, poscar_path: Path) -> None:
        sd = parse_poscar(poscar_path)
        assert sd.lattice.shape == (3, 3)

    def test_lattice_constant(self, poscar_path: Path) -> None:
        sd = parse_poscar(poscar_path)
        assert pytest.approx(sd.lattice_constant, abs=0.01) == 3.524

    def test_species_count(self, poscar_path: Path) -> None:
        sd = parse_poscar(poscar_path)
        assert len(sd.species) == 4
        assert all(s == "Ni" for s in sd.species)

    def test_positions_shape(self, poscar_path: Path) -> None:
        sd = parse_poscar(poscar_path)
        assert sd.positions.shape == (4, 3)

    def test_volume_positive(self, poscar_path: Path) -> None:
        sd = parse_poscar(poscar_path)
        assert sd.volume > 0


# ── OUTCAR tests ─────────────────────────────────────────────────────

class TestParseOutcar:
    def test_total_energy(self, outcar_path: Path) -> None:
        od = parse_outcar(outcar_path)
        assert od.total_energy is not None
        assert pytest.approx(od.total_energy, abs=0.001) == -21.568

    def test_num_atoms(self, outcar_path: Path) -> None:
        od = parse_outcar(outcar_path)
        assert od.num_atoms == 4

    def test_convergence(self, outcar_path: Path) -> None:
        od = parse_outcar(outcar_path)
        assert od.converged is True

    def test_energy_per_atom(self, outcar_path: Path) -> None:
        od = parse_outcar(outcar_path)
        assert od.energy_per_atom is not None
        assert pytest.approx(od.energy_per_atom, abs=0.01) == -21.568 / 4

    def test_fermi_energy(self, outcar_path: Path) -> None:
        od = parse_outcar(outcar_path)
        assert od.fermi_energy is not None
        assert pytest.approx(od.fermi_energy, abs=0.01) == 5.1234

    def test_ionic_steps(self, outcar_path: Path) -> None:
        od = parse_outcar(outcar_path)
        assert od.ionic_steps == 2

    def test_stress_tensor(self, outcar_path: Path) -> None:
        od = parse_outcar(outcar_path)
        assert od.stress_tensor is not None
        assert len(od.stress_tensor) == 6

    def test_elapsed_time(self, outcar_path: Path) -> None:
        od = parse_outcar(outcar_path)
        assert od.elapsed_time == 42.5


# ── DOSCAR tests ─────────────────────────────────────────────────────

class TestParseDoscar:
    def test_energies_length(self, doscar_path: Path) -> None:
        dd = parse_doscar(doscar_path)
        assert len(dd.energies) == 5

    def test_fermi_energy(self, doscar_path: Path) -> None:
        dd = parse_doscar(doscar_path)
        assert pytest.approx(dd.fermi_energy, abs=0.01) == 5.1234


# ── parse_calc_dir integration test ─────────────────────────────────

class TestParseCalcDir:
    def test_finds_poscar_and_outcar(self, tmp_path: Path,
                                      poscar_path: Path,
                                      outcar_path: Path) -> None:
        result = parse_calc_dir(tmp_path)
        assert "structure" in result
        assert "outcar" in result
        assert result["outcar"].num_atoms == 4
