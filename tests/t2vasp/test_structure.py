"""Tests for t2vasp.structure — ASE-based structure manipulation."""

from pathlib import Path

import numpy as np
import pytest

ase = pytest.importorskip("ase")

from t2vasp.structure import (
    apply_strain,
    create_b2,
    create_bulk,
    create_l12,
    generate_candidates,
    perturb_positions,
    write_poscar,
)


class TestCreateBulk:
    def test_fcc_ni(self) -> None:
        atoms = create_bulk("Ni", "fcc", a=3.524)
        assert len(atoms) == 1
        assert atoms.get_chemical_symbols() == ["Ni"]

    def test_bcc_fe(self) -> None:
        atoms = create_bulk("Fe", "bcc", a=2.87)
        assert len(atoms) == 1


class TestCreateL12:
    def test_atom_count(self) -> None:
        atoms = create_l12("Ni", "Al", a=3.56)
        assert len(atoms) == 4

    def test_species(self) -> None:
        atoms = create_l12("Ni", "Al", a=3.56)
        syms = sorted(atoms.get_chemical_symbols())
        assert syms.count("Al") == 1
        assert syms.count("Ni") == 3


class TestCreateB2:
    def test_atom_count(self) -> None:
        atoms = create_b2("Fe", "Al", a=2.90)
        assert len(atoms) == 2


class TestApplyStrain:
    def test_isotropic_strain(self) -> None:
        atoms = create_l12("Ni", "Al", a=3.524)
        a_orig = np.linalg.norm(atoms.cell[0])
        strained = apply_strain(atoms, 0.01, axis=0)
        a_new = np.linalg.norm(strained.cell[0])
        assert pytest.approx(a_new, abs=0.01) == a_orig * 1.01

    def test_uniaxial_strain(self) -> None:
        atoms = create_l12("Ni", "Al", a=3.524)
        a_orig = np.linalg.norm(atoms.cell[0])
        strained = apply_strain(atoms, 0.02, axis=1)
        a_new = np.linalg.norm(strained.cell[0])
        assert pytest.approx(a_new, abs=0.01) == a_orig * 1.02
        # b and c unchanged
        b_new = np.linalg.norm(strained.cell[1])
        assert pytest.approx(b_new, abs=0.01) == a_orig


class TestPerturbPositions:
    def test_positions_change(self) -> None:
        atoms = create_l12("Ni", "Al", a=3.56)
        perturbed = perturb_positions(atoms, amplitude=0.1, seed=42)
        assert not np.allclose(atoms.positions, perturbed.positions)


class TestWritePoscar:
    def test_creates_file(self, tmp_path: Path) -> None:
        atoms = create_l12("Ni", "Al", a=3.56)
        p = write_poscar(atoms, tmp_path / "POSCAR")
        assert p.is_file()

    def test_file_content(self, tmp_path: Path) -> None:
        atoms = create_l12("Ni", "Al", a=3.56)
        p = write_poscar(atoms, tmp_path / "POSCAR")
        text = p.read_text()
        assert "Ni" in text
        assert "Al" in text


class TestGenerateCandidates:
    def test_generates_poscars(self, tmp_path: Path) -> None:
        atoms = create_bulk("Ni", "fcc", a=3.524)
        paths = generate_candidates(atoms, output_dir=tmp_path / "cand")
        assert len(paths) == 4
        for p in paths:
            assert p.is_file()
