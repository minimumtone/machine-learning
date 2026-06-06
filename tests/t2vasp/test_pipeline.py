"""Tests for t2vasp.pipeline — batch processing and integration."""

import textwrap
from pathlib import Path

import pytest

from t2vasp.pipeline import process_batch, process_single


def _create_vasp_dir(base: Path, name: str) -> Path:
    """Create a minimal VASP directory with POSCAR + OUTCAR."""
    d = base / name
    d.mkdir(parents=True)

    poscar = textwrap.dedent("""\
        FCC test
        3.50
        1.0  0.0  0.0
        0.0  1.0  0.0
        0.0  0.0  1.0
        X
        4
        Direct
        0.0  0.0  0.0
        0.5  0.5  0.0
        0.5  0.0  0.5
        0.0  0.5  0.5
    """)
    (d / "POSCAR").write_text(poscar)

    outcar = textwrap.dedent("""\
        NIONS =      4
        free  energy   TOTEN  =      -20.00000000 eV
        energy  without entropy=     -19.80000000
        reached required accuracy
        FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)
    """)
    (d / "OUTCAR").write_text(outcar)
    return d


class TestProcessSingle:
    def test_returns_result(self, tmp_path: Path) -> None:
        d = _create_vasp_dir(tmp_path, "calc1")
        r = process_single(d)
        assert r.converged is True
        assert r.energy is not None
        assert r.structure is not None

    def test_energy_value(self, tmp_path: Path) -> None:
        d = _create_vasp_dir(tmp_path, "calc2")
        r = process_single(d)
        assert r.energy is not None
        assert pytest.approx(r.energy.energy_per_atom, abs=0.01) == -5.0


class TestProcessBatch:
    def test_batch_multiple(self, tmp_path: Path) -> None:
        _create_vasp_dir(tmp_path, "A")
        _create_vasp_dir(tmp_path, "B")
        results = process_batch(tmp_path, output_dir=tmp_path / "out",
                                export_formats=["csv", "json", "summary"])
        assert len(results) == 2
        assert (tmp_path / "out" / "results.csv").is_file()
        assert (tmp_path / "out" / "results.json").is_file()
        assert (tmp_path / "out" / "summary.txt").is_file()

    def test_empty_dir(self, tmp_path: Path) -> None:
        results = process_batch(tmp_path)
        assert results == []
