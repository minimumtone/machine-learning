"""Tests for t2vasp.exporter — CSV, JSON, and summary output."""

import csv
import json
from pathlib import Path

import pytest

from t2vasp.calculator import CalculationResult, EnergyResult, StructureResult
from t2vasp.exporter import export_csv, export_json, export_summary


def _sample_results() -> list:
    r1 = CalculationResult(
        label="calc_A",
        energy=EnergyResult(total_energy=-21.5, energy_per_atom=-5.375),
        structure=StructureResult(lattice_constant=3.52, volume=43.7,
                                  volume_per_atom=10.925, c_over_a=1.0),
        converged=True,
    )
    r2 = CalculationResult(
        label="calc_B",
        energy=EnergyResult(total_energy=-20.0, energy_per_atom=-5.0),
        structure=StructureResult(lattice_constant=3.60, volume=46.656,
                                  volume_per_atom=11.664, c_over_a=1.0),
        converged=False,
    )
    return [r1, r2]


class TestExportCSV:
    def test_creates_file(self, tmp_path: Path) -> None:
        out = tmp_path / "out.csv"
        export_csv(_sample_results(), out)
        assert out.is_file()

    def test_row_count(self, tmp_path: Path) -> None:
        out = tmp_path / "out.csv"
        export_csv(_sample_results(), out)
        with open(out) as f:
            reader = csv.reader(f)
            rows = list(reader)
        # header + 2 data rows
        assert len(rows) == 3

    def test_header_contains_label(self, tmp_path: Path) -> None:
        out = tmp_path / "out.csv"
        export_csv(_sample_results(), out)
        with open(out) as f:
            header = f.readline()
        assert "label" in header


class TestExportJSON:
    def test_creates_file(self, tmp_path: Path) -> None:
        out = tmp_path / "out.json"
        export_json(_sample_results(), out)
        assert out.is_file()

    def test_valid_json(self, tmp_path: Path) -> None:
        out = tmp_path / "out.json"
        export_json(_sample_results(), out)
        data = json.loads(out.read_text())
        assert isinstance(data, list)
        assert len(data) == 2

    def test_energy_present(self, tmp_path: Path) -> None:
        out = tmp_path / "out.json"
        export_json(_sample_results(), out)
        data = json.loads(out.read_text())
        assert "energy" in data[0]
        assert data[0]["energy"]["energy_per_atom"] == -5.375


class TestExportSummary:
    def test_returns_string(self) -> None:
        text = export_summary(_sample_results())
        assert isinstance(text, str)
        assert "calc_A" in text
        assert "calc_B" in text

    def test_convergence_count(self) -> None:
        text = export_summary(_sample_results())
        assert "Converged: 1/2" in text

    def test_writes_file(self, tmp_path: Path) -> None:
        out = tmp_path / "summary.txt"
        export_summary(_sample_results(), out)
        assert out.is_file()
        assert "t2vasp" in out.read_text()
