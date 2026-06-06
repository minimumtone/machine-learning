"""Tests for t2vasp.generator — end-to-end VASP input generation."""

import pytest
from pathlib import Path
from t2vasp.intent import classify
from t2vasp.entity import extract
from t2vasp.generator import generate, estimate_lattice_constant


# ── Lattice constant estimation ──────────────────────────────────────

def test_lattice_ni3al() -> None:
    a0 = estimate_lattice_constant(["Ni", "Al"], [3.0, 1.0], "L12")
    assert 3.5 < a0 < 3.8  # known ~3.57 Å


def test_lattice_bcc_fe() -> None:
    a0 = estimate_lattice_constant(["Fe"], [1.0], "BCC")
    assert 2.7 < a0 < 3.0  # known ~2.87 Å


# ── End-to-end generation ────────────────────────────────────────────

def test_generate_ni3al_l12(tmp_path: Path) -> None:
    intent = classify("Ni3AlのL12構造を最適化して")
    entity = extract("Ni3AlのL12構造を最適化して")
    plan = generate(intent, entity, tmp_path)

    assert (tmp_path / "INCAR").exists()
    assert (tmp_path / "POSCAR").exists()
    assert (tmp_path / "KPOINTS").exists()
    assert (tmp_path / "make_potcar.sh").exists()
    assert (tmp_path / "t2vasp_plan.yaml").exists()
    assert plan["calc_type"] == "relax"
    assert plan["prototype"] == "L12"


def test_generate_batio3_polarization(tmp_path: Path) -> None:
    query = "BaTiO3のペロブスカイト構造で自発分極をBerry phaseで計算"
    intent = classify(query)
    entity = extract(query)
    plan = generate(intent, entity, tmp_path)

    assert plan["calc_type"] == "polarization"
    assert plan["prototype"] == "perovskite"
    incar_text = (tmp_path / "INCAR").read_text()
    assert "LCALCPOL" in incar_text

    poscar_text = (tmp_path / "POSCAR").read_text()
    assert "perovskite" in poscar_text
    assert "Ba" in poscar_text
    assert "Ti" in poscar_text
    assert "O" in poscar_text


def test_generate_fe_bcc_magnetic(tmp_path: Path) -> None:
    query = "Fe BCC構造の磁性を計算"
    intent = classify(query)
    entity = extract(query)
    plan = generate(intent, entity, tmp_path)

    assert plan["calc_type"] == "magnetic"
    assert plan["spin_polarized"] is True
    incar_text = (tmp_path / "INCAR").read_text()
    assert "ISPIN = 2" in incar_text


def test_generate_hea_sqs(tmp_path: Path) -> None:
    query = "CrFeCoNiのBCC SQSを作って"
    intent = classify(query)
    entity = extract(query)
    plan = generate(intent, entity, tmp_path)

    assert plan["calc_type"] == "sqs"
    assert set(plan["elements"]) == {"Cr", "Fe", "Co", "Ni"}


def test_generate_phonon(tmp_path: Path) -> None:
    query = "Siのフォノン分散を計算"
    intent = classify(query)
    entity = extract(query)
    plan = generate(intent, entity, tmp_path)

    assert plan["calc_type"] == "phonon"
    incar_text = (tmp_path / "INCAR").read_text()
    assert "IBRION = 6" in incar_text


# ── Dry run ──────────────────────────────────────────────────────────

def test_dry_run(tmp_path: Path) -> None:
    intent = classify("Ni3Al relax")
    entity = extract("Ni3Al relax")
    plan = generate(intent, entity, tmp_path, dry_run=True)

    # Files should NOT be created
    assert not (tmp_path / "INCAR").exists()
    assert plan["formula"] == "Ni3Al"


# ── Scheduler variants ──────────────────────────────────────────────

def test_generate_pbs_scheduler(tmp_path: Path) -> None:
    intent = classify("Fe relax")
    entity = extract("Fe relax")
    plan = generate(intent, entity, tmp_path, scheduler="pbs")

    assert (tmp_path / "job_pbs.sh").exists()
    pbs_text = (tmp_path / "job_pbs.sh").read_text()
    assert "#PBS" in pbs_text


def test_generate_slurm_scheduler(tmp_path: Path) -> None:
    intent = classify("Fe relax")
    entity = extract("Fe relax")
    plan = generate(intent, entity, tmp_path, scheduler="slurm")

    assert (tmp_path / "job_slurm.sh").exists()
    slurm_text = (tmp_path / "job_slurm.sh").read_text()
    assert "#SBATCH" in slurm_text


def test_generate_local_scheduler(tmp_path: Path) -> None:
    intent = classify("Fe relax")
    entity = extract("Fe relax")
    plan = generate(intent, entity, tmp_path, scheduler="local")

    assert (tmp_path / "run_local.sh").exists()


# ── POTCAR script ────────────────────────────────────────────────────

def test_potcar_script_content(tmp_path: Path) -> None:
    intent = classify("Ni3AlのL12を最適化")
    entity = extract("Ni3AlのL12を最適化")
    generate(intent, entity, tmp_path)

    potcar_text = (tmp_path / "make_potcar.sh").read_text()
    assert "VASPPOT" in potcar_text
    assert "Ni_pv" in potcar_text
    assert "Al" in potcar_text


# ── KPOINTS ──────────────────────────────────────────────────────────

def test_kpoints_content(tmp_path: Path) -> None:
    intent = classify("Ni3Al relax")
    entity = extract("Ni3Al relax")
    generate(intent, entity, tmp_path)

    kpt_text = (tmp_path / "KPOINTS").read_text()
    assert "Gamma" in kpt_text
