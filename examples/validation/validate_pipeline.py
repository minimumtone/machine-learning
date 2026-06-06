#!/usr/bin/env python3
"""Validation 2: Synthetic VASP data — full pipeline verification.

Creates synthetic VASP output files (POSCAR, OUTCAR, DOSCAR) for
well-characterised test cases and verifies that the t2vasp pipeline
extracts physically correct values.

Test cases:
  A) Cubic Ni FCC (d8) — no Jahn-Teller distortion expected
  B) Cu²⁺ oxide-like (d9) — strong Jahn-Teller, tetragonal elongation
  C) Cr²⁺ (d4 HS) — strong Jahn-Teller
  D) Paired calculation — JTSE from undistorted/distorted energies

Each case defines known "ground truth" values and compares them
with t2vasp output.  Results are printed to console and optionally
saved as a Markdown report with embedded figures.

Usage:
    python examples/validation/validate_pipeline.py
    python examples/validation/validate_pipeline.py -o report.md
"""

import argparse
import logging
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# Ensure t2vasp is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from t2vasp.calculator import (  # noqa: E402
    CrystalFieldResult,
    EnergyResult,
    JahnTellerResult,
    StructureResult,
    compute_cfse,
    compute_crystal_field,
    compute_energy,
    compute_jahn_teller_energy,
    compute_structure_metrics,
    CalculationResult,
)
from t2vasp.parser import (  # noqa: E402
    DosData,
    OutcarData,
    StructureData,
    parse_poscar,
    parse_outcar,
    parse_doscar,
)
from t2vasp.exporter import export_csv, export_json, export_summary  # noqa: E402

logger = logging.getLogger(__name__)

# ── Figure output directory ──────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = SCRIPT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)


# ── Helpers: synthetic file generators ───────────────────────────────

def _write_poscar(path: Path, lattice: np.ndarray, species: list[str],
                  positions: np.ndarray, scale: float = 1.0) -> None:
    """Write a minimal POSCAR file."""
    from collections import Counter
    counts = Counter(species)
    unique = list(counts.keys())
    with open(path, "w") as f:
        f.write("Synthetic POSCAR\n")
        f.write(f"  {scale:.6f}\n")
        for row in lattice:
            f.write(f"  {row[0]:12.8f}  {row[1]:12.8f}  {row[2]:12.8f}\n")
        f.write("  " + "  ".join(unique) + "\n")
        f.write("  " + "  ".join(str(counts[s]) for s in unique) + "\n")
        f.write("Direct\n")
        for pos in positions:
            f.write(f"  {pos[0]:12.8f}  {pos[1]:12.8f}  {pos[2]:12.8f}\n")


def _write_outcar(path: Path, total_energy: float, num_atoms: int,
                  converged: bool = True, forces: Optional[np.ndarray] = None,
                  stress: Optional[np.ndarray] = None,
                  elapsed: float = 120.0) -> None:
    """Write a minimal OUTCAR with key fields."""
    with open(path, "w") as f:
        f.write(" vasp.6.4.0 (synthetic)\n")
        f.write(f" NIONS =       {num_atoms}\n")
        f.write(f"  free  energy   TOTEN  =      {total_energy:.8f} eV\n")
        if converged:
            f.write("                    aborting loop EDIFF is reached\n")
            f.write(" reached required accuracy - Loss function\n")
        if forces is not None:
            f.write(" POSITION                                       TOTAL-FORCE (eV/Angst)\n")
            f.write(" -----------------------------------------------------------------------------------\n")
            for i in range(num_atoms):
                pos = [0.0, 0.0, 0.0]
                f.write(f"  {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}"
                        f"  {forces[i,0]:12.6f} {forces[i,1]:12.6f} {forces[i,2]:12.6f}\n")
            f.write(" -----------------------------------------------------------------------------------\n")
        if stress is not None:
            f.write(f"  in kB  {stress[0]:12.4f}  {stress[1]:12.4f}  {stress[2]:12.4f}"
                    f"  {stress[3]:12.4f}  {stress[4]:12.4f}  {stress[5]:12.4f}\n")
        f.write(f"       Elapsed time (sec):      {elapsed:.3f}\n")


def _write_doscar(path: Path, energies: np.ndarray, total_dos: np.ndarray,
                  fermi: float, num_atoms: int = 4) -> None:
    """Write a minimal DOSCAR."""
    with open(path, "w") as f:
        f.write(f"    {num_atoms}    {num_atoms}    1    0\n")
        f.write("  0.0  0.0  0.0  0.0  0.0\n")
        f.write("  0.0  0.0  0.0  0.0  0.0\n")
        f.write("  0.0  0.0  0.0  0.0  0.0\n")
        f.write("  0.0  0.0  0.0  0.0  0.0\n")
        e_max = energies[-1]
        e_min = energies[0]
        nedos = len(energies)
        f.write(f"  {e_max:.6f}  {e_min:.6f}  {nedos}  {fermi:.6f}  1.0000\n")
        int_dos = np.cumsum(total_dos) * (energies[1] - energies[0]) if len(energies) > 1 else total_dos
        for e, d, intd in zip(energies, total_dos, int_dos):
            f.write(f"  {e:.6f}  {d:.6f}  {intd:.6f}\n")


def _make_gaussian_dos(center: float, sigma: float, energies: np.ndarray) -> np.ndarray:
    return np.exp(-((energies - center) ** 2) / (2 * sigma ** 2))


# ── Validation result container ──────────────────────────────────────

@dataclass
class CheckResult:
    name: str
    expected: str
    actual: str
    passed: bool
    note: str = ""


# ── Case A: Cubic Ni FCC (d8) ───────────────────────────────────────

def case_a_cubic_ni(tmpdir: Path) -> list[CheckResult]:
    """Ni FCC d8: CFSE = -1.2Δ, JT inactive."""
    checks: list[CheckResult] = []

    # Known values
    a = 3.524  # Å
    e_total = -21.568  # eV
    n_atoms = 4
    e_per_atom = e_total / n_atoms

    # Create synthetic files
    calc_dir = tmpdir / "Ni_fcc"
    calc_dir.mkdir()

    lattice = np.diag([a, a, a])
    positions = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    species = ["Ni"] * 4

    _write_poscar(calc_dir / "POSCAR", lattice, species, positions)
    _write_outcar(calc_dir / "OUTCAR", e_total, n_atoms,
                  forces=np.array([[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01], [0, 0, 0]]),
                  stress=np.array([-2.5, -2.5, -2.5, 0, 0, 0]))

    # Parse
    struct = parse_poscar(str(calc_dir / "POSCAR"))
    outcar = parse_outcar(str(calc_dir / "OUTCAR"))

    # Check POSCAR parsing
    checks.append(CheckResult(
        "A1: Lattice constant",
        f"{a:.3f} Å",
        f"{struct.lattice_constant:.3f} Å",
        abs(struct.lattice_constant - a) < 0.001,
    ))
    checks.append(CheckResult(
        "A2: Number of atoms",
        str(n_atoms),
        str(len(struct.species)),
        len(struct.species) == n_atoms,
    ))
    checks.append(CheckResult(
        "A3: Volume",
        f"{a**3:.3f} Å³",
        f"{struct.volume:.3f} Å³",
        abs(struct.volume - a**3) < 0.01,
    ))

    # Check OUTCAR parsing
    checks.append(CheckResult(
        "A4: Total energy",
        f"{e_total:.3f} eV",
        f"{outcar.total_energy:.3f} eV",
        abs(outcar.total_energy - e_total) < 0.001,
    ))
    checks.append(CheckResult(
        "A5: Convergence",
        "True",
        str(outcar.converged),
        outcar.converged,
    ))

    # Energy analysis
    er = compute_energy(outcar)
    checks.append(CheckResult(
        "A6: Energy per atom",
        f"{e_per_atom:.4f} eV/atom",
        f"{er.energy_per_atom:.4f} eV/atom",
        abs(er.energy_per_atom - e_per_atom) < 0.001,
    ))

    # Structure metrics
    sm = compute_structure_metrics(struct, outcar)
    checks.append(CheckResult(
        "A7: c/a ratio (cubic → 1.0)",
        "1.000",
        f"{sm.c_over_a:.3f}",
        abs(sm.c_over_a - 1.0) < 0.001,
    ))

    # CFSE for d8: -1.2Δ, JT inactive
    cfse_ev, cfse_delta, jt = compute_cfse(8, delta_oct=1.5)
    checks.append(CheckResult(
        "A8: CFSE(d8) in units of Δ",
        "-1.2",
        f"{cfse_delta:.1f}",
        abs(cfse_delta - (-1.2)) < 1e-10,
    ))
    checks.append(CheckResult(
        "A9: JT activity (d8 → inactive)",
        "None",
        str(jt),
        jt is None,
    ))

    return checks


# ── Case B: Cu²⁺ d9 — Strong Jahn-Teller ───────────────────────────

def case_b_cu_d9(tmpdir: Path) -> list[CheckResult]:
    """Cu²⁺ d9: strong JT, tetragonal distortion, eg splitting."""
    checks: list[CheckResult] = []

    # Synthetic projected DOS: t2g at -1 eV, eg split (dz2 at +1.5, dx2 at +2.5)
    e = np.linspace(-6, 4, 500)
    t2g_dos = (_make_gaussian_dos(-1.0, 0.4, e) +
               _make_gaussian_dos(-0.8, 0.4, e) +
               _make_gaussian_dos(-1.2, 0.4, e))
    dz2_dos = _make_gaussian_dos(1.5, 0.3, e)
    dx2_dos = _make_gaussian_dos(2.5, 0.3, e)

    projected = {
        "Cu_dxy": _make_gaussian_dos(-1.0, 0.4, e),
        "Cu_dxz": _make_gaussian_dos(-0.8, 0.4, e),
        "Cu_dyz": _make_gaussian_dos(-1.2, 0.4, e),
        "Cu_dz2": dz2_dos,
        "Cu_dx2": dx2_dos,
    }
    dos = DosData(
        energies=e, total_dos=t2g_dos + dz2_dos + dx2_dos,
        fermi_energy=0.0, projected_dos=projected,
    )

    # Crystal field with d9, tetragonal c/a=1.06
    cf = compute_crystal_field(dos, n_d_electrons=9, c_over_a=1.06)

    checks.append(CheckResult(
        "B1: Δ_oct (splitting) > 0",
        "> 0",
        f"{cf.splitting:.3f} eV" if cf.splitting else "None",
        cf.splitting is not None and cf.splitting > 0,
    ))
    checks.append(CheckResult(
        "B2: JT active (d9)",
        "True",
        str(cf.jt_active),
        cf.jt_active is True,
    ))
    checks.append(CheckResult(
        "B3: JT strength (d9 → strong)",
        "strong",
        str(cf.jt_strength),
        cf.jt_strength == "strong",
    ))
    checks.append(CheckResult(
        "B4: CFSE(d9) = -0.6Δ",
        "-0.6",
        f"{cf.cfse_over_delta:.1f}" if cf.cfse_over_delta is not None else "None",
        cf.cfse_over_delta is not None and abs(cf.cfse_over_delta - (-0.6)) < 1e-10,
    ))
    checks.append(CheckResult(
        "B5: eg splitting (dz²/dx²-y² separation) > 0",
        "> 0",
        f"{cf.eg_splitting:.3f} eV" if cf.eg_splitting else "None",
        cf.eg_splitting is not None and cf.eg_splitting > 0,
        note="Measures magnitude of Jahn-Teller distortion in DOS",
    ))
    checks.append(CheckResult(
        "B6: Tetragonality |c/a - 1|",
        "0.060",
        f"{cf.tetragonality:.3f}" if cf.tetragonality is not None else "None",
        cf.tetragonality is not None and abs(cf.tetragonality - 0.06) < 0.001,
    ))

    # Generate figure: projected DOS showing t2g/eg splitting
    _plot_dos_validation(e, projected, dos.fermi_energy, cf,
                         title="Case B: Cu$^{2+}$ (d$^9$) — Jahn-Teller Active",
                         filename="case_b_cu_d9_dos.png")

    return checks


# ── Case C: Cr²⁺ d4 HS — Strong Jahn-Teller ────────────────────────

def case_c_cr_d4(tmpdir: Path) -> list[CheckResult]:
    """Cr²⁺ d4 high-spin: strong JT, CFSE = -0.6Δ."""
    checks: list[CheckResult] = []

    e = np.linspace(-6, 4, 500)
    projected = {
        "Cr_dxy": _make_gaussian_dos(-1.5, 0.5, e),
        "Cr_dxz": _make_gaussian_dos(-1.3, 0.5, e),
        "Cr_dyz": _make_gaussian_dos(-1.7, 0.5, e),
        "Cr_dz2": _make_gaussian_dos(1.0, 0.4, e),
        "Cr_dx2": _make_gaussian_dos(1.2, 0.4, e),
    }
    total = sum(projected.values())
    dos = DosData(energies=e, total_dos=total, fermi_energy=0.0,
                  projected_dos=projected)

    cf = compute_crystal_field(dos, n_d_electrons=4, low_spin=False, c_over_a=0.98)

    checks.append(CheckResult(
        "C1: JT active (d4 HS)",
        "True",
        str(cf.jt_active),
        cf.jt_active is True,
    ))
    checks.append(CheckResult(
        "C2: JT strength (d4 HS → strong)",
        "strong",
        str(cf.jt_strength),
        cf.jt_strength == "strong",
    ))
    checks.append(CheckResult(
        "C3: CFSE(d4 HS) = -0.6Δ",
        "-0.6",
        f"{cf.cfse_over_delta:.1f}" if cf.cfse_over_delta is not None else "None",
        cf.cfse_over_delta is not None and abs(cf.cfse_over_delta - (-0.6)) < 1e-10,
    ))
    checks.append(CheckResult(
        "C4: Tetragonality |c/a - 1| = 0.02",
        "0.020",
        f"{cf.tetragonality:.3f}" if cf.tetragonality is not None else "None",
        cf.tetragonality is not None and abs(cf.tetragonality - 0.02) < 0.001,
    ))

    _plot_dos_validation(e, projected, dos.fermi_energy, cf,
                         title="Case C: Cr$^{2+}$ (d$^4$ HS) — Jahn-Teller Active",
                         filename="case_c_cr_d4_dos.png")

    return checks


# ── Case D: JTSE from paired calculations ───────────────────────────

def case_d_jtse_paired() -> list[CheckResult]:
    """Paired undistorted/distorted calculation → JTSE."""
    checks: list[CheckResult] = []

    # Undistorted: cubic, higher energy
    und = CalculationResult(
        label="CuO_cubic",
        energy=EnergyResult(total_energy=-40.000, energy_per_atom=-5.000),
        structure=StructureResult(
            lattice_constant=3.80, volume=54.872, volume_per_atom=6.859, c_over_a=1.0),
    )
    # Distorted: tetragonal elongation, lower energy (JT stabilised)
    dis = CalculationResult(
        label="CuO_tetragonal",
        energy=EnergyResult(total_energy=-40.350, energy_per_atom=-5.04375),
        structure=StructureResult(
            lattice_constant=3.80, volume=54.872, volume_per_atom=6.859, c_over_a=1.06),
    )

    jt = compute_jahn_teller_energy(und, dis)

    expected_jtse = 0.350  # eV
    expected_jtse_per_atom = 0.350 / 8  # 8 atoms
    expected_delta_ca = 0.06

    checks.append(CheckResult(
        "D1: JTSE > 0 (distortion is favorable)",
        "> 0",
        f"{jt.jtse:.4f} eV",
        jt.jtse > 0,
    ))
    checks.append(CheckResult(
        "D2: JTSE value",
        f"{expected_jtse:.3f} eV",
        f"{jt.jtse:.3f} eV",
        abs(jt.jtse - expected_jtse) < 0.001,
    ))
    checks.append(CheckResult(
        "D3: JTSE per atom",
        f"{expected_jtse_per_atom:.4f} eV/atom",
        f"{jt.jtse_per_atom:.4f} eV/atom",
        abs(jt.jtse_per_atom - expected_jtse_per_atom) < 0.001,
    ))
    checks.append(CheckResult(
        "D4: Δ(c/a)",
        f"{expected_delta_ca:.3f}",
        f"{jt.delta_c_over_a:.3f}",
        abs(jt.delta_c_over_a - expected_delta_ca) < 0.001,
    ))

    return checks


# ── Case E: Export pipeline ──────────────────────────────────────────

def case_e_export(tmpdir: Path) -> list[CheckResult]:
    """Verify CSV/JSON/summary export of analysis results."""
    checks: list[CheckResult] = []

    results = [
        CalculationResult(
            label="Ni_fcc", converged=True,
            energy=EnergyResult(-21.568, -5.392),
            structure=StructureResult(3.524, 43.77, 10.94, 1.0),
        ),
        CalculationResult(
            label="Cu_tet", converged=True,
            energy=EnergyResult(-15.200, -3.800),
            structure=StructureResult(3.60, 46.656, 11.664, 1.06),
        ),
    ]
    out_dir = tmpdir / "export_test"
    out_dir.mkdir()

    csv_path = export_csv(results, str(out_dir / "results.csv"))
    checks.append(CheckResult(
        "E1: CSV export creates file",
        "True",
        str(Path(csv_path).exists()),
        Path(csv_path).exists(),
    ))

    json_path = export_json(results, str(out_dir / "results.json"))
    checks.append(CheckResult(
        "E2: JSON export creates file",
        "True",
        str(Path(json_path).exists()),
        Path(json_path).exists(),
    ))

    summary = export_summary(results, str(out_dir / "summary.txt"))
    checks.append(CheckResult(
        "E3: Summary export non-empty",
        "True",
        str(len(summary) > 0),
        len(summary) > 0,
    ))

    # Read CSV and verify content
    import csv
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    checks.append(CheckResult(
        "E4: CSV has correct number of rows",
        "2",
        str(len(rows)),
        len(rows) == 2,
    ))

    return checks


# ── Plotting helper ──────────────────────────────────────────────────

def _plot_dos_validation(energies: np.ndarray, projected: dict,
                         fermi: float, cf: CrystalFieldResult,
                         title: str, filename: str) -> Optional[Path]:
    """Plot projected DOS with t2g/eg annotations."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping figure %s", filename)
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    # Sum t2g and eg contributions
    t2g_total = np.zeros_like(energies)
    eg_total = np.zeros_like(energies)
    for key, arr in projected.items():
        orbital = key.split("_")[-1]
        if orbital in ("dxy", "dxz", "dyz"):
            t2g_total += arr
        elif orbital in ("dz2", "dx2"):
            eg_total += arr

    e_shifted = energies - fermi
    ax.fill_between(e_shifted, t2g_total, alpha=0.3, color="tab:blue", label="t$_{2g}$")
    ax.fill_between(e_shifted, eg_total, alpha=0.3, color="tab:red", label="e$_g$")
    ax.plot(e_shifted, t2g_total, color="tab:blue", linewidth=1.5)
    ax.plot(e_shifted, eg_total, color="tab:red", linewidth=1.5)

    # Plot individual eg orbitals if available
    for key, arr in projected.items():
        orbital = key.split("_")[-1]
        if orbital == "dz2":
            ax.plot(e_shifted, arr, "--", color="darkred", linewidth=1, alpha=0.7,
                    label="d$_{z^2}$")
        elif orbital == "dx2":
            ax.plot(e_shifted, arr, ":", color="darkred", linewidth=1, alpha=0.7,
                    label="d$_{x^2-y^2}$")

    # Annotations
    if cf.t2g_center is not None:
        ax.axvline(cf.t2g_center, color="tab:blue", linestyle="--", alpha=0.5)
        ax.text(cf.t2g_center, ax.get_ylim()[1] * 0.9,
                f"t$_{{2g}}$ centre\n{cf.t2g_center:.2f} eV",
                ha="center", fontsize=9, color="tab:blue")
    if cf.eg_center is not None:
        ax.axvline(cf.eg_center, color="tab:red", linestyle="--", alpha=0.5)
        ax.text(cf.eg_center, ax.get_ylim()[1] * 0.9,
                f"e$_g$ centre\n{cf.eg_center:.2f} eV",
                ha="center", fontsize=9, color="tab:red")
    if cf.splitting is not None:
        y_arr = ax.get_ylim()[1] * 0.75
        ax.annotate("", xy=(cf.eg_center, y_arr), xytext=(cf.t2g_center, y_arr),
                     arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
        mid = (cf.t2g_center + cf.eg_center) / 2 if cf.t2g_center and cf.eg_center else 0
        ax.text(mid, y_arr * 1.05,
                f"Δ$_{{oct}}$ = {cf.splitting:.2f} eV",
                ha="center", fontsize=11, fontweight="bold")

    ax.axvline(0, color="gray", linestyle="-", alpha=0.3, label="E$_F$")
    ax.set_xlabel("E − E$_F$ (eV)", fontsize=12)
    ax.set_ylabel("DOS (states/eV)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10, loc="upper right")
    ax.set_xlim(-5, 4)

    # Add JT info box
    info_lines = []
    if cf.cfse is not None:
        info_lines.append(f"CFSE = {cf.cfse:.3f} eV ({cf.cfse_over_delta:.1f}Δ)")
    if cf.jt_strength is not None:
        info_lines.append(f"JT: {cf.jt_strength}")
    if cf.eg_splitting is not None:
        info_lines.append(f"eg split = {cf.eg_splitting:.3f} eV")
    if cf.tetragonality is not None:
        info_lines.append(f"|c/a−1| = {cf.tetragonality:.3f}")
    if info_lines:
        ax.text(0.02, 0.98, "\n".join(info_lines),
                transform=ax.transAxes, fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))

    fig.tight_layout()
    out_path = FIG_DIR / filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Figure saved: {out_path}")
    return out_path


# ── Report formatting ────────────────────────────────────────────────

def format_console_all(all_checks: dict[str, list[CheckResult]]) -> str:
    lines = []
    total_pass = 0
    total_count = 0
    for case_name, checks in all_checks.items():
        lines.append(f"\n{'='*60}")
        lines.append(f"  {case_name}")
        lines.append(f"{'='*60}")
        for c in checks:
            mark = "PASS" if c.passed else "FAIL"
            lines.append(f"  [{mark}] {c.name}: expected={c.expected}, actual={c.actual}")
            if c.note:
                lines.append(f"         Note: {c.note}")
            total_count += 1
            if c.passed:
                total_pass += 1
    lines.append(f"\n{'='*60}")
    lines.append(f"  TOTAL: {total_pass}/{total_count} passed")
    lines.append(f"{'='*60}")
    return "\n".join(lines)


def format_markdown_all(all_checks: dict[str, list[CheckResult]]) -> str:
    lines = []
    lines.append("# Validation 2: Synthetic VASP Data — Pipeline Verification")
    lines.append("")
    lines.append("End-to-end verification of t2vasp against synthetic VASP outputs")
    lines.append("with known ground-truth values.")
    lines.append("")

    total_pass = 0
    total_count = 0

    for case_name, checks in all_checks.items():
        lines.append(f"## {case_name}")
        lines.append("")
        lines.append("| Check | Expected | Actual | Result |")
        lines.append("|-------|----------|--------|--------|")
        for c in checks:
            mark = "PASS" if c.passed else "**FAIL**"
            note = f" ({c.note})" if c.note else ""
            lines.append(f"| {c.name}{note} | {c.expected} | {c.actual} | {mark} |")
            total_count += 1
            if c.passed:
                total_pass += 1
        lines.append("")

    # Embed figures
    fig_files = sorted(FIG_DIR.glob("*.png"))
    if fig_files:
        lines.append("## Figures")
        lines.append("")
        for fig in fig_files:
            rel = fig.relative_to(SCRIPT_DIR)
            lines.append(f"### {fig.stem.replace('_', ' ').title()}")
            lines.append(f"![{fig.stem}]({rel})")
            lines.append("")

    lines.append(f"## Summary")
    lines.append("")
    lines.append(f"**{total_pass}/{total_count}** checks passed.")
    if total_pass == total_count:
        lines.append("")
        lines.append("All validation checks match expected values.")
    else:
        lines.append("")
        lines.append("Some checks failed — see details above.")
    lines.append("")
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Validate t2vasp pipeline")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Write Markdown report to file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    tmpdir = Path(tempfile.mkdtemp(prefix="t2vasp_val_"))

    all_checks: dict[str, list[CheckResult]] = {}

    print("Running validation cases...")

    print("\nCase A: Cubic Ni FCC (d8) — file parsing & energy analysis")
    all_checks["Case A: Cubic Ni FCC (d$^8$) — Parsing & Energy"] = case_a_cubic_ni(tmpdir)

    print("\nCase B: Cu²⁺ (d9) — Strong Jahn-Teller")
    all_checks["Case B: Cu$^{2+}$ (d$^9$) — Strong Jahn-Teller"] = case_b_cu_d9(tmpdir)

    print("\nCase C: Cr²⁺ (d4 HS) — Strong Jahn-Teller")
    all_checks["Case C: Cr$^{2+}$ (d$^4$ HS) — Strong Jahn-Teller"] = case_c_cr_d4(tmpdir)

    print("\nCase D: JTSE from paired calculations")
    all_checks["Case D: Jahn-Teller Stabilisation Energy (Paired)"] = case_d_jtse_paired()

    print("\nCase E: Export pipeline (CSV/JSON/Summary)")
    all_checks["Case E: Export Pipeline"] = case_e_export(tmpdir)

    # Print results
    print(format_console_all(all_checks))

    if args.output:
        md = format_markdown_all(all_checks)
        Path(args.output).write_text(md, encoding="utf-8")
        print(f"\nMarkdown report written to: {args.output}")

    total_fail = sum(1 for checks in all_checks.values()
                     for c in checks if not c.passed)
    return 1 if total_fail else 0


if __name__ == "__main__":
    sys.exit(main())
