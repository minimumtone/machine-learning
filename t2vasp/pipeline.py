"""
End-to-end pipeline: parse → analyse → export.

Orchestrates a batch of VASP calculations and produces a unified report.
The pipeline also supports optional structure generation for the next
VASP cycle (USPEX-like loop).
"""

import logging
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .calculator import CalculationResult, analyse, compute_delta_energy
from .config import load_config
from .exporter import export_csv, export_json, export_summary
from .parser import parse_calc_dir
from .utils import find_calc_dirs

logger = logging.getLogger(__name__)


# ── Single-directory processing ──────────────────────────────────────
def process_single(
    calc_dir: str | Path,
    cfg: Dict[str, Any] | None = None,
) -> CalculationResult:
    """Parse and analyse one VASP calculation directory.

    Parameters
    ----------
    calc_dir : str or Path
    cfg : dict, optional
        Full merged configuration.

    Returns
    -------
    CalculationResult
    """
    cfg = cfg or {}
    parsed = parse_calc_dir(calc_dir)
    return analyse(parsed, cfg.get("calculator", {}))


# ── Batch processing ────────────────────────────────────────────────
def process_batch(
    base_dir: str | Path,
    cfg: Dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
    export_formats: List[str] | None = None,
) -> List[CalculationResult]:
    """Process all VASP calculations under *base_dir*.

    Parameters
    ----------
    base_dir : str or Path
    cfg : dict, optional
    output_dir : str or Path, optional
        Where to write exported results (defaults to ``base_dir/t2vasp_output``).
    export_formats : list[str]
        Subset of ``["csv", "json", "summary"]`` (default: all three).

    Returns
    -------
    list[CalculationResult]
    """
    cfg = cfg or load_config()
    dirs = find_calc_dirs(base_dir)
    if not dirs:
        logger.warning("No VASP calculation directories found in %s", base_dir)
        return []

    logger.info("Found %d calculation directories in %s", len(dirs), base_dir)

    results: List[CalculationResult] = []
    for d in dirs:
        try:
            r = process_single(d, cfg)
            results.append(r)
        except Exception as exc:
            logger.error("Failed to process %s: %s", d, exc)

    # Export
    out = Path(output_dir or Path(base_dir) / "t2vasp_output")
    formats = export_formats or ["csv", "json", "summary"]

    if "csv" in formats:
        export_csv(results, out / "results.csv",
                   delimiter=cfg.get("exporter", {}).get("csv_delimiter", ","),
                   precision=cfg.get("exporter", {}).get("float_precision", 6))
    if "json" in formats:
        export_json(results, out / "results.json",
                    indent=cfg.get("exporter", {}).get("json_indent", 2))
    if "summary" in formats:
        text = export_summary(results, out / "summary.txt")
        print(text)

    # ΔE ranking
    delta = compute_delta_energy(results)
    if delta:
        logger.info("ΔE ranking (eV/atom):")
        for lbl, de in sorted(delta.items(), key=lambda x: x[1]):
            logger.info("  %s: %.4f", lbl, de)

    return results


# ── VASP execution wrapper ──────────────────────────────────────────
def run_vasp(
    calc_dir: str | Path,
    vasp_command: str = "mpirun -np 4 vasp_std",
    timeout: int | None = None,
) -> bool:
    """Execute VASP in *calc_dir*.

    Parameters
    ----------
    calc_dir : str or Path
    vasp_command : str
    timeout : int, optional
        Seconds before killing the process.

    Returns
    -------
    bool
        True if the process exited successfully.
    """
    calc_dir = Path(calc_dir)
    logger.info("Starting VASP in %s: %s", calc_dir, vasp_command)

    try:
        proc = subprocess.run(
            vasp_command.split(),
            cwd=str(calc_dir),
            timeout=timeout,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            logger.error("VASP failed (rc=%d) in %s:\n%s",
                         proc.returncode, calc_dir, proc.stderr[-500:])
            return False
        logger.info("VASP completed in %s", calc_dir)
        return True
    except subprocess.TimeoutExpired:
        logger.error("VASP timed out after %ds in %s", timeout, calc_dir)
        return False
    except FileNotFoundError:
        logger.error("VASP executable not found: %s", vasp_command)
        return False


# ── Automated loop (VASP → analyse → generate next) ─────────────────
def optimisation_loop(
    initial_dir: str | Path,
    cfg: Dict[str, Any] | None = None,
    max_iterations: int = 5,
    vasp_command: str | None = None,
) -> List[CalculationResult]:
    """Run an iterative optimisation loop.

    1. Analyse current results.
    2. Generate candidate structures (strained variants).
    3. Run VASP on each candidate.
    4. Repeat until *max_iterations* or convergence.

    Parameters
    ----------
    initial_dir : str or Path
    cfg : dict, optional
    max_iterations : int
    vasp_command : str, optional

    Returns
    -------
    list[CalculationResult]
        All results across iterations.
    """
    cfg = cfg or load_config()
    cmd = vasp_command or cfg.get("pipeline", {}).get("vasp_command",
                                                       "mpirun -np 4 vasp_std")
    all_results: List[CalculationResult] = []

    current_dir = Path(initial_dir)
    for iteration in range(max_iterations):
        logger.info("=== Optimisation iteration %d/%d ===", iteration + 1,
                     max_iterations)

        result = process_single(current_dir, cfg)
        all_results.append(result)

        if result.energy is None:
            logger.warning("No energy found — stopping loop")
            break

        # Generate candidates via structure module (if ASE available)
        try:
            from .structure import generate_candidates
            from ase.io import read as ase_read

            struct_file = current_dir / "CONTCAR"
            if not struct_file.is_file():
                struct_file = current_dir / "POSCAR"

            atoms = ase_read(str(struct_file), format="vasp")
            candidate_dir = current_dir.parent / f"iter_{iteration + 1}"
            paths = generate_candidates(atoms, output_dir=candidate_dir)

            # Run VASP on each candidate
            for p in paths:
                calc = p.parent
                success = run_vasp(calc, vasp_command=cmd)
                if success:
                    r = process_single(calc, cfg)
                    all_results.append(r)

            # Select best for next iteration
            delta = compute_delta_energy(all_results)
            if delta:
                best_label = min(delta, key=delta.get)
                current_dir = Path(best_label)
                logger.info("Best candidate: %s (ΔE=%.4f)", best_label,
                            delta[best_label])

        except ImportError:
            logger.warning("ASE not available — skipping candidate generation")
            break

    return all_results
