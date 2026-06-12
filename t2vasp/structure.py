"""
Structure manipulation module — ASE-based helpers for POSCAR generation
and crystal-structure modification.

Provides the bridge between t2vasp analysis results and the next VASP
calculation cycle (Phase 5 of the development roadmap).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from ase import Atoms
    from ase.io import read as ase_read, write as ase_write
    from ase.build import bulk
    HAS_ASE = True
except ImportError:
    HAS_ASE = False


def _require_ase() -> None:
    if not HAS_ASE:
        raise ImportError("ASE is required. Install with: pip install ase")


# ── Create bulk structures ───────────────────────────────────────────
def create_bulk(
    symbol: str,
    crystal_type: str = "fcc",
    a: float | None = None,
) -> "Atoms":
    """Create a bulk crystal using ASE.

    Parameters
    ----------
    symbol : str
        Chemical symbol, e.g. ``"Ni"``.
    crystal_type : str
        One of ``fcc``, ``bcc``, ``hcp``, ``sc``.
    a : float, optional
        Lattice constant in Angstrom (ASE default if omitted).

    Returns
    -------
    ase.Atoms
    """
    _require_ase()
    kwargs: Dict = {"name": symbol, "crystalstructure": crystal_type}
    if a is not None:
        kwargs["a"] = a
    return bulk(**kwargs)


# ── Build L1₂ / B2 ordered supercells ───────────────────────────────
def create_l12(
    element_face: str,
    element_corner: str,
    a: float = 3.60,
) -> "Atoms":
    """Build an L1₂ (Cu₃Au-type) unit cell.

    Face centres → *element_face* (×3), corner → *element_corner* (×1).
    """
    _require_ase()
    cell = np.diag([a, a, a])
    positions = np.array([
        [0.0, 0.0, 0.0],           # corner
        [0.5 * a, 0.5 * a, 0.0],   # face
        [0.5 * a, 0.0, 0.5 * a],   # face
        [0.0, 0.5 * a, 0.5 * a],   # face
    ])
    symbols = [element_corner, element_face, element_face, element_face]
    return Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)


def create_b2(
    element_a: str,
    element_b: str,
    a: float = 3.10,
) -> "Atoms":
    """Build a B2 (CsCl-type) unit cell."""
    _require_ase()
    cell = np.diag([a, a, a])
    positions = np.array([
        [0.0, 0.0, 0.0],
        [0.5 * a, 0.5 * a, 0.5 * a],
    ])
    return Atoms(symbols=[element_a, element_b], positions=positions,
                 cell=cell, pbc=True)


# ── Apply strain ─────────────────────────────────────────────────────
def apply_strain(
    atoms: "Atoms",
    strain: float,
    axis: int = 0,
) -> "Atoms":
    """Return a strained copy of *atoms* (isotropic or uniaxial).

    Parameters
    ----------
    atoms : Atoms
    strain : float
        Fractional strain (0.01 = 1 %).
    axis : int
        0 = isotropic, 1/2/3 = uniaxial along a/b/c.
    """
    _require_ase()
    strained = atoms.copy()
    cell = strained.cell.array.copy()
    if axis == 0:
        cell *= 1 + strain
    else:
        cell[axis - 1] *= 1 + strain
    strained.set_cell(cell, scale_atoms=True)
    return strained


# ── Perturb positions ────────────────────────────────────────────────
def perturb_positions(
    atoms: "Atoms",
    amplitude: float = 0.05,
    seed: int | None = None,
) -> "Atoms":
    """Add random Gaussian noise to atomic positions (for SQS-like generation)."""
    _require_ase()
    rng = np.random.default_rng(seed)
    perturbed = atoms.copy()
    noise = rng.normal(scale=amplitude, size=perturbed.positions.shape)
    perturbed.positions += noise
    return perturbed


# ── Write POSCAR ─────────────────────────────────────────────────────
def write_poscar(
    atoms: "Atoms",
    path: str | Path,
    direct: bool = True,
    label: str = "t2vasp generated",
) -> Path:
    """Write an ASE ``Atoms`` object to VASP POSCAR format.

    Parameters
    ----------
    atoms : Atoms
    path : str or Path
    direct : bool
        Use fractional (Direct) coordinates.
    label : str
        Comment line.

    Returns
    -------
    Path
        Resolved output path.
    """
    _require_ase()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        ase_write(str(path), atoms, format="vasp", direct=direct, label=label)
    except TypeError:
        ase_write(str(path), atoms, format="vasp", direct=direct)
    logger.info("POSCAR written: %s", path)
    return path.resolve()


# ── Generate next-step POSCARs from ΔE analysis ─────────────────────
def generate_candidates(
    base_atoms: "Atoms",
    strain_values: List[float] | None = None,
    output_dir: str | Path = "candidates",
) -> List[Path]:
    """Generate a set of strained structures as POSCAR candidates.

    This is the first step toward USPEX-like functionality: from a base
    structure, create systematic variations for the next VASP cycle.

    Parameters
    ----------
    base_atoms : Atoms
    strain_values : list[float]
        Fractional strains to apply (defaults ±1 %, ±2 %).
    output_dir : str or Path

    Returns
    -------
    list[Path]
        Paths to the generated POSCAR files.
    """
    _require_ase()
    if strain_values is None:
        strain_values = [-0.02, -0.01, 0.01, 0.02]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []

    for eps in strain_values:
        tag = f"strain_{eps:+.3f}".replace(".", "p").replace("+", "pos").replace("-", "neg")
        strained = apply_strain(base_atoms, eps, axis=0)
        p = write_poscar(strained, out / tag / "POSCAR",
                         label=f"t2vasp strain={eps:+.3f}")
        paths.append(p)

    logger.info("Generated %d candidate POSCARs in %s", len(paths), out)
    return paths
