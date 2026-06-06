"""
Core calculation module for t2vasp.

All analysis routines are pure functions that accept physical parameters
and return derived quantities.  Constants and coefficients are loaded
from the YAML configuration — nothing is hard-coded.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .parser import DosData, OutcarData, StructureData

logger = logging.getLogger(__name__)


# ── Result containers ────────────────────────────────────────────────
@dataclass
class EnergyResult:
    """Computed energy-derived quantities for one calculation."""
    total_energy: float              # eV
    energy_per_atom: float           # eV/atom
    formation_energy: Optional[float] = None   # eV/atom (requires references)
    cohesive_energy: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class StructureResult:
    """Derived structural metrics."""
    lattice_constant: float          # Angstrom (cubic approx)
    volume: float                    # Angstrom^3
    volume_per_atom: float
    c_over_a: float                  # c/a ratio (1.0 for cubic)
    max_force: Optional[float] = None
    pressure: Optional[float] = None  # kBar (trace of stress / 3)

    def as_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class CrystalFieldResult:
    """t2g / eg splitting analysis from projected DOS."""
    t2g_center: Optional[float] = None   # eV (centre of t2g band)
    eg_center: Optional[float] = None    # eV
    splitting: Optional[float] = None    # eg_center - t2g_center
    t2g_occupation: Optional[float] = None
    eg_occupation: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class CalculationResult:
    """Aggregated analysis results for a single VASP calculation."""
    label: str
    energy: Optional[EnergyResult] = None
    structure: Optional[StructureResult] = None
    crystal_field: Optional[CrystalFieldResult] = None
    converged: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"label": self.label, "converged": self.converged}
        if self.energy:
            d["energy"] = self.energy.as_dict()
        if self.structure:
            d["structure"] = self.structure.as_dict()
        if self.crystal_field:
            d["crystal_field"] = self.crystal_field.as_dict()
        d.update(self.metadata)
        return d


# ── Energy analysis ──────────────────────────────────────────────────
def compute_energy(
    outcar: OutcarData,
    reference_energies: Dict[str, float] | None = None,
    species: List[str] | None = None,
) -> EnergyResult:
    """Derive energy metrics from parsed OUTCAR data.

    Parameters
    ----------
    outcar : OutcarData
        Parsed OUTCAR quantities.
    reference_energies : dict, optional
        ``{element: eV_per_atom}`` for formation-energy calculation.
    species : list[str], optional
        Per-atom species list (needed for formation energy).

    Returns
    -------
    EnergyResult
    """
    e_total = outcar.total_energy
    if e_total is None:
        raise ValueError("Total energy not found in OUTCAR data")

    natoms = outcar.num_atoms or 1
    e_per_atom = e_total / natoms

    formation = None
    if reference_energies and species:
        ref_sum = sum(reference_energies.get(s, 0.0) for s in species)
        formation = (e_total - ref_sum) / natoms

    return EnergyResult(
        total_energy=e_total,
        energy_per_atom=e_per_atom,
        formation_energy=formation,
    )


# ── Structure analysis ───────────────────────────────────────────────
def compute_structure_metrics(
    struct: StructureData,
    outcar: OutcarData | None = None,
) -> StructureResult:
    """Derive structural metrics from parsed structure data.

    Parameters
    ----------
    struct : StructureData
    outcar : OutcarData, optional
        Supplies forces and stress if available.
    """
    a = np.linalg.norm(struct.lattice[0])
    c = np.linalg.norm(struct.lattice[2])
    vol = struct.volume
    natoms = max(len(struct.species), 1)

    max_f: Optional[float] = None
    pressure: Optional[float] = None
    if outcar is not None:
        if outcar.forces is not None:
            max_f = float(np.max(np.linalg.norm(outcar.forces, axis=1)))
        if outcar.stress_tensor is not None:
            # Trace of Voigt → (σ_xx + σ_yy + σ_zz) / 3
            pressure = float(np.mean(outcar.stress_tensor[:3]))

    return StructureResult(
        lattice_constant=float(a),
        volume=vol,
        volume_per_atom=vol / natoms,
        c_over_a=c / a if a > 0 else 1.0,
        max_force=max_f,
        pressure=pressure,
    )


# ── Crystal-field (t2g / eg) analysis ────────────────────────────────
_T2G_ORBITALS = {"dxy", "dxz", "dyz"}
_EG_ORBITALS = {"dz2", "dx2"}  # dx2-y2 often labelled dx2


def compute_crystal_field(
    dos: DosData,
    cfg: Dict[str, Any] | None = None,
) -> CrystalFieldResult:
    """Estimate t2g/eg splitting from (projected) DOS.

    If projected DOS is unavailable, returns an empty result.
    """
    result = CrystalFieldResult()
    if dos.projected_dos is None:
        logger.debug("No projected DOS — crystal-field analysis skipped")
        return result

    window = (-6.0, 4.0)
    if cfg and "crystal_field" in cfg:
        w = cfg["crystal_field"].get("window", window)
        window = (w[0], w[1])

    e_shifted = dos.energies - dos.fermi_energy
    mask = (e_shifted >= window[0]) & (e_shifted <= window[1])
    e_win = e_shifted[mask]

    t2g_dos = np.zeros_like(e_win)
    eg_dos = np.zeros_like(e_win)

    for key, arr in dos.projected_dos.items():
        orbital = key.split("_")[-1] if "_" in key else key
        projected = arr[mask] if len(arr) == len(dos.energies) else np.zeros_like(e_win)
        if orbital in _T2G_ORBITALS:
            t2g_dos += projected
        elif orbital in _EG_ORBITALS:
            eg_dos += projected

    de = np.gradient(e_win) if len(e_win) > 1 else np.ones_like(e_win)

    _integrate = getattr(np, "trapezoid", None) or np.trapz
    t2g_total = float(_integrate(t2g_dos, e_win)) if len(e_win) > 1 else 0.0
    eg_total = float(_integrate(eg_dos, e_win)) if len(e_win) > 1 else 0.0

    if t2g_total > 1e-8:
        result.t2g_center = float(_integrate(t2g_dos * e_win, e_win) / t2g_total)
        result.t2g_occupation = t2g_total
    if eg_total > 1e-8:
        result.eg_center = float(_integrate(eg_dos * e_win, e_win) / eg_total)
        result.eg_occupation = eg_total
    if result.t2g_center is not None and result.eg_center is not None:
        result.splitting = result.eg_center - result.t2g_center

    return result


# ── Delta-E for structure ranking ────────────────────────────────────
def compute_delta_energy(
    results: List[CalculationResult],
    reference_label: str | None = None,
) -> Dict[str, float]:
    """Compute ΔE (eV/atom) relative to a reference or the global minimum.

    Parameters
    ----------
    results : list[CalculationResult]
    reference_label : str, optional
        Label of the reference calculation.  If ``None``, uses the lowest-energy
        entry as reference.

    Returns
    -------
    dict  ``{label: ΔE}``
    """
    valid = {r.label: r.energy.energy_per_atom
             for r in results if r.energy is not None}
    if not valid:
        return {}

    if reference_label and reference_label in valid:
        ref = valid[reference_label]
    else:
        ref = min(valid.values())

    return {lbl: e - ref for lbl, e in valid.items()}


# ── Convenience: full analysis on a parsed calc-dir dict ─────────────
def analyse(
    parsed: Dict,
    cfg: Dict[str, Any] | None = None,
) -> CalculationResult:
    """Run all applicable analyses on the output of :func:`parser.parse_calc_dir`.

    Parameters
    ----------
    parsed : dict
        Output of ``parse_calc_dir`` (keys: path, outcar, structure, vasprun, dos).
    cfg : dict, optional
        Calculator section of the YAML config.
    """
    cfg = cfg or {}
    label = parsed.get("path", "unknown")
    result = CalculationResult(label=label)

    outcar: Optional[OutcarData] = parsed.get("outcar")
    struct: Optional[StructureData] = parsed.get("structure")
    dos: Optional[DosData] = parsed.get("dos")

    if outcar is not None:
        result.converged = outcar.converged
        try:
            result.energy = compute_energy(
                outcar,
                reference_energies=cfg.get("reference_energies"),
                species=struct.species if struct else None,
            )
        except ValueError as exc:
            logger.warning("Energy computation failed for %s: %s", label, exc)

    if struct is not None:
        result.structure = compute_structure_metrics(struct, outcar)

    if dos is not None:
        result.crystal_field = compute_crystal_field(dos, cfg)

    logger.info("Analysis complete: %s (converged=%s)", label, result.converged)
    return result
