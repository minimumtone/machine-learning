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
    """t2g / eg splitting analysis from projected DOS.

    Includes Crystal Field Stabilization Energy (CFSE) and
    Jahn-Teller distortion indicators derived from octahedral
    crystal field theory.
    """
    t2g_center: Optional[float] = None   # eV (centre of t2g band)
    eg_center: Optional[float] = None    # eV
    splitting: Optional[float] = None    # Δ_oct = eg_center - t2g_center
    t2g_occupation: Optional[float] = None
    eg_occupation: Optional[float] = None
    # CFSE = n(t2g)×(-0.4Δ) + n(eg)×(+0.6Δ) + n_pairs×P
    cfse: Optional[float] = None         # eV (Crystal Field Stabilization Energy)
    cfse_over_delta: Optional[float] = None   # CFSE expressed in units of Δ_oct
    # Jahn-Teller analysis
    jt_active: Optional[bool] = None     # whether the config is JT-active
    jt_strength: Optional[str] = None    # "strong" | "weak" | None
    tetragonality: Optional[float] = None  # |c/a - 1| distortion parameter
    eg_splitting: Optional[float] = None   # eV (dz2 vs dx2-y2 splitting)

    def as_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class JahnTellerResult:
    """Jahn-Teller stabilization energy from paired calculations.

    E_JT = E(undistorted) - E(distorted), the energy gained by
    symmetry-lowering geometric distortion.  Positive E_JT means
    distortion is energetically favorable.
    """
    e_undistorted: float        # eV  (total energy of high-symmetry structure)
    e_distorted: float          # eV  (total energy of distorted structure)
    jtse: float                 # eV  (Jahn-Teller stabilisation energy)
    jtse_per_atom: float        # eV/atom
    delta_c_over_a: float       # |c/a(distorted) - c/a(undistorted)|

    def as_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class CalculationResult:
    """Aggregated analysis results for a single VASP calculation."""
    label: str
    energy: Optional[EnergyResult] = None
    structure: Optional[StructureResult] = None
    crystal_field: Optional[CrystalFieldResult] = None
    jahn_teller: Optional[JahnTellerResult] = None
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
        if self.jahn_teller:
            d["jahn_teller"] = self.jahn_teller.as_dict()
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

# High-spin CFSE coefficients (n_t2g, n_eg, extra_pairs) for d^0..d^10
# CFSE = n_t2g × (-0.4Δ) + n_eg × (+0.6Δ) + extra_pairs × P
# Jahn-Teller: "s" = strong (eg uneven), "w" = weak (t2g uneven), None = inactive
_CFSE_TABLE_HS: List[Tuple[int, int, int, Optional[str]]] = [
    # d0: (t2g, eg, pairs, JT)
    (0, 0, 0, None),        # d0
    (1, 0, 0, "w"),         # d1  — t2g^1
    (2, 0, 0, "w"),         # d2  — t2g^2
    (3, 0, 0, None),        # d3  — t2g^3 (half-filled, stable)
    (3, 1, 0, "s"),         # d4  HS — t2g^3 eg^1 (strong JT)
    (3, 2, 0, None),        # d5  HS — t2g^3 eg^2 (half-filled)
    (4, 2, 1, "w"),         # d6  HS — t2g^4 eg^2
    (5, 2, 2, "w"),         # d7  HS — t2g^5 eg^2
    (6, 2, 3, None),        # d8  — t2g^6 eg^2
    (6, 3, 4, "s"),         # d9  — t2g^6 eg^3 (strong JT, e.g. Cu²⁺)
    (6, 4, 5, None),        # d10 — fully filled
]

# Low-spin table (differs for d4–d7)
_CFSE_TABLE_LS: List[Tuple[int, int, int, Optional[str]]] = [
    (0, 0, 0, None),        # d0
    (1, 0, 0, "w"),         # d1
    (2, 0, 0, "w"),         # d2
    (3, 0, 0, None),        # d3
    (4, 0, 1, "w"),         # d4  LS — t2g^4 eg^0
    (5, 0, 2, "w"),         # d5  LS — t2g^5 eg^0
    (6, 0, 3, None),        # d6  LS — t2g^6 eg^0
    (6, 1, 3, "s"),         # d7  LS — t2g^6 eg^1 (strong JT)
    (6, 2, 3, None),        # d8
    (6, 3, 4, "s"),         # d9
    (6, 4, 5, None),        # d10
]


def compute_cfse(
    n_d_electrons: int,
    delta_oct: float,
    low_spin: bool = False,
    pairing_energy: float = 0.0,
) -> Tuple[float, float, Optional[str]]:
    """Compute Crystal Field Stabilization Energy for octahedral coordination.

    CFSE = n(t2g)×(-0.4Δ) + n(eg)×(+0.6Δ) + extra_pairs × P

    Parameters
    ----------
    n_d_electrons : int
        Number of d electrons (0–10).
    delta_oct : float
        Crystal field splitting energy Δ_oct (eV).
    low_spin : bool
        Whether the complex adopts a low-spin configuration.
    pairing_energy : float
        Electron pairing energy P (eV).  Often small relative to Δ;
        set to 0 to obtain CFSE without pairing contribution.

    Returns
    -------
    (cfse_eV, cfse_over_delta, jt_strength)
        cfse_eV: CFSE in eV.
        cfse_over_delta: CFSE expressed as a multiple of Δ_oct.
        jt_strength: ``"strong"`` / ``"weak"`` / ``None``.
    """
    if n_d_electrons < 0 or n_d_electrons > 10:
        raise ValueError(f"d-electron count must be 0–10, got {n_d_electrons}")

    table = _CFSE_TABLE_LS if low_spin else _CFSE_TABLE_HS
    n_t2g, n_eg, n_pairs, jt_code = table[n_d_electrons]
    _JT_MAP = {"s": "strong", "w": "weak"}
    jt = _JT_MAP.get(jt_code) if jt_code else None

    # CFSE = n_t2g × (-0.4Δ) + n_eg × (0.6Δ) + n_pairs × P
    cfse_delta = n_t2g * (-0.4) + n_eg * 0.6    # in units of Δ
    cfse_ev = cfse_delta * delta_oct + n_pairs * pairing_energy

    logger.debug(
        "CFSE(d%d, %s): t2g=%d eg=%d → %.2fΔ = %.4f eV (JT=%s)",
        n_d_electrons, "LS" if low_spin else "HS",
        n_t2g, n_eg, cfse_delta, cfse_ev, jt,
    )
    return cfse_ev, cfse_delta, jt


def compute_crystal_field(
    dos: DosData,
    cfg: Dict[str, Any] | None = None,
    n_d_electrons: int | None = None,
    low_spin: bool = False,
    c_over_a: float | None = None,
) -> CrystalFieldResult:
    """Estimate t2g/eg splitting, CFSE, and Jahn-Teller indicators.

    The crystal field splitting Δ_oct is obtained from the projected DOS
    as the separation between the t2g and eg band centres.  Given the
    number of d-electrons the CFSE and Jahn-Teller activity are derived
    from octahedral crystal field theory.

    Parameters
    ----------
    dos : DosData
        Parsed DOS with optional projected orbitals.
    cfg : dict, optional
        Calculator config section (``crystal_field`` sub-key).
    n_d_electrons : int, optional
        Number of d electrons for CFSE / JT analysis.
    low_spin : bool
        Whether to assume low-spin occupation.
    c_over_a : float, optional
        c/a ratio from structure; used to quantify tetragonal distortion.
    """
    result = CrystalFieldResult()
    if dos.projected_dos is None:
        logger.debug("No projected DOS — crystal-field analysis skipped")
        return result

    window = (-6.0, 4.0)
    pairing_energy = 0.0
    if cfg and "crystal_field" in cfg:
        cf_cfg = cfg["crystal_field"]
        w = cf_cfg.get("window", window)
        window = (w[0], w[1])
        pairing_energy = cf_cfg.get("pairing_energy", 0.0)

    e_shifted = dos.energies - dos.fermi_energy
    mask = (e_shifted >= window[0]) & (e_shifted <= window[1])
    e_win = e_shifted[mask]

    t2g_dos = np.zeros_like(e_win)
    eg_dos = np.zeros_like(e_win)
    # Track individual eg orbitals for JT eg-splitting
    dz2_dos = np.zeros_like(e_win)
    dx2_dos = np.zeros_like(e_win)

    for key, arr in dos.projected_dos.items():
        orbital = key.split("_")[-1] if "_" in key else key
        projected = arr[mask] if len(arr) == len(dos.energies) else np.zeros_like(e_win)
        if orbital in _T2G_ORBITALS:
            t2g_dos += projected
        elif orbital in _EG_ORBITALS:
            eg_dos += projected
        # Track dz2 and dx2 separately for eg splitting
        if orbital == "dz2":
            dz2_dos += projected
        elif orbital == "dx2":
            dx2_dos += projected

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

    # ── eg splitting (Jahn-Teller distortion lifts eg degeneracy) ─────
    dz2_total = float(_integrate(dz2_dos, e_win)) if len(e_win) > 1 else 0.0
    dx2_total = float(_integrate(dx2_dos, e_win)) if len(e_win) > 1 else 0.0
    if dz2_total > 1e-8 and dx2_total > 1e-8:
        dz2_center = float(_integrate(dz2_dos * e_win, e_win) / dz2_total)
        dx2_center = float(_integrate(dx2_dos * e_win, e_win) / dx2_total)
        result.eg_splitting = abs(dx2_center - dz2_center)

    # ── CFSE & Jahn-Teller indicators ────────────────────────────────
    if n_d_electrons is not None and result.splitting is not None:
        cfse_ev, cfse_delta, jt = compute_cfse(
            n_d_electrons, result.splitting, low_spin, pairing_energy,
        )
        result.cfse = cfse_ev
        result.cfse_over_delta = cfse_delta
        result.jt_active = jt is not None
        result.jt_strength = jt
        logger.info(
            "Crystal field: Δ=%.3f eV, CFSE=%.3f eV (%.2fΔ), JT=%s",
            result.splitting, cfse_ev, cfse_delta, jt or "inactive",
        )

    # ── Tetragonality from structural c/a ────────────────────────────
    if c_over_a is not None:
        result.tetragonality = abs(c_over_a - 1.0)

    return result


# ── Jahn-Teller stabilisation energy from paired calculations ───────
def compute_jahn_teller_energy(
    undistorted: CalculationResult,
    distorted: CalculationResult,
) -> JahnTellerResult:
    """Compute Jahn-Teller stabilisation energy from two calculations.

    E_JT = E(undistorted) - E(distorted).  A positive value indicates
    the distortion is energetically favorable (as expected for JT-active
    configurations such as d⁹ Cu²⁺ or HS d⁴ Cr²⁺).

    Parameters
    ----------
    undistorted, distorted : CalculationResult
        Results from high-symmetry and distorted structure calculations.
    """
    if undistorted.energy is None or distorted.energy is None:
        raise ValueError("Both calculations must have energy data")

    e_und = undistorted.energy.total_energy
    e_dis = distorted.energy.total_energy
    jtse = e_und - e_dis

    n_und = undistorted.energy.total_energy / undistorted.energy.energy_per_atom
    n_dis = distorted.energy.total_energy / distorted.energy.energy_per_atom
    natoms = max(round(n_dis), 1)
    jtse_per_atom = jtse / natoms

    ca_und = undistorted.structure.c_over_a if undistorted.structure else 1.0
    ca_dis = distorted.structure.c_over_a if distorted.structure else 1.0
    delta_ca = abs(ca_dis - ca_und)

    logger.info(
        "JTSE = %.4f eV (%.4f eV/atom), Δ(c/a) = %.4f",
        jtse, jtse_per_atom, delta_ca,
    )
    return JahnTellerResult(
        e_undistorted=e_und,
        e_distorted=e_dis,
        jtse=jtse,
        jtse_per_atom=jtse_per_atom,
        delta_c_over_a=delta_ca,
    )


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
        result.crystal_field = compute_crystal_field(
            dos, cfg,
            n_d_electrons=cfg.get("crystal_field", {}).get("n_d_electrons"),
            low_spin=cfg.get("crystal_field", {}).get("low_spin", False),
            c_over_a=result.structure.c_over_a if result.structure else None,
        )

    logger.info("Analysis complete: %s (converged=%s)", label, result.converged)
    return result
