"""
Feature Engineering Module for HEA Extrapolation Platform
HEA特徴量エンジニアリングモジュール

Provides systematic feature set construction from alloy compositions:
  FS_BASE     - Basic statistical descriptors (radius, entropy, VEC, etc.)
  FS_THERMO   - Thermodynamic proxy features (solid-solution index, phase separation risk)
  FS_SIZE     - Atomic size / elastic mismatch features
  FS_ELECTRON - Electronic structure proxy features (d-electron, DOS proxy)
  FS_ALL      - Union of all feature sets
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Element property database
# ---------------------------------------------------------------------------

class _ElementDB:
    """Internal element property database for HEA feature computation.

    Properties per element (metallic radius in Angstrom):
        symbol, atomic_number, vec, electronegativity (Pauling),
        atomic_radius, melting_point (K), atomic_mass (u),
        d_electrons, bulk_modulus_approx (GPa), atomic_volume (A^3)

    d_elec convention:
        Number of d-electrons in the *ground-state atomic* configuration.
        For transition metals this is the occupancy of the (n-1)d subshell.
        E.g. Ni = [Ar] 3d8 4s2 → d_elec = 8 (not 10).
        Cu = [Ar] 3d10 4s1 → d_elec = 10.
        This convention follows Mizutani (2010) for HEA solid-solution
        strengthening models. If a different convention is needed (e.g.
        d_elec based on metallic bonding state), override _DATA accordingly.
    """

    _DATA: Dict[str, Dict[str, float]] = {
        "Al": {"Z": 13, "vec": 3, "en": 1.61, "r": 1.43, "Tm": 933,
               "mass": 26.98, "d_elec": 0, "B": 76, "Vm": 10.0},
        "Ti": {"Z": 22, "vec": 4, "en": 1.54, "r": 1.47, "Tm": 1941,
               "mass": 47.87, "d_elec": 2, "B": 110, "Vm": 10.6},
        "V":  {"Z": 23, "vec": 5, "en": 1.63, "r": 1.34, "Tm": 2183,
               "mass": 50.94, "d_elec": 3, "B": 160, "Vm": 8.3},
        "Cr": {"Z": 24, "vec": 6, "en": 1.66, "r": 1.28, "Tm": 2180,
               "mass": 52.00, "d_elec": 5, "B": 160, "Vm": 7.2},
        "Mn": {"Z": 25, "vec": 7, "en": 1.55, "r": 1.27, "Tm": 1519,
               "mass": 54.94, "d_elec": 5, "B": 120, "Vm": 7.4},
        "Fe": {"Z": 26, "vec": 8, "en": 1.83, "r": 1.26, "Tm": 1811,
               "mass": 55.85, "d_elec": 6, "B": 170, "Vm": 7.1},
        "Co": {"Z": 27, "vec": 9, "en": 1.88, "r": 1.25, "Tm": 1768,
               "mass": 58.93, "d_elec": 7, "B": 180, "Vm": 6.7},
        "Ni": {"Z": 28, "vec": 10, "en": 1.91, "r": 1.24, "Tm": 1728,
               "mass": 58.69, "d_elec": 8, "B": 180, "Vm": 6.6},
        "Cu": {"Z": 29, "vec": 11, "en": 1.90, "r": 1.28, "Tm": 1358,
               "mass": 63.55, "d_elec": 10, "B": 140, "Vm": 7.1},
        "Zn": {"Z": 30, "vec": 12, "en": 1.65, "r": 1.34, "Tm": 693,
               "mass": 65.38, "d_elec": 10, "B": 70, "Vm": 9.2},
        "Zr": {"Z": 40, "vec": 4, "en": 1.33, "r": 1.60, "Tm": 2128,
               "mass": 91.22, "d_elec": 2, "B": 94, "Vm": 14.0},
        "Nb": {"Z": 41, "vec": 5, "en": 1.60, "r": 1.46, "Tm": 2750,
               "mass": 92.91, "d_elec": 4, "B": 170, "Vm": 10.8},
        "Mo": {"Z": 42, "vec": 6, "en": 2.16, "r": 1.39, "Tm": 2896,
               "mass": 95.95, "d_elec": 5, "B": 230, "Vm": 9.4},
        "Hf": {"Z": 72, "vec": 4, "en": 1.30, "r": 1.59, "Tm": 2506,
               "mass": 178.49, "d_elec": 2, "B": 110, "Vm": 13.4},
        "Ta": {"Z": 73, "vec": 5, "en": 1.50, "r": 1.46, "Tm": 3290,
               "mass": 180.95, "d_elec": 3, "B": 200, "Vm": 10.9},
        "W":  {"Z": 74, "vec": 6, "en": 2.36, "r": 1.39, "Tm": 3695,
               "mass": 183.84, "d_elec": 4, "B": 310, "Vm": 9.5},
        "Re": {"Z": 75, "vec": 7, "en": 1.90, "r": 1.37, "Tm": 3459,
               "mass": 186.21, "d_elec": 5, "B": 370, "Vm": 8.9},
        "Pd": {"Z": 46, "vec": 10, "en": 2.20, "r": 1.37, "Tm": 1828,
               "mass": 106.42, "d_elec": 10, "B": 180, "Vm": 8.6},
        "Ag": {"Z": 47, "vec": 11, "en": 1.93, "r": 1.44, "Tm": 1235,
               "mass": 107.87, "d_elec": 10, "B": 100, "Vm": 10.3},
        "Pt": {"Z": 78, "vec": 10, "en": 2.28, "r": 1.39, "Tm": 2041,
               "mass": 195.08, "d_elec": 9, "B": 230, "Vm": 9.1},
        "Au": {"Z": 79, "vec": 11, "en": 2.54, "r": 1.44, "Tm": 1337,
               "mass": 196.97, "d_elec": 10, "B": 220, "Vm": 10.2},
        "Si": {"Z": 14, "vec": 4, "en": 1.90, "r": 1.18, "Tm": 1687,
               "mass": 28.09, "d_elec": 0, "B": 100, "Vm": 12.1},
        "Mg": {"Z": 12, "vec": 2, "en": 1.31, "r": 1.60, "Tm": 923,
               "mass": 24.31, "d_elec": 0, "B": 45, "Vm": 14.0},
        "Sc": {"Z": 21, "vec": 3, "en": 1.36, "r": 1.64, "Tm": 1814,
               "mass": 44.96, "d_elec": 1, "B": 57, "Vm": 15.0},
        "Y":  {"Z": 39, "vec": 3, "en": 1.22, "r": 1.80, "Tm": 1799,
               "mass": 88.91, "d_elec": 1, "B": 41, "Vm": 19.9},
    }

    # Simplified Miedema binary mixing enthalpies (kJ/mol)
    _DELTA_H_BINARY: Dict[tuple, float] = {
        ("Co", "Cr"): -4, ("Co", "Fe"): -1, ("Co", "Mn"): -5,
        ("Co", "Ni"): 0,  ("Cr", "Fe"): -1, ("Cr", "Mn"): 2,
        ("Cr", "Ni"): -7, ("Fe", "Mn"): 0,  ("Fe", "Ni"): -2,
        ("Mn", "Ni"): -8, ("Ti", "V"): -2,  ("Ti", "Cr"): -7,
        ("Ti", "Fe"): -17, ("Ti", "Ni"): -35, ("Ti", "Co"): -28,
        ("V", "Cr"): -2,  ("V", "Fe"): -7,  ("V", "Ni"): -18,
        ("Cu", "Ni"): 4,  ("Cu", "Co"): 6,  ("Cu", "Fe"): 13,
        ("Cu", "Mn"): 4,  ("Cu", "Cr"): 12,
        ("Al", "Co"): -19, ("Al", "Cr"): -10, ("Al", "Fe"): -11,
        ("Al", "Mn"): -19, ("Al", "Ni"): -22, ("Al", "Ti"): -30,
        ("Al", "Cu"): -1,  ("Al", "Zr"): -44,
        ("Zr", "Ti"): 0,  ("Zr", "Ni"): -49, ("Zr", "Cu"): -23,
        ("Zr", "Co"): -41, ("Zr", "Fe"): -25,
        ("Nb", "Ti"): 2,  ("Nb", "Zr"): 4,  ("Mo", "Ti"): -4,
        ("Ta", "Ti"): 1,  ("W", "Ti"): -6,
        ("Hf", "Ti"): 0,  ("Hf", "Ni"): -42, ("Hf", "Co"): -35,
        ("Nb", "Ni"): -30, ("Mo", "Ni"): -7,  ("Ta", "Ni"): -29,
        ("W", "Ni"): -3,   ("Nb", "Co"): -25, ("Mo", "Co"): -5,
        ("Nb", "Fe"): -16, ("Mo", "Fe"): -2,
    }

    @classmethod
    def get(cls, symbol: str) -> Dict[str, float]:
        """Return property dict; raises KeyError for unknown elements."""
        if symbol not in cls._DATA:
            raise KeyError(
                f"Element '{symbol}' not in database. "
                f"Available: {sorted(cls._DATA.keys())}"
            )
        return cls._DATA[symbol]

    @classmethod
    def get_binary_enthalpy(cls, e1: str, e2: str) -> float:
        """Return binary mixing enthalpy (kJ/mol). Falls back to EN estimate."""
        if e1 == e2:
            return 0.0
        for key in [(e1, e2), (e2, e1)]:
            if key in cls._DELTA_H_BINARY:
                return cls._DELTA_H_BINARY[key]
        # Fallback: rough Miedema estimate from electronegativity
        try:
            en1 = cls._DATA[e1]["en"]
            en2 = cls._DATA[e2]["en"]
            return -10.0 * abs(en1 - en2)
        except KeyError:
            return 0.0

    @classmethod
    def available_elements(cls) -> List[str]:
        return sorted(cls._DATA.keys())


# ---------------------------------------------------------------------------
# Feature set definitions
# ---------------------------------------------------------------------------

class FeatureSetName(str, Enum):
    """Enumeration of feature set identifiers.

    Every non-base set *includes* FS_BASE columns by design (base is
    always a prerequisite).  The ``value`` is a simple identifier used
    as a dict key throughout the platform; see :class:`FeatureCatalog`
    for the actual column lists.
    """
    FS_BASE = "FS_BASE"
    FS_THERMO = "FS_THERMO"
    FS_SIZE = "FS_SIZE"
    FS_ELECTRON = "FS_ELECTRON"
    FS_ALL = "FS_ALL"


# Column-name constants for each group
_BASE_COLS = [
    "r_avg",        # atomic radius mean
    "delta_r",      # atomic radius difference (%)
    "dS_mix",       # mixing entropy (J/mol K)
    "dH_mix",       # mixing enthalpy proxy (kJ/mol)
    "VEC",          # valence electron concentration
    "delta_EN",     # electronegativity difference
    "Tm_avg",       # melting point average (K)
    "mass_avg",     # atomic mass average (u)
]

_THERMO_COLS = [
    "omega",            # Omega parameter (Tm*dS/|dH|)
    "ss_formation",     # solid solution formation index
    "phase_sep_risk",   # phase separation risk proxy
]

_SIZE_COLS = [
    "Vm_var",           # atomic volume variance
    "elastic_mismatch", # elastic (bulk modulus) mismatch index
]

_ELECTRON_COLS = [
    "d_elec_avg",   # d-electron count average
    "d_elec_std",   # d-electron count std
    "itinerant_proxy",  # itinerant electron proxy (VEC * EN_avg)
]


class FeatureCatalog:
    """Registry that maps FeatureSetName -> list of column names."""

    _SETS: Dict[FeatureSetName, List[str]] = {
        FeatureSetName.FS_BASE: _BASE_COLS,
        FeatureSetName.FS_THERMO: _BASE_COLS + _THERMO_COLS,
        FeatureSetName.FS_SIZE: _BASE_COLS + _SIZE_COLS,
        FeatureSetName.FS_ELECTRON: _BASE_COLS + _ELECTRON_COLS,
        FeatureSetName.FS_ALL: _BASE_COLS + _THERMO_COLS + _SIZE_COLS + _ELECTRON_COLS,
    }

    @classmethod
    def columns(cls, name: FeatureSetName) -> List[str]:
        """Return ordered list of feature column names for the given set."""
        return list(cls._SETS[name])

    @classmethod
    def all_columns(cls) -> List[str]:
        """Return all unique column names across every feature set."""
        return list(cls._SETS[FeatureSetName.FS_ALL])

    @classmethod
    def list_sets(cls) -> List[FeatureSetName]:
        return list(cls._SETS.keys())


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def _weighted_avg(values: np.ndarray, fracs: np.ndarray) -> float:
    return float(np.dot(fracs, values))


def _weighted_std(values: np.ndarray, fracs: np.ndarray) -> float:
    avg = np.dot(fracs, values)
    return float(np.sqrt(np.dot(fracs, (values - avg) ** 2)))


def _delta_percent(values: np.ndarray, fracs: np.ndarray) -> float:
    """Compute delta parameter (%) as in Yang & Zhang (2012)."""
    avg = np.dot(fracs, values)
    if avg == 0:
        return 0.0
    return 100.0 * float(np.sqrt(np.dot(fracs, ((1 - values / avg) ** 2))))


def compute_features_single(
    composition: Dict[str, float],
) -> Dict[str, float]:
    """Compute *all* features for one composition dict {element: fraction}.

    Parameters
    ----------
    composition : dict
        Element symbol -> atomic fraction (should sum to 1).

    Returns
    -------
    dict
        Feature name -> value.  Contains columns from FS_ALL.
    """
    R_GAS = 8.314  # J/(mol K)

    elems = list(composition.keys())
    fracs_raw = np.array([composition[e] for e in elems], dtype=np.float64)
    total = fracs_raw.sum()
    if total <= 0:
        raise ValueError("Composition fractions must be positive and sum > 0")
    fracs = fracs_raw / total  # normalise

    # Gather per-element properties
    props = [_ElementDB.get(e) for e in elems]
    r_arr = np.array([p["r"] for p in props])
    en_arr = np.array([p["en"] for p in props])
    vec_arr = np.array([p["vec"] for p in props])
    tm_arr = np.array([p["Tm"] for p in props])
    mass_arr = np.array([p["mass"] for p in props])
    d_arr = np.array([p["d_elec"] for p in props])
    B_arr = np.array([p["B"] for p in props])
    Vm_arr = np.array([p["Vm"] for p in props])

    # ---- FS_BASE ----
    r_avg = _weighted_avg(r_arr, fracs)
    delta_r = _delta_percent(r_arr, fracs)
    VEC = _weighted_avg(vec_arr, fracs)
    delta_EN = _weighted_std(en_arr, fracs)
    Tm_avg = _weighted_avg(tm_arr, fracs)
    mass_avg = _weighted_avg(mass_arr, fracs)

    # Mixing entropy
    safe_fracs = fracs[fracs > 0]
    dS_mix = -R_GAS * float(np.dot(safe_fracs, np.log(safe_fracs)))

    # Mixing enthalpy (Miedema pair-wise sum)
    dH_mix = 0.0
    for i, ei in enumerate(elems):
        for j, ej in enumerate(elems):
            if j <= i:
                continue
            dH_mix += 4.0 * fracs[i] * fracs[j] * _ElementDB.get_binary_enthalpy(ei, ej)

    # ---- FS_THERMO ----
    abs_dH = abs(dH_mix)
    # Omega parameter: clip to a sensible maximum instead of 1e6 to avoid
    # downstream numerical issues (e.g. ss_formation = omega * dS_mix).
    _OMEGA_MAX = 100.0
    if abs_dH > 1e-6:
        omega = min(Tm_avg * dS_mix / abs_dH, _OMEGA_MAX)
    else:
        omega = _OMEGA_MAX
    ss_formation = omega * dS_mix  # higher -> more likely solid-solution
    # Phase separation risk: positive dH + low omega -> higher risk
    phase_sep_risk = max(0.0, dH_mix) / (omega + 1.0)

    # ---- FS_SIZE ----
    Vm_avg = _weighted_avg(Vm_arr, fracs)
    Vm_var = float(np.dot(fracs, (Vm_arr - Vm_avg) ** 2))
    B_avg = _weighted_avg(B_arr, fracs)
    elastic_mismatch = _delta_percent(B_arr, fracs)

    # ---- FS_ELECTRON ----
    d_elec_avg = _weighted_avg(d_arr, fracs)
    d_elec_std = _weighted_std(d_arr, fracs)
    en_avg = _weighted_avg(en_arr, fracs)
    itinerant_proxy = VEC * en_avg  # proxy for itinerant electron behaviour

    return {
        # BASE
        "r_avg": r_avg,
        "delta_r": delta_r,
        "dS_mix": dS_mix,
        "dH_mix": dH_mix,
        "VEC": VEC,
        "delta_EN": delta_EN,
        "Tm_avg": Tm_avg,
        "mass_avg": mass_avg,
        # THERMO
        "omega": omega,
        "ss_formation": ss_formation,
        "phase_sep_risk": phase_sep_risk,
        # SIZE
        "Vm_var": Vm_var,
        "elastic_mismatch": elastic_mismatch,
        # ELECTRON
        "d_elec_avg": d_elec_avg,
        "d_elec_std": d_elec_std,
        "itinerant_proxy": itinerant_proxy,
    }


def compute_features(
    compositions: Sequence[Dict[str, float]],
    feature_set: FeatureSetName = FeatureSetName.FS_ALL,
) -> pd.DataFrame:
    """Compute features for a list of compositions.

    Parameters
    ----------
    compositions : list of dict
        Each dict maps element symbol -> atomic fraction.
    feature_set : FeatureSetName
        Which feature set to return.

    Returns
    -------
    pd.DataFrame
        Rows = samples, columns = selected feature columns.
    """
    logger.info(
        "Computing features for %d compositions (set=%s)",
        len(compositions), feature_set.value,
    )
    records: List[Dict[str, float]] = []
    for i, comp in enumerate(compositions):
        try:
            rec = compute_features_single(comp)
            records.append(rec)
        except Exception:
            logger.exception("Feature computation failed for sample %d: %s", i, comp)
            raise

    df_all = pd.DataFrame(records)
    cols = FeatureCatalog.columns(feature_set)
    missing = [c for c in cols if c not in df_all.columns]
    if missing:
        raise RuntimeError(f"Missing columns after computation: {missing}")
    return df_all[cols].copy()
