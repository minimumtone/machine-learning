"""
Feature Engineering Module for HEA Extrapolation Platform
HEA特徴量エンジニアリングモジュール

Provides systematic feature set construction from alloy compositions:
  FS_BASE     - Basic statistical descriptors (radius, entropy, VEC, etc.)
  FS_THERMO   - Thermodynamic proxy features (solid-solution index, phase separation risk)
  FS_SIZE     - Atomic size / elastic mismatch features
  FS_ELECTRON - Electronic structure proxy features (d-electron, DOS proxy)
  FS_ALL      - Union of all domain-specific feature sets
  FS_MAGPIE   - MAGPIE features (Ward et al. 2016 / matminer-compatible)
                22 elemental properties x 6 statistics = 132 composition-weighted
                descriptors following the matminer ElementProperty featurizer.
                Reference: Ward et al., npj Comput. Mater. 2, 16028 (2016).
"""

from __future__ import annotations

import logging
from collections import Counter
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

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

    # fmt: off
    # Core properties: Z, vec, en, r, Tm, mass, d_elec, B, Vm
    # MAGPIE additions (Ward et al. 2016 / matminer ElementProperty):
    #   mendeleev_no  - Pettifor Mendeleev number
    #   column        - Periodic table group number
    #   row           - Periodic table period number
    #   cov_r         - Covalent radius (pm)
    #   Ns/Np/Nd/Nf_val - Valence shell electron occupancy (s/p/d/f)
    #   bandgap       - Ground-state bandgap (eV), 0 for metals
    #   magmom        - Ground-state magnetic moment (muB/atom)
    #   space_group   - Ground-state crystal space group number
    # Sources: CRC Handbook, ASM, pymatgen, Pettifor (1984)
    _DATA: Dict[str, Dict[str, float]] = {
        "Al": {"Z": 13, "vec": 3, "en": 1.61, "r": 1.43, "Tm": 933,
               "mass": 26.98, "d_elec": 0, "B": 76, "Vm": 10.0,
               "mendeleev_no": 80, "column": 13, "row": 3, "cov_r": 121,
               "Ns_val": 2, "Np_val": 1, "Nd_val": 0, "Nf_val": 0,
               "bandgap": 0.0, "magmom": 0.0, "space_group": 225},
        "Ti": {"Z": 22, "vec": 4, "en": 1.54, "r": 1.47, "Tm": 1941,
               "mass": 47.87, "d_elec": 2, "B": 110, "Vm": 10.6,
               "mendeleev_no": 51, "column": 4, "row": 4, "cov_r": 160,
               "Ns_val": 2, "Np_val": 0, "Nd_val": 2, "Nf_val": 0,
               "bandgap": 0.0, "magmom": 0.0, "space_group": 194},
        "V":  {"Z": 23, "vec": 5, "en": 1.63, "r": 1.34, "Tm": 2183,
               "mass": 50.94, "d_elec": 3, "B": 160, "Vm": 8.3,
               "mendeleev_no": 54, "column": 5, "row": 4, "cov_r": 153,
               "Ns_val": 2, "Np_val": 0, "Nd_val": 3, "Nf_val": 0,
               "bandgap": 0.0, "magmom": 0.0, "space_group": 229},
        "Cr": {"Z": 24, "vec": 6, "en": 1.66, "r": 1.28, "Tm": 2180,
               "mass": 52.00, "d_elec": 5, "B": 160, "Vm": 7.2,
               "mendeleev_no": 57, "column": 6, "row": 4, "cov_r": 139,
               "Ns_val": 1, "Np_val": 0, "Nd_val": 5, "Nf_val": 0,
               "bandgap": 0.0, "magmom": 0.0, "space_group": 229},
        "Mn": {"Z": 25, "vec": 7, "en": 1.55, "r": 1.27, "Tm": 1519,
               "mass": 54.94, "d_elec": 5, "B": 120, "Vm": 7.4,
               "mendeleev_no": 60, "column": 7, "row": 4, "cov_r": 139,
               "Ns_val": 2, "Np_val": 0, "Nd_val": 5, "Nf_val": 0,
               "bandgap": 0.0, "magmom": 0.0, "space_group": 217},
        "Fe": {"Z": 26, "vec": 8, "en": 1.83, "r": 1.26, "Tm": 1811,
               "mass": 55.85, "d_elec": 6, "B": 170, "Vm": 7.1,
               "mendeleev_no": 61, "column": 8, "row": 4, "cov_r": 132,
               "Ns_val": 2, "Np_val": 0, "Nd_val": 6, "Nf_val": 0,
               "bandgap": 0.0, "magmom": 2.22, "space_group": 229},
        "Co": {"Z": 27, "vec": 9, "en": 1.88, "r": 1.25, "Tm": 1768,
               "mass": 58.93, "d_elec": 7, "B": 180, "Vm": 6.7,
               "mendeleev_no": 64, "column": 9, "row": 4, "cov_r": 126,
               "Ns_val": 2, "Np_val": 0, "Nd_val": 7, "Nf_val": 0,
               "bandgap": 0.0, "magmom": 1.72, "space_group": 194},
        "Ni": {"Z": 28, "vec": 10, "en": 1.91, "r": 1.24, "Tm": 1728,
               "mass": 58.69, "d_elec": 8, "B": 180, "Vm": 6.6,
               "mendeleev_no": 67, "column": 10, "row": 4, "cov_r": 124,
               "Ns_val": 2, "Np_val": 0, "Nd_val": 8, "Nf_val": 0,
               "bandgap": 0.0, "magmom": 0.61, "space_group": 225},
        "Cu": {"Z": 29, "vec": 11, "en": 1.90, "r": 1.28, "Tm": 1358,
               "mass": 63.55, "d_elec": 10, "B": 140, "Vm": 7.1,
               "mendeleev_no": 72, "column": 11, "row": 4, "cov_r": 132,
               "Ns_val": 1, "Np_val": 0, "Nd_val": 10, "Nf_val": 0,
               "bandgap": 0.0, "magmom": 0.0, "space_group": 225},
        "Zn": {"Z": 30, "vec": 12, "en": 1.65, "r": 1.34, "Tm": 693,
               "mass": 65.38, "d_elec": 10, "B": 70, "Vm": 9.2,
               "mendeleev_no": 76, "column": 12, "row": 4, "cov_r": 122,
               "Ns_val": 2, "Np_val": 0, "Nd_val": 10, "Nf_val": 0,
               "bandgap": 0.0, "magmom": 0.0, "space_group": 194},
        "Zr": {"Z": 40, "vec": 4, "en": 1.33, "r": 1.60, "Tm": 2128,
               "mass": 91.22, "d_elec": 2, "B": 94, "Vm": 14.0,
               "mendeleev_no": 48, "column": 4, "row": 5, "cov_r": 175,
               "Ns_val": 2, "Np_val": 0, "Nd_val": 2, "Nf_val": 0,
               "bandgap": 0.0, "magmom": 0.0, "space_group": 194},
        "Nb": {"Z": 41, "vec": 5, "en": 1.60, "r": 1.46, "Tm": 2750,
               "mass": 92.91, "d_elec": 4, "B": 170, "Vm": 10.8,
               "mendeleev_no": 53, "column": 5, "row": 5, "cov_r": 164,
               "Ns_val": 1, "Np_val": 0, "Nd_val": 4, "Nf_val": 0,
               "bandgap": 0.0, "magmom": 0.0, "space_group": 229},
        "Mo": {"Z": 42, "vec": 6, "en": 2.16, "r": 1.39, "Tm": 2896,
               "mass": 95.95, "d_elec": 5, "B": 230, "Vm": 9.4,
               "mendeleev_no": 56, "column": 6, "row": 5, "cov_r": 154,
               "Ns_val": 1, "Np_val": 0, "Nd_val": 5, "Nf_val": 0,
               "bandgap": 0.0, "magmom": 0.0, "space_group": 229},
        "Hf": {"Z": 72, "vec": 4, "en": 1.30, "r": 1.59, "Tm": 2506,
               "mass": 178.49, "d_elec": 2, "B": 110, "Vm": 13.4,
               "mendeleev_no": 50, "column": 4, "row": 6, "cov_r": 175,
               "Ns_val": 2, "Np_val": 0, "Nd_val": 2, "Nf_val": 0,
               "bandgap": 0.0, "magmom": 0.0, "space_group": 194},
        "Ta": {"Z": 73, "vec": 5, "en": 1.50, "r": 1.46, "Tm": 3290,
               "mass": 180.95, "d_elec": 3, "B": 200, "Vm": 10.9,
               "mendeleev_no": 52, "column": 5, "row": 6, "cov_r": 170,
               "Ns_val": 2, "Np_val": 0, "Nd_val": 3, "Nf_val": 0,
               "bandgap": 0.0, "magmom": 0.0, "space_group": 229},
        "W":  {"Z": 74, "vec": 6, "en": 2.36, "r": 1.39, "Tm": 3695,
               "mass": 183.84, "d_elec": 4, "B": 310, "Vm": 9.5,
               "mendeleev_no": 55, "column": 6, "row": 6, "cov_r": 162,
               "Ns_val": 2, "Np_val": 0, "Nd_val": 4, "Nf_val": 0,
               "bandgap": 0.0, "magmom": 0.0, "space_group": 229},
        "Re": {"Z": 75, "vec": 7, "en": 1.90, "r": 1.37, "Tm": 3459,
               "mass": 186.21, "d_elec": 5, "B": 370, "Vm": 8.9,
               "mendeleev_no": 58, "column": 7, "row": 6, "cov_r": 151,
               "Ns_val": 2, "Np_val": 0, "Nd_val": 5, "Nf_val": 0,
               "bandgap": 0.0, "magmom": 0.0, "space_group": 194},
        "Pd": {"Z": 46, "vec": 10, "en": 2.20, "r": 1.37, "Tm": 1828,
               "mass": 106.42, "d_elec": 10, "B": 180, "Vm": 8.6,
               "mendeleev_no": 69, "column": 10, "row": 5, "cov_r": 139,
               "Ns_val": 0, "Np_val": 0, "Nd_val": 10, "Nf_val": 0,
               "bandgap": 0.0, "magmom": 0.0, "space_group": 225},
        "Ag": {"Z": 47, "vec": 11, "en": 1.93, "r": 1.44, "Tm": 1235,
               "mass": 107.87, "d_elec": 10, "B": 100, "Vm": 10.3,
               "mendeleev_no": 71, "column": 11, "row": 5, "cov_r": 145,
               "Ns_val": 1, "Np_val": 0, "Nd_val": 10, "Nf_val": 0,
               "bandgap": 0.0, "magmom": 0.0, "space_group": 225},
        "Pt": {"Z": 78, "vec": 10, "en": 2.28, "r": 1.39, "Tm": 2041,
               "mass": 195.08, "d_elec": 9, "B": 230, "Vm": 9.1,
               "mendeleev_no": 68, "column": 10, "row": 6, "cov_r": 136,
               "Ns_val": 1, "Np_val": 0, "Nd_val": 9, "Nf_val": 0,
               "bandgap": 0.0, "magmom": 0.0, "space_group": 225},
        "Au": {"Z": 79, "vec": 11, "en": 2.54, "r": 1.44, "Tm": 1337,
               "mass": 196.97, "d_elec": 10, "B": 220, "Vm": 10.2,
               "mendeleev_no": 70, "column": 11, "row": 6, "cov_r": 136,
               "Ns_val": 1, "Np_val": 0, "Nd_val": 10, "Nf_val": 0,
               "bandgap": 0.0, "magmom": 0.0, "space_group": 225},
        "Si": {"Z": 14, "vec": 4, "en": 1.90, "r": 1.18, "Tm": 1687,
               "mass": 28.09, "d_elec": 0, "B": 100, "Vm": 12.1,
               "mendeleev_no": 85, "column": 14, "row": 3, "cov_r": 111,
               "Ns_val": 2, "Np_val": 2, "Nd_val": 0, "Nf_val": 0,
               "bandgap": 1.12, "magmom": 0.0, "space_group": 227},
        "Mg": {"Z": 12, "vec": 2, "en": 1.31, "r": 1.60, "Tm": 923,
               "mass": 24.31, "d_elec": 0, "B": 45, "Vm": 14.0,
               "mendeleev_no": 73, "column": 2, "row": 3, "cov_r": 141,
               "Ns_val": 2, "Np_val": 0, "Nd_val": 0, "Nf_val": 0,
               "bandgap": 0.0, "magmom": 0.0, "space_group": 194},
        "Sc": {"Z": 21, "vec": 3, "en": 1.36, "r": 1.64, "Tm": 1814,
               "mass": 44.96, "d_elec": 1, "B": 57, "Vm": 15.0,
               "mendeleev_no": 19, "column": 3, "row": 4, "cov_r": 170,
               "Ns_val": 2, "Np_val": 0, "Nd_val": 1, "Nf_val": 0,
               "bandgap": 0.0, "magmom": 0.0, "space_group": 194},
        "Y":  {"Z": 39, "vec": 3, "en": 1.22, "r": 1.80, "Tm": 1799,
               "mass": 88.91, "d_elec": 1, "B": 41, "Vm": 19.9,
               "mendeleev_no": 12, "column": 3, "row": 5, "cov_r": 190,
               "Ns_val": 2, "Np_val": 0, "Nd_val": 1, "Nf_val": 0,
               "bandgap": 0.0, "magmom": 0.0, "space_group": 194},
    }
    # fmt: on

    # Miedema binary mixing enthalpies (kJ/mol).
    # Sources: de Boer et al. (1988), Takeuchi & Inoue (2005).
    # Coverage expanded from 52 to ~120 pairs for common HEA elements.
    _DELTA_H_BINARY: Dict[tuple, float] = {
        # --- 3d transition metal pairs ---
        ("Co", "Cr"): -4, ("Co", "Fe"): -1, ("Co", "Mn"): -5,
        ("Co", "Ni"): 0,  ("Cr", "Fe"): -1, ("Cr", "Mn"): 2,
        ("Cr", "Ni"): -7, ("Fe", "Mn"): 0,  ("Fe", "Ni"): -2,
        ("Mn", "Ni"): -8, ("Co", "V"): -14, ("Cr", "V"): -2,
        ("Fe", "V"): -7,  ("Mn", "V"): -1,  ("Ni", "V"): -18,
        ("Co", "Cu"): 6,  ("Cr", "Cu"): 12, ("Fe", "Cu"): 13,
        ("Mn", "Cu"): 4,  ("Ni", "Cu"): 4,  ("V", "Cu"): 5,
        # --- Ti binary pairs ---
        ("Ti", "V"): -2,  ("Ti", "Cr"): -7, ("Ti", "Mn"): -8,
        ("Ti", "Fe"): -17, ("Ti", "Co"): -28, ("Ti", "Ni"): -35,
        ("Ti", "Cu"): -9, ("Ti", "Zn"): -15,
        # --- Al binary pairs ---
        ("Al", "Co"): -19, ("Al", "Cr"): -10, ("Al", "Fe"): -11,
        ("Al", "Mn"): -19, ("Al", "Ni"): -22, ("Al", "Ti"): -30,
        ("Al", "Cu"): -1,  ("Al", "V"): -16, ("Al", "Zr"): -44,
        ("Al", "Nb"): -18, ("Al", "Mo"): -5, ("Al", "Hf"): -39,
        ("Al", "Ta"): -19, ("Al", "W"): -2,  ("Al", "Si"): -4,
        ("Al", "Mg"): -2,  ("Al", "Sc"): -38,
        # --- Refractory pairs (4d/5d) ---
        ("Nb", "Ti"): 2,   ("Nb", "Zr"): 4,   ("Nb", "Hf"): 4,
        ("Nb", "Ta"): 0,   ("Nb", "Mo"): -6,  ("Nb", "W"): -8,
        ("Nb", "V"): -1,   ("Nb", "Cr"): -7,  ("Nb", "Mn"): -4,
        ("Nb", "Fe"): -16, ("Nb", "Co"): -25, ("Nb", "Ni"): -30,
        ("Nb", "Cu"): 3,
        ("Mo", "Ti"): -4,  ("Mo", "Zr"): -6,  ("Mo", "Hf"): -4,
        ("Mo", "Ta"): -5,  ("Mo", "W"): 0,    ("Mo", "V"): -1,
        ("Mo", "Cr"): 0,   ("Mo", "Mn"): -5,  ("Mo", "Fe"): -2,
        ("Mo", "Co"): -5,  ("Mo", "Ni"): -7,  ("Mo", "Cu"): 19,
        ("Ta", "Ti"): 1,   ("Ta", "Zr"): 3,   ("Ta", "Hf"): 3,
        ("Ta", "V"): -1,   ("Ta", "Cr"): -7,  ("Ta", "Mn"): -5,
        ("Ta", "Fe"): -15, ("Ta", "Co"): -24, ("Ta", "Ni"): -29,
        ("Ta", "Cu"): 2,   ("Ta", "W"): -7,
        ("W", "Ti"): -6,   ("W", "Zr"): -9,   ("W", "Hf"): -6,
        ("W", "V"): -1,    ("W", "Cr"): 1,    ("W", "Mn"): -4,
        ("W", "Fe"): -1,   ("W", "Co"): -1,   ("W", "Ni"): -3,
        ("W", "Cu"): 22,
        ("Hf", "Ti"): 0,   ("Hf", "Zr"): 0,   ("Hf", "V"): -2,
        ("Hf", "Cr"): -9,  ("Hf", "Mn"): -12, ("Hf", "Fe"): -21,
        ("Hf", "Co"): -35, ("Hf", "Ni"): -42, ("Hf", "Cu"): -17,
        ("Zr", "Ti"): 0,   ("Zr", "V"): -4,   ("Zr", "Cr"): -12,
        ("Zr", "Mn"): -15, ("Zr", "Fe"): -25, ("Zr", "Co"): -41,
        ("Zr", "Ni"): -49, ("Zr", "Cu"): -23,
        # --- Mg pairs (important: many are positive/near-zero) ---
        ("Mg", "Ti"): -4,  ("Mg", "V"): 0,    ("Mg", "Cr"): 2,
        ("Mg", "Mn"): -3,  ("Mg", "Fe"): 4,   ("Mg", "Co"): 3,
        ("Mg", "Ni"): -4,  ("Mg", "Cu"): -3,  ("Mg", "Zn"): -4,
        ("Mg", "Al"): -2,  ("Mg", "Si"): -3,  ("Mg", "Sc"): -7,
        ("Mg", "Zr"): -6,  ("Mg", "Nb"): 3,   ("Mg", "Mo"): 10,
        ("Mg", "Hf"): -4,  ("Mg", "Ta"): 13,  ("Mg", "W"): 13,
        # --- Si pairs ---
        ("Si", "Ti"): -66, ("Si", "V"): -48,  ("Si", "Cr"): -37,
        ("Si", "Mn"): -37, ("Si", "Fe"): -35, ("Si", "Co"): -38,
        ("Si", "Ni"): -40, ("Si", "Cu"): -19, ("Si", "Zr"): -84,
        ("Si", "Nb"): -56, ("Si", "Mo"): -35, ("Si", "Hf"): -80,
        # --- Sc pairs ---
        ("Sc", "Ti"): -4,  ("Sc", "V"): -4,   ("Sc", "Cr"): -4,
        ("Sc", "Fe"): -15, ("Sc", "Co"): -22, ("Sc", "Ni"): -27,
        # --- Precious/rare (Re, Pd, Pt, Au, Ag, Y) ---
        ("Re", "Ti"): -8,  ("Re", "Cr"): -6,  ("Re", "Fe"): -1,
        ("Re", "Ni"): -9,  ("Re", "W"): -3,   ("Re", "Mo"): -2,
        ("Pd", "Ti"): -52, ("Pd", "Cr"): -3,  ("Pd", "Fe"): -4,
        ("Pd", "Co"): -1,  ("Pd", "Ni"): 0,   ("Pd", "Cu"): -14,
        ("Pt", "Ti"): -74, ("Pt", "Cr"): -12, ("Pt", "Fe"): -6,
        ("Pt", "Ni"): -5,  ("Pt", "Cu"): -12,
        ("Au", "Ti"): -48, ("Au", "Cr"): 3,   ("Au", "Fe"): 3,
        ("Au", "Ni"): -9,  ("Au", "Cu"): -9,  ("Au", "Al"): -22,
        ("Ag", "Ti"): -15, ("Ag", "Cr"): 15,  ("Ag", "Fe"): 15,
        ("Ag", "Ni"): 15,  ("Ag", "Cu"): 2,   ("Ag", "Al"): -4,
        ("Y", "Ti"): 5,    ("Y", "Cr"): 2,    ("Y", "Fe"): -1,
        ("Y", "Ni"): -31,  ("Y", "Al"): -38,
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

    FS_MAGPIE is an independent, general-purpose compositional feature
    set following Ward et al. (2016) / matminer's ElementProperty
    featurizer.  It computes 6 statistics (mean, avg_dev, range,
    maximum, minimum, mode) for each of 22 elemental properties,
    yielding 132 features.
    """
    FS_BASE = "FS_BASE"
    FS_THERMO = "FS_THERMO"
    FS_SIZE = "FS_SIZE"
    FS_ELECTRON = "FS_ELECTRON"
    FS_ALL = "FS_ALL"
    FS_MAGPIE = "FS_MAGPIE"


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
    "B_avg",            # bulk modulus average (GPa) — Hall-Petch proxy
    "Vm_avg",           # atomic volume average (Å³) — packing density
    "Vm_var",           # atomic volume variance
    "elastic_mismatch", # elastic (bulk modulus) mismatch index
]

_ELECTRON_COLS = [
    "d_elec_avg",   # d-electron count average
    "d_elec_std",   # d-electron count std
    "itinerant_proxy",  # itinerant electron proxy (VEC * EN_avg)
]

# ---------------------------------------------------------------------------
# MAGPIE feature definitions (Ward et al. 2016 / matminer ElementProperty)
# ---------------------------------------------------------------------------
# 22 elemental properties × 6 statistics = 132 features.
# Property keys map to _ElementDB._DATA keys or are derived at runtime.

_MAGPIE_PROP_KEYS: List[Tuple[str, str]] = [
    # (display_name, _DATA key or "DERIVED:xxx")
    ("Number", "Z"),
    ("MendeleevNumber", "mendeleev_no"),
    ("AtomicWeight", "mass"),
    ("MeltingT", "Tm"),
    ("Column", "column"),
    ("Row", "row"),
    ("CovalentRadius", "cov_r"),
    ("Electronegativity", "en"),
    ("NsValence", "Ns_val"),
    ("NpValence", "Np_val"),
    ("NdValence", "Nd_val"),
    ("NfValence", "Nf_val"),
    ("NValence", "DERIVED:N_valence"),
    ("NsUnfilled", "DERIVED:Ns_unfilled"),
    ("NpUnfilled", "DERIVED:Np_unfilled"),
    ("NdUnfilled", "DERIVED:Nd_unfilled"),
    ("NfUnfilled", "DERIVED:Nf_unfilled"),
    ("NUnfilled", "DERIVED:N_unfilled"),
    ("GSvolume_pa", "Vm"),
    ("GSbandgap", "bandgap"),
    ("GSmagmom", "magmom"),
    ("SpaceGroupNumber", "space_group"),
]

_MAGPIE_STATS = ["mean", "avg_dev", "range", "maximum", "minimum", "mode"]

# Build MAGPIE column names: "MagpieData {stat} {property}"
_MAGPIE_COLS: List[str] = []
for _prop_name, _ in _MAGPIE_PROP_KEYS:
    for _stat in _MAGPIE_STATS:
        _MAGPIE_COLS.append(f"MagpieData {_stat} {_prop_name}")


class FeatureCatalog:
    """Registry that maps FeatureSetName -> list of column names."""

    _SETS: Dict[FeatureSetName, List[str]] = {
        FeatureSetName.FS_BASE: _BASE_COLS,
        FeatureSetName.FS_THERMO: _BASE_COLS + _THERMO_COLS,
        FeatureSetName.FS_SIZE: _BASE_COLS + _SIZE_COLS,
        FeatureSetName.FS_ELECTRON: _BASE_COLS + _ELECTRON_COLS,
        FeatureSetName.FS_ALL: _BASE_COLS + _THERMO_COLS + _SIZE_COLS + _ELECTRON_COLS,
        FeatureSetName.FS_MAGPIE: _MAGPIE_COLS,
    }

    @classmethod
    def columns(cls, name: FeatureSetName) -> List[str]:
        """Return ordered list of feature column names for the given set."""
        return list(cls._SETS[name])

    @classmethod
    def all_columns(cls) -> List[str]:
        """Return all unique column names across every feature set."""
        seen: set = set()
        result: List[str] = []
        for cols in cls._SETS.values():
            for c in cols:
                if c not in seen:
                    seen.add(c)
                    result.append(c)
        return result

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


# ---------------------------------------------------------------------------
# MAGPIE statistics (matminer-compatible)
# ---------------------------------------------------------------------------

def _magpie_avg_dev(values: np.ndarray, fracs: np.ndarray) -> float:
    """Composition-weighted mean absolute deviation from the weighted mean.

    This matches matminer's ``avg_dev`` statistic for ElementProperty.
    avg_dev = sum_i( x_i * |p_i - mean| )
    """
    mean = float(np.dot(fracs, values))
    return float(np.dot(fracs, np.abs(values - mean)))


def _magpie_mode(values: np.ndarray, fracs: np.ndarray) -> float:
    """Mode of the elemental property weighted by composition.

    Following matminer convention: the property value of the element
    with the highest atomic fraction.  If multiple elements share the
    maximum fraction, return the mean of their property values.
    """
    max_frac = float(np.max(fracs))
    mask = np.isclose(fracs, max_frac, atol=1e-10)
    return float(np.mean(values[mask]))


def _get_magpie_property_values(
    props: List[Dict[str, float]],
    key: str,
) -> np.ndarray:
    """Extract property array for one MAGPIE property.

    Handles both direct _DATA keys and derived properties (prefixed
    with ``DERIVED:``).
    """
    if not key.startswith("DERIVED:"):
        return np.array([p[key] for p in props], dtype=np.float64)

    derived_name = key.split(":", 1)[1]
    n = len(props)
    result = np.zeros(n, dtype=np.float64)

    for i, p in enumerate(props):
        ns = p["Ns_val"]
        np_val = p["Np_val"]
        nd = p["Nd_val"]
        nf = p["Nf_val"]

        if derived_name == "N_valence":
            result[i] = ns + np_val + nd + nf
        elif derived_name == "Ns_unfilled":
            result[i] = (2 - ns) if ns > 0 else 0
        elif derived_name == "Np_unfilled":
            result[i] = (6 - np_val) if np_val > 0 else 0
        elif derived_name == "Nd_unfilled":
            result[i] = (10 - nd) if nd > 0 else 0
        elif derived_name == "Nf_unfilled":
            result[i] = (14 - nf) if nf > 0 else 0
        elif derived_name == "N_unfilled":
            ns_u = (2 - ns) if ns > 0 else 0
            np_u = (6 - np_val) if np_val > 0 else 0
            nd_u = (10 - nd) if nd > 0 else 0
            nf_u = (14 - nf) if nf > 0 else 0
            result[i] = ns_u + np_u + nd_u + nf_u
        else:
            raise ValueError(f"Unknown derived MAGPIE property: {derived_name}")

    return result


def _compute_magpie_stats(
    values: np.ndarray,
    fracs: np.ndarray,
) -> Dict[str, float]:
    """Compute the 6 MAGPIE statistics for one property.

    Returns dict with keys: mean, avg_dev, range, maximum, minimum, mode.
    """
    w_mean = float(np.dot(fracs, values))
    w_avg_dev = _magpie_avg_dev(values, fracs)
    v_max = float(np.max(values))
    v_min = float(np.min(values))
    v_range = v_max - v_min
    v_mode = _magpie_mode(values, fracs)

    return {
        "mean": w_mean,
        "avg_dev": w_avg_dev,
        "range": v_range,
        "maximum": v_max,
        "minimum": v_min,
        "mode": v_mode,
    }


def compute_magpie_features(
    elems: List[str],
    fracs: np.ndarray,
    props: List[Dict[str, float]],
) -> Dict[str, float]:
    """Compute all 132 MAGPIE features for one composition.

    Parameters
    ----------
    elems : list of str
        Element symbols.
    fracs : np.ndarray
        Normalised atomic fractions (same order as *elems*).
    props : list of dict
        Per-element property dicts from ``_ElementDB.get()``.

    Returns
    -------
    dict
        Feature name -> value.  132 features total
        (22 properties x 6 statistics).
    """
    result: Dict[str, float] = {}

    for prop_name, data_key in _MAGPIE_PROP_KEYS:
        values = _get_magpie_property_values(props, data_key)
        stats = _compute_magpie_stats(values, fracs)
        for stat_name, stat_val in stats.items():
            col = f"MagpieData {stat_name} {prop_name}"
            result[col] = stat_val

    return result


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
        Feature name -> value.  Contains columns from FS_ALL and FS_MAGPIE.
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
    # Omega parameter (Yang & Zhang 2012): Ω = Tm * ΔS_mix / |ΔH_mix|
    # Units: dS_mix is J/(mol·K), dH_mix is kJ/mol → convert kJ to J (*1000).
    # Typical real-HEA Ω ∈ [1, 50]; clip at 10 to keep it a useful discriminator.
    _OMEGA_MAX = 10.0
    if abs_dH > 1e-6:
        omega = min(Tm_avg * dS_mix / (abs_dH * 1000.0), _OMEGA_MAX)
    else:
        omega = _OMEGA_MAX
    # Use the clipped omega for ss_formation to avoid extreme feature values
    # when abs_dH is very small (numerical stability for downstream ML).
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

    # ---- FS_MAGPIE ----
    magpie_feats = compute_magpie_features(elems, fracs, props)

    result = {
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
        "B_avg": B_avg,           # bulk modulus average (GPa)
        "Vm_avg": Vm_avg,         # atomic volume average (Å³)
        "Vm_var": Vm_var,
        "elastic_mismatch": elastic_mismatch,
        # ELECTRON
        "d_elec_avg": d_elec_avg,
        "d_elec_std": d_elec_std,
        "itinerant_proxy": itinerant_proxy,
    }
    # Merge MAGPIE features into the result dict
    result.update(magpie_feats)
    return result


def compute_features(
    compositions: Sequence[Dict[str, float]],
    feature_set: Optional[FeatureSetName] = FeatureSetName.FS_ALL,
) -> pd.DataFrame:
    """Compute features for a list of compositions.

    Parameters
    ----------
    compositions : list of dict
        Each dict maps element symbol -> atomic fraction.
    feature_set : FeatureSetName or None
        Which feature set to return.  If ``None``, return *all*
        computed columns (domain-specific + MAGPIE).

    Returns
    -------
    pd.DataFrame
        Rows = samples, columns = selected feature columns.
    """
    logger.info(
        "Computing features for %d compositions (set=%s)",
        len(compositions), feature_set.value if feature_set else "ALL_COLUMNS",
    )
    records: List[Dict[str, float]] = []
    for i, comp in enumerate(compositions):
        try:
            rec = compute_features_single(comp)
            records.append(rec)
        except Exception:
            logger.exception("Feature computation failed for sample %d: %s", i, comp)
            raise

    # Build DataFrame from dict-of-lists (columnar) instead of
    # list-of-dicts (row-wise).  pd.DataFrame(list-of-dicts) creates
    # one internal memory block per dict key, leading to a highly
    # fragmented BlockManager (148+ blocks for MAGPIE features).
    # Downstream .describe() / numpy interop on such a frame can
    # trigger a SIGSEGV in the pandas/numpy C layer.
    # Building from {col: [values]} creates a single consolidated block.
    if records:
        col_names = list(records[0].keys())
        columns_dict = {
            k: [r[k] for r in records] for k in col_names
        }
        df_all = pd.DataFrame(columns_dict)
    else:
        df_all = pd.DataFrame()

    if feature_set is None:
        # Return all computed columns (domain + MAGPIE)
        return df_all

    cols = FeatureCatalog.columns(feature_set)
    missing = [c for c in cols if c not in df_all.columns]
    if missing:
        raise RuntimeError(f"Missing columns after computation: {missing}")
    return df_all[cols]
