"""
HEA Dataset Generation Module
HEAデータセット生成モジュール

Generates a synthetic HEA dataset with compositions drawn from
metallurgically realistic element pools (FCC / BCC / mixed-phase).
A structure-aware yield-strength proxy distinguishes FCC-type
(Cantor-family) from BCC-type (refractory) alloys.

Solid-solution stability filters (delta < 8 %, dH_mix range, dS_mix
floor, fraction bounds) reject unphysical compositions before they
enter the dataset.

References:
  - Yang & Zhang, Mater. Chem. Phys. 132 (2012) 233
  - Guo et al., J. Appl. Phys. 109 (2011) 103505  (VEC phase boundary)
  - Toda-Caraballo & Rivera-Diaz-del-Castillo, Acta Mat. 85 (2015) 14
  - Senkov et al., J. Alloys Compd. 509 (2011) 6043
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from hea_extrapolation_platform.features import (
    _ElementDB,
    compute_features,
    compute_features_single,
    FeatureSetName,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Element pools split by expected crystal structure
# ---------------------------------------------------------------------------

# FCC-type: Cantor-family + Al extension (VEC >= ~8 when equimolar)
_POOL_FCC = ["Co", "Cr", "Fe", "Mn", "Ni", "Cu", "Al"]

# BCC-type: refractory HEAs (VEC < ~6.87 when equimolar)
_POOL_BCC = ["Mo", "Nb", "Ta", "Ti", "V", "W", "Hf", "Zr", "Cr"]

# Mixed-phase candidates: elements that appear in both families
_POOL_MIXED = ["Al", "Co", "Cr", "Fe", "Mn", "Ni", "Ti", "V", "Cu",
               "Mo", "Nb", "Hf"]

# ---------------------------------------------------------------------------
# Solid-solution stability filter constants (Yang & Zhang 2012)
# ---------------------------------------------------------------------------
_DELTA_R_MAX = 8.0          # atomic radius mismatch (%)
_DH_MIX_MIN = -20.0         # mixing enthalpy lower bound (kJ/mol)
_DH_MIX_MAX = 5.0           # mixing enthalpy upper bound (kJ/mol)
_DS_MIX_MIN = 12.0          # minimum mixing entropy (J/mol K)
_FRAC_MIN = 0.05            # minimum atomic fraction per element
_FRAC_MAX = 0.35            # maximum atomic fraction per element
_MAX_REJECT_RATIO = 200     # max rejected / n_samples before relaxing


# ---------------------------------------------------------------------------
# Composition generation with stability filter
# ---------------------------------------------------------------------------

def _random_composition(
    elements: List[str],
    rng: np.random.Generator,
    min_frac: float = _FRAC_MIN,
    max_frac: float = _FRAC_MAX,
) -> Dict[str, float]:
    """Generate a random composition with fraction bounds.

    Uses a Dirichlet distribution and rejects samples where any
    fraction falls outside [min_frac, max_frac].  When the bounds
    are infeasible for the given number of elements (e.g. n=2 with
    max_frac=0.35), ``max_frac`` is automatically relaxed so that
    the constraint is satisfiable.
    """
    n = len(elements)
    # Feasibility guard: n * max_frac must be >= 1.0 for fractions
    # that sum to 1.0 to exist within [min_frac, max_frac].
    effective_max = max_frac
    if n * max_frac < 1.0:
        effective_max = 1.0  # relax max_frac for small n
    _MAX_ITER = 10_000
    for _ in range(_MAX_ITER):
        fracs = rng.dirichlet(np.ones(n) * 2.0)
        if np.all(fracs >= min_frac) and np.all(fracs <= effective_max):
            return {e: float(f) for e, f in zip(elements, fracs)}
    # Fallback: accept last draw with only min_frac constraint
    logger.warning(
        "Composition sampling did not converge in %d iterations for "
        "%d elements; relaxing max_frac constraint", _MAX_ITER, n,
    )
    while True:
        fracs = rng.dirichlet(np.ones(n) * 2.0)
        if np.all(fracs >= min_frac):
            return {e: float(f) for e, f in zip(elements, fracs)}


def _passes_stability_filter(feat: Dict[str, float]) -> bool:
    """Return True if the composition satisfies solid-solution criteria.

    Criteria (Yang & Zhang 2012, relaxed for synthetic data):
      - delta_r < 8 %
      - -20 <= dH_mix <= 5 kJ/mol
      - dS_mix >= 12 J/mol K
    """
    if feat["delta_r"] >= _DELTA_R_MAX:
        return False
    if feat["dH_mix"] < _DH_MIX_MIN or feat["dH_mix"] > _DH_MIX_MAX:
        return False
    if feat["dS_mix"] < _DS_MIX_MIN:
        return False
    return True


# ---------------------------------------------------------------------------
# Structure-aware yield-strength proxy
# ---------------------------------------------------------------------------

def _yield_strength_proxy(
    feat: Dict[str, float],
    noise: float,
) -> float:
    """Compute a synthetic yield strength (MPa) with FCC/BCC-aware model.

    The VEC-based phase boundary follows Guo et al. (2011):
      VEC >= 8.0   -> FCC  (Cantor-type, 600-1000 MPa range)
      VEC <  6.87  -> BCC  (refractory, 900-2000 MPa range)
      otherwise    -> mixed phase

    Parameters
    ----------
    feat : dict
        Pre-computed feature dict (from FS_ALL).
    noise : float
        Pre-drawn standard-normal value.  Actual noise is proportional
        to predicted strength (7 % coefficient of variation).
    """
    vec = feat["VEC"]

    if vec >= 8.0:
        # FCC regime: lower base, moderate lattice-strain hardening
        base = 300.0
        k_lattice = 18.0
        k_Tm = 50.0
    elif vec < 6.87:
        # BCC regime: high base, strong lattice-strain hardening
        base = 600.0
        k_lattice = 35.0
        k_Tm = 150.0
    else:
        # Mixed-phase regime
        base = 450.0
        k_lattice = 26.0
        k_Tm = 100.0

    ys = (
        base
        + k_lattice * feat["delta_r"] * np.sqrt(max(vec, 1.0))
        + 3.0 * abs(feat["dH_mix"])
        + k_Tm * feat["Tm_avg"] / 1000.0
        + 5.0 * feat["elastic_mismatch"]
    )

    # Proportional noise: high-strength alloys have larger scatter
    noise_std = 0.07 * ys
    ys += noise * noise_std
    return max(ys, 100.0)


# ---------------------------------------------------------------------------
# Main dataset generator
# ---------------------------------------------------------------------------

def generate_hea_dataset(
    n_samples: int = 200,
    seed: int = 42,
    min_elements: int = 5,
    max_elements: int = 7,
    noise_std: float = 50.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Generate a synthetic HEA dataset with realistic compositions.

    Sampling strategy:
      - 40 % FCC-pool (Cantor-family)
      - 40 % BCC-pool (refractory)
      - 20 % mixed-phase pool

    Each candidate composition is checked against solid-solution
    stability filters; unphysical compositions are rejected and
    re-sampled (up to ``_MAX_REJECT_RATIO * n_samples`` attempts).

    Parameters
    ----------
    n_samples : int
        Number of compositions to generate.
    seed : int
        Random seed for reproducibility.
    min_elements : int
        Minimum number of constituent elements per alloy.
    max_elements : int
        Maximum number of constituent elements per alloy.
    noise_std : float
        Kept for API compatibility; strength noise is now proportional
        to predicted YS (7 % coefficient of variation).

    Returns
    -------
    compositions_df : pd.DataFrame
        Composition table (element columns, fraction values).
    features_df : pd.DataFrame
        All features (FS_ALL) for each sample.
    target : pd.Series
        Yield strength (MPa).
    """
    rng = np.random.default_rng(seed)
    logger.info(
        "Generating HEA dataset: n=%d, seed=%d, elements=%d-%d",
        n_samples, seed, min_elements, max_elements,
    )

    # Pre-filter pools to only include elements in the database
    available = set(_ElementDB.available_elements())
    pool_fcc = [e for e in _POOL_FCC if e in available]
    pool_bcc = [e for e in _POOL_BCC if e in available]
    pool_mix = [e for e in _POOL_MIXED if e in available]

    # Determine how many samples from each pool
    n_fcc = int(n_samples * 0.4)
    n_bcc = int(n_samples * 0.4)
    n_mix = n_samples - n_fcc - n_bcc

    compositions: List[Dict[str, float]] = []
    noise_values: List[float] = []
    all_elements_set: set = set()
    rejected_total = 0
    reject_limit = _MAX_REJECT_RATIO * n_samples
    filter_relaxed = False

    for pool, n_pool, label in [
        (pool_fcc, n_fcc, "FCC"),
        (pool_bcc, n_bcc, "BCC"),
        (pool_mix, n_mix, "MIXED"),
    ]:
        accepted = 0
        while accepted < n_pool:
            n_elems = int(rng.integers(min_elements, max_elements + 1))
            if len(pool) < n_elems:
                n_elems = len(pool)
            chosen = list(rng.choice(pool, size=n_elems, replace=False))
            comp = _random_composition(chosen, rng)

            # Draw noise *before* the filter check to preserve RNG order
            noise_val = float(rng.standard_normal())

            # Stability filter (disabled once reject limit is exceeded)
            try:
                feat = compute_features_single(comp)
            except Exception:
                rejected_total += 1
                if rejected_total > reject_limit and not filter_relaxed:
                    filter_relaxed = True
                    logger.warning(
                        "Reject limit (%d) reached at %s pool; "
                        "disabling stability filter for remaining samples",
                        reject_limit, label,
                    )
                continue

            if filter_relaxed or _passes_stability_filter(feat):
                compositions.append(comp)
                noise_values.append(noise_val)
                all_elements_set.update(chosen)
                accepted += 1
            else:
                rejected_total += 1
                if rejected_total > reject_limit and not filter_relaxed:
                    filter_relaxed = True
                    logger.warning(
                        "Reject limit (%d) reached at %s pool; "
                        "disabling stability filter for remaining samples",
                        reject_limit, label,
                    )

    if len(compositions) < n_samples:
        logger.warning(
            "Generated %d / %d requested samples (shortfall due to "
            "feature computation errors)", len(compositions), n_samples,
        )

    logger.info(
        "Composition generation: %d accepted, %d rejected by stability filter",
        len(compositions), rejected_total,
    )

    # Build composition DataFrame (sparse: missing element -> 0)
    all_elems_sorted = sorted(all_elements_set)
    comp_records = []
    for comp in compositions:
        rec = {e: comp.get(e, 0.0) for e in all_elems_sorted}
        comp_records.append(rec)
    compositions_df = pd.DataFrame(comp_records)

    # Build features DataFrame (all features)
    features_df = compute_features(compositions, FeatureSetName.FS_ALL)

    # Compute target from the *already computed* FS_ALL features
    features_records = features_df.to_dict(orient="records")
    targets = [
        _yield_strength_proxy(f, n)
        for f, n in zip(features_records, noise_values)
    ]
    target = pd.Series(targets, name="yield_strength_MPa")

    logger.info(
        "Dataset generated: %d samples, %d elements, features shape %s, "
        "YS range [%.0f, %.0f] MPa",
        len(target), len(all_elems_sorted), features_df.shape,
        target.min(), target.max(),
    )
    return compositions_df, features_df, target
