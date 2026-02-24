"""
HEA Dataset Generation Module
HEAデータセット生成モジュール

Generates a synthetic HEA dataset with ~200 compositions (5+ element systems)
and a literature-inspired yield-strength proxy based on solid-solution
strengthening models (Toda-Caraballo & Rivera-Diaz-del-Castillo, Acta Mat 2015).

The proxy is intentionally noisy to simulate real experimental scatter.
"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from hea_extrapolation_platform.features import (
    _ElementDB,
    compute_features,
    FeatureSetName,
)

logger = logging.getLogger(__name__)

# Common HEA element pools
_POOL_3D = ["Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu"]
_POOL_REFRACTORY = ["Ti", "V", "Cr", "Zr", "Nb", "Mo", "Hf", "Ta", "W"]
_POOL_LIGHT = ["Mg", "Al", "Si", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni"]


def _random_composition(
    elements: List[str],
    rng: np.random.Generator,
    min_frac: float = 0.05,
) -> Dict[str, float]:
    """Generate a random composition for the given elements.

    Uses a Dirichlet distribution to ensure fractions sum to 1, with a
    minimum fraction constraint to avoid trace-level components.
    """
    n = len(elements)
    while True:
        fracs = rng.dirichlet(np.ones(n) * 2.0)
        if np.all(fracs >= min_frac):
            break
    comp = {e: float(f) for e, f in zip(elements, fracs)}
    return comp


def _yield_strength_proxy(
    composition: Dict[str, float],
    rng: np.random.Generator,
    noise_std: float = 50.0,
) -> float:
    """Compute a synthetic yield strength (MPa) using a simplified
    solid-solution strengthening model.

    YS ~ base + k1 * delta_r * sqrt(VEC) + k2 * |dH_mix| + k3 * dS_mix
         + k4 * Tm_avg / 1000 + noise

    This is deliberately an *approximate* physics-inspired proxy, not a
    real predictive model.  The noise term simulates experimental scatter.
    """
    from hea_extrapolation_platform.features import compute_features_single

    feat = compute_features_single(composition)
    base = 200.0
    ys = (
        base
        + 15.0 * feat["delta_r"] * np.sqrt(max(feat["VEC"], 1.0))
        + 3.0 * abs(feat["dH_mix"])
        + 0.02 * feat["dS_mix"]
        + 80.0 * feat["Tm_avg"] / 1000.0
        + 5.0 * feat["d_elec_avg"]
        - 2.0 * feat["phase_sep_risk"]
        + 0.5 * feat["elastic_mismatch"]
    )
    # Add noise
    ys += rng.normal(0.0, noise_std)
    return max(ys, 50.0)  # floor at 50 MPa


def generate_hea_dataset(
    n_samples: int = 200,
    seed: int = 42,
    min_elements: int = 5,
    max_elements: int = 7,
    noise_std: float = 50.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Generate a synthetic HEA dataset.

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
        Standard deviation of Gaussian noise added to yield strength.

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

    available = _ElementDB.available_elements()
    # Bias toward common HEA elements
    common_pool = list(set(_POOL_3D + _POOL_REFRACTORY + _POOL_LIGHT))
    common_pool = [e for e in common_pool if e in available]

    compositions: List[Dict[str, float]] = []
    targets: List[float] = []
    all_elements_set: set = set()

    for i in range(n_samples):
        n_elems = rng.integers(min_elements, max_elements + 1)
        # 80% chance to pick from common pool, 20% from full available
        if rng.random() < 0.8:
            pool = common_pool
        else:
            pool = available
        if len(pool) < n_elems:
            pool = available
        chosen = list(rng.choice(pool, size=n_elems, replace=False))
        comp = _random_composition(chosen, rng)
        compositions.append(comp)
        all_elements_set.update(chosen)
        ys = _yield_strength_proxy(comp, rng, noise_std=noise_std)
        targets.append(ys)

    # Build composition DataFrame (sparse: NaN -> 0)
    all_elems_sorted = sorted(all_elements_set)
    comp_records = []
    for comp in compositions:
        rec = {e: comp.get(e, 0.0) for e in all_elems_sorted}
        comp_records.append(rec)
    compositions_df = pd.DataFrame(comp_records)

    # Build features DataFrame (all features)
    features_df = compute_features(compositions, FeatureSetName.FS_ALL)

    target = pd.Series(targets, name="yield_strength_MPa")

    logger.info(
        "Dataset generated: %d samples, %d elements, features shape %s",
        len(target), len(all_elems_sorted), features_df.shape,
    )
    return compositions_df, features_df, target
