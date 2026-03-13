"""
Feature Validity Evaluation Engine
特徴量妥当性評価エンジン

Scores each feature set along six axes:
  1. Effect size   - performance delta vs FS_BASE
  2. Stability     - variance across seeds / folds
  3. Generalisation - sign consistency between RandomCV and Block splits
  4. Leak suspicion - Random-only improvement with Block degradation
  5. Extrapolation safety - uncertainty behaviour on OOD points
  6. Multicollinearity penalty - VIF-based collinearity penalty (Phase 0)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

from hea_extrapolation_platform.features import FeatureSetName
from hea_extrapolation_platform.workflows import RunResult

if TYPE_CHECKING:
    from hea_extrapolation_platform.multicollinearity import MulticollinearityReport

logger = logging.getLogger(__name__)


# Default weight configuration for ValidityScore.total.
# Positive-direction weights sum to 1.00; penalty weights are subtracted.
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "effect_size": 0.30,
    "stability": 0.20,
    "generalisation": 0.30,
    "leak_penalty": -0.15,
    "extrapolation_safety": 0.20,
    "multicollinearity_penalty": -0.10,
}


@dataclass
class ValidityScore:
    """Feature-set validity score across six dimensions."""

    feature_set: str
    effect_size: float = 0.0
    stability: float = 0.0
    generalisation: float = 0.0
    leak_penalty: float = 0.0
    extrapolation_safety: float = 0.0
    multicollinearity_penalty: float = 0.0

    # Weights can be overridden per-instance via FeatureValidityEvaluator
    _weights: Dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_WEIGHTS), repr=False,
    )

    @property
    def total(self) -> float:
        """Weighted total score (higher = better).

        Uses weights from ``_weights`` dict (configurable via
        ``FeatureValidityEvaluator(weights=...)``).  Default weights:
          effect_size=0.30, stability=0.20, generalisation=0.30,
          leak_penalty=-0.15, extrapolation_safety=0.20,
          multicollinearity_penalty=-0.10.

        Score range:
          - Best case (all 1.0, no penalties):  1.00
          - Worst case (all 0.0, max penalties): -0.25
        """
        w = self._weights
        return (
            w.get("effect_size", 0.30) * self.effect_size
            + w.get("stability", 0.20) * self.stability
            + w.get("generalisation", 0.30) * self.generalisation
            + w.get("leak_penalty", -0.15) * self.leak_penalty
            + w.get("extrapolation_safety", 0.20) * self.extrapolation_safety
            + w.get("multicollinearity_penalty", -0.10) * self.multicollinearity_penalty
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "feature_set": self.feature_set,
            "effect_size": round(self.effect_size, 4),
            "stability": round(self.stability, 4),
            "generalisation": round(self.generalisation, 4),
            "leak_penalty": round(self.leak_penalty, 4),
            "extrapolation_safety": round(self.extrapolation_safety, 4),
            "multicollinearity_penalty": round(self.multicollinearity_penalty, 4),
            "total": round(self.total, 4),
        }


class FeatureValidityEvaluator:
    """Evaluate feature-set validity from a collection of RunResult objects.

    Usage::

        evaluator = FeatureValidityEvaluator()
        scores = evaluator.evaluate(runs, ood_errors)

        # Custom weights:
        evaluator = FeatureValidityEvaluator(weights={
            "effect_size": 0.40,
            "stability": 0.20,
            "generalisation": 0.20,
            "leak_penalty": -0.15,
            "extrapolation_safety": 0.10,
            "multicollinearity_penalty": -0.10,
        })
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self._weights = dict(weights) if weights else dict(_DEFAULT_WEIGHTS)

    def evaluate(
        self,
        runs: List[RunResult],
        ood_errors: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
        mc_reports: Optional[Dict[str, "MulticollinearityReport"]] = None,
    ) -> List[ValidityScore]:
        """Compute validity scores for every feature set present in *runs*.

        Parameters
        ----------
        runs : list of RunResult
            All experiment run results.
        ood_errors : dict, optional
            {feature_set: {"errors": ..., "uncertainties": ..., "is_ood": ...}}
            Per-sample data for extrapolation safety assessment.
        mc_reports : dict, optional
            {feature_set: MulticollinearityReport} from Phase 0.
            Used to compute multicollinearity_penalty.

        Returns
        -------
        list of ValidityScore, sorted by total descending.
        """
        # Group runs by feature set
        fs_runs: Dict[str, List[RunResult]] = {}
        for r in runs:
            fs_runs.setdefault(r.feature_set, []).append(r)

        # Identify baseline (FS_BASE) performance
        base_key = FeatureSetName.FS_BASE.value
        base_rmse = self._mean_test_rmse(fs_runs.get(base_key, []))
        if base_rmse <= 0:
            # FS_BASE has no runs or all-zero RMSE — cannot compute meaningful
            # effect sizes so every feature set will get 0.
            logger.warning(
                "Baseline (FS_BASE) RMSE is 0 or has no runs; "
                "effect_size for all feature sets will be 0. "
                "Check that FS_BASE experiments completed successfully."
            )

        scores: List[ValidityScore] = []
        for fs_name, fs_run_list in fs_runs.items():
            vs = ValidityScore(feature_set=fs_name, _weights=self._weights)

            # 1. Effect size
            fs_rmse = self._mean_test_rmse(fs_run_list)
            if base_rmse > 0:
                vs.effect_size = max(0.0, (base_rmse - fs_rmse) / base_rmse)
            else:
                vs.effect_size = 0.0

            # 2. Stability (inverse of coefficient of variation of RMSE across runs)
            rmses = [
                r.rmse_test for r in fs_run_list
                if r.rmse_test > 0 and np.isfinite(r.rmse_test)
            ]
            if len(rmses) > 1:
                _mean = sum(rmses) / len(rmses)
                _var = sum((x - _mean) ** 2 for x in rmses) / len(rmses)
                _std = _var ** 0.5
                cv = _std / _mean if _mean > 0 else 1.0
                vs.stability = max(0.0, 1.0 - cv)
            else:
                vs.stability = 0.5  # neutral

            # 3. Generalisation (Random vs Block sign consistency)
            # Use exact policy names to avoid accidentally matching
            # ElementExclusion or future split policies.
            random_runs = [r for r in fs_run_list if r.split_policy == "RandomCV"]
            block_runs = [r for r in fs_run_list if r.split_policy == "CompositionBlock"]
            vs.generalisation = self._generalisation_score(random_runs, block_runs, base_rmse)

            # 4. Leak suspicion
            vs.leak_penalty = self._leak_penalty(random_runs, block_runs, base_rmse)

            # 5. Extrapolation safety
            if ood_errors and fs_name in ood_errors:
                vs.extrapolation_safety = self._extrapolation_safety(
                    ood_errors[fs_name]
                )
            else:
                vs.extrapolation_safety = 0.5  # neutral when data not available

            # 6. Multicollinearity penalty (Phase 0)
            if mc_reports and fs_name in mc_reports:
                rpt = mc_reports[fs_name]
                vs.multicollinearity_penalty = min(1.0, rpt.high_vif_ratio)
            else:
                vs.multicollinearity_penalty = 0.0  # no info = neutral

            scores.append(vs)

        scores.sort(key=lambda s: s.total, reverse=True)
        logger.info(
            "Validity evaluation complete. Top feature set: %s (total=%.4f)",
            scores[0].feature_set if scores else "N/A",
            scores[0].total if scores else 0.0,
        )
        return scores

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mean_test_rmse(runs: List[RunResult]) -> float:
        if not runs:
            return 0.0
        # Use pure Python to avoid numpy C-extension SIGSEGV on
        # pandas 3.0 F-contiguous memory (even for list-of-floats).
        vals = [float(r.rmse_test) for r in runs]
        return sum(vals) / len(vals)

    @staticmethod
    def _generalisation_score(
        random_runs: List[RunResult],
        block_runs: List[RunResult],
        base_rmse: float,
    ) -> float:
        """Score [0, 1]: both splits improve -> 1, divergent -> 0."""
        if not random_runs or not block_runs or base_rmse <= 0:
            return 0.5
        _rand_vals = [float(r.rmse_test) for r in random_runs]
        _block_vals = [float(r.rmse_test) for r in block_runs]
        rand_rmse = sum(_rand_vals) / len(_rand_vals)
        block_rmse = sum(_block_vals) / len(_block_vals)
        rand_improve = (base_rmse - rand_rmse) / base_rmse
        block_improve = (base_rmse - block_rmse) / base_rmse
        # Both improve -> high score; divergent -> low.
        # Use a small tolerance for "no change" to avoid dead-code with
        # exact floating-point zero comparisons.
        _eps = 1e-9
        if rand_improve > _eps and block_improve > _eps:
            # Use geometric mean instead of min to avoid bottleneck
            # (Review: min causes asymmetric improvements to be undervalued)
            geo_mean = math.sqrt(rand_improve * block_improve)
            return min(1.0, geo_mean)
        elif rand_improve < -_eps and block_improve < -_eps:
            # Both degrade -> low but not worst
            return 0.3
        elif abs(rand_improve) <= _eps and abs(block_improve) <= _eps:
            # Negligible change from baseline is neutral
            return 0.5
        else:
            return 0.1  # divergent (one improves, other degrades)

    @staticmethod
    def _leak_penalty(
        random_runs: List[RunResult],
        block_runs: List[RunResult],
        base_rmse: float,
    ) -> float:
        """Detect leak: Random improves a lot but Block degrades."""
        if not random_runs or not block_runs or base_rmse <= 0:
            return 0.0
        _rand_vals = [float(r.rmse_test) for r in random_runs]
        _block_vals = [float(r.rmse_test) for r in block_runs]
        rand_rmse = sum(_rand_vals) / len(_rand_vals)
        block_rmse = sum(_block_vals) / len(_block_vals)
        rand_improve = (base_rmse - rand_rmse) / base_rmse
        block_change = (base_rmse - block_rmse) / base_rmse
        if rand_improve > 0.05 and block_change < -0.02:
            return min(1.0, rand_improve - block_change)
        return 0.0

    @staticmethod
    def _extrapolation_safety(ood_data: Dict[str, np.ndarray]) -> float:
        """Score [0, 1] based on uncertainty + error behaviour on OOD points.

        Desired: uncertainty increases for OOD, errors do not explode.
        """
        errors = np.ascontiguousarray(np.asarray(ood_data.get("errors", [])))
        uncertainties = np.ascontiguousarray(np.asarray(ood_data.get("uncertainties", [])))
        is_ood = np.ascontiguousarray(np.asarray(ood_data.get("is_ood", []), dtype=bool))

        if len(errors) == 0 or is_ood.sum() == 0 or (~is_ood).sum() == 0:
            return 0.5

        ood_err = np.abs(errors[is_ood])
        id_err = np.abs(errors[~is_ood])
        # Ratio of OOD error to ID error (want < 2x)
        ratio = ood_err.mean() / max(id_err.mean(), 1e-6)
        err_score = max(0.0, 1.0 - (ratio - 1.0) / 2.0)

        # Uncertainty should increase for OOD — use gradient score
        # (Review: binary score doesn't reflect magnitude of uncertainty increase)
        if len(uncertainties) > 0 and is_ood.sum() > 0 and (~is_ood).sum() > 0:
            ood_unc = float(uncertainties[is_ood].mean())
            id_unc = float(uncertainties[~is_ood].mean())
            if id_unc > 1e-10:
                unc_ratio = ood_unc / id_unc
                # ratio<1 → 0.0, ratio=1 → 0.5, ratio=2 → 1.0
                unc_score = max(0.0, min(1.0, 0.5 * unc_ratio - 0.5)) if unc_ratio >= 1.0 else 0.0
            else:
                unc_score = 1.0 if ood_unc > 1e-10 else 0.5
        else:
            unc_score = 0.5

        return 0.6 * err_score + 0.4 * unc_score
