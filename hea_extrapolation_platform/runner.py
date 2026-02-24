"""
Experiment Runner for Extrapolation Discovery Platform
実験オーケストレータ

Orchestrates the full experiment grid:
  3 workflows x 3 split policies x 3 seeds x 5 feature sets = 135 runs

Provides MLflow-style run tracking with:
  - Run registry (in-memory + JSON export)
  - Progress logging
  - Single-pass execution (all outputs captured in one run)

NOTE: HEA is used as a concrete example; the runner is domain-agnostic.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from hea_extrapolation_platform.features import (
    FeatureCatalog,
    FeatureSetName,
    compute_features,
)
from hea_extrapolation_platform.splitters import (
    BaseSplitter,
    CompositionBlockSplitter,
    ElementExclusionSplitter,
    RandomCVSplitter,
)
from hea_extrapolation_platform.workflows import (
    BaseWorkflow,
    RunResult,
    WorkflowENS,
    WorkflowLIN,
    WorkflowXGB,
)
from hea_extrapolation_platform.ood import OODDetector, OODResult
from hea_extrapolation_platform.evaluation import FeatureValidityEvaluator, ValidityScore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Run Registry (MLflow-style)
# ---------------------------------------------------------------------------

class RunRegistry:
    """In-memory registry of experiment runs with JSON export."""

    def __init__(self) -> None:
        self._runs: List[RunResult] = []

    def add(self, run: RunResult) -> None:
        self._runs.append(run)

    @property
    def runs(self) -> List[RunResult]:
        return list(self._runs)

    def __len__(self) -> int:
        return len(self._runs)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all runs to a summary DataFrame."""
        records = []
        for r in self._runs:
            rec = {
                "workflow": r.workflow,
                "feature_set": r.feature_set,
                "split_policy": r.split_policy,
                "seed": r.seed,
                "fold": r.fold,
                "rmse_train": r.rmse_train,
                "rmse_test": r.rmse_test,
                "mae_train": r.mae_train,
                "mae_test": r.mae_test,
                "r2_train": r.r2_train,
                "r2_test": r.r2_test,
                "elapsed_sec": r.elapsed_sec,
            }
            records.append(rec)
        return pd.DataFrame(records)

    def export_json(self, path: Path) -> None:
        """Export run summaries to JSON."""
        df = self.to_dataframe()
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_json(path, orient="records", indent=2, force_ascii=False)
        logger.info("Exported %d runs to %s", len(self._runs), path)


# ---------------------------------------------------------------------------
# Experiment Runner
# ---------------------------------------------------------------------------

class ExperimentRunner:
    """Orchestrate the full experiment grid.

    Parameters
    ----------
    seeds : list of int
        Random seeds for reproducibility (default [42, 123, 456]).
    quick : bool
        If True, use reduced hyperparameter grids for faster execution.
    exclude_elements : list of str
        Elements to use for ElementExclusion splits (default ["Co", "Ni", "Ti"]).
    """

    def __init__(
        self,
        seeds: Optional[List[int]] = None,
        quick: bool = False,
        exclude_elements: Optional[List[str]] = None,
    ) -> None:
        self._seeds = seeds or [42, 123, 456]
        self._quick = quick
        self._exclude_elements = exclude_elements or ["Co", "Ni", "Ti"]
        self._registry = RunRegistry()

    @property
    def registry(self) -> RunRegistry:
        return self._registry

    def _build_workflows(self) -> Dict[str, BaseWorkflow]:
        return {
            "WF-LIN": WorkflowLIN(),
            "WF-XGB": WorkflowXGB(quick=self._quick),
            "WF-ENS": WorkflowENS(n_members=3 if self._quick else 5, quick=self._quick),
        }

    def _build_splitters(self, seed: int) -> Dict[str, BaseSplitter]:
        return {
            "RandomCV": RandomCVSplitter(n_folds=5, seed=seed),
            "CompositionBlock": CompositionBlockSplitter(n_folds=5, seed=seed),
            "ElementExclusion": ElementExclusionSplitter(
                target_elements=self._exclude_elements,
            ),
        }

    def run(
        self,
        compositions_df: pd.DataFrame,
        features_all: pd.DataFrame,
        target: pd.Series,
    ) -> Tuple[List[RunResult], List[ValidityScore], Dict[str, OODResult]]:
        """Execute the full experiment grid.

        Parameters
        ----------
        compositions_df : pd.DataFrame
            Composition table (element columns, fraction values).
        features_all : pd.DataFrame
            All features (FS_ALL columns) for each sample.
        target : pd.Series
            Target variable (e.g. yield strength).

        Returns
        -------
        runs : list of RunResult
        validity_scores : list of ValidityScore
        ood_results : dict of {feature_set_name: OODResult}
        """
        t_start = time.time()
        workflows = self._build_workflows()
        feature_sets = FeatureCatalog.list_sets()

        total_expected = (
            len(workflows)
            * len(self._seeds)
            * len(feature_sets)
            * 8  # rough upper bound on total folds across 3 splitters
        )
        logger.info(
            "Starting experiment: %d workflows x %d seeds x %d feature sets "
            "(estimated ~%d runs)",
            len(workflows), len(self._seeds), len(feature_sets), total_expected,
        )

        ood_results: Dict[str, OODResult] = {}
        ood_errors_for_eval: Dict[str, Dict[str, np.ndarray]] = {}

        run_count = 0

        for seed in self._seeds:
            splitters = self._build_splitters(seed)

            for fs_name in feature_sets:
                # Select feature columns for this set
                cols = FeatureCatalog.columns(fs_name)
                X_fs = features_all[cols].copy()

                for sp_name, splitter in splitters.items():
                    fold_idx = 0
                    for train_idx, test_idx in splitter.split(
                        X_fs, target, compositions=compositions_df
                    ):
                        X_train = X_fs.iloc[train_idx].reset_index(drop=True)
                        X_test = X_fs.iloc[test_idx].reset_index(drop=True)
                        y_train = target.iloc[train_idx].reset_index(drop=True)
                        y_test = target.iloc[test_idx].reset_index(drop=True)

                        for wf_name, wf in workflows.items():
                            try:
                                result = wf.run(
                                    X_train, y_train, X_test, y_test,
                                    seed=seed,
                                    feature_set=fs_name.value,
                                    split_policy=sp_name,
                                    fold=fold_idx,
                                    test_indices=test_idx,
                                )
                                self._registry.add(result)
                                run_count += 1

                                if run_count % 20 == 0:
                                    logger.info(
                                        "Progress: %d runs completed (%.1f sec)",
                                        run_count, time.time() - t_start,
                                    )
                            except Exception:
                                logger.exception(
                                    "Run failed: wf=%s fs=%s sp=%s seed=%d fold=%d",
                                    wf_name, fs_name.value, sp_name, seed, fold_idx,
                                )

                        fold_idx += 1

                # OOD detection per feature set (using last seed's full data)
                fs_key = fs_name.value
                if fs_key not in ood_results:
                    try:
                        detector = OODDetector(k=10)
                        detector.fit(X_fs)
                        ood_res = detector.score(X_fs)
                        ood_results[fs_key] = ood_res

                        # Collect OOD error data for evaluation
                        # Use ensemble runs for uncertainty
                        ens_runs = [
                            r for r in self._registry.runs
                            if r.feature_set == fs_key and r.workflow == "WF-ENS"
                        ]
                        if ens_runs:
                            last_ens = ens_runs[-1]
                            pred_std = np.array(
                                last_ens.artifacts.get("pred_std_test", [])
                            )
                            if (
                                last_ens.y_test_true is not None
                                and last_ens.y_test_pred is not None
                                and last_ens.test_indices is not None
                                and len(pred_std) > 0
                            ):
                                errors = last_ens.y_test_true - last_ens.y_test_pred
                                is_ood_test = ood_res.is_ood[last_ens.test_indices]
                                ood_errors_for_eval[fs_key] = {
                                    "errors": errors,
                                    "uncertainties": pred_std,
                                    "is_ood": is_ood_test,
                                }
                    except Exception:
                        logger.exception("OOD detection failed for %s", fs_key)

        # Evaluation
        evaluator = FeatureValidityEvaluator()
        validity_scores = evaluator.evaluate(
            self._registry.runs,
            ood_errors=ood_errors_for_eval,
        )

        elapsed = time.time() - t_start
        logger.info(
            "Experiment complete: %d runs in %.1f sec. Top feature set: %s",
            run_count, elapsed,
            validity_scores[0].feature_set if validity_scores else "N/A",
        )

        return self._registry.runs, validity_scores, ood_results

    def export(self, out_dir: Path) -> None:
        """Export run registry to JSON."""
        self._registry.export_json(out_dir / "run_registry.json")
