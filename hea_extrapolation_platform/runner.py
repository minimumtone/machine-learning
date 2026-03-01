"""
Experiment Runner for Extrapolation Discovery Platform
実験オーケストレータ

Orchestrates the full experiment grid:
  3 workflows x 3 split policies x 3 seeds x 5 feature sets = 135 runs

Provides experiment tracking with:
  - Run registry (in-memory + JSON export)
  - MLflow integration (optional — falls back to in-memory)
  - Feast feature store integration (optional — falls back to FeatureCatalog)
  - MInt workflow adapter integration (optional — falls back to built-in)
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
from hea_extrapolation_platform.integrations.mlflow_tracker import (
    MLflowTracker,
    is_mlflow_available,
)
from hea_extrapolation_platform.integrations.feast_store import (
    FeastFeatureStore,
    is_feast_available,
)
from hea_extrapolation_platform.integrations.mint_adapter import (
    MIntWorkflowAdapter,
    MIntWorkflowConfig,
    MIntWorkflowRegistry,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Run Registry (MLflow-style)
# ---------------------------------------------------------------------------

class RunRegistry:
    """In-memory registry of experiment runs with JSON export."""

    def __init__(self) -> None:
        self._runs: List[RunResult] = []

    def reset(self) -> None:
        """Clear all stored runs (useful when calling run() multiple times)."""
        self._runs.clear()

    def add(self, run: RunResult) -> None:
        self._runs.append(run)

    @property
    def runs(self) -> List[RunResult]:
        return list(self._runs)

    def __len__(self) -> int:
        return len(self._runs)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all runs to a summary DataFrame.

        Uses columnar (dict-of-lists) construction to avoid DataFrame
        fragmentation that can cause SIGSEGV in numpy/pandas C layer.
        """
        if not self._runs:
            return pd.DataFrame()
        col_names = [
            "workflow", "feature_set", "split_policy", "seed", "fold",
            "rmse_train", "rmse_test", "mae_train", "mae_test",
            "r2_train", "r2_test", "elapsed_sec",
        ]
        columns: dict = {k: [] for k in col_names}
        for r in self._runs:
            columns["workflow"].append(r.workflow)
            columns["feature_set"].append(r.feature_set)
            columns["split_policy"].append(r.split_policy)
            columns["seed"].append(r.seed)
            columns["fold"].append(r.fold)
            columns["rmse_train"].append(r.rmse_train)
            columns["rmse_test"].append(r.rmse_test)
            columns["mae_train"].append(r.mae_train)
            columns["mae_test"].append(r.mae_test)
            columns["r2_train"].append(r.r2_train)
            columns["r2_test"].append(r.r2_test)
            columns["elapsed_sec"].append(r.elapsed_sec)
        return pd.DataFrame(columns)

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
    mlflow_tracker : MLflowTracker or None
        Optional MLflow tracker for experiment logging.  If None and
        ``use_mlflow=True``, a default tracker is created.
    feature_store : FeastFeatureStore or None
        Optional Feast feature store for feature management.
    mint_registry : MIntWorkflowRegistry or None
        Optional MInt workflow registry.  If provided, MInt workflows
        are executed *in addition to* built-in workflows.
    use_mlflow : bool
        If True, create a default MLflow tracker when none is provided.
    use_feast : bool
        If True, create a default Feast store when none is provided.
    use_mint : bool
        If True, create a default MInt registry when none is provided.
    """

    def __init__(
        self,
        seeds: Optional[List[int]] = None,
        quick: bool = False,
        exclude_elements: Optional[List[str]] = None,
        mlflow_tracker: Optional[MLflowTracker] = None,
        feature_store: Optional[FeastFeatureStore] = None,
        mint_registry: Optional[MIntWorkflowRegistry] = None,
        use_mlflow: bool = False,
        use_feast: bool = False,
        use_mint: bool = False,
    ) -> None:
        self._seeds = seeds or [42, 123, 456]
        self._quick = quick
        self._exclude_elements = exclude_elements or ["Co", "Ni", "Ti"]
        self._registry = RunRegistry()
        self._ood_split_indices: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        # MLflow tracker
        if mlflow_tracker is not None:
            self._tracker = mlflow_tracker
        elif use_mlflow:
            self._tracker = MLflowTracker(
                experiment_name="extrapolation_discovery",
                enabled=True,
            )
        else:
            self._tracker = MLflowTracker(enabled=False)

        # Feast feature store
        if feature_store is not None:
            self._feature_store = feature_store
        elif use_feast:
            self._feature_store = FeastFeatureStore(enabled=True)
        else:
            self._feature_store = FeastFeatureStore(enabled=False)

        # MInt workflow registry
        if mint_registry is not None:
            self._mint_registry = mint_registry
        elif use_mint:
            self._mint_registry = MIntWorkflowRegistry.create_default()
        else:
            self._mint_registry = None

    @property
    def registry(self) -> RunRegistry:
        return self._registry

    @property
    def tracker(self) -> MLflowTracker:
        """Return the MLflow tracker instance."""
        return self._tracker

    @property
    def feature_store(self) -> FeastFeatureStore:
        """Return the Feast feature store instance."""
        return self._feature_store

    @property
    def mint_registry(self) -> Optional[MIntWorkflowRegistry]:
        """Return the MInt workflow registry (or None)."""
        return self._mint_registry

    @property
    def ood_split_indices(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Return {feature_set: (train_indices, test_indices)} used for OOD."""
        return self._ood_split_indices

    def _build_workflows(
        self,
        selected_workflows: Optional[List[str]] = None,
    ) -> Dict[str, BaseWorkflow]:
        """Build workflow instances.

        Parameters
        ----------
        selected_workflows : list of str, optional
            Workflow names to include (e.g. ["WF-LIN", "WF-XGB"]).
            If *None* or empty, all built-in workflows are used.
        """
        all_wfs: Dict[str, BaseWorkflow] = {
            "WF-LIN": WorkflowLIN(),
            "WF-XGB": WorkflowXGB(quick=self._quick),
            "WF-ENS": WorkflowENS(n_members=3 if self._quick else 5, quick=self._quick),
        }
        if selected_workflows:
            filtered = {
                k: v for k, v in all_wfs.items()
                if k in selected_workflows
            }
            return filtered if filtered else all_wfs
        return all_wfs

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
        progress_callback: Optional[Any] = None,
        selected_workflows: Optional[List[str]] = None,
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
        progress_callback : callable, optional
            Called as ``progress_callback(completed, total, message)``
            after each individual run to allow the caller (e.g. a GUI)
            to display granular progress.  *completed* and *total* are
            ints; *message* is a short status string.
        selected_workflows : list of str, optional
            Workflow names to run (e.g. ["WF-LIN", "WF-XGB"]).
            If *None*, all built-in workflows are used.

        Returns
        -------
        runs : list of RunResult
        validity_scores : list of ValidityScore
        ood_results : dict of {feature_set_name: OODResult}
        """
        t_start = time.time()
        workflows = self._build_workflows(selected_workflows)

        # Add MInt workflows if registry is provided
        if self._mint_registry is not None:
            for wf_info in self._mint_registry.list_workflows():
                wf_name = wf_info["name"]
                if wf_name not in workflows:
                    adapter = self._mint_registry.get_adapter(wf_name)
                    workflows[wf_name] = adapter
                    logger.info("Added MInt workflow: %s", wf_name)

        feature_sets = FeatureCatalog.list_sets()

        # Store features in Feast store if enabled
        self._feature_store.store_features(features_all)

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

        # Start MLflow parent run for the entire experiment
        self._tracker.start_run(
            run_name=f"experiment_{int(t_start)}",
            tags={
                "seeds": str(self._seeds),
                "quick": str(self._quick),
                "n_feature_sets": str(len(feature_sets)),
                "n_workflows": str(len(workflows)),
                "mlflow_active": str(self._tracker.is_mlflow_active),
                "feast_active": str(self._feature_store.is_feast_active),
                "mint_active": str(self._mint_registry is not None),
            },
        )

        ood_results: Dict[str, OODResult] = {}
        # Reset per-run state so repeated calls don't accumulate.
        self._registry.reset()
        self._ood_split_indices.clear()
        ood_errors_for_eval: Dict[str, Dict[str, np.ndarray]] = {}

        run_count = 0
        fail_count = 0

        try:
            for seed in self._seeds:
                splitters = self._build_splitters(seed)

                for fs_name in feature_sets:
                    # Select feature columns for this set.
                    # Rebuild from numpy to get a single contiguous block;
                    # column-subset slicing on a wide DataFrame (150+ cols)
                    # creates a fragmented BlockManager that triggers
                    # PerformanceWarning and can SIGSEGV downstream.
                    cols = FeatureCatalog.columns(fs_name)
                    _arr = features_all[cols].to_numpy(dtype="float64")
                    X_fs = pd.DataFrame(_arr, columns=cols, index=features_all.index)

                    for sp_name, splitter in splitters.items():
                        fold_idx = 0
                        for train_idx, test_idx in splitter.split(
                            X_fs, target, compositions=compositions_df
                        ):
                            # Rebuild from numpy after iloc to guarantee
                            # a single contiguous memory block; iloc on a
                            # wide DataFrame creates fragmented views that
                            # SIGSEGV when .values is accessed downstream.
                            X_train = pd.DataFrame(
                                X_fs.iloc[train_idx].to_numpy(dtype="float64"),
                                columns=cols,
                            )
                            X_test = pd.DataFrame(
                                X_fs.iloc[test_idx].to_numpy(dtype="float64"),
                                columns=cols,
                            )
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

                                    # Notify caller of progress
                                    if progress_callback is not None:
                                        try:
                                            progress_callback(
                                                run_count,
                                                total_expected,
                                                f"{wf_name} | {fs_name.value} | "
                                                f"{sp_name} fold {fold_idx}",
                                            )
                                        except Exception:
                                            pass  # never let callback errors stop the experiment
                                except Exception:
                                    fail_count += 1
                                    logger.exception(
                                        "Run failed: wf=%s fs=%s sp=%s seed=%d fold=%d",
                                        wf_name, fs_name.value, sp_name, seed, fold_idx,
                                    )

                            fold_idx += 1

                    # OOD detection per feature set (dedicated split).
                    fs_key = fs_name.value
                    if fs_key not in ood_results:
                        ood_res = self._run_ood_detection(
                            X_fs, target, compositions_df, fs_key,
                        )
                        if ood_res is not None:
                            ood_results[fs_key] = ood_res
                            self._collect_ood_errors_for_eval(
                                fs_key, ood_res, ood_errors_for_eval,
                            )

            # Evaluation
            evaluator = FeatureValidityEvaluator()
            validity_scores = evaluator.evaluate(
                self._registry.runs,
                ood_errors=ood_errors_for_eval,
            )

            elapsed = time.time() - t_start
            logger.info(
                "Experiment complete: %d runs (%d failed) in %.1f sec. "
                "Top feature set: %s",
                run_count, fail_count, elapsed,
                validity_scores[0].feature_set if validity_scores else "N/A",
            )

            # Log experiment summary to MLflow
            self._tracker.log_experiment_summary(
                n_runs=run_count,
                validity_scores=validity_scores,
                ood_results=ood_results,
                elapsed_sec=elapsed,
            )
            self._tracker.end_run()

        except Exception:
            # Ensure MLflow run is properly closed even on failure
            self._tracker.end_run(status="FAILED")
            raise

        return self._registry.runs, validity_scores, ood_results

    # ------------------------------------------------------------------
    # Private helpers extracted from run() for readability
    # ------------------------------------------------------------------

    def _run_ood_detection(
        self,
        X_fs: pd.DataFrame,
        target: pd.Series,
        compositions_df: pd.DataFrame,
        fs_key: str,
    ) -> Optional[OODResult]:
        """Run OOD detection on *one* feature set.

        Uses a dedicated RandomCV split (first seed, first fold) so OOD
        detection is deterministic and independent of the main experiment
        loop.

        Returns
        -------
        OODResult or None if detection failed.
        """
        try:
            ood_splitter = RandomCVSplitter(n_folds=5, seed=self._seeds[0])
            ood_folds = list(ood_splitter.split(
                X_fs, target, compositions=compositions_df
            ))
            ood_train_idx, ood_test_idx = ood_folds[0]  # first fold
            # Rebuild from numpy after iloc to avoid fragmented
            # BlockManager that can SIGSEGV in .values calls.
            _cols = X_fs.columns
            X_train_ood = pd.DataFrame(
                X_fs.iloc[ood_train_idx].to_numpy(dtype="float64"),
                columns=_cols,
            )
            X_test_ood = pd.DataFrame(
                X_fs.iloc[ood_test_idx].to_numpy(dtype="float64"),
                columns=_cols,
            )

            detector = OODDetector(k=10)
            detector.fit(X_train_ood)
            ood_res = detector.score(X_test_ood)

            self._ood_split_indices[fs_key] = (
                np.asarray(ood_train_idx),
                np.asarray(ood_test_idx),
            )
            return ood_res
        except Exception:
            logger.exception("OOD detection failed for %s", fs_key)
            return None

    def _collect_ood_errors_for_eval(
        self,
        fs_key: str,
        ood_res: OODResult,
        ood_errors_for_eval: Dict[str, Dict[str, np.ndarray]],
    ) -> None:
        """Match an ENS run to the OOD partition and collect prediction errors.

        The matched ENS run must share *exactly* the same ``test_indices``
        as the OOD split to guarantee a valid comparison.
        """
        ood_test_indices = self._ood_split_indices[fs_key][1]

        ens_runs = [
            r for r in self._registry.runs
            if r.feature_set == fs_key and r.workflow == "WF-ENS"
               and r.test_indices is not None
        ]
        matched_ens = None
        for er in reversed(ens_runs):
            if np.array_equal(
                np.asarray(er.test_indices), ood_test_indices
            ):
                matched_ens = er
                break

        if matched_ens is not None:
            pred_std = np.array(
                matched_ens.artifacts.get("pred_std_test", [])
            )
            if (
                matched_ens.y_test_true is not None
                and matched_ens.y_test_pred is not None
                and len(pred_std) > 0
            ):
                errors = (
                    matched_ens.y_test_true
                    - matched_ens.y_test_pred
                )
                ood_errors_for_eval[fs_key] = {
                    "errors": errors,
                    "uncertainties": pred_std,
                    "is_ood": ood_res.is_ood,
                }
        else:
            logger.info(
                "No ENS run with matching test_indices "
                "for OOD eval on %s", fs_key,
            )

    def export(self, out_dir: Path) -> None:
        """Export run registry to JSON."""
        self._registry.export_json(out_dir / "run_registry.json")
