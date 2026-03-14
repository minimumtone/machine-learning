"""Experiment Runner — orchestrates the 7-phase ML pipeline.

Phases: 1 Multicollinearity  2 Feature-selection  3 Fold-precompute
        4 Job-list build     5 Parallel training  6 OOD  7 Evaluate
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
from collections import defaultdict
try:
    import resource
    _HAS_RESOURCE = True
except ImportError:
    _HAS_RESOURCE = False  # Windows
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd

from extrapolation_discovery_platform.features import (
    FeatureCatalog,
    FeatureSetName,
    compute_features,
)
from extrapolation_discovery_platform.splitters import (
    BaseSplitter,
    CompositionBlockSplitter,
    ElementExclusionSplitter,
    RandomCVSplitter,
)
from extrapolation_discovery_platform.workflows import (
    BaseWorkflow,
    RunResult,
    WorkflowARD,
    WorkflowENS,
    WorkflowLASSO,
    WorkflowLIN,
    WorkflowRF,
    WorkflowXGB,
)
from extrapolation_discovery_platform.ood import OODDetector, OODResult
from extrapolation_discovery_platform.evaluation import FeatureValidityEvaluator, ValidityScore
from extrapolation_discovery_platform.integrations.mlflow_tracker import (
    MLflowTracker,
    is_mlflow_available,
)
from extrapolation_discovery_platform.integrations.feast_store import (
    FeastFeatureStore,
    is_feast_available,
)
from extrapolation_discovery_platform.integrations.mint_adapter import (
    MIntWorkflowAdapter,
    MIntWorkflowConfig,
    MIntWorkflowRegistry,
)
from extrapolation_discovery_platform.multicollinearity import (
    MulticollinearityReport,
    run_phase0_multicollinearity,
)
from extrapolation_discovery_platform.feature_selection import run_feature_selection
from extrapolation_discovery_platform.model_selection import (
    ModelSelectionResult,
    run_model_selection,
)
from extrapolation_discovery_platform._compat import as_serializable

logger = logging.getLogger(__name__)


from extrapolation_discovery_platform._utils import safe_array  # noqa: E402


class RunRegistry:
    """In-memory registry of experiment runs with JSON export."""

    def __init__(self) -> None:
        self._runs: List[RunResult] = []

    def reset(self) -> None:
        self._runs.clear()

    def add(self, run: RunResult) -> None:
        self._runs.append(run)

    def add_many(self, runs: List[RunResult]) -> None:
        self._runs.extend(runs)

    @property
    def runs(self) -> List[RunResult]:
        return list(self._runs)

    def __len__(self) -> int:
        return len(self._runs)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all runs to a summary DataFrame."""
        import dataclasses as _dc

        if not self._runs:
            return pd.DataFrame()

        _SKIP_TYPES = (np.ndarray, dict)
        col_names = [
            f.name for f in _dc.fields(RunResult)
            if not isinstance(getattr(self._runs[0], f.name, None), _SKIP_TYPES)
            and getattr(self._runs[0], f.name, None) is not None
        ]
        columns: Dict[str, list] = {k: [] for k in col_names}
        for r in self._runs:
            for col in col_names:
                columns[col].append(getattr(r, col))
        return pd.DataFrame(columns)

    def export_json(self, path: Path) -> None:
        """Export runs to JSON (numpy-safe serialization)."""
        df = self.to_dataframe()
        path.parent.mkdir(parents=True, exist_ok=True)
        records = as_serializable(df.to_dict(orient="records"))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        logger.info("Exported %d runs to %s", len(self._runs), path)


# ---------------------------------------------------------------------------
# Internal job descriptor — lightweight, picklable
# ---------------------------------------------------------------------------

class _Job(NamedTuple):
    """Lightweight, picklable job descriptor for one training run."""
    wf_name: str
    fs_name: str
    sp_name: str
    seed: int
    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    quick: bool
    dim_reduction: bool = True
    force_pca: bool = False  # moderate VIF: force PCA on for linear models


def _run_job(
    job: _Job,
    X_fs: np.ndarray,
    feature_cols: List[str],
    y: np.ndarray,
    mint_configs: Optional[Dict[str, "MIntWorkflowConfig"]] = None,
) -> RunResult:
    """Execute one training run (pure, picklable)."""
    import os as _os
    import pandas as _pd

    # Signal to inner estimators (GridSearchCV etc.) that we are inside
    # a ProcessPoolExecutor worker — they must use n_jobs=1 to avoid
    # double parallelization and thread contention.
    _os.environ["_EDP_INSIDE_WORKER"] = "1"

    # Row slices of a C-contiguous 2-D array are themselves contiguous.
    # Wrap them in DataFrames only for the workflow API.
    X_train = _pd.DataFrame(X_fs[job.train_idx], columns=feature_cols)
    X_test  = _pd.DataFrame(X_fs[job.test_idx],  columns=feature_cols)
    y_train = _pd.Series(y[job.train_idx])
    y_test  = _pd.Series(y[job.test_idx])

    from extrapolation_discovery_platform.workflows import (
        WorkflowARD, WorkflowENS, WorkflowLASSO, WorkflowLIN, WorkflowRF,
        WorkflowXGB,
    )
    _dr = job.dim_reduction
    _pca = _dr or job.force_pca  # force_pca=True → always PCA ON (WF-LIN only)

    # Lazy instantiation: create only the one workflow needed for this job.
    # Previously all 6 workflows were instantiated per job, wasting resources
    # when running hundreds of parallel jobs via ProcessPoolExecutor.
    _BUILTIN_FACTORIES = {
        "WF-LIN":   lambda: WorkflowLIN(dim_reduction=_pca),
        "WF-LASSO": lambda: WorkflowLASSO(dim_reduction=_dr),
        "WF-ARD":   lambda: WorkflowARD(dim_reduction=_dr),
        "WF-RF":    lambda: WorkflowRF(quick=job.quick, dim_reduction=_dr),
        "WF-XGB":   lambda: WorkflowXGB(quick=job.quick, dim_reduction=_dr),
        "WF-ENS":   lambda: WorkflowENS(
            n_members=3 if job.quick else 5, quick=job.quick,
            dim_reduction=_dr,
        ),
    }

    if job.wf_name in _BUILTIN_FACTORIES:
        wf = _BUILTIN_FACTORIES[job.wf_name]()
    elif mint_configs is not None and job.wf_name in mint_configs:
        from extrapolation_discovery_platform.integrations.mint_adapter import (
            MIntWorkflowAdapter,
        )
        wf = MIntWorkflowAdapter(config=mint_configs[job.wf_name])
    else:
        raise KeyError(
            f"Unknown workflow '{job.wf_name}'. "
            f"Built-in: {list(_BUILTIN_FACTORIES)}, MInt: {list(mint_configs or {})}"
        )

    return wf.run(
        X_train, y_train, X_test, y_test,
        seed=job.seed,
        feature_set=job.fs_name,
        split_policy=job.sp_name,
        fold=job.fold,
        test_indices=job.test_idx,
    )


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
        Reduced HPO grids for faster execution.
    exclude_elements : list of str
        Elements to use for ElementExclusion splits.
    n_workers : int or None
        Parallel worker processes.  ``1`` → serial (debug-friendly).
        ``None`` → ``os.cpu_count()``.
    """

    def __init__(
        self,
        seeds: Optional[List[int]] = None,
        quick: bool = False,
        exclude_elements: Optional[List[str]] = None,
        n_workers: Optional[int] = 1,
        mlflow_tracker: Optional[MLflowTracker] = None,
        feature_store: Optional[FeastFeatureStore] = None,
        mint_registry: Optional[MIntWorkflowRegistry] = None,
        use_mlflow: bool = False,
        use_feast: bool = False,
        use_mint: bool = False,
        dim_reduction: bool = True,
        leak_auto_exclude: bool = True,
        leak_corr_threshold: float = 0.85,
    ) -> None:
        self._seeds = seeds or [42, 123, 456]
        self._quick = quick
        self._dim_reduction = dim_reduction
        self._leak_auto_exclude = leak_auto_exclude
        self._leak_corr_threshold = leak_corr_threshold
        self._exclude_elements = exclude_elements or ["Co", "Ni", "Ti"]
        self._n_workers = n_workers if n_workers is not None else os.cpu_count()

        # Guard against n_jobs over-subscription (Bug #3):
        # If ProcessPoolExecutor workers AND inner estimators both use
        # parallelism, CPU contention prevents model convergence and
        # produces unstable / suppressed R² values.
        _inner = int(os.environ.get("HEA_INNER_N_JOBS", "1") or "1")
        if self._n_workers > 1 and _inner > 1:
            logger.warning(
                "Over-subscription detected: n_workers=%d × HEA_INNER_N_JOBS=%d. "
                "This causes CPU contention that can suppress R² scores. "
                "Set HEA_INNER_N_JOBS=1 (default) when using parallel workers.",
                self._n_workers, _inner,
            )

        self._registry = RunRegistry()
        self._ood_split_indices: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._mc_reports: Optional[Dict[str, MulticollinearityReport]] = None
        self._model_selection_result: Optional[ModelSelectionResult] = None
        self._effective_cols: Dict[str, List[str]] = {}

        # Unified interface: passing an object implies "use it".
        # The boolean flags (use_mlflow, use_feast, use_mint) are kept for
        # backward compatibility but are redundant when the object is given.
        if mlflow_tracker is not None:
            use_mlflow = True
            self._tracker = mlflow_tracker
        elif use_mlflow:
            self._tracker = MLflowTracker(
                experiment_name="extrapolation_discovery", enabled=True,
            )
        else:
            self._tracker = MLflowTracker(enabled=False)

        if feature_store is not None:
            use_feast = True
            self._feature_store = feature_store
        elif use_feast:
            self._feature_store = FeastFeatureStore(enabled=True)
        else:
            self._feature_store = FeastFeatureStore(enabled=False)

        if mint_registry is not None:
            use_mint = True
            self._mint_registry = mint_registry
        elif use_mint:
            self._mint_registry = MIntWorkflowRegistry.create_default()
        else:
            self._mint_registry = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def registry(self) -> RunRegistry:
        return self._registry

    @property
    def tracker(self) -> MLflowTracker:
        return self._tracker

    @property
    def feature_store(self) -> FeastFeatureStore:
        return self._feature_store

    @property
    def mint_registry(self) -> Optional[MIntWorkflowRegistry]:
        return self._mint_registry

    @property
    def ood_split_indices(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        return self._ood_split_indices

    @property
    def mc_reports(self) -> Optional[Dict[str, MulticollinearityReport]]:
        """Multicollinearity reports from Phase 1 (None before run)."""
        return self._mc_reports

    @property
    def model_selection_result(self) -> Optional[ModelSelectionResult]:
        """Model selection result (None if not run)."""
        return self._model_selection_result

    @property
    def fs_summaries(self) -> Dict[str, Any]:
        """Feature selection summaries from Phase 2 (empty before run)."""
        return getattr(self, "_fs_summaries", {})

    @property
    def effective_cols(self) -> Dict[str, List[str]]:
        """Effective columns per feature set after Phases 1-2."""
        return self._effective_cols

    def run(
        self,
        compositions_df: pd.DataFrame,
        features_all: pd.DataFrame,
        target: pd.Series,
        progress_callback: Optional[Any] = None,
        selected_workflows: Optional[List[str]] = None,
        selected_feature_sets: Optional[List[str]] = None,
    ) -> Tuple[List[RunResult], List[ValidityScore], Dict[str, OODResult]]:
        """Execute the full experiment grid in 7 phases."""
        t_start = time.time()

        # ── Reset per-run state so repeated GUI calls don't accumulate ──
        self._registry.reset()
        self._ood_split_indices.clear()

        # Rebuild as C-contiguous to prevent BLAS SIGSEGV from F-order arrays
        _cols = list(features_all.columns)
        features_all = pd.DataFrame(
            safe_array(features_all),
            columns=_cols,
            index=features_all.index,
        )

        self._feature_store.store_features(features_all)

        all_wf_names = ["WF-LIN", "WF-LASSO", "WF-ARD", "WF-RF", "WF-XGB", "WF-ENS"]
        mint_configs: Dict[str, MIntWorkflowConfig] = {}
        if self._mint_registry is not None:
            for wf_info in self._mint_registry.list_workflows():
                wf_name = wf_info["name"]
                if wf_name not in all_wf_names:
                    all_wf_names.append(wf_name)
                    cfg = self._mint_registry.get_config(wf_name)
                    mint_configs[wf_name] = cfg
                    logger.info("Added MInt workflow: %s", wf_name)
        wf_names = (
            [w for w in all_wf_names if w in (selected_workflows or all_wf_names)]
        )

        feature_sets = FeatureCatalog.list_sets()
        if selected_feature_sets is not None:
            feature_sets = [
                fs for fs in feature_sets
                if fs.value in selected_feature_sets
            ]
        y_arr = np.asarray(target, dtype=float)

        self._tracker.start_run(
            run_name=f"experiment_{int(t_start)}",
            tags={
                "seeds": str(self._seeds),
                "quick": str(self._quick),
                "n_feature_sets": str(len(feature_sets)),
                "n_workflows": str(len(wf_names)),
                "n_workers": str(self._n_workers),
                "mlflow_active": str(self._tracker.is_mlflow_active),
                "feast_active": str(self._feature_store.is_feast_active),
                "mint_active": str(self._mint_registry is not None),
            },
        )

        try:
            # ── Phase 1: Multicollinearity diagnostics & leak detection ──
            mc_reports = run_phase0_multicollinearity(
                features_all, feature_sets, wf_names, len(features_all),
                target=target,
                leak_corr_threshold=self._leak_corr_threshold,
            )
            self._mc_reports = mc_reports

            # ── Effective columns per feature set (subtract Phase 1 drops) ──
            self._effective_cols = {}
            for _fs in feature_sets:
                _fs_key = _fs.value
                _orig = [c for c in FeatureCatalog.columns(_fs)
                         if c in features_all.columns]
                if mc_reports and _fs_key in mc_reports:
                    _rpt = mc_reports[_fs_key]
                    _dropped = set(_rpt.dropped_constant + _rpt.dropped_perfect)
                    _orig = [c for c in _orig if c not in _dropped]
                    # Auto-exclude leaked features (high |r| with target)
                    if self._leak_auto_exclude and _rpt.leak_suspects:
                        _leak_set = set(_rpt.leak_suspects.keys())
                        _before = len(_orig)
                        _orig = [c for c in _orig if c not in _leak_set]
                        if _before != len(_orig):
                            logger.info(
                                "Leak auto-exclude [%s]: removed %d features "
                                "(|r| > %.2f): %s",
                                _fs_key, _before - len(_orig),
                                self._leak_corr_threshold,
                                ", ".join(sorted(_leak_set & set(
                                    FeatureCatalog.columns(_fs)
                                ))),
                            )
                self._effective_cols[_fs_key] = _orig
                logger.info(
                    "Effective columns [%s]: %d features (after Phase 1 drops)",
                    _fs_key, len(_orig),
                )

            # ── Phase 3: Precompute folds ──
            fold_plan = self._phase3_precompute_folds(
                compositions_df, features_all, target
            )

            # ── Phase 2: Feature selection (train-only to prevent leakage) ──
            _primary_train_idx = fold_plan["CompositionBlock"][0][0]
            self._fs_summaries: Dict[str, Any] = {}  # store for GUI display

            for _fs_key, _cols in list(self._effective_cols.items()):
                if len(_cols) <= 3:
                    continue  # too few to select from
                try:
                    # Slice to train-only rows to prevent data leakage
                    _X_fs_train = features_all.iloc[_primary_train_idx][_cols]
                    _y_train = target.iloc[_primary_train_idx]
                    _fs_summary = run_feature_selection(
                        _X_fs_train, _y_train,
                        methods=None,  # ALL methods: Lasso, AIC, BIC, ARD
                        consensus_threshold=2,
                        feature_set=_fs_key,
                    )
                    self._fs_summaries[_fs_key] = _fs_summary

                    # Prefer consensus features (selected by >= 2 methods)
                    if _fs_summary.consensus_features and len(_fs_summary.consensus_features) >= 2:
                        _sel = _fs_summary.consensus_features
                        logger.info(
                            "Feature selection [%s]: %d → %d consensus features"
                            " (%d methods, train-only, n_train=%d)",
                            _fs_key, len(_cols), len(_sel),
                            len(_fs_summary.results),
                            len(_primary_train_idx),
                        )
                        self._effective_cols[_fs_key] = _sel
                    else:
                        # Fall back to Lasso-only if consensus is too small
                        _lasso = _fs_summary.results.get("Lasso")
                        if _lasso and _lasso.selected_features and len(_lasso.selected_features) >= 2:
                            _sel = _lasso.selected_features
                            logger.info(
                                "Feature selection [%s]: %d → %d features"
                                " (Lasso fallback, train-only, n_train=%d)",
                                _fs_key, len(_cols), len(_sel),
                                len(_primary_train_idx),
                            )
                            self._effective_cols[_fs_key] = _sel
                    # Log per-method results
                    for _m_name, _m_res in _fs_summary.results.items():
                        logger.info(
                            "  %s [%s]: %d features selected",
                            _m_name, _fs_key, _m_res.n_selected,
                        )
                except Exception:
                    logger.warning(
                        "Feature selection failed for %s; using all cleaned features",
                        _fs_key,
                    )

            if progress_callback is not None:
                try:
                    progress_callback(
                        0, 0,
                        f"Phases 1-2 complete: multicollinearity + feature "
                        f"selection for {len(mc_reports)} feature sets",
                    )
                except Exception:
                    pass
            # ── Phase 4: Build job list ──
            jobs = self._phase4_build_jobs(
                feature_sets, wf_names, fold_plan,
                mc_reports=mc_reports,
            )
            logger.info(
                "Starting experiment: %d jobs, %d workers",
                len(jobs), self._n_workers,
            )

            # ── Phase 5: Parallel training ──
            all_results = self._phase5_train(
                jobs, features_all, feature_sets, y_arr,
                progress_callback, mint_configs or None,
            )
            self._registry.add_many(all_results)

            # Log each run as a separate tracked entry
            self._tracker.end_run()
            for _rr in all_results:
                try:
                    _run_label = (
                        f"{_rr.workflow}_{_rr.feature_set}"
                        f"_{_rr.split_policy}_s{_rr.seed}_f{_rr.fold}"
                    )
                    self._tracker.start_run(run_name=_run_label)
                    self._tracker.log_run_result(_rr)
                    self._tracker.end_run()
                except Exception:
                    logger.debug(
                        "Failed to log run result to tracker: %s %s",
                        _rr.workflow, _rr.feature_set,
                    )
                    try:
                        self._tracker.end_run(status="FAILED")
                    except Exception:
                        pass

            # Register effective columns per feature set with Feast store
            for _fs_key, _fs_cols in self._effective_cols.items():
                try:
                    self._feature_store.register_feature_set(
                        _fs_key, _fs_cols,
                    )
                except Exception:
                    logger.debug(
                        "Failed to register feature set %s in Feast",
                        _fs_key,
                    )

            # Re-open a summary run to log experiment summary
            self._tracker.start_run(
                run_name=f"experiment_summary_{int(t_start)}",
                tags={"type": "experiment_summary"},
            )

            # ── Phase 6: OOD detection ──
            self._ood_split_indices = {}
            ood_results, ood_errors_for_eval = self._phase6_ood(
                features_all, feature_sets, fold_plan
            )

            # ── Phase 7: Evaluation ──
            validity_scores = self._phase7_evaluate(
                ood_errors_for_eval, mc_reports=mc_reports,
            )

            elapsed = time.time() - t_start
            run_count = len(all_results)
            logger.info(
                "Experiment complete: %d runs in %.1f sec. Top feature set: %s",
                run_count, elapsed,
                validity_scores[0].feature_set if validity_scores else "N/A",
            )
            self._tracker.log_experiment_summary(
                n_runs=run_count,
                validity_scores=validity_scores,
                ood_results=ood_results,
                elapsed_sec=elapsed,
            )
            self._tracker.end_run()

        except Exception:
            self._tracker.end_run(status="FAILED")
            raise

        return self._registry.runs, validity_scores, ood_results

    def run_model_selection(
        self,
        features_all: pd.DataFrame,
        target: pd.Series,
        out_dir: Optional[Path] = None,
        n_outer: int = 5,
        n_inner: int = 3,
        n_iter: int = 20,
        progress_callback: Optional[Any] = None,
    ) -> ModelSelectionResult:
        """Optional nested-CV model selection (runs after the 7-phase grid)."""
        # Pick effective columns for the best-performing feature set.
        best_cols: Optional[List[str]] = None
        if self._registry.runs:
            rmse_sums: dict = defaultdict(float)
            rmse_counts: dict = defaultdict(int)
            for r in self._registry.runs:
                if np.isfinite(r.rmse_test) and r.rmse_test > 0:
                    rmse_sums[r.feature_set] += r.rmse_test
                    rmse_counts[r.feature_set] += 1
            if rmse_sums:
                best_fs = min(
                    rmse_sums,
                    key=lambda k: rmse_sums[k] / max(rmse_counts[k], 1),
                )
                best_cols = self._effective_cols.get(best_fs)
                logger.info(
                    "run_model_selection: best feature set by mean RMSE = %s "
                    "(%d cols, mean_rmse=%.4f)",
                    best_fs, len(best_cols or []),
                    rmse_sums[best_fs] / max(rmse_counts[best_fs], 1),
                )

        # Fallback: pick by FS priority
        _FS_PRIORITY = ["FS_BASE", "FS_THERMO", "FS_SIZE", "FS_ELECTRON", "FS_ALL", "FS_MAGPIE"]
        if not best_cols:
            for fs_key in _FS_PRIORITY:
                if fs_key in self._effective_cols and self._effective_cols[fs_key]:
                    best_cols = self._effective_cols[fs_key]
                    break

        if not best_cols:
            best_cols = list(features_all.columns)

        X = features_all[best_cols]

        def _ms_callback(msg: str) -> None:
            logger.info("Model selection: %s", msg)
            if progress_callback is not None:
                try:
                    progress_callback(0, 0, f"Model selection: {msg}")
                except Exception:
                    pass

        result = run_model_selection(
            X, target,
            out_dir=out_dir,
            n_outer=n_outer,
            n_inner=n_inner,
            n_iter=n_iter,
            quick=self._quick,
            random_state=self._seeds[0] if self._seeds else 42,
            progress_callback=_ms_callback,
        )
        self._model_selection_result = result
        return result

    def _phase3_precompute_folds(
        self,
        compositions_df: pd.DataFrame,
        features_all: pd.DataFrame,
        target: pd.Series,
    ) -> Dict[str, List[Tuple[np.ndarray, np.ndarray]]]:
        """Phase 3: compute all splits exactly once."""
        fold_plan: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}

        logger.debug("Phase 3: CompositionBlock (once)")
        cb = CompositionBlockSplitter(n_folds=5, seed=self._seeds[0])
        fold_plan["CompositionBlock"] = list(
            cb.split(features_all, target, compositions=compositions_df)
        )

        logger.debug("Phase 3: ElementExclusion (once)")
        ee = ElementExclusionSplitter(target_elements=self._exclude_elements)
        fold_plan["ElementExclusion"] = list(
            ee.split(features_all, target, compositions=compositions_df)
        )

        for seed in self._seeds:
            logger.debug("Phase 3: RandomCV seed=%d", seed)
            rc = RandomCVSplitter(n_folds=5, seed=seed)
            fold_plan[f"RandomCV_seed{seed}"] = list(
                rc.split(features_all, target, compositions=compositions_df)
            )

        logger.info(
            "Phase 3 complete: CompositionBlock=%d folds, "
            "ElementExclusion=%d folds, RandomCV=%d folds/seed × %d seeds",
            len(fold_plan["CompositionBlock"]),
            len(fold_plan["ElementExclusion"]),
            len(fold_plan[f"RandomCV_seed{self._seeds[0]}"]),
            len(self._seeds),
        )
        return fold_plan

    def _phase4_build_jobs(
        self,
        feature_sets: List[FeatureSetName],
        wf_names: List[str],
        fold_plan: Dict[str, List[Tuple[np.ndarray, np.ndarray]]],
        mc_reports: Optional[Dict[str, MulticollinearityReport]] = None,
    ) -> List[_Job]:
        jobs: List[_Job] = []
        blocked_count = 0
        for seed in self._seeds:
            splitter_folds = {
                "CompositionBlock": fold_plan["CompositionBlock"],
                "ElementExclusion": fold_plan["ElementExclusion"],
                "RandomCV":         fold_plan[f"RandomCV_seed{seed}"],
            }
            for fs_name in feature_sets:
                fs_key = fs_name.value

                # ── Model selection filter from Phase 1 ──
                if mc_reports and fs_key in mc_reports:
                    recs = mc_reports[fs_key].recommended_workflows
                    # If recommended_workflows is empty, allow all
                    # workflows instead of silently blocking everything.
                    allowed_wf = set(recs) if recs else set(wf_names)
                else:
                    allowed_wf = set(wf_names)  # fallback: all allowed

                # ── PCA force flag from Phase 1 ──
                force_pca = bool(
                    mc_reports
                    and fs_key in mc_reports
                    and mc_reports[fs_key].multicollinearity_level == 'moderate'
                )

                for sp_name, folds in splitter_folds.items():
                    for fold_idx, (train_idx, test_idx) in enumerate(folds):
                        for wf_name in wf_names:
                            if wf_name not in allowed_wf:
                                blocked_count += 1
                                continue  # blocked workflow
                            jobs.append(_Job(
                                wf_name=wf_name,
                                fs_name=fs_key,
                                sp_name=sp_name,
                                seed=seed,
                                fold=fold_idx,
                                train_idx=train_idx,
                                test_idx=test_idx,
                                quick=self._quick,
                                dim_reduction=self._dim_reduction,
                                force_pca=force_pca,
                            ))
        if blocked_count > 0:
            logger.info(
                "Phase 4: %d jobs (%d blocked by multicollinearity filter)",
                len(jobs), blocked_count,
            )
        else:
            logger.debug("Phase 4: %d jobs", len(jobs))
        return jobs

    def _phase5_train(
        self,
        jobs: List[_Job],
        features_all: pd.DataFrame,
        feature_sets: List[FeatureSetName],
        y_arr: np.ndarray,
        progress_callback: Optional[Any],
        mint_configs: Optional[Dict[str, MIntWorkflowConfig]] = None,
    ) -> List[RunResult]:
        """Phase 5: train all jobs (serial or parallel)."""
        # Build one C-contiguous array per feature set
        fs_arrays: Dict[str, Tuple[np.ndarray, List[str]]] = {}
        for fs_name in feature_sets:
            cols = self._effective_cols.get(
                fs_name.value,
                list(FeatureCatalog.columns(fs_name)),
            )
            missing = [c for c in cols if c not in features_all.columns]
            if missing:
                logger.warning(
                    "Feature set %s skipped: columns not in data: %s",
                    fs_name.value, missing[:5],
                )
                continue  # skip this FS — columns not present
            arr = safe_array(features_all[cols])
            fs_arrays[fs_name.value] = (arr, cols)
            logger.debug(
                "Phase 5 prep: %s → array shape=%s C-contiguous=%s",
                fs_name.value, arr.shape, arr.flags["C_CONTIGUOUS"],
            )

        # Filter jobs to only those whose feature set was successfully prepared
        jobs = [j for j in jobs if j.fs_name in fs_arrays]
        all_results: List[RunResult] = []
        n_total = len(jobs)
        completed = 0
        _t0 = time.time()

        def _log_progress(n: int, last_job: _Job) -> None:
            if _HAS_RESOURCE:
                try:
                    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                except Exception:
                    rss_kb = -1
            else:
                rss_kb = -1  # Windows: resource module not available
            logger.info(
                "Progress: %d / %d runs (%.1f sec, RSS peak ~%d MB)",
                n, n_total, time.time() - _t0, rss_kb // 1024,
            )

        if self._n_workers == 1:
            for job in jobs:
                arr, cols = fs_arrays[job.fs_name]
                try:
                    result = _run_job(job, arr, cols, y_arr, mint_configs)
                    all_results.append(result)
                except Exception:
                    logger.exception(
                        "Job failed: wf=%s fs=%s sp=%s seed=%d fold=%d",
                        job.wf_name, job.fs_name, job.sp_name, job.seed, job.fold,
                    )
                completed += 1
                if completed % 20 == 0:
                    _log_progress(completed, job)
                if progress_callback is not None:
                    try:
                        progress_callback(
                            completed, n_total,
                            f"{job.wf_name} | {job.fs_name} | "
                            f"{job.sp_name} fold {job.fold}",
                        )
                    except Exception:
                        pass
        else:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self._n_workers
            ) as executor:
                future_to_job = {
                    executor.submit(
                        _run_job, job,
                        fs_arrays[job.fs_name][0],
                        fs_arrays[job.fs_name][1],
                        y_arr,
                        mint_configs,
                    ): job
                    for job in jobs
                }
                for future in concurrent.futures.as_completed(future_to_job):
                    job = future_to_job[future]
                    try:
                        all_results.append(future.result())
                    except Exception:
                        logger.exception(
                            "Job failed: wf=%s fs=%s sp=%s seed=%d fold=%d",
                            job.wf_name, job.fs_name, job.sp_name,
                            job.seed, job.fold,
                        )
                    completed += 1
                    if completed % 20 == 0:
                        _log_progress(completed, job)
                    if progress_callback is not None:
                        try:
                            progress_callback(
                                completed, n_total,
                                f"{job.wf_name} | {job.fs_name} | "
                                f"{job.sp_name} fold {job.fold}",
                            )
                        except Exception:
                            pass

        logger.info(
            "Phase 5 complete: %d / %d runs in %.1f sec",
            len(all_results), n_total, time.time() - _t0,
        )
        return all_results

    def _phase6_ood(
        self,
        features_all: pd.DataFrame,
        feature_sets: List[FeatureSetName],
        fold_plan: Dict[str, List[Tuple[np.ndarray, np.ndarray]]],
    ) -> Tuple[Dict[str, OODResult], Dict[str, Dict[str, np.ndarray]]]:
        """Phase 6: OOD detection (once per feature set, multi-fold ensemble)."""
        ood_results: Dict[str, OODResult] = {}
        ood_errors_for_eval: Dict[str, Dict[str, np.ndarray]] = {}

        n_samples = len(features_all)

        # Collect all RandomCV folds across all seeds for ensemble OOD
        all_ood_folds: List[Tuple[np.ndarray, np.ndarray]] = []
        for seed in self._seeds:
            key = f"RandomCV_seed{seed}"
            if key in fold_plan:
                all_ood_folds.append(fold_plan[key][0])  # first fold per seed
        if not all_ood_folds:
            logger.warning("Phase 6: No RandomCV folds available for OOD")
            return ood_results, ood_errors_for_eval

        # Primary fold (used for ENS error collection and split indices)
        ood_train_idx, ood_test_idx = all_ood_folds[0]

        logger.info(
            "Phase 6: OOD for %d feature sets "
            "(ensemble over %d folds, primary train=%d test=%d)",
            len(feature_sets), len(all_ood_folds),
            len(ood_train_idx), len(ood_test_idx),
        )

        for fs_name in feature_sets:
            fs_key = fs_name.value
            cols = self._effective_cols.get(
                fs_key, list(FeatureCatalog.columns(fs_name)),
            )

            # Skip feature sets with missing columns
            missing = [c for c in cols if c not in features_all.columns]
            if missing:
                logger.warning(
                    "OOD: Feature set %s skipped: columns not in data: %s",
                    fs_key, missing[:5],
                )
                continue

            # C-contiguous feature array — shared across folds
            X_fs_arr = safe_array(features_all[cols])

            try:
                score_sum = np.zeros(n_samples, dtype=np.float64)
                score_count = np.zeros(n_samples, dtype=np.int32)
                primary_res: Optional[OODResult] = None

                for fold_i, (tr_idx, te_idx) in enumerate(all_ood_folds):
                    X_tr = pd.DataFrame(X_fs_arr[tr_idx], columns=cols)
                    X_te = pd.DataFrame(X_fs_arr[te_idx], columns=cols)
                    detector = OODDetector(k=10)
                    detector.fit(X_tr)
                    res = detector.score(X_te)
                    # Map scores back to global sample indices
                    score_sum[te_idx] += res.composite_scores
                    score_count[te_idx] += 1
                    if fold_i == 0:
                        primary_res = res

                if primary_res is None:
                    continue

                primary_te = ood_test_idx
                scored_mask = score_count[primary_te] > 0
                avg_scores = np.where(
                    scored_mask,
                    score_sum[primary_te] / np.maximum(score_count[primary_te], 1),
                    primary_res.composite_scores,
                )

                if len(all_ood_folds) > 1:
                    avg_threshold = primary_res.ood_threshold
                    is_ood_avg = avg_scores > avg_threshold
                    n_ood = int(is_ood_avg.sum())
                    ood_res = OODResult(
                        mahalanobis_scores=primary_res.mahalanobis_scores,
                        knn_scores=primary_res.knn_scores,
                        composite_scores=np.ascontiguousarray(avg_scores),
                        is_ood=np.ascontiguousarray(is_ood_avg),
                        ood_threshold=avg_threshold,
                        ood_ratio=n_ood / max(len(avg_scores), 1),
                        n_total=len(avg_scores),
                        n_ood=n_ood,
                    )
                else:
                    ood_res = primary_res

                ood_results[fs_key] = ood_res
                self._ood_split_indices[fs_key] = (
                    np.ascontiguousarray(np.asarray(ood_train_idx)),
                    np.ascontiguousarray(np.asarray(ood_test_idx)),
                )

                logger.debug(
                    "OOD %s: %d/%d flagged (ensemble over %d folds)",
                    fs_key, ood_res.n_ood, ood_res.n_total,
                    len(all_ood_folds),
                )

                self._collect_ood_errors(
                    fs_key, ood_res, ood_test_idx, ood_errors_for_eval
                )

            except Exception:
                logger.exception("OOD detection failed for %s", fs_key)

        logger.info(
            "Phase 6 complete: OOD done for %d / %d feature sets",
            len(ood_results), len(feature_sets),
        )
        return ood_results, ood_errors_for_eval

    def _phase7_evaluate(
        self,
        ood_errors_for_eval: Dict[str, Dict[str, np.ndarray]],
        mc_reports: Optional[Dict[str, MulticollinearityReport]] = None,
    ) -> List[ValidityScore]:
        evaluator = FeatureValidityEvaluator()
        return evaluator.evaluate(
            self._registry.runs,
            ood_errors=ood_errors_for_eval,
            mc_reports=mc_reports,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _collect_ood_errors(
        self,
        fs_key: str,
        ood_res: OODResult,
        ood_test_idx: np.ndarray,
        out: Dict[str, Dict[str, np.ndarray]],
    ) -> None:
        ood_test_indices = np.ascontiguousarray(np.asarray(ood_test_idx))
        ood_key = ood_test_indices.tobytes()

        ens_runs = [
            r for r in self._registry.runs
            if r.feature_set == fs_key
            and r.workflow == "WF-ENS"
            and r.test_indices is not None
        ]
        ens_index: Dict[bytes, "RunResult"] = {}
        for er in ens_runs:
            key = np.ascontiguousarray(np.asarray(er.test_indices)).tobytes()
            ens_index[key] = er  # last write wins (most recent run)

        matched = ens_index.get(ood_key)
        if matched is None:
            logger.info("No matching ENS run for OOD eval on %s", fs_key)
            return
        pred_std = np.ascontiguousarray(
            np.array(matched.artifacts.get("pred_std_test", []))
        )
        if (
            matched.y_test_true is not None
            and matched.y_test_pred is not None
            and len(pred_std) > 0
        ):
            _errors = np.ascontiguousarray(
                matched.y_test_true - matched.y_test_pred
            )
            _is_ood = np.ascontiguousarray(np.asarray(ood_res.is_ood))
            out[fs_key] = {
                "errors":        _errors,
                "uncertainties": pred_std,
                "is_ood":        _is_ood,
            }

    def export(self, out_dir: Path) -> None:
        self._registry.export_json(out_dir / "run_registry.json")

    def export_experiment_log(self, out_dir: Path) -> Path:
        """Write experiment log (JSON) with all phases' metadata."""
        import datetime as _dt

        log_path = out_dir / "experiment_log.json"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Per-run metrics
        run_records = []
        for r in self._registry.runs:
            run_records.append({
                "workflow": r.workflow,
                "feature_set": r.feature_set,
                "split_policy": r.split_policy,
                "seed": int(r.seed),
                "fold": int(r.fold),
                "rmse_train": round(float(r.rmse_train), 6),
                "rmse_test": round(float(r.rmse_test), 6),
                "r2_train": round(float(r.r2_train), 6),
                "r2_test": round(float(r.r2_test), 6),
                "mae_train": round(float(r.mae_train), 6),
                "mae_test": round(float(r.mae_test), 6),
                "elapsed_sec": round(float(r.elapsed_sec), 3),
                "params": {k: str(v) for k, v in (r.params or {}).items()},
            })

        # Phase 1 multicollinearity summary
        mc_summary = {}
        for fs_key, rpt in (self._mc_reports or {}).items():
            mc_summary[fs_key] = {
                "n_features_before": rpt.n_features_before,
                "n_features_after": rpt.n_features_after,
                "dropped_constant": rpt.dropped_constant,
                "dropped_perfect": rpt.dropped_perfect,
                "high_vif_count": rpt.high_vif_count,
                "moderate_vif_count": rpt.moderate_vif_count,
                "multicollinearity_level": rpt.multicollinearity_level,
                "high_vif_ratio": round(rpt.high_vif_ratio, 4),
                "recommended_workflows": rpt.recommended_workflows,
                "blocked_workflows": rpt.blocked_workflows,
                "leak_suspects": rpt.leak_suspects or {},
            }

        # Effective columns
        eff_cols = {
            k: {"n_features": len(v), "columns": v}
            for k, v in self._effective_cols.items()
        }

        # Tracker summary
        tracker_runs = self._tracker.list_runs()

        log_data = {
            "timestamp": _dt.datetime.now().isoformat(),
            "seeds": self._seeds,
            "quick": self._quick,
            "n_runs": len(self._registry.runs),
            "leak_auto_exclude": self._leak_auto_exclude,
            "leak_corr_threshold": self._leak_corr_threshold,
            "multicollinearity_reports": mc_summary,
            "effective_columns": eff_cols,
            "runs": run_records,
            "tracker_runs": as_serializable(tracker_runs),
        }

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(as_serializable(log_data), f, indent=2,
                      ensure_ascii=False)
        logger.info("Experiment log written to %s", log_path)
        return log_path
