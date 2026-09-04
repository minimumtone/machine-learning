"""Experiment Runner — orchestrates the 7-phase ML pipeline.

Phases: 1 Multicollinearity  2 Fold-precompute   3 Feature-selection
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
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

from extrapolation_discovery_platform.features import (
    FeatureCatalog,
    FeatureSetName,
    compute_features,
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
)
from extrapolation_discovery_platform.model_selection import (
    ModelSelectionResult,
    run_model_selection,
)
from extrapolation_discovery_platform._compat import as_serializable

if TYPE_CHECKING:
    from extrapolation_discovery_platform.ood import OODResult

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
    """Execute one training run (pure, picklable).

    Delegates workflow instantiation to individual_runner._WORKFLOW_FACTORIES
    so that runner.py and individual_runner.py share a single source of truth
    for how each workflow is constructed.  Previously both modules duplicated
    the factory dict, so changes to one (e.g. adding StandardScaler to WF-XGB)
    were not automatically reflected in the other.
    """
    import os as _os
    import pandas as _pd

    _os.environ["_EDP_INSIDE_WORKER"] = "1"

    X_train = _pd.DataFrame(X_fs[job.train_idx], columns=feature_cols)
    X_test  = _pd.DataFrame(X_fs[job.test_idx],  columns=feature_cols)
    y_train = _pd.Series(y[job.train_idx])
    y_test  = _pd.Series(y[job.test_idx])
    from extrapolation_discovery_platform.pipeline import impute_by_train_median
    X_train, X_test = impute_by_train_median(X_train, X_test)

    # --- single source of truth: delegate to individual_runner ---
    from extrapolation_discovery_platform.individual_runner import (
        _WORKFLOW_FACTORIES as _IR_FACTORIES,
    )
    _dr  = job.dim_reduction
    _pca = _dr or job.force_pca

    # WF-LIN gets force_pca treatment; all others use the standard _dr flag
    _dim_r_for_wf = _pca if job.wf_name == "WF-LIN" else _dr

    if job.wf_name in _IR_FACTORIES:
        wf = _IR_FACTORIES[job.wf_name](job.quick, _dim_r_for_wf)
    elif mint_configs is not None and job.wf_name in mint_configs:
        from extrapolation_discovery_platform.integrations.mint_adapter import (
            MIntWorkflowAdapter,
        )
        wf = MIntWorkflowAdapter(config=mint_configs[job.wf_name])
    else:
        raise KeyError(
            f"Unknown workflow '{job.wf_name}'. "
            f"Built-in: {list(_IR_FACTORIES)}, MInt: {list(mint_configs or {})}"
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
        n_folds: int = 5,
        test_size: float = 0.2,
    ) -> None:
        self._seeds = seeds or [42, 123, 456]
        self._quick = quick
        self._dim_reduction = dim_reduction
        self._leak_auto_exclude = leak_auto_exclude
        self._leak_corr_threshold = leak_corr_threshold
        self._n_folds = max(2, int(n_folds))
        self._test_size = float(test_size)
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
        selected_split_policies: Optional[List[str]] = None,
    ) -> Tuple[List[RunResult], List[ValidityScore], Dict[str, OODResult]]:
        """Execute the full experiment grid in 7 phases.

        Parameters
        ----------
        selected_split_policies : list of str, optional
            Subset of ["CompositionBlock", "ElementExclusion", "RandomCV"].
            Default (None) uses ["CompositionBlock", "ElementExclusion"] —
            RandomCV is intentionally excluded by default because:

            1. **Data leakage risk**: RandomCV randomly assigns near-duplicate
               or related samples to both train and test sets.  For materials
               datasets where alloys with similar compositions appear multiple
               times, this inflates apparent test-set performance.
            2. **Misleading generalisation score**: The validity evaluator
               computes a "generalisation" score by comparing RandomCV vs
               CompositionBlock performance.  If RandomCV is always included,
               this score reflects random-split variance rather than true
               compositional extrapolation ability.
            3. **Redundancy**: CompositionBlock already provides rigorous
               k-fold cross-validation; adding RandomCV doubles run count
               without adding information about extrapolation safety.

            Set ``selected_split_policies=["CompositionBlock",
            "ElementExclusion", "RandomCV"]`` to include RandomCV when
            needed (e.g. for baselines or debugging).
        """
        t_start = time.time()

        # ── Resolve split policies (default: exclude RandomCV) ──────────
        _ALL_POLICIES = ["CompositionBlock", "ElementExclusion", "RandomCV"]
        if selected_split_policies is None:
            # Default: CompositionBlock + ElementExclusion only.
            # RandomCV is excluded because it leaks compositionally similar
            # samples between train/test and inflates test metrics.
            # See run() docstring for full rationale.
            active_policies = ["CompositionBlock", "ElementExclusion"]
        else:
            active_policies = [
                p for p in _ALL_POLICIES if p in selected_split_policies
            ]
            if not active_policies:
                logger.warning(
                    "selected_split_policies contained no valid policies; "
                    "falling back to ['CompositionBlock', 'ElementExclusion']"
                )
                active_policies = ["CompositionBlock", "ElementExclusion"]
        logger.info("Active split policies: %s", active_policies)

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
            # ── Stage 1: 前処理（Phase1 + Phase3 + Phase2 を一本化） ──────
            # pipeline.stage1_preprocess が:
            #   1. 多重共線性・リーク検出
            #   2. 有効列決定（drop + leak 除外）
            #   3. 分割計画計算（fold_plan）
            #   4. 特徴量選択（訓練データのみ・リーク防止）
            # を同じ順序で実行する。individual_runner も同じ関数を使うため
            # 同一条件なら同一結果が保証される。
            from extrapolation_discovery_platform.pipeline import stage1_preprocess

            _fs_names = [fs.value for fs in feature_sets]
            _generic = getattr(self, "_generic_csv_mode", False)
            prep = stage1_preprocess(
                features_df=features_all,
                target=target,
                compositions_df=compositions_df,
                feature_set_names=_fs_names,
                workflow_names=wf_names,
                seeds=self._seeds,
                active_policies=active_policies,
                leak_auto_exclude=self._leak_auto_exclude,
                leak_corr_threshold=self._leak_corr_threshold,
                generic_csv_mode=_generic,
                n_folds=self._n_folds,
                test_size=self._test_size,
            )
            if not prep.success:
                raise RuntimeError(f"Stage1 前処理失敗:\n{prep.error_message}")

            mc_reports = prep.mc_reports
            self._mc_reports = mc_reports
            self._effective_cols = prep.effective_cols
            self._fs_summaries = prep.fs_summaries
            fold_plan = prep.fold_plan

            if progress_callback is not None:
                try:
                    progress_callback(
                        0, 0,
                        f"Stage1 完了: 前処理 + 特徴量選択 ({len(mc_reports)} FS, "
                        f"{len(fold_plan)} split policies)",
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

            # ── Stage 3: OOD 検出（学習とは完全独立） ───────────────────
            # pipeline.stage3_detect_ood が各 FS の fold_plan を使って
            # 全 fold × 全 split でアンサンブル OOD を計算する。
            # OOD は RunResult に含まれず、ここで独立して計算される。
            from extrapolation_discovery_platform.pipeline import stage3_detect_ood

            self._ood_split_indices = {}
            ood_results: Dict[str, OODResult] = {}
            ood_errors_for_eval: Dict[str, Dict[str, np.ndarray]] = {}

            for _fs in feature_sets:
                _fs_key = _fs.value
                _ood_cols = self._effective_cols.get(
                    _fs_key, list(FeatureCatalog.columns(_fs)),
                )
                _ood_cols = [c for c in _ood_cols if c in features_all.columns]
                if not _ood_cols:
                    logger.warning("Stage3: %s — OOD列なし、スキップ", _fs_key)
                    continue
                ood_stage = stage3_detect_ood(
                    features_df=features_all,
                    effective_columns=_ood_cols,
                    fold_plan=fold_plan,
                )
                if ood_stage.success and ood_stage.ood_result is not None:
                    ood_results[_fs_key] = ood_stage.ood_result
                    if ood_stage.primary_train_idx is not None:
                        self._ood_split_indices[_fs_key] = (
                            ood_stage.primary_train_idx,
                            ood_stage.primary_test_idx,
                        )
                    self._collect_ood_errors(
                        _fs_key, ood_stage.ood_result,
                        ood_stage.primary_test_idx, ood_errors_for_eval,
                    )
                else:
                    logger.warning("Stage3 OOD失敗 [%s]: %s",
                                   _fs_key, ood_stage.error_message[:200])

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
            # Build splitter_folds from fold_plan keys that were actually
            # computed in Phase 3 (depends on active_policies).
            splitter_folds: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}
            if "CompositionBlock" in fold_plan:
                splitter_folds["CompositionBlock"] = fold_plan["CompositionBlock"]
            if "ElementExclusion" in fold_plan:
                splitter_folds["ElementExclusion"] = fold_plan["ElementExclusion"]
            rc_key = f"RandomCV_seed{seed}"
            if rc_key in fold_plan:
                splitter_folds["RandomCV"] = fold_plan[rc_key]
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
