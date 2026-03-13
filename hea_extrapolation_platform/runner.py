"""
Experiment Runner for Extrapolation Discovery Platform
実験オーケストレータ

設計方針 (PR#116 redesign)
--------------------------
旧実装は `seed > fs > splitter > fold > workflow` の5重ループを1つの run() メソッドに
詰め込んでいた。この構造には3つの根本的な問題があった:

  1. **重複計算**: CompositionBlock/ElementExclusion の分割はデータにのみ依存し
     seed・fs とは無関係だが、旧実装は seed x fs = 18 回再計算していた。
     RandomCV も seed が同じなら fs が変わっても同じ fold になる。

  2. **ループ内 OOD**: OOD 検出が「seed ループ内 fs ループの末尾」で実行されていた
     ため、タイミングが不明確で fs の X コピーが長寿命になっていた。

  3. **逐次実行**: 全ジョブが直列実行されていた。各ジョブは入力/出力が
     完全に独立しており、ProcessPoolExecutor による並列化が可能。

新スキーム: 5 フェーズ
----------------------
  Phase 1 – 分割事前計算 (最小回数)
      CompositionBlock/ElementExclusion: 1 回
      RandomCV: seed 毎に 1 回 (fs に非依存)

  Phase 2 – ジョブリスト構築
      _Job NamedTuple (スカラ + インデックス配列のみ)

  Phase 3 – 並列学習 (ProcessPoolExecutor)
      各 worker は C-contiguous numpy 配列からスライスして学習
      n_workers=1 で逐次実行 (デバッグ用)

  Phase 4 – OOD (全学習完了後、fs 毎に 1 回)

  Phase 5 – 評価・ログ
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
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
    WorkflowARD,
    WorkflowENS,
    WorkflowLASSO,
    WorkflowLIN,
    WorkflowRF,
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
from hea_extrapolation_platform.multicollinearity import (
    MulticollinearityReport,
    run_phase0_multicollinearity,
)
from hea_extrapolation_platform.feature_selection import run_feature_selection

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Run Registry
# ---------------------------------------------------------------------------

from hea_extrapolation_platform._utils import safe_array  # noqa: E402

# Re-export for backward compatibility
__all__ = ["safe_array"]


class RunRegistry:
    """In-memory registry of experiment runs with JSON export."""

    def __init__(self) -> None:
        self._runs: List[RunResult] = []

    def reset(self) -> None:
        """Clear all stored runs (called at the start of each run())."""
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
        """Convert all runs to a summary DataFrame (columnar construction).

        Column names are derived dynamically from ``RunResult`` dataclass
        fields, excluding non-scalar fields (numpy arrays, dicts) that
        cannot be represented as DataFrame columns.
        """
        import dataclasses as _dc

        if not self._runs:
            return pd.DataFrame()

        # Dynamically extract scalar column names from RunResult fields.
        # Exclude ndarray / dict fields that are not suitable for tabular
        # representation (y_test_true, y_test_pred, test_indices, params,
        # artifacts).
        _SKIP_TYPES = (np.ndarray, dict)
        col_names = [
            f.name for f in _dc.fields(RunResult)
            if f.default_factory is not dict.__class__  # exclude dict fields
            and not (
                isinstance(getattr(self._runs[0], f.name, None), _SKIP_TYPES)
            )
        ]
        # Fallback: filter to fields whose first-run value is a scalar
        col_names = [
            name for name in col_names
            if not isinstance(getattr(self._runs[0], name, None), (np.ndarray, dict))
        ]
        columns: Dict[str, list] = {k: [] for k in col_names}
        for r in self._runs:
            for col in col_names:
                columns[col].append(getattr(r, col))
        return pd.DataFrame(columns)

    def export_json(self, path: Path) -> None:
        df = self.to_dataframe()
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_json(path, orient="records", indent=2, force_ascii=False)
        logger.info("Exported %d runs to %s", len(self._runs), path)


# ---------------------------------------------------------------------------
# Internal job descriptor — lightweight, picklable
# ---------------------------------------------------------------------------

class _Job(NamedTuple):
    """All information needed to execute one (wf, fs, sp, seed, fold) run.

    Only scalars and index arrays — no DataFrames.  The worker slices its
    own train/test views from the shared feature array.
    """
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


# ---------------------------------------------------------------------------
# Top-level worker function (module-level for pickling)
# ---------------------------------------------------------------------------

def _run_job(
    job: _Job,
    X_fs: np.ndarray,
    feature_cols: List[str],
    y: np.ndarray,
    mint_configs: Optional[Dict[str, "MIntWorkflowConfig"]] = None,
) -> RunResult:
    """Execute a single training run.  Pure function — no side effects.

    X_fs must be C-contiguous so that row slices have unit stride, avoiding
    the BLAS SIGSEGV that occurs with F-contiguous (pandas 3.0) layouts.

    Parameters
    ----------
    mint_configs : dict, optional
        Mapping of MInt workflow name → MIntWorkflowConfig.  When the job's
        ``wf_name`` is not a built-in workflow, the config is used to
        reconstruct a ``MIntWorkflowAdapter`` in the worker process.
    """
    import pandas as _pd

    # Row slices of a C-contiguous 2-D array are themselves contiguous.
    # Wrap them in DataFrames only for the workflow API.
    X_train = _pd.DataFrame(X_fs[job.train_idx], columns=feature_cols)
    X_test  = _pd.DataFrame(X_fs[job.test_idx],  columns=feature_cols)
    y_train = _pd.Series(y[job.train_idx])
    y_test  = _pd.Series(y[job.test_idx])

    from hea_extrapolation_platform.workflows import (
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
        from hea_extrapolation_platform.integrations.mint_adapter import (
            MIntWorkflowAdapter,
        )
        wf = MIntWorkflowAdapter(config=mint_configs[job.wf_name])
    else:
        raise KeyError(
            f"Unknown workflow '{job.wf_name}'. "
            f"Built-in: {list(wf_map)}, MInt: {list(mint_configs or {})}"
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
    ) -> None:
        self._seeds = seeds or [42, 123, 456]
        self._quick = quick
        self._dim_reduction = dim_reduction
        self._exclude_elements = exclude_elements or ["Co", "Ni", "Ti"]
        self._n_workers = n_workers if n_workers is not None else os.cpu_count()
        self._registry = RunRegistry()
        self._ood_split_indices: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._mc_reports: Optional[Dict[str, MulticollinearityReport]] = None
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
        """Multicollinearity reports from Phase 0 (None before run)."""
        return self._mc_reports

    def run(
        self,
        compositions_df: pd.DataFrame,
        features_all: pd.DataFrame,
        target: pd.Series,
        progress_callback: Optional[Any] = None,
        selected_workflows: Optional[List[str]] = None,
        selected_feature_sets: Optional[List[str]] = None,
    ) -> Tuple[List[RunResult], List[ValidityScore], Dict[str, OODResult]]:
        """Execute the full experiment grid in 5 phases."""
        t_start = time.time()

        # ── Reset per-run state so repeated GUI calls don't accumulate ──
        self._registry.reset()
        self._ood_split_indices.clear()

        # ── Consolidate features_all to a single C-contiguous block ──
        # This is the ROOT FIX for the SIGSEGV: pandas DataFrames built from
        # heterogeneous sources (CSV upload, concat, column-subset) carry a
        # fragmented BlockManager whose .values returns F-contiguous arrays.
        # We rebuild once here so every downstream consumer (Phase 3 training,
        # Phase 4 OOD, post-processing visualizations) gets safe arrays.
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
            # ── Phase 0: Multicollinearity diagnostics & model selection ──
            mc_reports = run_phase0_multicollinearity(
                features_all, feature_sets, wf_names, len(features_all),
            )
            self._mc_reports = mc_reports

            # ── Compute effective columns per feature set ──
            # Phase 0 identifies constant / perfectly-collinear columns
            # to drop.  Until now these drops were *not* reflected in
            # the actual training arrays — Phase 3 still used the
            # original FeatureCatalog columns.  Fix: subtract drops.
            self._effective_cols = {}
            for _fs in feature_sets:
                _fs_key = _fs.value
                _orig = [c for c in FeatureCatalog.columns(_fs)
                         if c in features_all.columns]
                if mc_reports and _fs_key in mc_reports:
                    _rpt = mc_reports[_fs_key]
                    _dropped = set(_rpt.dropped_constant + _rpt.dropped_perfect)
                    _orig = [c for c in _orig if c not in _dropped]
                self._effective_cols[_fs_key] = _orig
                logger.info(
                    "Effective columns [%s]: %d features (after Phase 0 drops)",
                    _fs_key, len(_orig),
                )

            # ── Phase 0.5: Feature selection on cleaned columns ──
            # Lasso-based feature selection removes uninformative
            # features before training.  This addresses the issue
            # where high-dimensional / collinear features were passed
            # straight to learning without any selection step.
            for _fs_key, _cols in list(self._effective_cols.items()):
                if len(_cols) <= 3:
                    continue  # too few to select from
                try:
                    _X_fs = features_all[_cols]
                    _fs_summary = run_feature_selection(
                        _X_fs, target,
                        methods=["Lasso"],
                        feature_set=_fs_key,
                    )
                    _lasso = _fs_summary.results.get("Lasso")
                    if _lasso and _lasso.selected_features:
                        _sel = _lasso.selected_features
                        if len(_sel) >= 2:
                            logger.info(
                                "Feature selection [%s]: %d → %d features",
                                _fs_key, len(_cols), len(_sel),
                            )
                            self._effective_cols[_fs_key] = _sel
                except Exception:
                    logger.warning(
                        "Feature selection failed for %s; using all cleaned features",
                        _fs_key,
                    )

            if progress_callback is not None:
                try:
                    progress_callback(
                        0, 0,
                        f"Phase 0 complete: multicollinearity diagnosed "
                        f"for {len(mc_reports)} feature sets, "
                        f"feature selection applied",
                    )
                except Exception:
                    pass

            fold_plan = self._phase1_precompute_folds(
                compositions_df, features_all, target
            )
            jobs = self._phase2_build_jobs(
                feature_sets, wf_names, fold_plan,
                mc_reports=mc_reports,
            )
            logger.info(
                "Starting experiment: %d jobs, %d workers",
                len(jobs), self._n_workers,
            )

            all_results = self._phase3_train(
                jobs, features_all, feature_sets, y_arr,
                progress_callback, mint_configs or None,
            )
            self._registry.add_many(all_results)

            self._ood_split_indices = {}
            ood_results, ood_errors_for_eval = self._phase4_ood(
                features_all, feature_sets, fold_plan
            )

            validity_scores = self._phase5_evaluate(
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

    # ------------------------------------------------------------------
    # Phase 1: Pre-compute fold splits (minimum necessary calls)
    # ------------------------------------------------------------------

    def _phase1_precompute_folds(
        self,
        compositions_df: pd.DataFrame,
        features_all: pd.DataFrame,
        target: pd.Series,
    ) -> Dict[str, List[Tuple[np.ndarray, np.ndarray]]]:
        """Compute all splits exactly once.

        CompositionBlock and ElementExclusion depend only on compositions,
        not on seed or feature set.  The old runner re-computed them
        seed × fs = 18 times.  Here each is computed once.

        RandomCV depends on seed but not feature set.  Computed once per seed.
        """
        fold_plan: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}

        logger.debug("Phase 1: CompositionBlock (once)")
        cb = CompositionBlockSplitter(n_folds=5, seed=self._seeds[0])
        fold_plan["CompositionBlock"] = list(
            cb.split(features_all, target, compositions=compositions_df)
        )

        logger.debug("Phase 1: ElementExclusion (once)")
        ee = ElementExclusionSplitter(target_elements=self._exclude_elements)
        fold_plan["ElementExclusion"] = list(
            ee.split(features_all, target, compositions=compositions_df)
        )

        for seed in self._seeds:
            logger.debug("Phase 1: RandomCV seed=%d", seed)
            rc = RandomCVSplitter(n_folds=5, seed=seed)
            fold_plan[f"RandomCV_seed{seed}"] = list(
                rc.split(features_all, target, compositions=compositions_df)
            )

        logger.info(
            "Phase 1 complete: CompositionBlock=%d folds, "
            "ElementExclusion=%d folds, RandomCV=%d folds/seed × %d seeds",
            len(fold_plan["CompositionBlock"]),
            len(fold_plan["ElementExclusion"]),
            len(fold_plan[f"RandomCV_seed{self._seeds[0]}"]),
            len(self._seeds),
        )
        return fold_plan

    # ------------------------------------------------------------------
    # Phase 2: Build job list (no DataFrame duplication)
    # ------------------------------------------------------------------

    def _phase2_build_jobs(
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

                # ── Model selection filter from Phase 0 ──
                if mc_reports and fs_key in mc_reports:
                    allowed_wf = set(mc_reports[fs_key].recommended_workflows)
                else:
                    allowed_wf = set(wf_names)  # fallback: all allowed

                # ── PCA force flag from Phase 0 ──
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
                "Phase 2: %d jobs (%d blocked by multicollinearity filter)",
                len(jobs), blocked_count,
            )
        else:
            logger.debug("Phase 2: %d jobs", len(jobs))
        return jobs

    # ------------------------------------------------------------------
    # Phase 3: Training (serial or parallel)
    # ------------------------------------------------------------------

    def _phase3_train(
        self,
        jobs: List[_Job],
        features_all: pd.DataFrame,
        feature_sets: List[FeatureSetName],
        y_arr: np.ndarray,
        progress_callback: Optional[Any],
        mint_configs: Optional[Dict[str, MIntWorkflowConfig]] = None,
    ) -> List[RunResult]:
        """Train all jobs, optionally in parallel.

        Feature matrices are prepared once per feature set as C-contiguous
        numpy arrays, eliminating the pandas F-contiguous layout issue
        (root cause of the BLAS SIGSEGV bugs encountered previously).
        Each worker receives a read-only view and slices its own train/test.
        """
        # Build one C-contiguous array per feature set (not per seed/job)
        fs_arrays: Dict[str, Tuple[np.ndarray, List[str]]] = {}
        for fs_name in feature_sets:
            # Use effective columns (Phase 0 drops + feature selection)
            # instead of the raw FeatureCatalog columns.
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
                "Phase 3 prep: %s → array shape=%s C-contiguous=%s",
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
            "Phase 3 complete: %d / %d runs in %.1f sec",
            len(all_results), n_total, time.time() - _t0,
        )
        return all_results

    # ------------------------------------------------------------------
    # Phase 4: OOD detection (once per fs, after all training)
    # ------------------------------------------------------------------

    def _phase4_ood(
        self,
        features_all: pd.DataFrame,
        feature_sets: List[FeatureSetName],
        fold_plan: Dict[str, List[Tuple[np.ndarray, np.ndarray]]],
    ) -> Tuple[Dict[str, OODResult], Dict[str, Dict[str, np.ndarray]]]:
        """Run OOD detection once per feature set.

        Uses **multiple folds across all seeds** for more representative OOD
        detection.  Each fold produces an OOD composite score per sample;
        the final score is the mean across folds.  This avoids the previous
        bias of relying on a single fold/seed pair.

        Runs *after* all training so the full RunRegistry is available
        for ENS error collection.
        """
        ood_results: Dict[str, OODResult] = {}
        ood_errors_for_eval: Dict[str, Dict[str, np.ndarray]] = {}

        # Collect all RandomCV folds across all seeds for ensemble OOD
        all_ood_folds: List[Tuple[np.ndarray, np.ndarray]] = []
        for seed in self._seeds:
            key = f"RandomCV_seed{seed}"
            if key in fold_plan:
                all_ood_folds.append(fold_plan[key][0])  # first fold per seed
        if not all_ood_folds:
            logger.warning("Phase 4: No RandomCV folds available for OOD")
            return ood_results, ood_errors_for_eval

        # Primary fold (used for ENS error collection and split indices)
        ood_train_idx, ood_test_idx = all_ood_folds[0]

        logger.info(
            "Phase 4: OOD for %d feature sets "
            "(ensemble over %d folds, primary train=%d test=%d)",
            len(feature_sets), len(all_ood_folds),
            len(ood_train_idx), len(ood_test_idx),
        )

        for fs_name in feature_sets:
            fs_key = fs_name.value
            # Use effective columns (Phase 0 drops + feature selection)
            cols = self._effective_cols.get(
                fs_key, list(FeatureCatalog.columns(fs_name)),
            )

            # Guard: skip feature sets with missing columns (same as Phase 3)
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
                # Ensemble OOD: run detector on each fold and average scores
                fold_composites: List[np.ndarray] = []
                primary_res: Optional[OODResult] = None

                for fold_i, (tr_idx, te_idx) in enumerate(all_ood_folds):
                    X_tr = pd.DataFrame(X_fs_arr[tr_idx], columns=cols)
                    X_te = pd.DataFrame(X_fs_arr[te_idx], columns=cols)
                    detector = OODDetector(k=10)
                    detector.fit(X_tr)
                    res = detector.score(X_te)
                    fold_composites.append(res.composite_scores)
                    if fold_i == 0:
                        primary_res = res

                if primary_res is None:
                    continue

                # Average composite scores across folds for more stable OOD
                if len(fold_composites) > 1:
                    avg_composite = np.mean(fold_composites, axis=0)
                    avg_threshold = float(np.quantile(avg_composite, 0.95))
                    is_ood_avg = avg_composite > avg_threshold
                    n_ood = int(is_ood_avg.sum())
                    ood_res = OODResult(
                        mahalanobis_scores=primary_res.mahalanobis_scores,
                        knn_scores=primary_res.knn_scores,
                        composite_scores=avg_composite,
                        is_ood=is_ood_avg,
                        ood_threshold=avg_threshold,
                        ood_ratio=n_ood / max(len(avg_composite), 1),
                        n_total=len(avg_composite),
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
                    len(fold_composites),
                )

                self._collect_ood_errors(
                    fs_key, ood_res, ood_test_idx, ood_errors_for_eval
                )

            except Exception:
                logger.exception("OOD detection failed for %s", fs_key)

        logger.info(
            "Phase 4 complete: OOD done for %d / %d feature sets",
            len(ood_results), len(feature_sets),
        )
        return ood_results, ood_errors_for_eval

    # ------------------------------------------------------------------
    # Phase 5: Evaluation
    # ------------------------------------------------------------------

    def _phase5_evaluate(
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

        # O(1) lookup via hash dict instead of O(n) linear scan with
        # np.array_equal.  Build a dict keyed by test_indices.tobytes().
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
