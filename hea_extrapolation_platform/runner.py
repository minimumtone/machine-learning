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
import resource
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
# Run Registry
# ---------------------------------------------------------------------------

def safe_array(source: Any, dtype: str = "float64") -> np.ndarray:
    """Convert *source* to a C-contiguous numpy array.

    This is the single choke-point for every DataFrame → numpy conversion
    in the platform.  pandas 3.0 returns F-contiguous (column-major) arrays
    from ``.values`` and ``.to_numpy()`` when the BlockManager is fragmented.
    Many C extensions (BLAS, LAPACK, scipy, sklearn) assume C-contiguous
    (row-major) layout and SIGSEGV on F-contiguous input.

    Accepts: DataFrame, Series, ndarray, or list.
    Returns: C-contiguous ndarray with requested dtype.
    """
    if isinstance(source, pd.DataFrame):
        arr = source.to_numpy(dtype=dtype, na_value=np.nan)
    elif isinstance(source, pd.Series):
        arr = source.to_numpy(dtype=dtype)
    elif isinstance(source, np.ndarray):
        arr = np.array(source, dtype=dtype)
    else:
        arr = np.asarray(source, dtype=dtype)
    return np.ascontiguousarray(arr)


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
        """Convert all runs to a summary DataFrame (columnar construction)."""
        if not self._runs:
            return pd.DataFrame()
        col_names = [
            "workflow", "feature_set", "split_policy", "seed", "fold",
            "rmse_train", "rmse_test", "mae_train", "mae_test",
            "r2_train", "r2_test", "elapsed_sec",
        ]
        columns: Dict[str, list] = {k: [] for k in col_names}
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
        WorkflowLIN, WorkflowXGB, WorkflowENS,
    )
    wf_map: Dict[str, Any] = {
        "WF-LIN": WorkflowLIN(),
        "WF-XGB": WorkflowXGB(quick=job.quick),
        "WF-ENS": WorkflowENS(
            n_members=3 if job.quick else 5, quick=job.quick
        ),
    }

    if job.wf_name in wf_map:
        wf = wf_map[job.wf_name]
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
    ) -> None:
        self._seeds = seeds or [42, 123, 456]
        self._quick = quick
        self._exclude_elements = exclude_elements or ["Co", "Ni", "Ti"]
        self._n_workers = n_workers if n_workers is not None else os.cpu_count()
        self._registry = RunRegistry()
        self._ood_split_indices: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        if mlflow_tracker is not None:
            self._tracker = mlflow_tracker
        elif use_mlflow:
            self._tracker = MLflowTracker(
                experiment_name="extrapolation_discovery", enabled=True,
            )
        else:
            self._tracker = MLflowTracker(enabled=False)

        if feature_store is not None:
            self._feature_store = feature_store
        elif use_feast:
            self._feature_store = FeastFeatureStore(enabled=True)
        else:
            self._feature_store = FeastFeatureStore(enabled=False)

        if mint_registry is not None:
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

    def run(
        self,
        compositions_df: pd.DataFrame,
        features_all: pd.DataFrame,
        target: pd.Series,
        progress_callback: Optional[Any] = None,
        selected_workflows: Optional[List[str]] = None,
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

        all_wf_names = ["WF-LIN", "WF-XGB", "WF-ENS"]
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
            fold_plan = self._phase1_precompute_folds(
                compositions_df, features_all, target
            )
            jobs = self._phase2_build_jobs(feature_sets, wf_names, fold_plan)
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

            validity_scores = self._phase5_evaluate(ood_errors_for_eval)

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
    ) -> List[_Job]:
        jobs: List[_Job] = []
        for seed in self._seeds:
            splitter_folds = {
                "CompositionBlock": fold_plan["CompositionBlock"],
                "ElementExclusion": fold_plan["ElementExclusion"],
                "RandomCV":         fold_plan[f"RandomCV_seed{seed}"],
            }
            for fs_name in feature_sets:
                for sp_name, folds in splitter_folds.items():
                    for fold_idx, (train_idx, test_idx) in enumerate(folds):
                        for wf_name in wf_names:
                            jobs.append(_Job(
                                wf_name=wf_name,
                                fs_name=fs_name.value,
                                sp_name=sp_name,
                                seed=seed,
                                fold=fold_idx,
                                train_idx=train_idx,
                                test_idx=test_idx,
                                quick=self._quick,
                            ))
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
            cols = list(FeatureCatalog.columns(fs_name))
            arr = safe_array(features_all[cols])
            fs_arrays[fs_name.value] = (arr, cols)
            logger.debug(
                "Phase 3 prep: %s → array shape=%s C-contiguous=%s",
                fs_name.value, arr.shape, arr.flags["C_CONTIGUOUS"],
            )

        all_results: List[RunResult] = []
        n_total = len(jobs)
        completed = 0
        _t0 = time.time()

        def _log_progress(n: int, last_job: _Job) -> None:
            try:
                rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            except Exception:
                rss_kb = -1
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

        Uses the first fold of the first seed's RandomCV split.
        Runs *after* all training so the full RunRegistry is available
        for ENS error collection.
        """
        ood_results: Dict[str, OODResult] = {}
        ood_errors_for_eval: Dict[str, Dict[str, np.ndarray]] = {}

        first_seed = self._seeds[0]
        ood_train_idx, ood_test_idx = fold_plan[f"RandomCV_seed{first_seed}"][0]

        logger.info(
            "Phase 4: OOD for %d feature sets "
            "(train=%d, test=%d, seed=%d fold=0)",
            len(feature_sets), len(ood_train_idx), len(ood_test_idx), first_seed,
        )

        for fs_name in feature_sets:
            fs_key = fs_name.value
            cols = list(FeatureCatalog.columns(fs_name))

            # C-contiguous slices — same pattern as Phase 3
            X_fs_arr = safe_array(features_all[cols])
            X_train_ood = pd.DataFrame(X_fs_arr[ood_train_idx], columns=cols)
            X_test_ood  = pd.DataFrame(X_fs_arr[ood_test_idx],  columns=cols)

            try:
                try:
                    rss_pre = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                except Exception:
                    rss_pre = -1

                detector = OODDetector(k=10)
                detector.fit(X_train_ood)
                ood_res = detector.score(X_test_ood)
                ood_results[fs_key] = ood_res
                self._ood_split_indices[fs_key] = (
                    np.asarray(ood_train_idx),
                    np.asarray(ood_test_idx),
                )

                try:
                    rss_post = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                except Exception:
                    rss_post = -1
                logger.debug(
                    "OOD %s: %d/%d flagged, RSS delta ~%d MB",
                    fs_key, ood_res.n_ood, ood_res.n_total,
                    (rss_post - rss_pre) // 1024,
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
    ) -> List[ValidityScore]:
        evaluator = FeatureValidityEvaluator()
        return evaluator.evaluate(
            self._registry.runs,
            ood_errors=ood_errors_for_eval,
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
        ood_test_indices = np.asarray(ood_test_idx)
        ens_runs = [
            r for r in self._registry.runs
            if r.feature_set == fs_key
            and r.workflow == "WF-ENS"
            and r.test_indices is not None
        ]
        matched = None
        for er in reversed(ens_runs):
            if np.array_equal(np.asarray(er.test_indices), ood_test_indices):
                matched = er
                break
        if matched is None:
            logger.info("No matching ENS run for OOD eval on %s", fs_key)
            return
        pred_std = np.array(matched.artifacts.get("pred_std_test", []))
        if (
            matched.y_test_true is not None
            and matched.y_test_pred is not None
            and len(pred_std) > 0
        ):
            out[fs_key] = {
                "errors":        matched.y_test_true - matched.y_test_pred,
                "uncertainties": pred_std,
                "is_ood":        ood_res.is_ood,
            }

    def export(self, out_dir: Path) -> None:
        self._registry.export_json(out_dir / "run_registry.json")
