"""
EDP 3-Stage パイプライン（共通処理）
======================================

「データ選択 → 一括計算(runner.py) → 可視化」と
「データ選択 → 個別計算(individual_runner.py) → 可視化」の両ルートが
**同じ計算条件なら同じ結果** になるよう、処理を 3 ステージに分離する。

    Stage 1 : stage1_preprocess()   前処理
    Stage 2 : stage2_train()        ML 学習（OOD なし）
    Stage 3 : stage3_detect_ood()   OOD 検出（学習と完全独立）

OOD は Stage 3 に完全分離されており、RunResult には含まれない。
runner.py / individual_runner.py のどちらも同一の Stage1→2→3 を通るため
同一条件なら同一結果が保証される。

Public API
----------
stage1_preprocess(features_df, target, compositions_df,
                  feature_set_names, workflow_names,
                  seeds, active_policies,
                  leak_auto_exclude, leak_corr_threshold,
                  generic_csv_mode)
    -> PreprocessResult

stage2_train(preprocess_result, features_df, target,
             workflow_name, split_policy_name, feature_set_name,
             quick, dim_reduction, seed, generic_csv_mode)
    -> TrainResult

stage3_detect_ood(features_df, effective_columns, fold_plan)
    -> OODStageResult
"""
from __future__ import annotations

import logging
import math
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from extrapolation_discovery_platform._utils import safe_array
from extrapolation_discovery_platform.features import FeatureCatalog, FeatureSetName
from extrapolation_discovery_platform.multicollinearity import (
    MulticollinearityReport,
    detect_target_leakage,
    remove_constant_columns,
    remove_perfect_collinear,
    run_phase0_multicollinearity,
)
from extrapolation_discovery_platform.ood import OODDetector, OODResult
from extrapolation_discovery_platform.splitters import (
    CompositionBlockSplitter,
    ElementExclusionSplitter,
    RandomCVSplitter,
)
from extrapolation_discovery_platform.workflows import RunResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 結果コンテナ
# ---------------------------------------------------------------------------

@dataclass
class PreprocessResult:
    """Stage 1 の出力。runner/individual の両方に渡す。

    Attributes
    ----------
    effective_cols : dict {fs_key: [col, ...]}
        多重共線性除去・リーク除去・特徴量選択後の有効列。
    fold_plan : dict {policy_key: [(train_idx, test_idx), ...]}
        分割計画。CompositionBlock / ElementExclusion / RandomCV_seedN。
    mc_reports : dict {fs_key: MulticollinearityReport}
        多重共線性診断レポート。
    fs_summaries : dict
        特徴量選択サマリ（GUI 表示用）。
    active_policies : list of str
        fold_plan に実際に含まれる分割ポリシー名。
    """
    effective_cols: Dict[str, List[str]] = field(default_factory=dict)
    fold_plan: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = field(default_factory=dict)
    mc_reports: Dict[str, MulticollinearityReport] = field(default_factory=dict)
    fs_summaries: Dict[str, Any] = field(default_factory=dict)
    active_policies: List[str] = field(default_factory=list)
    elapsed_sec: float = 0.0
    success: bool = False
    error_message: str = ""


@dataclass
class TrainResult:
    """Stage 2 の出力。OOD 情報は含まない。

    Attributes
    ----------
    runs : list of RunResult
        全 fold の学習結果。
    rmse_test_mean / r2_test_mean など
        fold 平均のメトリクス（GUI サマリ表示用）。
    n_folds_executed : int
        実際に完了した fold 数。
    n_features_used : int
        Stage 1 から引き継いだ有効列数。
    """
    runs: List[RunResult] = field(default_factory=list)
    rmse_test_mean:  float = float("nan")
    rmse_test_std:   float = float("nan")
    rmse_train_mean: float = float("nan")
    r2_test_mean:    float = float("nan")
    r2_test_std:     float = float("nan")
    mae_test_mean:   float = float("nan")
    n_folds_executed: int = 0
    n_features_used:  int = 0
    elapsed_sec: float = 0.0
    success: bool = False
    error_message: str = ""


@dataclass
class OODStageResult:
    """Stage 3 の出力。RunResult とは完全に分離。

    Attributes
    ----------
    ood_result : OODResult
        全 fold アンサンブル後の OOD 検出結果（primary fold 基準）。
    primary_train_idx / primary_test_idx : np.ndarray
        GUI の OOD Map 可視化に使う primary fold のインデックス。
    ensemble_scores : np.ndarray
        全サンプルの全 fold 累積スコア（デバッグ・詳細分析用）。
    """
    ood_result: Optional[OODResult] = None
    primary_train_idx: Optional[np.ndarray] = None
    primary_test_idx:  Optional[np.ndarray] = None
    ensemble_scores:   Optional[np.ndarray] = None
    elapsed_sec: float = 0.0
    success: bool = False
    error_message: str = ""


# ---------------------------------------------------------------------------
# Stage 1: 前処理
# ---------------------------------------------------------------------------

def stage1_preprocess(
    features_df: pd.DataFrame,
    target: pd.Series,
    compositions_df: Optional[pd.DataFrame],
    feature_set_names: List[str],
    workflow_names: List[str],
    seeds: List[int],
    active_policies: List[str],
    leak_auto_exclude: bool = True,
    leak_corr_threshold: float = 0.85,
    generic_csv_mode: bool = False,
    n_folds: int = 5,
) -> PreprocessResult:
    """Stage 1: 前処理。

    runner.py / individual_runner.py の両方が呼ぶ共通実装。
    同一引数なら同一の PreprocessResult を返すことで、
    一括計算と個別計算の結果一致を保証する。

    処理順序（リーク防止のため分割→特徴量選択の順）:
        1. 多重共線性・リーク検出
        2. 有効列決定（定数列除去 + 完全共線除去 + leak_suspect除去）
        3. 分割計算（fold_plan）  ← 特徴量選択より先（訓練 idx が必要）
        4. 特徴量選択（訓練データのみ）← データリーク防止

    Parameters
    ----------
    feature_set_names : list of str
        HEA モード: ["FS_BASE", "FS_ALL", ...] など FeatureSetName.value の文字列。
        generic_csv_mode=True のとき無視し features_df 全列を 1 セットとして使用。
    active_policies : list of str
        有効にする分割ポリシー。["CompositionBlock", "ElementExclusion"] が推奨。
        "RandomCV" を含めるとデータリーク懸念あり（デフォルト無効）。
    n_folds : int
        分割数（デフォルト 5）。2〜10 の範囲で指定する。
        小さいほど1 fold あたりの訓練データが増え、大きいほど評価が安定する。
    """
    t0 = time.time()
    result = PreprocessResult(active_policies=list(active_policies))

    try:
        # ── Step 1: 多重共線性・リーク検出 ───────────────────────────
        if generic_csv_mode:
            # generic CSV モード: features_df 全列に直接 VIF + リーク検出を適用
            mc_reports = _run_generic_mc(features_df, target, leak_corr_threshold)
            fs_key_list = ["generic"]
        else:
            # HEA モード: FeatureSetName ベースで各 FS の MC 解析を実行
            fs_enums: List[FeatureSetName] = []
            for name in feature_set_names:
                try:
                    fs_enums.append(FeatureSetName(name))
                except ValueError:
                    logger.warning("Stage1: 未知の FS '%s' をスキップ", name)
            mc_reports = (
                run_phase0_multicollinearity(
                    features_df, fs_enums, workflow_names, len(features_df),
                    target=target,
                    leak_corr_threshold=leak_corr_threshold,
                ) if fs_enums else {}
            )
            fs_key_list = list(feature_set_names)

        result.mc_reports = mc_reports

        # ── Step 2: 有効列決定（FS ごとに drop + leak 除外） ─────────
        effective_cols: Dict[str, List[str]] = {}
        for fs_key in fs_key_list:
            # 初期列リストを取得
            if generic_csv_mode:
                orig = list(features_df.columns)
            else:
                try:
                    orig = [c for c in FeatureCatalog.columns(FeatureSetName(fs_key))
                            if c in features_df.columns]
                except (ValueError, KeyError):
                    logger.warning("Stage1 [%s]: FS列取得失敗 — 全列を使用", fs_key)
                    orig = list(features_df.columns)

            # MC レポートに基づいて不要列を除去
            rpt = mc_reports.get(fs_key)
            if rpt is not None:
                drop_set = set(rpt.dropped_constant + rpt.dropped_perfect)
                orig = [c for c in orig if c not in drop_set]
                if leak_auto_exclude and rpt.leak_suspects:
                    n_before = len(orig)
                    orig = [c for c in orig if c not in rpt.leak_suspects]
                    logger.info("Stage1 [%s]: leak除外 %d列", fs_key, n_before - len(orig))

            if not orig:
                logger.warning("Stage1 [%s]: 有効列 0 — このFSをスキップ", fs_key)
                continue

            effective_cols[fs_key] = orig
            logger.info("Stage1 [%s]: 有効列 %d 列", fs_key, len(orig))

        result.effective_cols = effective_cols

        # ── Step 3: 分割計算（特徴量選択より先に実行） ───────────────
        # 特徴量選択は訓練 idx のスライスが必要なため、分割を先に行う。
        _seed0 = seeds[0] if seeds else 42
        fold_plan: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}

        if "CompositionBlock" in active_policies:
            if compositions_df is not None:
                try:
                    cb = CompositionBlockSplitter(n_folds=n_folds, seed=_seed0)
                    folds = list(cb.split(features_df, target, compositions=compositions_df))
                    if folds:
                        fold_plan["CompositionBlock"] = folds
                        logger.info("Stage1: CompositionBlock %d folds", len(folds))
                    else:
                        logger.warning("Stage1: CompositionBlock — fold 0件")
                except Exception:
                    logger.warning("Stage1: CompositionBlock 分割失敗:\n%s",
                                   traceback.format_exc())
            else:
                logger.warning("Stage1: CompositionBlock — compositions_df が None")

        if "ElementExclusion" in active_policies:
            if compositions_df is not None:
                try:
                    ee = ElementExclusionSplitter()
                    folds = list(ee.split(features_df, target, compositions=compositions_df))
                    if folds:
                        fold_plan["ElementExclusion"] = folds
                        logger.info("Stage1: ElementExclusion %d folds", len(folds))
                    else:
                        logger.warning("Stage1: ElementExclusion — fold 0件")
                except Exception:
                    logger.warning("Stage1: ElementExclusion 分割失敗:\n%s",
                                   traceback.format_exc())
            else:
                logger.warning("Stage1: ElementExclusion — compositions_df が None")

        if "RandomCV" in active_policies:
            # RandomCV は seed ごとに別キーで保持（evaluation.py の base_rmse 計算で参照）
            for seed in seeds:
                try:
                    rc = RandomCVSplitter(n_folds=n_folds, seed=seed)
                    folds = list(rc.split(features_df, target, compositions=compositions_df))
                    if folds:
                        fold_plan[f"RandomCV_seed{seed}"] = folds
                        logger.info("Stage1: RandomCV seed=%d %d folds", seed, len(folds))
                except Exception:
                    logger.warning("Stage1: RandomCV seed=%d 失敗:\n%s",
                                   seed, traceback.format_exc())

        # 全ポリシーで fold が空の場合のフォールバック
        if not fold_plan:
            logger.warning(
                "Stage1: 全分割ポリシーで fold 0。"
                "RandomCV seed=%d でフォールバック", _seed0
            )
            rc_fb = RandomCVSplitter(n_folds=n_folds, seed=_seed0)
            folds = list(rc_fb.split(features_df, target, compositions=compositions_df))
            if folds:
                fold_plan[f"RandomCV_seed{_seed0}"] = folds

        result.fold_plan = fold_plan

        # ── Step 4: 特徴量選択（訓練データのみ・リーク防止） ─────────
        # CompositionBlock の最初の fold の train_idx をスライスに使用。
        # generic CSV モードでは特徴量選択をスキップ（汎用データに FS 基準なし）。
        primary_train_idx: Optional[np.ndarray] = None
        if "CompositionBlock" in fold_plan and fold_plan["CompositionBlock"]:
            primary_train_idx = fold_plan["CompositionBlock"][0][0]
        elif fold_plan:
            # CompositionBlock がない場合は先頭 fold の train_idx を使用
            primary_train_idx = next(iter(fold_plan.values()))[0][0]

        if primary_train_idx is not None and not generic_csv_mode:
            from extrapolation_discovery_platform.feature_selection import run_feature_selection
            fs_summaries: Dict[str, Any] = {}
            for fs_key, cols in list(effective_cols.items()):
                if len(cols) <= 3:
                    # 列数が少なすぎる場合は選択不要
                    continue
                try:
                    X_tr = features_df.iloc[primary_train_idx][cols]
                    y_tr = target.iloc[primary_train_idx]
                    summary = run_feature_selection(
                        X_tr, y_tr,
                        methods=None,           # 全手法: Lasso, AIC, BIC, ARD
                        consensus_threshold=2,  # 2手法以上で選択された列を採用
                        feature_set=fs_key,
                    )
                    fs_summaries[fs_key] = summary

                    # コンセンサス特徴量（2手法以上で選択）が十分あれば採用
                    # ただし元の列数の 20% 未満になる場合はスキップ
                    min_cols = max(3, len(cols) // 5)
                    consensus = summary.consensus_features or []
                    if len(consensus) >= min_cols:
                        effective_cols[fs_key] = consensus
                        logger.info("Stage1 特徴量選択 [%s]: %d→%d (consensus)",
                                    fs_key, len(cols), len(effective_cols[fs_key]))
                    else:
                        # Lasso フォールバック：最低 min_cols 列を保証
                        lasso = summary.results.get("Lasso")
                        lasso_feats = (lasso.selected_features if lasso else []) or []
                        if len(lasso_feats) >= min_cols:
                            effective_cols[fs_key] = lasso_feats
                            logger.info("Stage1 特徴量選択 [%s]: %d→%d (lasso fallback)",
                                        fs_key, len(cols), len(effective_cols[fs_key]))
                        else:
                            # 選択結果が不十分 → 全列を維持（特徴量選択をスキップ）
                            logger.info(
                                "Stage1 特徴量選択 [%s]: 選択結果が不十分 "
                                "(consensus=%d, lasso=%d, min_required=%d) — 全 %d 列を維持",
                                fs_key, len(consensus), len(lasso_feats), min_cols, len(cols),
                            )
                except Exception:
                    logger.warning("Stage1 特徴量選択失敗 [%s] — 全列を維持:\n%s",
                                   fs_key, traceback.format_exc())
            result.fs_summaries = fs_summaries

        result.effective_cols = effective_cols
        result.elapsed_sec = time.time() - t0
        result.success = True
        logger.info(
            "Stage1 完了: %d FS, %d split-policies, %.2fs",
            len(effective_cols), len(fold_plan), result.elapsed_sec,
        )

    except Exception:
        result.error_message = traceback.format_exc()
        result.elapsed_sec = time.time() - t0
        result.success = False
        logger.exception("Stage1 前処理失敗")

    return result


# ---------------------------------------------------------------------------
# Stage 2: ML 学習（OOD なし）
# ---------------------------------------------------------------------------

def stage2_train(
    preprocess_result: PreprocessResult,
    features_df: pd.DataFrame,
    target: pd.Series,
    workflow_name: str,
    split_policy_name: str,
    feature_set_name: str,
    quick: bool = True,
    dim_reduction: bool = True,
    seed: int = 42,
    generic_csv_mode: bool = False,
) -> TrainResult:
    """Stage 2: ML 学習。

    PreprocessResult の effective_cols と fold_plan を参照し、
    指定された (WF × FS × split_policy) で全 fold を学習する。
    OOD 計算はここに含まない（Stage 3 で独立して実行）。

    Parameters
    ----------
    split_policy_name : str
        "CompositionBlock" / "ElementExclusion" / "RandomCV" のいずれか。
        fold_plan に対応するキーが存在しない場合はエラーを返す。
    """
    t0 = time.time()
    result = TrainResult()

    try:
        # _WORKFLOW_FACTORIES は individual_runner.py で定義した
        # {wf_name: (quick, dim_reduction) -> BaseWorkflow} の辞書
        from extrapolation_discovery_platform.individual_runner import _WORKFLOW_FACTORIES

        # ── 有効列を取得 ──────────────────────────────────────────────
        fs_key = "generic" if generic_csv_mode else feature_set_name
        effective_cols = preprocess_result.effective_cols.get(fs_key)

        if not effective_cols:
            # フォールバック: Stage1 がスキップされた場合など
            if generic_csv_mode:
                effective_cols = list(features_df.columns)
            else:
                try:
                    effective_cols = [
                        c for c in FeatureCatalog.columns(FeatureSetName(feature_set_name))
                        if c in features_df.columns
                    ]
                except (ValueError, KeyError):
                    effective_cols = list(features_df.columns)

        if not effective_cols:
            raise ValueError(f"有効列が空: feature_set='{feature_set_name}'")

        # ── fold を取得 ───────────────────────────────────────────────
        fold_plan = preprocess_result.fold_plan
        if split_policy_name == "RandomCV":
            # seed に対応する RandomCV キーを優先し、なければ任意の RC fold を使用
            splits = fold_plan.get(f"RandomCV_seed{seed}")
            if splits is None:
                for k, v in fold_plan.items():
                    if k.startswith("RandomCV_") and v:
                        splits = v
                        logger.warning(
                            "Stage2: seed=%d の RandomCV fold なし。%s を使用", seed, k
                        )
                        break
        else:
            splits = fold_plan.get(split_policy_name)

        if not splits:
            raise ValueError(
                f"Stage2: '{split_policy_name}' の fold が存在しない。"
                f"利用可能 keys: {list(fold_plan.keys())}"
            )

        # ── ワークフロー取得 ──────────────────────────────────────────
        factory = _WORKFLOW_FACTORIES.get(workflow_name)
        if factory is None:
            raise ValueError(
                f"未知のワークフロー '{workflow_name}'. "
                f"使用可能: {sorted(_WORKFLOW_FACTORIES)}"
            )

        # ── 各 fold で独立学習 ───────────────────────────────────────
        X = pd.DataFrame(safe_array(features_df[effective_cols]), columns=effective_cols)
        y = target.reset_index(drop=True)
        runs: List[RunResult] = []

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            X_tr = pd.DataFrame(safe_array(X.iloc[train_idx]), columns=effective_cols)
            X_te = pd.DataFrame(safe_array(X.iloc[test_idx]),  columns=effective_cols)
            y_tr = y.iloc[train_idx].reset_index(drop=True)
            y_te = y.iloc[test_idx].reset_index(drop=True)

            # 毎 fold で新インスタンス（fold 間の独立性を保証）
            wf = factory(quick, dim_reduction)
            run = wf.run(
                X_tr, y_tr, X_te, y_te,
                seed=seed,
                feature_set=feature_set_name,
                split_policy=split_policy_name,
                fold=fold_idx,
                test_indices=np.asarray(test_idx),
            )
            runs.append(run)
            logger.info(
                "Stage2 [%s/%s/%s] fold=%d: RMSE=%.4f R²=%.4f",
                workflow_name, feature_set_name, split_policy_name,
                fold_idx, run.rmse_test, run.r2_test,
            )

        result.runs = runs
        result.n_folds_executed = len(runs)
        result.n_features_used  = len(effective_cols)

        # ── 集約メトリクス ────────────────────────────────────────────
        def _mean(xs: list) -> float:
            return float(np.mean(xs)) if xs else float("nan")
        def _std(xs: list) -> float:
            return float(np.std(xs)) if len(xs) > 1 else 0.0

        valid_te  = [r.rmse_test  for r in runs if r.rmse_test  > 0 and math.isfinite(r.rmse_test)]
        valid_tr  = [r.rmse_train for r in runs if r.rmse_train > 0 and math.isfinite(r.rmse_train)]
        valid_mae = [r.mae_test   for r in runs if r.mae_test   > 0 and math.isfinite(r.mae_test)]
        valid_r2  = [r.r2_test    for r in runs if math.isfinite(r.r2_test)]

        result.rmse_test_mean  = _mean(valid_te)
        result.rmse_test_std   = _std(valid_te)
        result.rmse_train_mean = _mean(valid_tr)
        result.mae_test_mean   = _mean(valid_mae)
        result.r2_test_mean    = _mean(valid_r2)
        result.r2_test_std     = _std(valid_r2)
        result.elapsed_sec = time.time() - t0
        result.success = True

    except Exception:
        result.error_message = traceback.format_exc()
        result.elapsed_sec = time.time() - t0
        result.success = False
        logger.exception(
            "Stage2 学習失敗: %s/%s/%s",
            workflow_name, feature_set_name, split_policy_name,
        )

    return result


# ---------------------------------------------------------------------------
# Stage 3: OOD 検出（学習と完全独立）
# ---------------------------------------------------------------------------

def stage3_detect_ood(
    features_df: pd.DataFrame,
    effective_columns: List[str],
    fold_plan: Dict[str, List[Tuple[np.ndarray, np.ndarray]]],
) -> OODStageResult:
    """Stage 3: OOD 検出。

    Stage 2 とは完全に独立した処理。
    PreprocessResult の fold_plan を使い、全 fold × 全 split policy で
    独立した OODDetector を fit/score し、スコアをアンサンブルする。

    設計原則:
        - RunResult には OOD 情報を含めない（分離の原則）
        - 各 fold で独立 fit → train set が変われば OOD スコアも変わる
        - 全 fold スコアをアンサンブル（平均）して代表値を決定
        - primary fold（CompositionBlock の先頭 fold 優先）の結果を GUI に使用
        - GUI の OOD Map タブ・OOD サマリーのみがこの出力を参照する

    Parameters
    ----------
    effective_columns : list of str
        Stage 1 の effective_cols から取得した OOD 計算対象列。
    fold_plan : dict
        Stage 1 の fold_plan。全 split policy の train/test インデックス。
    """
    t0 = time.time()
    result = OODStageResult()

    try:
        # OOD 計算対象列を features_df に存在するものに絞る
        ood_cols = [c for c in effective_columns if c in features_df.columns]
        if not ood_cols:
            raise ValueError(
                f"effective_columns の列が features_df に存在しない: "
                f"{effective_columns[:5]}"
            )

        # 全 split policy の全 fold を統合してアンサンブルに使用
        all_folds: List[Tuple[np.ndarray, np.ndarray]] = []
        for folds in fold_plan.values():
            all_folds.extend(folds)

        if not all_folds:
            raise ValueError("fold_plan が空。Stage1 が正常に完了していない。")

        # primary fold の決定（CompositionBlock 優先、なければ先頭 fold）
        if "CompositionBlock" in fold_plan and fold_plan["CompositionBlock"]:
            primary_train_idx, primary_test_idx = fold_plan["CompositionBlock"][0]
        else:
            primary_train_idx, primary_test_idx = all_folds[0]

        n_samples = len(features_df)
        X_arr = safe_array(features_df[ood_cols])

        # fold ごとにスコアを累積（アンサンブル用）
        score_sum   = np.zeros(n_samples, dtype=np.float64)
        score_count = np.zeros(n_samples, dtype=np.int32)
        primary_res: Optional[OODResult] = None

        for tr_idx, te_idx in all_folds:
            if len(tr_idx) < 2 or len(te_idx) < 1:
                continue
            try:
                actual_k = min(10, len(tr_idx) - 1)
                X_tr = pd.DataFrame(X_arr[tr_idx], columns=ood_cols)
                X_te = pd.DataFrame(X_arr[te_idx], columns=ood_cols)

                # 各 fold で独立 fit（train が変われば OOD も変わる）
                detector = OODDetector(k=actual_k)
                detector.fit(X_tr)
                fold_res = detector.score(X_te)

                score_sum[te_idx]   += fold_res.composite_scores
                score_count[te_idx] += 1

                # primary fold を特定（bytes 比較で同一インデックスを判定）
                if (tr_idx.tobytes() == primary_train_idx.tobytes()
                        and te_idx.tobytes() == primary_test_idx.tobytes()):
                    primary_res = fold_res

            except Exception:
                logger.warning("Stage3: OOD fold 失敗 (non-fatal):\n%s",
                               traceback.format_exc())

        # primary が見つからなかった場合のフォールバック
        if primary_res is None:
            for tr_idx, te_idx in all_folds:
                if len(tr_idx) >= 2:
                    try:
                        X_tr = pd.DataFrame(X_arr[tr_idx], columns=ood_cols)
                        X_te = pd.DataFrame(X_arr[te_idx], columns=ood_cols)
                        det = OODDetector(k=min(10, len(tr_idx) - 1))
                        det.fit(X_tr)
                        primary_res = det.score(X_te)
                        primary_test_idx = te_idx
                        break
                    except Exception:
                        pass

        if primary_res is None:
            raise ValueError("全 fold で OOD 計算失敗（有効 fold なし）")

        # primary test set のアンサンブルスコアを計算
        te = primary_test_idx
        scored = score_count[te] > 0
        avg_scores = np.where(
            scored,
            score_sum[te] / np.maximum(score_count[te], 1),
            primary_res.composite_scores,  # スコアなし fold は primary を使用
        )
        is_ood_avg = avg_scores > primary_res.ood_threshold
        n_ood = int(is_ood_avg.sum())

        result.ood_result = OODResult(
            mahalanobis_scores=primary_res.mahalanobis_scores,
            knn_scores=primary_res.knn_scores,
            composite_scores=np.ascontiguousarray(avg_scores),
            is_ood=np.ascontiguousarray(is_ood_avg),
            ood_threshold=primary_res.ood_threshold,
            ood_ratio=n_ood / max(len(avg_scores), 1),
            n_total=len(avg_scores),
            n_ood=n_ood,
        )
        result.ensemble_scores   = score_sum
        result.primary_train_idx = np.ascontiguousarray(primary_train_idx)
        result.primary_test_idx  = np.ascontiguousarray(primary_test_idx)
        result.elapsed_sec = time.time() - t0
        result.success = True

        logger.info(
            "Stage3 OOD完了: %d/%d OOD (%.1f%%), %d folds, %.2fs",
            n_ood, len(avg_scores),
            100 * n_ood / max(len(avg_scores), 1),
            len(all_folds), result.elapsed_sec,
        )

    except Exception:
        result.error_message = traceback.format_exc()
        result.elapsed_sec = time.time() - t0
        result.success = False
        logger.exception("Stage3 OOD検出失敗")

    return result


# ---------------------------------------------------------------------------
# Internal helper: generic CSV モード用の簡易 MC レポート
# ---------------------------------------------------------------------------

def _run_generic_mc(
    features_df: pd.DataFrame,
    target: pd.Series,
    leak_corr_threshold: float,
) -> Dict[str, MulticollinearityReport]:
    """generic CSV モード専用の多重共線性レポート生成。

    HEA モードの run_phase0_multicollinearity は FeatureSetName を前提と
    しているため、汎用 CSV では直接 VIF + リーク検出を適用する。
    """
    try:
        from extrapolation_discovery_platform.multicollinearity import compute_vif
        X_tmp, dropped_c = remove_constant_columns(features_df.copy())
        X_tmp, dropped_p = remove_perfect_collinear(X_tmp)
        n_after = X_tmp.shape[1]

        try:
            # VIF は列数が多いと計算コストが高いため 50 列以下のみ実行
            vif = compute_vif(X_tmp) if n_after <= 50 else None
        except Exception:
            vif = None

        high_vif   = int((vif > 10).sum()) if vif is not None else 0
        high_ratio = high_vif / max(n_after, 1)
        mc_level   = ("high"     if high_ratio > 0.5
                      else "moderate" if high_ratio > 0.2
                      else "low")

        # リーク検出
        leak: Dict[str, float] = {}
        if target is not None:
            try:
                leak = detect_target_leakage(X_tmp, target, threshold=leak_corr_threshold)
            except Exception:
                pass

        rpt = MulticollinearityReport(
            feature_set="generic",
            n_features_before=len(features_df.columns),
            n_features_after=n_after,
            dropped_constant=list(dropped_c),
            dropped_perfect=list(dropped_p),
            vif_series=vif if vif is not None else pd.Series(dtype=float),
            high_vif_count=high_vif,
            moderate_vif_count=0,
            multicollinearity_level=mc_level,
            recommended_workflows=[],
            blocked_workflows=[],
            leak_suspects=leak,
        )
        return {"generic": rpt}

    except Exception:
        logger.warning("generic MC 失敗:\n%s", traceback.format_exc())
        return {}
