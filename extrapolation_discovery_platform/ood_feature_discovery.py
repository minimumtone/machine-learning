"""
OOD-Aware Feature Discovery
=============================

OOD 検出後に「OOD に近い内挿サンプル」を特定し、
追加特徴量を組み込んで再学習することで外挿性能の向上を探索する。

フロー:
  1. identify_boundary_samples()
       OOD スコアが上位にある「境界サンプル」を特定
  2. augment_dataset()
       元データ + 境界サンプルを合わせた拡張データセットを構築
  3. run_feature_discovery_round()
       追加特徴量候補を 1 つずつ組み込んで再学習し、OOD 予測性能を比較
  4. FeatureDiscoveryResult
       結果コンテナ（WF × 候補特徴量 の RMSE@OOD を格納）
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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 結果コンテナ
# ---------------------------------------------------------------------------

@dataclass
class BoundarySampleInfo:
    """OOD 境界付近のサンプル情報。"""
    # OOD スコアが threshold 以上 (OOD フラグ = True)
    ood_indices: np.ndarray           # 元データフレームのインデックス
    ood_scores: np.ndarray            # composite_score

    # 境界サンプル: score が threshold × margin 以内（内挿側の境界）
    boundary_indices: np.ndarray      # 元データフレームのインデックス
    boundary_scores: np.ndarray       # composite_score
    ood_threshold: float
    margin: float                     # threshold × (1 + margin) を境界の上限とする

    @property
    def n_ood(self) -> int:
        return len(self.ood_indices)

    @property
    def n_boundary(self) -> int:
        return len(self.boundary_indices)


@dataclass
class DiscoveryRoundResult:
    """1 候補特徴量での再学習結果。"""
    candidate_feature: str            # 追加した特徴量名（"" = ベースライン）
    workflow: str
    feature_set: str
    split_policy: str

    # 元データのみでの性能（ベースライン）
    baseline_rmse: float = float("nan")
    baseline_r2:   float = float("nan")

    # OOD サンプルへの予測性能（再学習後）
    ood_rmse: float = float("nan")
    ood_r2:   float = float("nan")

    # 拡張データ全体での CV 性能
    augmented_rmse: float = float("nan")
    augmented_r2:   float = float("nan")

    improvement: float = float("nan")  # (baseline_rmse - ood_rmse) / baseline_rmse
    success: bool = False
    error_message: str = ""
    elapsed_sec: float = 0.0


@dataclass
class FeatureDiscoveryResult:
    """全候補・全 WF の探索結果。"""
    rounds: List[DiscoveryRoundResult] = field(default_factory=list)
    best_feature: str = ""
    best_improvement: float = float("nan")
    n_boundary_samples: int = 0
    elapsed_sec: float = 0.0
    success: bool = False
    error_message: str = ""


# ---------------------------------------------------------------------------
# Step 1: OOD 境界サンプル特定
# ---------------------------------------------------------------------------

def identify_boundary_samples(
    ood_result,           # OODResult (composite_scores, is_ood, ood_threshold)
    n_ood_samples: int,
    margin: float = 0.5,  # threshold × (1 + margin) を境界上限とする
) -> BoundarySampleInfo:
    """OOD スコアに基づき OOD サンプルと境界サンプルを特定する。

    Parameters
    ----------
    ood_result : OODResult
        Stage 3 の出力。composite_scores が全テストサンプルのスコア。
    n_ood_samples : int
        OOD と判定するサンプル数（上位 N 件）。0 → threshold 以上を全選択。
    margin : float
        境界サンプルの範囲係数。threshold 以下かつ threshold × (1+margin) 以内。
    """
    scores    = np.asarray(ood_result.composite_scores)
    threshold = float(ood_result.ood_threshold)
    n         = len(scores)
    all_idx   = np.arange(n)

    # OOD サンプル（スコア ≥ threshold）
    ood_mask = scores >= threshold
    if n_ood_samples > 0:
        # 上位 N 件に限定
        sorted_desc = np.argsort(-scores)
        ood_idx   = sorted_desc[:min(n_ood_samples, n_ood_samples)]
    else:
        ood_idx   = all_idx[ood_mask]
    ood_scores = scores[ood_idx]

    # 境界サンプル（threshold × margin 以内の内挿側）
    upper = threshold * (1.0 + margin)
    boundary_mask = (~ood_mask) & (scores >= threshold * (1.0 - margin))
    boundary_idx  = all_idx[boundary_mask]
    boundary_scr  = scores[boundary_idx]

    logger.info(
        "boundary: n_ood=%d  n_boundary=%d  threshold=%.4f  margin=%.2f",
        len(ood_idx), len(boundary_idx), threshold, margin,
    )

    return BoundarySampleInfo(
        ood_indices=ood_idx,
        ood_scores=ood_scores,
        boundary_indices=boundary_idx,
        boundary_scores=boundary_scr,
        ood_threshold=threshold,
        margin=margin,
    )


# ---------------------------------------------------------------------------
# Step 2: データセット拡張
# ---------------------------------------------------------------------------

def augment_dataset(
    features_df: pd.DataFrame,
    target: pd.Series,
    boundary_info: BoundarySampleInfo,
    ood_test_idx: Optional[np.ndarray],
    extra_features_df: Optional[pd.DataFrame] = None,
    candidate_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.Series, np.ndarray, np.ndarray]:
    """境界サンプルを訓練データに追加した拡張データセットを構築する。

    Returns
    -------
    X_aug : pd.DataFrame
        拡張後の特徴量行列（元データ + 境界サンプルの重複を除去）
    y_aug : pd.Series
        拡張後のターゲット
    train_idx_aug : np.ndarray
        拡張データ内の訓練インデックス（元の訓練データ + 境界サンプル）
    ood_eval_idx : np.ndarray
        OOD 評価用インデックス（拡張データ内での位置）
    """
    n_orig = len(features_df)

    # 追加特徴量を結合
    if extra_features_df is not None and candidate_col is not None:
        if candidate_col in extra_features_df.columns:
            col_data = extra_features_df[[candidate_col]].reset_index(drop=True)
            X_base = pd.concat(
                [features_df.reset_index(drop=True), col_data], axis=1
            )
        else:
            X_base = features_df.copy()
    else:
        X_base = features_df.copy()

    y_base = target.reset_index(drop=True)

    # 境界サンプルの OOD テストインデックスへの変換
    # ood_test_idx が与えられている場合はそれを使ってオリジナルインデックスに変換
    boundary_orig_idx = (
        ood_test_idx[boundary_info.boundary_indices]
        if ood_test_idx is not None
        else boundary_info.boundary_indices
    )
    # 有効なインデックスのみ
    boundary_orig_idx = boundary_orig_idx[boundary_orig_idx < n_orig]

    if len(boundary_orig_idx) == 0:
        # 境界サンプルなし → 元データをそのまま返す
        train_idx = np.arange(n_orig)
        ood_idx   = (
            ood_test_idx[boundary_info.ood_indices]
            if ood_test_idx is not None
            else boundary_info.ood_indices
        )
        ood_idx = ood_idx[ood_idx < n_orig]
        return X_base, y_base, train_idx, ood_idx

    # 境界サンプルを訓練に追加（重複を除いたすべての元データが訓練、
    # 境界サンプルも訓練に追加する構造）
    # → 元データ全体 (0..n_orig-1) + 境界サンプルの複製（拡張行）
    boundary_X = X_base.iloc[boundary_orig_idx].reset_index(drop=True)
    boundary_y = y_base.iloc[boundary_orig_idx].reset_index(drop=True)

    X_aug = pd.concat([X_base, boundary_X], ignore_index=True)
    y_aug = pd.concat([y_base, boundary_y], ignore_index=True)

    n_aug = len(X_aug)
    # 訓練: 元データ全体 + 境界追加行
    boundary_aug_idx = np.arange(n_orig, n_aug)
    train_idx = np.concatenate([np.arange(n_orig), boundary_aug_idx])

    # OOD 評価: 元インデックス内の OOD サンプル
    ood_orig_idx = (
        ood_test_idx[boundary_info.ood_indices]
        if ood_test_idx is not None
        else boundary_info.ood_indices
    )
    ood_eval_idx = ood_orig_idx[ood_orig_idx < n_orig]

    logger.info(
        "augment: n_orig=%d  n_boundary_added=%d  n_aug=%d  n_ood_eval=%d",
        n_orig, len(boundary_orig_idx), n_aug, len(ood_eval_idx),
    )

    return X_aug, y_aug, train_idx, ood_eval_idx


# ---------------------------------------------------------------------------
# Step 3: 特徴量探索ラウンド
# ---------------------------------------------------------------------------

def run_feature_discovery_round(
    workflow_name: str,
    feature_set_name: str,
    split_policy: str,
    features_df: pd.DataFrame,
    target: pd.Series,
    compositions_df: Optional[pd.DataFrame],
    ood_result,
    ood_test_idx: np.ndarray,
    candidate_feature: str,
    extra_features_df: Optional[pd.DataFrame],
    seed: int = 42,
    n_folds: int = 5,
    quick: bool = True,
    n_ood_samples: int = 0,
    boundary_margin: float = 0.5,
) -> DiscoveryRoundResult:
    """1 候補特徴量での再学習と OOD 予測性能評価を実行する。"""
    t0 = time.time()
    result = DiscoveryRoundResult(
        candidate_feature=candidate_feature,
        workflow=workflow_name,
        feature_set=feature_set_name,
        split_policy=split_policy,
    )

    try:
        from extrapolation_discovery_platform.pipeline import (
            stage1_preprocess, stage2_train,
        )
        from extrapolation_discovery_platform.individual_runner import _WORKFLOW_FACTORIES

        # ── ベースライン（元データのみ、追加特徴量なし） ─────────────────
        prep_base = stage1_preprocess(
            features_df=features_df,
            target=target,
            compositions_df=compositions_df,
            feature_set_names=[feature_set_name],
            workflow_names=[workflow_name],
            seeds=[seed],
            active_policies=[split_policy],
            n_folds=n_folds,
        )
        if not prep_base.success:
            raise RuntimeError(f"Stage1 失敗: {prep_base.error_message}")

        tr_base = stage2_train(
            prep_base, features_df, target,
            workflow_name, split_policy, feature_set_name,
            quick=quick, seed=seed,
        )
        if not tr_base.success:
            raise RuntimeError(f"Stage2 失敗: {tr_base.error_message}")

        result.baseline_rmse = tr_base.rmse_test_mean
        result.baseline_r2   = tr_base.r2_test_mean

        # ── OOD 境界サンプルを特定 ──────────────────────────────────────
        boundary = identify_boundary_samples(
            ood_result, n_ood_samples, margin=boundary_margin,
        )

        # ── 拡張データセット構築 ─────────────────────────────────────────
        X_aug, y_aug, train_aug_idx, ood_eval_idx = augment_dataset(
            features_df=features_df,
            target=target,
            boundary_info=boundary,
            ood_test_idx=ood_test_idx,
            extra_features_df=extra_features_df,
            candidate_col=candidate_feature if candidate_feature else None,
        )

        if len(ood_eval_idx) == 0:
            logger.warning("OOD 評価サンプルが 0 件 — スコアを計算できません")
            result.elapsed_sec = time.time() - t0
            result.success = True
            return result

        # ── 拡張データで再学習 ───────────────────────────────────────────
        # 有効列を取得（追加特徴量を含む）
        aug_fs_name = feature_set_name
        prep_aug = stage1_preprocess(
            features_df=X_aug,
            target=y_aug,
            compositions_df=compositions_df,
            feature_set_names=[aug_fs_name],
            workflow_names=[workflow_name],
            seeds=[seed],
            active_policies=[split_policy],
            n_folds=n_folds,
        )
        if not prep_aug.success:
            raise RuntimeError(f"拡張 Stage1 失敗: {prep_aug.error_message}")

        # 拡張データの全行を訓練に使う（fold は元の fold を踏まえ再計算）
        factory = _WORKFLOW_FACTORIES.get(workflow_name)
        if factory is None:
            raise ValueError(f"未知のWF: {workflow_name}")

        effective_cols = prep_aug.effective_cols.get(aug_fs_name, list(X_aug.columns))
        effective_cols = [c for c in effective_cols if c in X_aug.columns]

        X_tr = X_aug.iloc[train_aug_idx][effective_cols]
        y_tr = y_aug.iloc[train_aug_idx]
        X_ood = X_aug.iloc[ood_eval_idx][effective_cols]
        y_ood = y_aug.iloc[ood_eval_idx]

        from extrapolation_discovery_platform._utils import safe_array
        wf = factory(quick, True)
        run_aug = wf.run(
            pd.DataFrame(safe_array(X_tr), columns=effective_cols),
            y_tr.reset_index(drop=True),
            pd.DataFrame(safe_array(X_ood), columns=effective_cols),
            y_ood.reset_index(drop=True),
            seed=seed,
            feature_set=aug_fs_name,
            split_policy=split_policy,
            fold=0,
        )

        result.ood_rmse      = run_aug.rmse_test
        result.ood_r2        = run_aug.r2_test
        result.augmented_rmse = run_aug.rmse_train  # 拡張訓練データでの train RMSE

        if math.isfinite(result.baseline_rmse) and result.baseline_rmse > 0:
            result.improvement = (
                (result.baseline_rmse - result.ood_rmse) / result.baseline_rmse
            )

        result.elapsed_sec = time.time() - t0
        result.success = True

        logger.info(
            "Discovery [%s/%s]: baseline_rmse=%.4f  ood_rmse=%.4f  "
            "improvement=%.3f  feature=%s",
            workflow_name, candidate_feature or "baseline",
            result.baseline_rmse, result.ood_rmse,
            result.improvement if math.isfinite(result.improvement) else float("nan"),
            candidate_feature or "(none)",
        )

    except Exception:
        result.error_message = traceback.format_exc()
        result.elapsed_sec   = time.time() - t0
        result.success       = False
        logger.exception("DiscoveryRound 失敗: %s / %s", workflow_name, candidate_feature)

    return result


# ---------------------------------------------------------------------------
# Step 4: 複数候補を一括探索
# ---------------------------------------------------------------------------

def run_feature_discovery(
    workflow_names: List[str],
    feature_set_name: str,
    split_policy: str,
    features_df: pd.DataFrame,
    target: pd.Series,
    compositions_df: Optional[pd.DataFrame],
    ood_result,
    ood_test_idx: np.ndarray,
    candidate_features: List[str],
    extra_features_df: Optional[pd.DataFrame],
    seed: int = 42,
    n_folds: int = 5,
    quick: bool = True,
    n_ood_samples: int = 0,
    boundary_margin: float = 0.5,
    progress_callback: Optional[Any] = None,
) -> FeatureDiscoveryResult:
    """複数候補特徴量 × 複数 WF の探索を一括実行する。

    Parameters
    ----------
    candidate_features : list of str
        探索する特徴量名のリスト。extra_features_df のカラム名と対応。
        空リストの場合はベースライン（追加なし）のみ評価する。
    extra_features_df : pd.DataFrame or None
        追加候補特徴量を格納した DataFrame。
        行数は features_df と一致している必要がある。
    """
    t0 = time.time()
    result = FeatureDiscoveryResult()

    try:
        boundary = identify_boundary_samples(ood_result, n_ood_samples, boundary_margin)
        result.n_boundary_samples = boundary.n_boundary

        # 候補リスト: ベースライン（""）+ 各候補特徴量
        all_candidates = [""] + list(candidate_features)
        total = len(all_candidates) * len(workflow_names)
        done = 0

        for wf in workflow_names:
            for cand in all_candidates:
                if progress_callback:
                    try:
                        progress_callback(
                            done, total,
                            f"探索中: WF={wf}  特徴量={cand or 'baseline'} "
                            f"({done}/{total})",
                        )
                    except Exception:
                        pass

                round_res = run_feature_discovery_round(
                    workflow_name=wf,
                    feature_set_name=feature_set_name,
                    split_policy=split_policy,
                    features_df=features_df,
                    target=target,
                    compositions_df=compositions_df,
                    ood_result=ood_result,
                    ood_test_idx=ood_test_idx,
                    candidate_feature=cand,
                    extra_features_df=extra_features_df,
                    seed=seed,
                    n_folds=n_folds,
                    quick=quick,
                    n_ood_samples=n_ood_samples,
                    boundary_margin=boundary_margin,
                )
                result.rounds.append(round_res)
                done += 1

        # 最良候補を選択（ベースライン除く、改善率が最大のもの）
        non_baseline = [
            r for r in result.rounds
            if r.candidate_feature and r.success and math.isfinite(r.improvement)
        ]
        if non_baseline:
            best = max(non_baseline, key=lambda r: r.improvement)
            result.best_feature    = best.candidate_feature
            result.best_improvement = best.improvement

        result.elapsed_sec = time.time() - t0
        result.success = True

        logger.info(
            "FeatureDiscovery 完了: %d rounds  best=%s  improvement=%.3f  %.2fs",
            len(result.rounds), result.best_feature,
            result.best_improvement if math.isfinite(result.best_improvement) else 0.0,
            result.elapsed_sec,
        )

    except Exception:
        result.error_message = traceback.format_exc()
        result.elapsed_sec   = time.time() - t0
        result.success       = False
        logger.exception("FeatureDiscovery 全体失敗")

    return result
