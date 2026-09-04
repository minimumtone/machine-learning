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

    # 境界サンプル: score が threshold × (1-margin) 以上 threshold 未満（内挿側の境界）
    boundary_indices: np.ndarray      # 元データフレームのインデックス
    boundary_scores: np.ndarray       # composite_score
    ood_threshold: float
    margin: float                     # threshold × (1 - margin) を境界の下限とする

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

    # 拡張訓練データでの当てはめ RMSE（参考値）
    augmented_rmse: float = float("nan")
    augmented_r2:   float = float("nan")

    # 同一の train/OOD 評価分割で候補列なしに学習したときの OOD RMSE。
    # improvement はこの値との比較（同一評価データ・同一指標）で算出する。
    baseline_ood_rmse: float = float("nan")

    improvement: float = float("nan")  # (baseline_ood_rmse - ood_rmse) / baseline_ood_rmse

    # 訓練スコープの記録（結果の再現・解釈に必須。局所訓練では n が小さい）
    train_scope: str = "global"
    n_train_rows: int = 0            # 訓練に使った元データ行数（複製前・重複なし）
    n_train_aug: int = 0             # 複製を含む訓練行数
    n_ood_eval: int = 0              # OOD 評価行数
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
    n_ood_eval: int = 0
    train_scope: str = "global"
    n_train_rows: int = 0
    elapsed_sec: float = 0.0
    success: bool = False
    error_message: str = ""


# ---------------------------------------------------------------------------
# 訓練スコープ（OOD 近辺のみで学習するための行選択／複製計画）
# ---------------------------------------------------------------------------

TRAIN_SCOPES = ("global", "neighborhood", "kernel")


@dataclass
class NeighborhoodPlan:
    """OOD 評価行との特徴量空間距離に基づく訓練行の複製回数。

    copies[i] は元データ行 i を訓練に何回入れるか（0 = 訓練から除外）。
    OOD 評価行は常に 0。候補特徴量・目的変数には一切依存しない
    （全候補で同一の訓練行集合を使うため、また y を見ないため）。
    """
    scope: str
    copies: np.ndarray               # int, shape (n_orig,)
    distances: np.ndarray            # 各行から最近傍 OOD 評価行までの標準化距離
    bandwidth: float                 # kernel の h（neighborhood では閾値距離）
    n_train_rows: int                # copies >= 1 の行数
    n_train_aug: int                 # copies の総和

    @property
    def is_global(self) -> bool:
        return self.scope == "global"


def compute_neighborhood_plan(
    features_df: pd.DataFrame,
    ood_eval_idx: np.ndarray,
    scope: str = "global",
    neighborhood_quantile: float = 0.3,
    kernel_max_copies: int = 4,
    min_train_rows: int = 30,
) -> NeighborhoodPlan:
    """OOD 評価行の近傍に訓練を絞る／重み付けする計画を作る。

    Parameters
    ----------
    features_df : pd.DataFrame
        元の特徴量行列（候補特徴量を含まないこと）。数値列のみ距離計算に使う。
    ood_eval_idx : np.ndarray
        OOD 評価行の元インデックス。これらは訓練から必ず除外する。
    scope : {"global", "neighborhood", "kernel"}
        global      : 全行を 1 回ずつ（従来動作。境界複製は augment_dataset 側）。
        neighborhood: 最近傍 OOD 評価行までの距離が下位 `neighborhood_quantile`
                      に入る行だけを訓練に使う（min_train_rows を下回らない）。
        kernel      : 全行を残しつつ、距離に応じた Gaussian 重み
                      w = exp(-d^2 / 2h^2)（h = 距離の中央値）で
                      1 + round(w * (kernel_max_copies - 1)) 回複製する。
                      sample_weight を持たない WF でも同じ効果が得られる。
    """
    if scope not in TRAIN_SCOPES:
        raise ValueError(f"train_scope は {TRAIN_SCOPES} のいずれか: {scope!r}")

    n_orig = len(features_df)
    ood_eval_idx = np.unique(np.asarray(ood_eval_idx, dtype=int))
    ood_eval_idx = ood_eval_idx[(ood_eval_idx >= 0) & (ood_eval_idx < n_orig)]
    is_eval = np.zeros(n_orig, dtype=bool)
    is_eval[ood_eval_idx] = True

    # 標準化 Euclid 距離（分散ゼロ列は除外、NaN は列中央値）
    num = features_df.select_dtypes(include=[np.number]).astype("float64")
    num = num.fillna(num.median())
    std = num.std(ddof=0)
    num = num.loc[:, std > 0]
    Z = ((num - num.mean()) / num.std(ddof=0)).to_numpy()

    distances = np.full(n_orig, np.nan)
    if len(ood_eval_idx) > 0 and Z.shape[1] > 0:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=1).fit(Z[is_eval])
        d, _ = nn.kneighbors(Z[~is_eval])
        distances[~is_eval] = d[:, 0]
    distances[is_eval] = 0.0

    copies = np.ones(n_orig, dtype=int)
    bandwidth = float("nan")
    cand = np.flatnonzero(~is_eval)

    if scope == "neighborhood" and len(cand) > 0:
        d_c = distances[cand]
        q = float(np.clip(neighborhood_quantile, 0.0, 1.0))
        thr = float(np.quantile(d_c, q))
        keep = d_c <= thr
        # 最低行数を保証（距離の近い順に補充）
        if keep.sum() < min(min_train_rows, len(cand)):
            order = np.argsort(d_c)
            keep = np.zeros(len(cand), dtype=bool)
            keep[order[:min(min_train_rows, len(cand))]] = True
            thr = float(d_c[order[min(min_train_rows, len(cand)) - 1]])
        copies[cand] = keep.astype(int)
        bandwidth = thr
    elif scope == "kernel" and len(cand) > 0:
        d_c = distances[cand]
        h = float(np.median(d_c))
        if not np.isfinite(h) or h <= 0:
            h = 1.0
        w = np.exp(-0.5 * (d_c / h) ** 2)
        copies[cand] = 1 + np.rint(w * max(kernel_max_copies - 1, 0)).astype(int)
        bandwidth = h

    copies[is_eval] = 0

    plan = NeighborhoodPlan(
        scope=scope, copies=copies, distances=distances, bandwidth=bandwidth,
        n_train_rows=int((copies >= 1).sum()), n_train_aug=int(copies.sum()),
    )
    logger.info(
        "train_scope=%s: n_ood_eval=%d  n_train_rows=%d  n_train_aug=%d  h/thr=%.3f",
        scope, len(ood_eval_idx), plan.n_train_rows, plan.n_train_aug, bandwidth,
    )
    return plan


# ---------------------------------------------------------------------------
# Step 1: OOD 境界サンプル特定
# ---------------------------------------------------------------------------

def identify_boundary_samples(
    ood_result,           # OODResult (composite_scores, is_ood, ood_threshold)
    n_ood_samples: int,
    margin: float = 0.5,  # threshold × (1 - margin) を境界下限とする
) -> BoundarySampleInfo:
    """OOD スコアに基づき OOD サンプルと境界サンプルを特定する。

    Parameters
    ----------
    ood_result : OODResult
        Stage 3 の出力。composite_scores が全テストサンプルのスコア。
    n_ood_samples : int
        OOD と判定するサンプル数の上限。threshold 以上のサンプルのうち
        スコア上位 N 件に限定する。0 → threshold 以上を全選択。
    margin : float
        境界サンプルの範囲係数。threshold × (1-margin) 以上かつ threshold 未満。
    """
    scores    = np.asarray(ood_result.composite_scores)
    threshold = float(ood_result.ood_threshold)
    n         = len(scores)
    all_idx   = np.arange(n)

    # OOD サンプル（スコア ≥ threshold）。n_ood_samples > 0 のときは
    # threshold 以上のサンプルの中からスコア上位 N 件に限定する。
    ood_mask = scores >= threshold
    ood_idx  = all_idx[ood_mask]
    if n_ood_samples > 0 and len(ood_idx) > n_ood_samples:
        order   = np.argsort(-scores[ood_idx])
        ood_idx = ood_idx[order[:n_ood_samples]]
    ood_scores = scores[ood_idx]

    # 境界サンプル（threshold 未満かつ threshold × (1-margin) 以上の内挿側）
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
    neighborhood_plan: Optional[NeighborhoodPlan] = None,
) -> Tuple[pd.DataFrame, pd.Series, np.ndarray, np.ndarray]:
    """境界サンプルを訓練データに追加した拡張データセットを構築する。

    neighborhood_plan が None または scope="global" のときは従来どおり
    「全行 + 境界サンプル複製」。それ以外では plan.copies に従って
    訓練行を選択・複製し、境界複製は行わない（plan が近傍性を担う）。

    Returns
    -------
    X_aug : pd.DataFrame
        拡張後の特徴量行列（元データ + 境界サンプルの重複を除去）
    y_aug : pd.Series
        拡張後のターゲット
    train_idx_aug : np.ndarray
        拡張データ内の訓練インデックス。OOD 評価行は含まない
        （元データから OOD 評価行を除いた行 + 境界サンプルの複製行）。
    ood_eval_idx : np.ndarray
        OOD 評価用インデックス（拡張データ内での位置）。train_idx_aug と素。
    """
    n_orig = len(features_df)

    # 追加特徴量を結合
    if extra_features_df is not None and candidate_col is not None:
        if len(extra_features_df) != n_orig:
            raise ValueError(
                f"extra_features_df の行数 ({len(extra_features_df)}) が "
                f"features_df の行数 ({n_orig}) と一致しません"
            )
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

    # OOD 評価行（元インデックス内の OOD サンプル）— 訓練から完全に除外する
    ood_orig_idx = (
        ood_test_idx[boundary_info.ood_indices]
        if ood_test_idx is not None
        else boundary_info.ood_indices
    )
    ood_orig_idx = np.unique(ood_orig_idx[ood_orig_idx < n_orig])

    # ── 局所訓練スコープ（neighborhood / kernel）────────────────────
    if neighborhood_plan is not None and not neighborhood_plan.is_global:
        if len(neighborhood_plan.copies) != n_orig:
            raise ValueError(
                f"neighborhood_plan.copies の長さ ({len(neighborhood_plan.copies)}) が "
                f"features_df の行数 ({n_orig}) と一致しません"
            )
        copies = neighborhood_plan.copies.copy()
        copies[ood_orig_idx] = 0            # OOD 評価行は常に訓練から除外
        base_rows = np.flatnonzero(copies >= 1)
        extra_rows = np.repeat(np.arange(n_orig), np.clip(copies - 1, 0, None))
        if len(extra_rows) > 0:
            X_aug = pd.concat(
                [X_base, X_base.iloc[extra_rows].reset_index(drop=True)],
                ignore_index=True,
            )
            y_aug = pd.concat(
                [y_base, y_base.iloc[extra_rows].reset_index(drop=True)],
                ignore_index=True,
            )
        else:
            X_aug, y_aug = X_base, y_base
        train_idx = np.concatenate(
            [base_rows, np.arange(n_orig, n_orig + len(extra_rows))]
        )
        logger.info(
            "augment[%s]: n_orig=%d  n_train_rows=%d  n_train_aug=%d  n_ood_eval=%d",
            neighborhood_plan.scope, n_orig, len(base_rows), len(train_idx),
            len(ood_orig_idx),
        )
        return X_aug, y_aug, train_idx, ood_orig_idx

    if len(boundary_orig_idx) == 0:
        # 境界サンプルなし → 元データから OOD 評価行を除いた行を訓練にする
        train_idx = np.setdiff1d(np.arange(n_orig), ood_orig_idx)
        return X_base, y_base, train_idx, ood_orig_idx

    # 境界サンプル（OOD 評価行に含まれないもの）を複製して訓練の重みを増やす
    boundary_orig_idx = np.setdiff1d(boundary_orig_idx, ood_orig_idx)
    boundary_X = X_base.iloc[boundary_orig_idx].reset_index(drop=True)
    boundary_y = y_base.iloc[boundary_orig_idx].reset_index(drop=True)

    X_aug = pd.concat([X_base, boundary_X], ignore_index=True)
    y_aug = pd.concat([y_base, boundary_y], ignore_index=True)

    n_aug = len(X_aug)
    # 訓練: 元データから OOD 評価行を除いた行 + 境界複製行
    boundary_aug_idx = np.arange(n_orig, n_aug)
    train_idx = np.concatenate(
        [np.setdiff1d(np.arange(n_orig), ood_orig_idx), boundary_aug_idx]
    )
    ood_eval_idx = ood_orig_idx

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
    generic_csv_mode: Optional[bool] = None,
    baseline_cache: Optional[Dict[str, Tuple[float, float]]] = None,
    neighborhood_plan: Optional[NeighborhoodPlan] = None,
) -> DiscoveryRoundResult:
    """1 候補特徴量での再学習と OOD 予測性能評価を実行する。

    generic_csv_mode : bool, optional
        None のときは feature_set_name が FeatureSetName に存在するかで自動判定。
    baseline_cache : dict, optional
        {workflow_name: (baseline_rmse, baseline_r2)}。候補間で不変のベースライン
        を再計算しないためのキャッシュ。
    neighborhood_plan : NeighborhoodPlan, optional
        訓練スコープ。None は "global"（従来動作）。
    """
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

        # generic CSV モードの自動判定（FeatureSetName にない FS 名は generic 扱い）
        if generic_csv_mode is None:
            from extrapolation_discovery_platform.features import FeatureSetName
            try:
                FeatureSetName(feature_set_name)
                generic_csv_mode = False
            except ValueError:
                generic_csv_mode = True

        # ── ベースライン（元データのみ、追加特徴量なし、候補間で不変） ─────
        cached = baseline_cache.get(workflow_name) if baseline_cache is not None else None
        if cached is not None:
            result.baseline_rmse, result.baseline_r2 = cached
        else:
            prep_base = stage1_preprocess(
                features_df=features_df,
                target=target,
                compositions_df=compositions_df,
                feature_set_names=[feature_set_name],
                workflow_names=[workflow_name],
                seeds=[seed],
                active_policies=[split_policy],
                n_folds=n_folds,
                generic_csv_mode=generic_csv_mode,
            )
            if not prep_base.success:
                raise RuntimeError(f"Stage1 失敗: {prep_base.error_message}")

            tr_base = stage2_train(
                prep_base, features_df, target,
                workflow_name, split_policy, feature_set_name,
                quick=quick, seed=seed, generic_csv_mode=generic_csv_mode,
            )
            if not tr_base.success:
                raise RuntimeError(f"Stage2 失敗: {tr_base.error_message}")

            result.baseline_rmse = tr_base.rmse_test_mean
            result.baseline_r2   = tr_base.r2_test_mean
            if baseline_cache is not None:
                baseline_cache[workflow_name] = (
                    result.baseline_rmse, result.baseline_r2,
                )

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
            neighborhood_plan=neighborhood_plan,
        )
        result.train_scope = neighborhood_plan.scope if neighborhood_plan else "global"
        result.n_train_aug = int(len(train_aug_idx))
        # 複製行（>= n_orig）は元行の写しなので、元行部分だけ数える
        result.n_train_rows = int((np.asarray(train_aug_idx) < len(features_df)).sum())
        result.n_ood_eval = int(len(ood_eval_idx))

        if len(ood_eval_idx) == 0:
            logger.warning("OOD 評価サンプルが 0 件 — スコアを計算できません")
            result.elapsed_sec = time.time() - t0
            result.success = True
            return result

        # ── 拡張データで再学習 ───────────────────────────────────────────
        # 有効列を取得（追加特徴量を含む）。
        # 列の選別（近ゼロ分散・共線性・リーク）は複製前の元行だけで行う。
        # 複製行を含めると、局所スコープで OOD 近辺の行が多重化された結果、
        # 近辺に存在しない元素列などが「近ゼロ分散」として落ちてしまい、
        # スコープ間で特徴量集合が変わって比較にならない。
        aug_fs_name = "generic" if generic_csv_mode else feature_set_name
        n_orig_rows = len(features_df)
        prep_aug = stage1_preprocess(
            features_df=X_aug.iloc[:n_orig_rows].reset_index(drop=True),
            target=y_aug.iloc[:n_orig_rows].reset_index(drop=True),
            compositions_df=compositions_df,
            feature_set_names=[feature_set_name],
            workflow_names=[workflow_name],
            seeds=[seed],
            active_policies=[split_policy],
            n_folds=n_folds,
            generic_csv_mode=generic_csv_mode,
        )
        if not prep_aug.success:
            raise RuntimeError(f"拡張 Stage1 失敗: {prep_aug.error_message}")

        factory = _WORKFLOW_FACTORIES.get(workflow_name)
        if factory is None:
            raise ValueError(f"未知のWF: {workflow_name}")

        effective_cols = prep_aug.effective_cols.get(aug_fs_name, list(X_aug.columns))
        effective_cols = [c for c in effective_cols if c in X_aug.columns]
        # 候補列はリーク容疑でない限り必ず学習に含める
        # （HEA モードでは FeatureCatalog 外の列が除外されるため）
        if candidate_feature and candidate_feature in X_aug.columns:
            rpt = (prep_aug.mc_reports.get(aug_fs_name)
                   or prep_aug.mc_reports.get(feature_set_name))
            leak_suspects = set(rpt.leak_suspects) if rpt is not None else set()
            if (candidate_feature not in effective_cols
                    and candidate_feature not in leak_suspects):
                effective_cols = effective_cols + [candidate_feature]

        X_tr = X_aug.iloc[train_aug_idx][effective_cols]
        y_tr = y_aug.iloc[train_aug_idx]
        X_ood = X_aug.iloc[ood_eval_idx][effective_cols]
        y_ood = y_aug.iloc[ood_eval_idx]

        from extrapolation_discovery_platform._utils import safe_array
        from extrapolation_discovery_platform.pipeline import (
            apply_extrapolation_guard,
            impute_by_train_median,
        )
        X_tr, X_ood = impute_by_train_median(X_tr, X_ood)
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
        apply_extrapolation_guard(run_aug, safe_array(y_tr))

        result.ood_rmse      = run_aug.rmse_test
        result.ood_r2        = run_aug.r2_test
        result.augmented_rmse = run_aug.rmse_train  # 拡張訓練データでの当てはめ RMSE（参考値）

        # 同一の train/OOD 分割で候補列なしに学習したベースライン OOD RMSE。
        # baseline_rmse（CV 平均）とは評価データが異なるため直接比較しない。
        _ood_cache_key = f"{workflow_name}__ood_baseline__{result.train_scope}"
        if not candidate_feature:
            result.baseline_ood_rmse = run_aug.rmse_test
            if baseline_cache is not None:
                baseline_cache[_ood_cache_key] = (run_aug.rmse_test, run_aug.r2_test)
        else:
            cached_ood = (
                baseline_cache.get(_ood_cache_key)
                if baseline_cache is not None else None
            )
            if cached_ood is not None:
                result.baseline_ood_rmse = cached_ood[0]
            else:
                base_cols = [c for c in effective_cols if c != candidate_feature]
                X_base_tr, X_base_ood = impute_by_train_median(
                    X_aug.iloc[train_aug_idx][base_cols],
                    X_aug.iloc[ood_eval_idx][base_cols],
                )
                wf_base = factory(quick, True)
                run_base = wf_base.run(
                    pd.DataFrame(
                        safe_array(X_base_tr),
                        columns=base_cols,
                    ),
                    y_tr.reset_index(drop=True),
                    pd.DataFrame(
                        safe_array(X_base_ood),
                        columns=base_cols,
                    ),
                    y_ood.reset_index(drop=True),
                    seed=seed,
                    feature_set=aug_fs_name,
                    split_policy=split_policy,
                    fold=0,
                )
                apply_extrapolation_guard(run_base, safe_array(y_tr))
                result.baseline_ood_rmse = run_base.rmse_test
                if baseline_cache is not None:
                    baseline_cache[_ood_cache_key] = (
                        run_base.rmse_test, run_base.r2_test,
                    )

        if math.isfinite(result.baseline_ood_rmse) and result.baseline_ood_rmse > 0:
            result.improvement = (
                (result.baseline_ood_rmse - result.ood_rmse)
                / result.baseline_ood_rmse
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
    generic_csv_mode: Optional[bool] = None,
    include_negative_control: bool = True,
    train_scope: str = "global",
    neighborhood_quantile: float = 0.3,
    kernel_max_copies: int = 4,
    min_train_rows: int = 30,
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
    generic_csv_mode : bool, optional
        None のときは feature_set_name から自動判定。
    include_negative_control : bool
        True のとき、先頭候補を行方向にシャッフルしたネガティブ
        コントロール列（情報ゼロ）を自動追加する。有用な候補はこの
        列を上回る improvement を示すべきである。
    train_scope : {"global", "neighborhood", "kernel"}
        訓練行のスコープ。"global" は従来動作（全行 + 境界複製）。
        "neighborhood" は OOD 評価行に近い訓練行のみ、"kernel" は距離に応じた
        複製で OOD 近辺を重み付けする。compute_neighborhood_plan() を参照。
        近辺のみで学習すると全データ使用時に平均化されて見えなくなる
        OOD 近辺の分散が損失に反映されるが、n が小さくなるため
        n_train_rows / n_ood_eval を必ず結果と一緒に報告すること。
    neighborhood_quantile, kernel_max_copies, min_train_rows
        compute_neighborhood_plan() に渡すパラメータ。
    """
    t0 = time.time()
    result = FeatureDiscoveryResult()

    try:
        boundary = identify_boundary_samples(ood_result, n_ood_samples, boundary_margin)
        result.n_boundary_samples = boundary.n_boundary

        # 訓練スコープ計画（候補・目的変数に依存しないので全ラウンドで共有）
        n_orig = len(features_df)
        ood_orig_idx = (
            np.asarray(ood_test_idx)[boundary.ood_indices]
            if ood_test_idx is not None else boundary.ood_indices
        )
        ood_orig_idx = np.unique(ood_orig_idx[ood_orig_idx < n_orig])
        plan = compute_neighborhood_plan(
            features_df, ood_orig_idx, scope=train_scope,
            neighborhood_quantile=neighborhood_quantile,
            kernel_max_copies=kernel_max_copies,
            min_train_rows=min_train_rows,
        )
        result.train_scope = train_scope
        result.n_ood_eval = int(len(ood_orig_idx))
        result.n_train_rows = (
            plan.n_train_rows if not plan.is_global
            else int(n_orig - len(ood_orig_idx))
        )

        # ネガティブコントロール: 先頭候補をシャッフルした列を追加
        candidate_features = list(candidate_features)
        if (include_negative_control and candidate_features
                and extra_features_df is not None
                and candidate_features[0] in extra_features_df.columns):
            ctrl_name = "__shuffled_control__"
            if ctrl_name not in extra_features_df.columns:
                rng = np.random.default_rng(seed)
                extra_features_df = extra_features_df.copy()
                extra_features_df[ctrl_name] = rng.permutation(
                    extra_features_df[candidate_features[0]].to_numpy()
                )
            if ctrl_name not in candidate_features:
                candidate_features.append(ctrl_name)

        # 候補リスト: ベースライン（""）+ 各候補特徴量
        all_candidates = [""] + list(candidate_features)
        baseline_cache: Dict[str, Tuple[float, float]] = {}
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
                    generic_csv_mode=generic_csv_mode,
                    baseline_cache=baseline_cache,
                    neighborhood_plan=plan,
                )
                result.rounds.append(round_res)
                done += 1

        # 最良候補を選択（ベースライン・ネガティブコントロール除く、改善率最大）
        non_baseline = [
            r for r in result.rounds
            if r.candidate_feature
            and r.candidate_feature != "__shuffled_control__"
            and r.success and math.isfinite(r.improvement)
        ]
        if non_baseline:
            best = max(non_baseline, key=lambda r: r.improvement)
            result.best_feature    = best.candidate_feature
            result.best_improvement = best.improvement

        result.elapsed_sec = time.time() - t0
        result.success = True

        logger.info(
            "FeatureDiscovery 完了: %d rounds  best=%s  improvement=%.3f  "
            "scope=%s n_train=%d n_ood_eval=%d  %.2fs",
            len(result.rounds), result.best_feature,
            result.best_improvement if math.isfinite(result.best_improvement) else 0.0,
            result.train_scope, result.n_train_rows, result.n_ood_eval,
            result.elapsed_sec,
        )

    except Exception:
        result.error_message = traceback.format_exc()
        result.elapsed_sec   = time.time() - t0
        result.success       = False
        logger.exception("FeatureDiscovery 全体失敗")

    return result
