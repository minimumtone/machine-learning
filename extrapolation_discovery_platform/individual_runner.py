"""
Individual ML Experiment Runner
================================
グリッド全体ではなく、ユーザーが指定した1アルゴリズム × 1特徴量セット × 1分割ポリシー
を即座に実行するための軽量ランナー。

設計方針:
  - 特徴量データ・目的変数・ハイパーパラメータをすべて毎回直接渡す（冗長OK）
  - ExperimentRunner に一切依存しない。ステートレス関数群で構成
  - セッション状態を書き換えずに結果だけを返す
  - 呼び出し元（app.py）がセッションにマージするかどうかを自由に決める

Public API:
    run_individual(
        workflow_name, feature_set_name, split_policy_name,
        features_df, target, compositions_df,
        seed, test_size, n_folds,
        quick, dim_reduction,
        leak_auto_exclude, leak_corr_threshold,
    ) -> IndividualRunResult
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
from sklearn.model_selection import train_test_split

from extrapolation_discovery_platform._utils import safe_array
from extrapolation_discovery_platform.workflows import (
    RunResult,
    WorkflowARD,
    WorkflowENS,
    WorkflowLASSO,
    WorkflowLIN,
    WorkflowRF,
    WorkflowXGB,
)
from extrapolation_discovery_platform.features import (
    FeatureCatalog,
    FeatureSetName,
)
from extrapolation_discovery_platform.splitters import (
    CompositionBlockSplitter,
    ElementExclusionSplitter,
    RandomCVSplitter,
)
from extrapolation_discovery_platform.ood import OODDetector, OODResult
from extrapolation_discovery_platform.multicollinearity import (
    run_phase0_multicollinearity,
    MulticollinearityReport,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class IndividualRunResult:
    """すべての個別実行結果を保持するコンテナ。

    フィールドはすべて任意（失敗時は error_message に詳細が入る）。
    """

    # ── 識別子 ────────────────────────────────────────────────────────────
    workflow: str = ""
    feature_set: str = ""
    split_policy: str = ""
    seed: int = 42

    # ── 個別RunResultのリスト（fold × split を展開） ─────────────────────
    runs: List[RunResult] = field(default_factory=list)

    # ── 集約メトリクス ────────────────────────────────────────────────────
    rmse_test_mean: float = float("nan")
    rmse_test_std: float = float("nan")
    rmse_train_mean: float = float("nan")
    mae_test_mean: float = float("nan")
    r2_test_mean: float = float("nan")
    r2_test_std: float = float("nan")

    # ── 個別ランのアーティファクト（係数・特徴量重要度など） ──────────────
    artifacts: Dict[str, Any] = field(default_factory=dict)

    # ── OOD検出結果 ────────────────────────────────────────────────────────
    ood_result: Optional[OODResult] = None
    ood_train_idx: Optional[np.ndarray] = None
    ood_test_idx: Optional[np.ndarray] = None

    # ── 前処理情報（冗長に保持） ────────────────────────────────────────────
    effective_columns: List[str] = field(default_factory=list)
    n_features_before: int = 0
    n_features_after: int = 0
    dropped_columns: List[str] = field(default_factory=list)
    leak_suspects: Dict[str, float] = field(default_factory=dict)
    mc_report: Optional[MulticollinearityReport] = None

    # ── 実行メタ情報 ────────────────────────────────────────────────────────
    elapsed_sec: float = 0.0
    n_train_samples: int = 0
    n_test_samples: int = 0
    n_folds_executed: int = 0
    error_message: str = ""
    success: bool = False

    def summary_md(self) -> str:
        """GUIに表示する Markdown サマリーを生成する。"""
        if not self.success:
            return (
                f"## ❌ 実行失敗\n\n"
                f"**アルゴリズム**: {self.workflow}  \n"
                f"**特徴量セット**: {self.feature_set}  \n"
                f"**エラー**:\n```\n{self.error_message}\n```"
            )

        lines = [
            f"## ✅ 個別実行結果",
            f"",
            f"| 項目 | 値 |",
            f"|---|---|",
            f"| アルゴリズム | **{self.workflow}** |",
            f"| 特徴量セット | **{self.feature_set}** |",
            f"| 分割方法 | {self.split_policy} |",
            f"| Seed | {self.seed} |",
            f"| 実行Fold数 | {self.n_folds_executed} |",
            f"| 学習サンプル数 | {self.n_train_samples} |",
            f"| テストサンプル数 | {self.n_test_samples} |",
            f"| 有効特徴量数 | {self.n_features_after} / {self.n_features_before} |",
            f"| 実行時間 | {self.elapsed_sec:.2f} 秒 |",
            f"",
            f"### 📊 性能指標（{self.n_folds_executed} Fold 平均）",
            f"",
            f"| 指標 | 値 |",
            f"|---|---|",
            f"| RMSE (Test) | **{self.rmse_test_mean:.4f}** ± {self.rmse_test_std:.4f} |",
            f"| RMSE (Train) | {self.rmse_train_mean:.4f} |",
            f"| MAE (Test) | {self.mae_test_mean:.4f} |",
            f"| R² (Test) | **{self.r2_test_mean:.4f}** ± {self.r2_test_std:.4f} |",
        ]

        if self.leak_suspects:
            lines += [
                f"",
                f"### ⚠️ リーク疑い特徴量",
                f"",
            ]
            for feat, r in sorted(self.leak_suspects.items(), key=lambda x: -abs(x[1])):
                lines.append(f"- `{feat}` (|r| = {abs(r):.4f})")

        if self.dropped_columns:
            lines += [
                f"",
                f"### 🗑️ 除去された特徴量",
                f"（定数列・完全共線）: {', '.join(f'`{c}`' for c in self.dropped_columns[:10])}",
            ]

        if self.ood_result is not None:
            ood = self.ood_result
            lines += [
                f"",
                f"### 🗺️ OOD 検出",
                f"",
                f"| 項目 | 値 |",
                f"|---|---|",
                f"| OOD サンプル数 | {ood.n_ood} / {ood.n_total} |",
                f"| OOD 比率 | {ood.ood_ratio:.1%} |",
                f"| OOD 閾値 | {ood.ood_threshold:.4f} |",
            ]

        # モデル固有のアーティファクト
        if "coef_raw" in self.artifacts:
            coef = self.artifacts["coef_raw"]
            if isinstance(coef, dict) and coef:
                top = sorted(coef.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
                lines += [
                    f"",
                    f"### 📐 係数 Top 10 (|coef|降順)",
                    f"",
                    f"| 特徴量 | 係数 |",
                    f"|---|---|",
                ]
                for feat, val in top:
                    lines.append(f"| `{feat}` | {val:.6f} |")

        if "feature_importance" in self.artifacts:
            fi = self.artifacts["feature_importance"]
            if isinstance(fi, dict) and fi:
                top = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:10]
                lines += [
                    f"",
                    f"### 🌲 特徴量重要度 Top 10",
                    f"",
                    f"| 特徴量 | 重要度 |",
                    f"|---|---|",
                ]
                for feat, val in top:
                    lines.append(f"| `{feat}` | {val:.6f} |")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ワークフローファクトリ
# ---------------------------------------------------------------------------

_WORKFLOW_FACTORIES = {
    "WF-LIN":   lambda quick, dim_r: WorkflowLIN(dim_reduction=dim_r),
    "WF-LASSO": lambda quick, dim_r: WorkflowLASSO(dim_reduction=dim_r),
    "WF-ARD":   lambda quick, dim_r: WorkflowARD(dim_reduction=dim_r),
    "WF-RF":    lambda quick, dim_r: WorkflowRF(quick=quick, dim_reduction=dim_r),
    "WF-XGB":   lambda quick, dim_r: WorkflowXGB(quick=quick, dim_reduction=dim_r),
    "WF-ENS":   lambda quick, dim_r: WorkflowENS(
        n_members=3 if quick else 5, quick=quick, dim_reduction=dim_r,
    ),
}


# ---------------------------------------------------------------------------
# 分割ポリシーファクトリ
# ---------------------------------------------------------------------------

def _make_splits(
    split_policy: str,
    features_df: pd.DataFrame,
    target: pd.Series,
    compositions_df: Optional[pd.DataFrame],
    seed: int,
    n_folds: int,
    test_size: float,
    exclude_elements: Optional[List[str]],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """指定された分割ポリシーに従い (train_idx, test_idx) のリストを返す。"""

    if split_policy == "RandomCV":
        splitter = RandomCVSplitter(n_folds=n_folds, seed=seed)
        return list(splitter.split(features_df, target, compositions=compositions_df))

    elif split_policy == "CompositionBlock":
        if compositions_df is None:
            raise ValueError("CompositionBlock 分割には compositions_df が必要です。")
        splitter = CompositionBlockSplitter(n_folds=n_folds, seed=seed)
        return list(splitter.split(features_df, target, compositions=compositions_df))

    elif split_policy == "ElementExclusion":
        if compositions_df is None:
            raise ValueError("ElementExclusion 分割には compositions_df が必要です。")
        elems = exclude_elements or ["Co", "Ni", "Ti"]
        splitter = ElementExclusionSplitter(target_elements=elems)
        return list(splitter.split(features_df, target, compositions=compositions_df))

    elif split_policy == "Holdout":
        # 単純なホールドアウト（fold=1）
        n = len(features_df)
        idx = np.arange(n)
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(n)
        n_test = max(1, int(n * test_size))
        test_idx  = shuffled[:n_test]
        train_idx = shuffled[n_test:]
        return [(train_idx, test_idx)]

    else:
        raise ValueError(
            f"未知の分割ポリシー: '{split_policy}'. "
            f"使用可能: RandomCV, CompositionBlock, ElementExclusion, Holdout"
        )


# ---------------------------------------------------------------------------
# メイン公開関数
# ---------------------------------------------------------------------------

def run_individual(
    # ── 実行設定 ────────────────────────────────────────────────────────────
    workflow_name: str,
    feature_set_name: str,
    split_policy_name: str,
    # ── データ（毎回直接渡す） ────────────────────────────────────────────
    features_df: pd.DataFrame,
    target: pd.Series,
    compositions_df: Optional[pd.DataFrame] = None,
    # ── 分割パラメータ ─────────────────────────────────────────────────────
    seed: int = 42,
    test_size: float = 0.2,       # Holdout 用
    n_folds: int = 5,             # RandomCV / CompositionBlock 用
    exclude_elements: Optional[List[str]] = None,
    # ── モデル設定 ─────────────────────────────────────────────────────────
    quick: bool = True,
    dim_reduction: bool = True,
    # ── 前処理設定（毎回渡す） ─────────────────────────────────────────────
    leak_auto_exclude: bool = True,
    leak_corr_threshold: float = 0.85,
    # ── 汎用CSV モードフラグ ───────────────────────────────────────────────
    generic_csv_mode: bool = False,
) -> IndividualRunResult:
    """1アルゴリズム × 1特徴量セット × 1分割ポリシーを即座に実行する。

    特徴量データ・目的変数・設定はすべて引数で直接渡す。
    セッションオブジェクトには一切アクセスしない。

    Parameters
    ----------
    workflow_name : str
        実行するワークフロー名。WF-LIN / WF-LASSO / WF-ARD / WF-RF / WF-XGB / WF-ENS
    feature_set_name : str
        使用する特徴量セット名。FS_BASE / FS_THERMO / FS_SIZE / FS_ELECTRON / FS_ALL / FS_MAGPIE
    split_policy_name : str
        分割ポリシー。RandomCV / CompositionBlock / ElementExclusion / Holdout
    features_df : pd.DataFrame
        特徴量行列（全特徴量を含む、FS列の選択はここで行う）
    target : pd.Series
        目的変数
    compositions_df : pd.DataFrame, optional
        元素組成行列（CompositionBlock / ElementExclusion 分割に必要）
    seed : int
        乱数シード
    test_size : float
        Holdout 分割時のテスト比率
    n_folds : int
        RandomCV / CompositionBlock の分割数
    exclude_elements : list of str, optional
        ElementExclusion で除外する元素リスト
    quick : bool
        True にするとハイパーパラメータグリッドを小さくして高速化
    dim_reduction : bool
        True にすると StandardScaler + PCA(95%) を適用
    leak_auto_exclude : bool
        True にするとターゲットと高相関な特徴量を自動除去
    leak_corr_threshold : float
        リーク検出の相関係数閾値
    generic_csv_mode : bool
        True のとき、features_df 全列を1つの特徴量セットとして使う
        （HEA固有のFS列選択をスキップ）

    Returns
    -------
    IndividualRunResult
    """
    t0 = time.time()
    result = IndividualRunResult(
        workflow=workflow_name,
        feature_set=feature_set_name,
        split_policy=split_policy_name,
        seed=seed,
    )

    try:
        # ── 1. 特徴量列の選択（毎回直接渡されたデータから行う） ──────────
        if generic_csv_mode:
            # Generic CSV: 全列を使う
            fs_cols = list(features_df.columns)
        else:
            try:
                fs_enum = FeatureSetName(feature_set_name)
                fs_cols = [c for c in FeatureCatalog.columns(fs_enum) if c in features_df.columns]
            except (ValueError, KeyError):
                # 未知のFS名 → 全列フォールバック
                logger.warning(
                    "未知の特徴量セット '%s'。全列を使用します。", feature_set_name
                )
                fs_cols = list(features_df.columns)

        if not fs_cols:
            raise ValueError(
                f"特徴量セット '{feature_set_name}' に対応する列が "
                f"features_df に見つかりません。"
            )

        result.n_features_before = len(fs_cols)

        # ── 2. 特徴量行列を抽出（C-contiguous保証） ─────────────────────
        X_full = pd.DataFrame(
            safe_array(features_df[fs_cols]),
            columns=fs_cols,
            index=features_df.index,
        )

        # ── 3. Phase1: 多重共線性 + リーク検出（毎回直接渡す） ───────────
        try:
            if not generic_csv_mode and fs_cols:
                # HEA mode: use FeatureSetName-based MC analysis
                try:
                    fs_enum_for_mc = FeatureSetName(feature_set_name)
                    fs_enum_list = [fs_enum_for_mc]
                except ValueError:
                    fs_enum_list = []
                mc_reports = run_phase0_multicollinearity(
                    X_full,
                    fs_enum_list,
                    [workflow_name],
                    len(X_full),
                    target=target,
                    leak_corr_threshold=leak_corr_threshold,
                )
                mc_rpt = mc_reports.get(feature_set_name) if mc_reports else None
            else:
                # Generic CSV mode: run MC analysis directly on X_full columns
                # (run_phase0_multicollinearity needs FeatureSetName objects,
                #  so we build a temporary FeatureSetName-free report instead)
                from extrapolation_discovery_platform.multicollinearity import (
                    remove_constant_columns,
                    remove_perfect_collinear,
                    compute_vif,
                    detect_target_leakage,
                )
                _X_tmp, _dropped_c = remove_constant_columns(X_full.copy())
                _X_tmp, _dropped_p = remove_perfect_collinear(_X_tmp)
                _n_after = _X_tmp.shape[1]
                _vif = compute_vif(_X_tmp) if _n_after <= 50 else None
                _high_vif = (
                    int((_vif > 10).sum()) if _vif is not None else 0
                )
                _high_ratio = _high_vif / max(_n_after, 1)
                _mc_level = (
                    "High" if _high_ratio > 0.5
                    else "Moderate" if _high_ratio > 0.2
                    else "Low"
                )
                _leak = {}
                if target is not None:
                    try:
                        _leak = detect_target_leakage(
                            _X_tmp, target, threshold=leak_corr_threshold,
                        )
                    except Exception:
                        pass
                from extrapolation_discovery_platform.multicollinearity import (
                    MulticollinearityReport as _MCR,
                )
                mc_rpt = _MCR(
                    feature_set=feature_set_name,
                    n_features_before=len(fs_cols),
                    n_features_after=_n_after,
                    dropped_constant=list(_dropped_c),
                    dropped_perfect=list(_dropped_p),
                    vif_series=_vif if _vif is not None else pd.Series(dtype=float),
                    high_vif_count=_high_vif,
                    moderate_vif_count=0,
                    multicollinearity_level=_mc_level.lower(),
                    recommended_workflows=[workflow_name],
                    blocked_workflows=[],
                    leak_suspects=_leak,
                )
            result.mc_report = mc_rpt
        except Exception:
            logger.warning("Phase1 multicollinearity failed (non-fatal):\n%s", traceback.format_exc())
            mc_rpt = None

        # ── 4. 有効列を決定（drop + リーク除外）────────────────────────
        effective_cols = list(fs_cols)
        dropped: List[str] = []
        leak_suspects: Dict[str, float] = {}

        if mc_rpt is not None:
            _drop_set = set(mc_rpt.dropped_constant + mc_rpt.dropped_perfect)
            if _drop_set:
                dropped = [c for c in effective_cols if c in _drop_set]
                effective_cols = [c for c in effective_cols if c not in _drop_set]

            if leak_auto_exclude and mc_rpt.leak_suspects:
                leak_suspects = dict(mc_rpt.leak_suspects)
                _leak_set = set(leak_suspects.keys())
                _before = len(effective_cols)
                effective_cols = [c for c in effective_cols if c not in _leak_set]
                if _before != len(effective_cols):
                    logger.info(
                        "リーク自動除外: %d 特徴量除去 (|r|>%.2f)",
                        _before - len(effective_cols), leak_corr_threshold,
                    )

        if not effective_cols:
            raise ValueError("有効な特徴量列がありません。前処理後にすべての列が除去されました。")

        result.effective_columns = effective_cols
        result.n_features_after  = len(effective_cols)
        result.dropped_columns   = dropped
        result.leak_suspects     = leak_suspects

        # ── 5. 有効特徴量行列を再構築（C-contiguous）─────────────────
        X = pd.DataFrame(
            safe_array(X_full[effective_cols]),
            columns=effective_cols,
            index=X_full.index,
        )
        y = target.reset_index(drop=True)

        # ── 6. 分割を生成（毎回直接渡す）───────────────────────────────
        splits = _make_splits(
            split_policy=split_policy_name,
            features_df=X,
            target=y,
            compositions_df=compositions_df,
            seed=seed,
            n_folds=n_folds,
            test_size=test_size,
            exclude_elements=exclude_elements,
        )

        if not splits:
            if split_policy_name == "ElementExclusion":
                logger.warning(
                    "ElementExclusion で有効な分割が生成されませんでした。"
                    "RandomCV (n_folds=%d) にフォールバックします。", n_folds,
                )
                splits = _make_splits(
                    split_policy="RandomCV",
                    features_df=X, target=y,
                    compositions_df=compositions_df,
                    seed=seed, n_folds=n_folds,
                    test_size=test_size,
                    exclude_elements=exclude_elements,
                )
                if splits:
                    # 分割ポリシー名を実際に使ったものに更新
                    split_policy_name = "RandomCV"
                    result.split_policy = f"RandomCV (ElementExclusion fallback)"
            if not splits:
                raise ValueError(
                    f"分割ポリシー '{split_policy_name}' で有効な分割が生成されませんでした。"
                    "データ数が少すぎるか、指定した元素が存在しない可能性があります。"
                )

        # ── 7. ワークフローを生成（毎回 new instance）──────────────────
        factory = _WORKFLOW_FACTORIES.get(workflow_name)
        if factory is None:
            raise ValueError(
                f"未知のワークフロー '{workflow_name}'. "
                f"使用可能: {list(_WORKFLOW_FACTORIES.keys())}"
            )

        # ── 8. 各Foldで学習・推論（毎回データを直接渡す）──────────────
        runs: List[RunResult] = []
        n_train_list: List[int] = []
        n_test_list:  List[int] = []

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            X_train = pd.DataFrame(
                safe_array(X.iloc[train_idx]),
                columns=effective_cols,
            )
            X_test = pd.DataFrame(
                safe_array(X.iloc[test_idx]),
                columns=effective_cols,
            )
            y_train = y.iloc[train_idx].reset_index(drop=True)
            y_test  = y.iloc[test_idx].reset_index(drop=True)

            n_train_list.append(len(X_train))
            n_test_list.append(len(X_test))

            # ワークフローインスタンスを毎回新規作成（冗長だが独立性を保証）
            wf = factory(quick, dim_reduction)

            run = wf.run(
                X_train, y_train, X_test, y_test,
                seed=seed,
                feature_set=feature_set_name,
                split_policy=split_policy_name,
                fold=fold_idx,
                test_indices=np.asarray(test_idx),
            )
            runs.append(run)
            logger.info(
                "[個別実行] %s / %s / %s / seed=%d / fold=%d  "
                "RMSE=%.4f  R²=%.4f  (%.2f s)",
                workflow_name, feature_set_name, split_policy_name,
                seed, fold_idx,
                run.rmse_test, run.r2_test, run.elapsed_sec,
            )

        result.runs = runs
        result.n_folds_executed  = len(runs)
        result.n_train_samples   = int(np.mean(n_train_list)) if n_train_list else 0
        result.n_test_samples    = int(np.mean(n_test_list))  if n_test_list  else 0

        # ── 9. メトリクス集計 ─────────────────────────────────────────
        valid_rmse_te = [r.rmse_test  for r in runs if r.rmse_test  > 0 and math.isfinite(r.rmse_test)]
        valid_rmse_tr = [r.rmse_train for r in runs if r.rmse_train > 0 and math.isfinite(r.rmse_train)]
        valid_mae_te  = [r.mae_test   for r in runs if r.mae_test   > 0 and math.isfinite(r.mae_test)]
        valid_r2_te   = [r.r2_test    for r in runs if math.isfinite(r.r2_test)]

        def _mean(xs): return float(np.mean(xs)) if xs else float("nan")
        def _std(xs):  return float(np.std(xs))  if len(xs) > 1 else 0.0

        result.rmse_test_mean  = _mean(valid_rmse_te)
        result.rmse_test_std   = _std(valid_rmse_te)
        result.rmse_train_mean = _mean(valid_rmse_tr)
        result.mae_test_mean   = _mean(valid_mae_te)
        result.r2_test_mean    = _mean(valid_r2_te)
        result.r2_test_std     = _std(valid_r2_te)

        # ── 10. アーティファクト集約（最後のfoldのものを代表として保持）
        if runs and runs[-1].artifacts:
            result.artifacts = dict(runs[-1].artifacts)

        # ── 11. OOD 検出（各foldで独立にfit/score、毎回直接データを渡す）────
        # 学習のたびにOODが変わるべき：train_idxが変わればfit結果も変わる。
        # 全foldのスコアを平均して代表値とし、GUIには最初のfoldのsplit indexを渡す。
        if splits:
            n_total_samples = len(X)
            ood_score_sum   = np.zeros(n_total_samples, dtype=np.float64)
            ood_score_count = np.zeros(n_total_samples, dtype=np.int32)
            primary_ood_res  = None
            primary_train_idx, primary_test_idx = splits[0]

            for fold_idx_ood, (tr_idx_ood, te_idx_ood) in enumerate(splits):
                if len(tr_idx_ood) < 2 or len(te_idx_ood) < 1:
                    continue
                try:
                    actual_k = min(10, len(tr_idx_ood) - 1)
                    X_ood_tr = pd.DataFrame(
                        safe_array(X.iloc[tr_idx_ood]), columns=effective_cols,
                    )
                    X_ood_te = pd.DataFrame(
                        safe_array(X.iloc[te_idx_ood]), columns=effective_cols,
                    )
                    # 各foldで独立にOODDetectorをfit → train setが変わればOODも変わる
                    detector = OODDetector(k=actual_k)
                    detector.fit(X_ood_tr)
                    fold_ood = detector.score(X_ood_te)
                    # グローバルインデックスにスコアを蓄積
                    ood_score_sum[te_idx_ood]   += fold_ood.composite_scores
                    ood_score_count[te_idx_ood] += 1
                    if fold_idx_ood == 0:
                        primary_ood_res = fold_ood
                except Exception:
                    logger.warning(
                        "OOD fold=%d 失敗 (non-fatal)", fold_idx_ood, exc_info=True,
                    )

            if primary_ood_res is not None:
                # primary test setの平均スコアを計算
                te = primary_test_idx
                scored = ood_score_count[te] > 0
                avg_composite = np.where(
                    scored,
                    ood_score_sum[te] / np.maximum(ood_score_count[te], 1),
                    primary_ood_res.composite_scores,
                )
                is_ood_avg = avg_composite > primary_ood_res.ood_threshold
                n_ood_avg  = int(is_ood_avg.sum())
                # 全fold平均のOODResultを保存
                result.ood_result = OODResult(
                    mahalanobis_scores=primary_ood_res.mahalanobis_scores,
                    knn_scores=primary_ood_res.knn_scores,
                    composite_scores=np.ascontiguousarray(avg_composite),
                    is_ood=np.ascontiguousarray(is_ood_avg),
                    ood_threshold=primary_ood_res.ood_threshold,
                    ood_ratio=n_ood_avg / max(len(avg_composite), 1),
                    n_total=len(avg_composite),
                    n_ood=n_ood_avg,
                )
                result.ood_train_idx = np.asarray(primary_train_idx)
                result.ood_test_idx  = np.asarray(primary_test_idx)
                logger.info(
                    "OOD完了: %d/%d OOD (%d fold平均)",
                    n_ood_avg, len(avg_composite), len(splits),
                )

        result.elapsed_sec = time.time() - t0
        result.success = True

    except Exception:
        result.error_message = traceback.format_exc()
        result.elapsed_sec   = time.time() - t0
        result.success       = False
        logger.exception(
            "個別実行失敗: %s / %s / %s",
            workflow_name, feature_set_name, split_policy_name,
        )

    return result


# ---------------------------------------------------------------------------
# 複数ワークフロー一括比較（同一データを各WFに直接渡す）
# ---------------------------------------------------------------------------

def run_individual_compare(
    workflow_names: List[str],
    feature_set_name: str,
    split_policy_name: str,
    features_df: pd.DataFrame,
    target: pd.Series,
    compositions_df: Optional[pd.DataFrame] = None,
    seed: int = 42,
    test_size: float = 0.2,
    n_folds: int = 5,
    exclude_elements: Optional[List[str]] = None,
    quick: bool = True,
    dim_reduction: bool = True,
    leak_auto_exclude: bool = True,
    leak_corr_threshold: float = 0.85,
    generic_csv_mode: bool = False,
    progress_callback: Optional[Any] = None,
) -> List[IndividualRunResult]:
    """複数ワークフローを同一データで比較実行する。

    各ワークフローに毎回 features_df / target を直接渡す（冗長だが独立性保証）。

    Parameters
    ----------
    workflow_names : list of str
        比較するワークフロー名のリスト
    progress_callback : callable, optional
        (completed: int, total: int, message: str) を受け取るコールバック

    Returns
    -------
    list of IndividualRunResult  （workflow_names と同順）
    """
    results: List[IndividualRunResult] = []
    total = len(workflow_names)

    for i, wf_name in enumerate(workflow_names):
        if progress_callback is not None:
            try:
                progress_callback(i, total, f"{wf_name} / {feature_set_name} 実行中...")
            except Exception:
                pass

        res = run_individual(
            workflow_name=wf_name,
            feature_set_name=feature_set_name,
            split_policy_name=split_policy_name,
            features_df=features_df,          # 毎回直接渡す
            target=target,                     # 毎回直接渡す
            compositions_df=compositions_df,   # 毎回直接渡す
            seed=seed,
            test_size=test_size,
            n_folds=n_folds,
            exclude_elements=exclude_elements,
            quick=quick,
            dim_reduction=dim_reduction,
            leak_auto_exclude=leak_auto_exclude,
            leak_corr_threshold=leak_corr_threshold,
            generic_csv_mode=generic_csv_mode,
        )
        results.append(res)

    if progress_callback is not None:
        try:
            progress_callback(total, total, "完了")
        except Exception:
            pass

    return results
