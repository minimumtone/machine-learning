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
from extrapolation_discovery_platform.ood import OODResult
from extrapolation_discovery_platform.multicollinearity import (
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
    r2_global: float = float("nan")  # 全 fold 集積での全体 R²

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
            "## ✅ 個別実行結果",
            "",
            "| 項目 | 値 |",
            "|---|---|",
            f"| アルゴリズム | **{self.workflow}** |",
            f"| 特徴量セット | **{self.feature_set}** |",
            f"| 分割方法 | {self.split_policy} |",
            f"| Seed | {self.seed} |",
            f"| 実行Fold数 | {self.n_folds_executed} |",
            f"| 学習サンプル数 | {self.n_train_samples} |",
            f"| テストサンプル数 | {self.n_test_samples} |",
            f"| 有効特徴量数 | {self.n_features_after} / {self.n_features_before} |",
            f"| 実行時間 | {self.elapsed_sec:.2f} 秒 |",
            "",
            f"### 📊 性能指標（{self.n_folds_executed} Fold 平均）",
            "",
            "| 指標 | 値 |",
            "|---|---|",
            f"| RMSE (Test) | **{self.rmse_test_mean:.4f}** ± {self.rmse_test_std:.4f} |",
            f"| RMSE (Train) | {self.rmse_train_mean:.4f} |",
            f"| MAE (Test) | {self.mae_test_mean:.4f} |",
            f"| R² (Test, 全体) | **{self.r2_global:.4f}** |",
            f"| R² (Test, fold平均) | {self.r2_test_mean:.4f} ± {self.r2_test_std:.4f} |",
            f"| ※ CompositionBlock等ではfold平均R²<0になりますが全体R²が正しい指標です | |",
        ]

        if self.leak_suspects:
            lines += [
                "",
                "### ⚠️ リーク疑い特徴量",
                "",
            ]
            for feat, r in sorted(self.leak_suspects.items(), key=lambda x: -abs(x[1])):
                lines.append(f"- `{feat}` (|r| = {abs(r):.4f})")

        if self.dropped_columns:
            lines += [
                "",
                "### 🗑️ 除去された特徴量",
                f"（定数列・完全共線）: {', '.join(f'`{c}`' for c in self.dropped_columns[:10])}",
            ]

        if self.ood_result is not None:
            ood = self.ood_result
            lines += [
                "",
                "### 🗺️ OOD 検出",
                "",
                "| 項目 | 値 |",
                "|---|---|",
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
                    "",
                    "### 📐 係数 Top 10 (|coef|降順)",
                    "",
                    "| 特徴量 | 係数 |",
                    "|---|---|",
                ]
                for feat, val in top:
                    lines.append(f"| `{feat}` | {val:.6f} |")

        if "feature_importance" in self.artifacts:
            fi = self.artifacts["feature_importance"]
            if isinstance(fi, dict) and fi:
                top = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:10]
                lines += [
                    "",
                    "### 🌲 特徴量重要度 Top 10",
                    "",
                    "| 特徴量 | 重要度 |",
                    "|---|---|",
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

    if split_policy in ("RandomCV", "RandomCV ⚠️(リーク懸念)"):
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
    # ── runner実行後の有効列を引き継ぐ（Phase2特徴量選択を反映） ────────────
    precomputed_columns: Optional[List[str]] = None,
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
    precomputed_columns : list of str, optional
        runner.ExperimentRunner.run() 実行後に得られた effective_cols
        (Phase1+Phase2で絞り込まれた列リスト) を渡すと、全実行と完全に
        同じ列でindividual実行が行われる。
        None のときは Phase1 のみで列を決定する（runner未実行時の単体使用）。

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
        # ══════════════════════════════════════════════════════════════
        # Stage 1 → Stage 2 → Stage 3 の 3 段階パイプラインに委譲
        #
        # 一括計算(runner.py) も個別計算(ここ) も同じ pipeline.py を呼ぶ。
        # 同一の計算条件なら同一の結果が保証される。
        #
        # Stage 1: stage1_preprocess — 前処理
        #   多重共線性検出 → 有効列決定 → 分割計算 → 特徴量選択
        # Stage 2: stage2_train      — ML 学習（OOD なし）
        #   各 fold × WF で学習・推論 → RunResult[]
        # Stage 3: stage3_detect_ood — OOD 検出（学習と完全独立）
        #   全 fold アンサンブル OOD → OODStageResult
        # ══════════════════════════════════════════════════════════════
        from extrapolation_discovery_platform.pipeline import (
            stage1_preprocess,
            stage2_train,
            stage3_detect_ood,
        )

        # ── Stage 1: 前処理 ──────────────────────────────────────────
        # precomputed_columns が渡されている場合（一括計算後の個別追加計算）は
        # Stage1 をスキップして既存の有効列をそのまま使う。
        if precomputed_columns is not None:
            # 既に runner が Stage1 を実行済み → effective_cols だけ引き継ぐ
            _valid_cols = [c for c in precomputed_columns if c in features_df.columns]
            _fs_key = "generic" if generic_csv_mode else feature_set_name
            _fold_plan_hint: dict = {}  # 分割は Stage1 相当を再実行
            # 分割だけは再計算が必要（seed 統一のため）
            try:
                _sp = split_policy_name.replace(" ⚠️(リーク懸念)", "")
                _splits = _make_splits(
                    split_policy=_sp,
                    features_df=features_df,
                    target=target,
                    compositions_df=compositions_df,
                    seed=seed,
                    n_folds=n_folds,
                    test_size=test_size,
                    exclude_elements=exclude_elements,
                )
                _plan_key = f"RandomCV_seed{seed}" if _sp == "RandomCV" else _sp
                _fold_plan_hint[_plan_key] = _splits
            except Exception:
                logger.warning("個別実行: 分割再計算失敗:\n%s", traceback.format_exc())

            # PreprocessResult の簡易版を作る
            from extrapolation_discovery_platform.pipeline import PreprocessResult
            prep = PreprocessResult(
                effective_cols={_fs_key: _valid_cols if _valid_cols else list(features_df.columns)},
                fold_plan=_fold_plan_hint,
                active_policies=[split_policy_name.replace(" ⚠️(リーク懸念)", "")],
                success=True,
            )
        else:
            # 新規実行 → Stage1 を完全実行
            _sp_clean = split_policy_name.replace(" ⚠️(リーク懸念)", "")
            prep = stage1_preprocess(
                features_df=features_df,
                target=target,
                compositions_df=compositions_df,
                feature_set_names=[feature_set_name],
                workflow_names=[workflow_name],
                seeds=[seed],
                active_policies=[_sp_clean],
                leak_auto_exclude=leak_auto_exclude,
                leak_corr_threshold=leak_corr_threshold,
                generic_csv_mode=generic_csv_mode,
            )
            if not prep.success:
                raise RuntimeError(f"Stage1 前処理失敗:\n{prep.error_message}")

        result.mc_report = next(iter(prep.mc_reports.values()), None)

        # ── Stage 2: ML 学習 ─────────────────────────────────────────
        _sp_clean = split_policy_name.replace(" ⚠️(リーク懸念)", "")
        train_res = stage2_train(
            preprocess_result=prep,
            features_df=features_df,
            target=target,
            workflow_name=workflow_name,
            split_policy_name=_sp_clean,
            feature_set_name=feature_set_name,
            quick=quick,
            dim_reduction=dim_reduction,
            seed=seed,
            generic_csv_mode=generic_csv_mode,
        )
        if not train_res.success:
            raise RuntimeError(f"Stage2 学習失敗:\n{train_res.error_message}")

        runs = train_res.runs
        result.runs              = runs
        result.n_folds_executed  = train_res.n_folds_executed
        result.n_features_before = len(prep.effective_cols.get(
            "generic" if generic_csv_mode else feature_set_name,
            []
        ))
        result.n_features_after  = train_res.n_features_used
        # RandomCV の fold_plan キーは "RandomCV_seed{seed}" 形式
        _plan_lookup_key = f"RandomCV_seed{seed}" if _sp_clean == "RandomCV" else _sp_clean
        result.n_train_samples   = (
            int(np.mean([len(s[0]) for s in prep.fold_plan.get(_plan_lookup_key, [])]))
            if prep.fold_plan.get(_plan_lookup_key) else 0
        )
        result.n_test_samples = (
            int(np.mean([len(s[1]) for s in prep.fold_plan.get(_plan_lookup_key, [])]))
            if prep.fold_plan.get(_plan_lookup_key) else 0
        )

        # ── 集約メトリクス（Stage2 からコピー） ─────────────────────
        result.rmse_test_mean  = train_res.rmse_test_mean
        result.rmse_test_std   = train_res.rmse_test_std
        result.rmse_train_mean = train_res.rmse_train_mean
        result.mae_test_mean   = train_res.mae_test_mean
        result.r2_test_mean    = train_res.r2_test_mean
        result.r2_test_std     = train_res.r2_test_std

        # 全 fold を集積した全体 R² を計算（fold平均R²はCompositionBlockで負になるため）
        try:
            from sklearn.metrics import r2_score as _r2s
            _yt, _yp = [], []
            for _r in runs:
                if _r.y_test_true is not None and _r.y_test_pred is not None:
                    _yt.extend(_r.y_test_true.ravel())
                    _yp.extend(_r.y_test_pred.ravel())
            if len(_yt) >= 2:
                result.r2_global = float(_r2s(_yt, _yp))
        except Exception:
            result.r2_global = result.r2_test_mean

        # ── アーティファクト集約（最後の fold を代表に） ─────────────
        if runs and runs[-1].artifacts:
            result.artifacts = dict(runs[-1].artifacts)

        # ── Stage 3: OOD 検出（学習と完全独立） ─────────────────────
        _ood_cols_key = "generic" if generic_csv_mode else feature_set_name
        _ood_cols = prep.effective_cols.get(_ood_cols_key, list(features_df.columns))
        _ood_cols = [c for c in _ood_cols if c in features_df.columns]

        if _ood_cols and prep.fold_plan:
            ood_stage = stage3_detect_ood(
                features_df=features_df,
                effective_columns=_ood_cols,
                fold_plan=prep.fold_plan,
            )
            if ood_stage.success and ood_stage.ood_result is not None:
                result.ood_result    = ood_stage.ood_result
                result.ood_train_idx = ood_stage.primary_train_idx
                result.ood_test_idx  = ood_stage.primary_test_idx
            else:
                logger.warning("Stage3 OOD失敗 (non-fatal): %s",
                               ood_stage.error_message[:200])

        result.elapsed_sec = time.time() - t0


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
