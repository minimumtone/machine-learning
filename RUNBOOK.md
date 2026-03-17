# Extrapolation Discovery Platform — 実行手順書 (PR#148)

## 概要

このドキュメントは PR#148 の変更内容・構成・起動方法・運用手順をまとめたものです。

---

## 1. PR#148 の主な変更点

### 1-1. 3-Stage パイプライン導入（最重要）

`pipeline.py` を新設し、データ処理を以下の3ステージに分離しました。

```
Stage 1: stage1_preprocess()   前処理
  └── 多重共線性・リーク検出
  └── 有効列決定（drop + leak除外）
  └── 分割計算（fold_plan）     ← 特徴量選択より先（訓練idx必要）
  └── 特徴量選択（訓練データのみ）← リーク防止

Stage 2: stage2_train()        ML 学習（OOD なし）
  └── 各 fold × WF で学習・推論
  └── RunResult[] を返す（OOD情報は含まない）

Stage 3: stage3_detect_ood()   OOD 検出（学習と完全独立）
  └── 全 fold × 全 split policy でアンサンブル
  └── OODStageResult を返す
```

`runner.py`（一括計算）と `individual_runner.py`（個別計算）の両方が同じ
Stage1→2→3 を呼ぶため、**同一条件なら同一結果が保証されます**。

### 1-2. 削除した重複実装

| ファイル | 削除したコード | 理由 |
|---|---|---|
| `runner.py` | `_phase3_precompute_folds()`（53行） | `stage1_preprocess()` に統合 |
| `runner.py` | `_phase6_ood()`（157行） | `stage3_detect_ood()` に統合 |
| `individual_runner.py` | Step1〜11 の独自前処理・OOD実装 | `pipeline.py` の3ステージに委譲 |

### 1-3. RandomCV デフォルト無効化

**データリーク懸念**があるため、`runner.run()` の `selected_split_policies` の
デフォルトを `["CompositionBlock", "ElementExclusion"]` に変更しました。

理由:
- 組成が類似した合金がtrain/testに混在し、テスト性能が過大評価される
- evaluationスコアが真の外挿能力でなくランダム分散を反映してしまう
- CompositionBlock が既に厳密なk-fold CVを提供しており冗長

RandomCV が必要な場合（ベースライン比較・診断目的）は GUI の「分割ポリシー設定」
アコーディオンから明示的に有効化できます。

### 1-3-2. 分割数 (n_folds) の設定

一括計算の交差検証分割数を GUI から設定できるようになりました。

- GUI: `Config & Run` → 「分割ポリシー設定」アコーディオン内の **分割数 (n_folds)** スライダー
- 範囲: 2〜10、デフォルト: **5**
- API: `ExperimentRunner(n_folds=3)` のように指定

| n_folds | 速度 | 評価安定性 | 1 fold あたりの訓練データ |
|---|---|---|---|
| 2 | 最速 | 低 | 75% |
| 5 | 標準 | 中 | 80% |
| 10 | 遅い | 高 | 90% |

```python
# 例: 3-fold CV で高速実行
runner = ExperimentRunner(seeds=[42], quick=True, n_folds=3)
runs, validity, ood = runner.run(comp, features, target)
```

### 1-4. WF-ENS の修正

`WorkflowENS` の `base_workflow` を `"xgb"` → `"ridge"` に変更し、
WF-XGB と同一結果になるバグを修正しました。

### 1-5. Individual Run タブの追加

GUI に「🔬 Individual Run」タブを追加。単一 WF × FS × split policy の
詳細結果（パリティプロット・OODマップ・fold別テーブル）を確認できます。

---

## 2. ファイル構成

```
extrapolation_discovery_platform/
├── pipeline.py          ★ 新設: 3-Stage パイプライン共通処理
├── runner.py            ★ 変更: Stage1/Stage3 を pipeline.py に委譲
├── individual_runner.py ★ 変更: Stage1/2/3 を pipeline.py に委譲
├── workflows.py         ★ 変更: WF-ENS base_workflow修正 / WF-XGB・WF-RF に StandardScaler追加
├── evaluation.py        ★ 変更: RandomCV限定 base_rmse / generalisation_score 修正
├── gui/
│   ├── app.py           ★ 変更: Individual Runタブ / RandomCV設定UI / Dashboardフィルタ
│   └── plotly_charts.py ★ 変更: plotly_parity_grid_by_algorithm等の新規チャート追加
├── features.py          （変更なし）
├── multicollinearity.py （変更なし）
├── ood.py               （変更なし）
├── splitters.py         （変更なし）
├── feature_selection.py （変更なし）
└── ...
```

---

## 3. 環境構築

### 3-1. 必要条件

- Python 3.10 以上（3.11 推奨）
- 依存パッケージは `requirements.txt` を参照

### 3-2. インストール

```bash
# 仮想環境を作成（推奨）
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# パッケージをインストール
pip install -r requirements.txt

# 開発モードでインストール（ソースを直接編集する場合）
pip install -e .
```

### 3-3. 依存サービス（オプション）

以下は `use_mlflow=False` / `use_feast=False` を指定すればスキップできます。
GUI 起動時は自動的にスタブモードで動作します。

- **MLflow**: 実験トラッキング。`mlflow ui` で確認可能
- **Feast**: 特徴量ストア。`feast apply` でセットアップ

---

## 4. アプリケーション起動

### 4-1. GUI 起動（通常）

```bash
cd extrapolation_discovery_platform_PR148
PYTHONPATH=. python3 -m extrapolation_discovery_platform
```

または

```bash
python3 -m extrapolation_discovery_platform.gui.app
```

デフォルトは `http://localhost:7860` で起動します。

### 4-2. ポートを変更する場合

```bash
python3 -m extrapolation_discovery_platform --port 8080
```

### 4-3. 外部公開する場合（Gradio share）

```bash
python3 -m extrapolation_discovery_platform --share
```

---

## 5. GUI の操作手順

### 5-1. 一括計算フロー

```
1. [Config & Run] タブを開く
2. データをアップロード（CSV）またはサンプルデータを生成
3. ワークフロー・特徴量セットのチェックボックスを選択
4. [分割ポリシー設定] アコーディオンで必要に応じて設定
   - CompositionBlock（推奨・デフォルト ON）
   - ElementExclusion（推奨・デフォルト ON）
   - RandomCV（デフォルト OFF・リーク懸念あり）
5. [Run Analysis] ボタンをクリック
6. [Dashboard] / [Results] / [OOD Map] タブで結果を確認
```

### 5-2. 個別計算フロー

```
1. 先に一括計算を実行する（データをロードするため）
2. [🔬 Individual Run] タブを開く
3. WF・FS・split policy を選択
4. [実行] ボタンをクリック
5. パリティプロット・OODマップ・fold別テーブルを確認
```

**注意:** 個別計算は一括計算が完了している場合、自動的に runner の
`effective_cols`（特徴量選択済み列）と `seeds[0]`（同一 seed）を引き継ぎます。
これにより **一括計算と同一条件で比較可能** になります。

---

## 6. プログラムから使用する場合（API）

### 6-1. 3-Stage パイプラインを直接使う

```python
from extrapolation_discovery_platform.pipeline import (
    stage1_preprocess,
    stage2_train,
    stage3_detect_ood,
)
import pandas as pd

# データ準備
features_df   = pd.read_csv("features.csv")
target        = pd.read_csv("target.csv").iloc[:, 0]
compositions  = pd.read_csv("compositions.csv")  # optional

# Stage 1: 前処理
prep = stage1_preprocess(
    features_df=features_df,
    target=target,
    compositions_df=compositions,
    feature_set_names=["FS_BASE", "FS_ALL"],
    workflow_names=["WF-LIN", "WF-XGB"],
    seeds=[42],
    active_policies=["CompositionBlock"],  # RandomCV はデフォルト除外
    leak_auto_exclude=True,
    leak_corr_threshold=0.85,
)
assert prep.success, prep.error_message

# Stage 2: ML 学習（OOD なし）
train_res = stage2_train(
    preprocess_result=prep,
    features_df=features_df,
    target=target,
    workflow_name="WF-LIN",
    split_policy_name="CompositionBlock",
    feature_set_name="FS_ALL",
    quick=False,  # 本番実行は False
    seed=42,
)
assert train_res.success, train_res.error_message
print(f"RMSE: {train_res.rmse_test_mean:.4f} ± {train_res.rmse_test_std:.4f}")
print(f"R²:   {train_res.r2_test_mean:.4f}")

# Stage 3: OOD 検出（学習とは独立）
ood_res = stage3_detect_ood(
    features_df=features_df,
    effective_columns=prep.effective_cols["FS_ALL"],
    fold_plan=prep.fold_plan,
)
if ood_res.success:
    print(f"OOD: {ood_res.ood_result.n_ood}/{ood_res.ood_result.n_total} サンプル")
```

### 6-2. 一括実行（従来の runner.py API）

```python
from extrapolation_discovery_platform.runner import ExperimentRunner

runner = ExperimentRunner(
    seeds=[42, 123],
    quick=False,
    leak_auto_exclude=True,
    leak_corr_threshold=0.85,
)
runs, validity_scores, ood_results = runner.run(
    compositions_df=compositions,
    features_df=features_df,
    target=target,
    selected_workflows=["WF-LIN", "WF-XGB", "WF-ENS"],
    selected_feature_sets=["FS_BASE", "FS_ALL"],
    selected_split_policies=["CompositionBlock"],  # RandomCV は明示的に追加
)
# runner._effective_cols に Stage1 で選択された有効列が保存されている
```

### 6-3. 個別実行

```python
from extrapolation_discovery_platform.individual_runner import run_individual

result = run_individual(
    workflow_name="WF-XGB",
    feature_set_name="FS_ALL",
    split_policy_name="CompositionBlock",
    features_df=features_df,
    target=target,
    compositions_df=compositions,
    seed=42,
    n_folds=5,
    quick=False,
    # 一括計算済みの場合は precomputed_columns を渡すと同一結果になる
    precomputed_columns=runner._effective_cols.get("FS_ALL"),
)
print(f"RMSE: {result.rmse_test_mean:.4f}")
print(f"OOD:  {result.ood_result.n_ood} サンプル")
```

---

## 7. よくある問題と対処

| 症状 | 原因 | 対処 |
|---|---|---|
| FS_MAGPIE で全 fold 失敗 | テストデータに MAGPIE 列がない | 正常。本番データでは自動的に計算される |
| ElementExclusion で fold 0件 | 組成データに対象元素がない | 正常。他の split policy にフォールバックする |
| 一括と個別の RMSE が異なる | precomputed_columns を渡していない | `runner._effective_cols.get("FS_ALL")` を引数に渡す |
| RandomCV の結果が含まれない | デフォルトで無効化されている | GUI「分割ポリシー設定」または `selected_split_policies=["RandomCV"]` で有効化 |
| サイドバーが消えない | ブラウザのキャッシュ | ブラウザのハードリロード（Ctrl+Shift+R） |

---

## 8. ログの確認

アプリケーションは Python の標準 `logging` を使用しています。

```bash
# DEBUG レベルでログを出力する場合
PYTHONPATH=. python3 -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from extrapolation_discovery_platform.gui.app import launch
launch()
"
```

ログの主要なプレフィックス:
- `Stage1:` — 前処理フェーズ
- `Stage2 [WF/FS/SP]:` — ML 学習フェーズ
- `Stage3 OOD:` — OOD 検出フェーズ
- `[個別実行]` — individual_runner.py からの実行

---

## 9. バージョン情報

| 項目 | 内容 |
|---|---|
| PR | #148 |
| GUI バージョンタグ | PR#148（画面右上に表示） |
| Python 要件 | 3.10 以上 |
| 主要変更ファイル | pipeline.py（新設）, runner.py, individual_runner.py, gui/app.py |
