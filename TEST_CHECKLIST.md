# EDP PR#148 — テスト確認チェックリスト

テスト実施時にこのドキュメントに従って確認してください。
**全項目が ✅ になることを目標とします。**

---

## T-A: サイドバー削除確認

**確認コマンド:**
```python
with open("gui/app.py") as f:
    app = f.read()
assert "_SIDEBAR_CSS" not in app
assert "_SIDEBAR_JS"  not in app
assert "css=_SIDEBAR_CSS" not in app
```

**確認すべき点:**
- [ ] `_SIDEBAR_CSS` の定義が app.py に存在しない
- [ ] `_SIDEBAR_JS` の定義が app.py に存在しない
- [ ] Gradio Blocks の `css=` / `js=` にサイドバー参照がない
- [ ] GUI 起動時に左サイドバーが表示されない（標準タブバーのみ）

---

## T-B: test_size / Holdout 分割の確認

**確認コマンド:**
```python
from extrapolation_discovery_platform.pipeline import stage1_preprocess
prep = stage1_preprocess(X, y, comp, ["FS_BASE"], ["WF-LIN"],
    seeds=[42], active_policies=["Holdout"], test_size=0.3)
tr, te = prep.fold_plan["Holdout"][0]
ratio = len(te) / (len(tr) + len(te))
assert abs(ratio - 0.3) < 0.05
```

**確認すべき点:**
- [ ] `active_policies=["Holdout"]` で fold_plan に "Holdout" キーが入る
- [ ] `test_size=0.3` でテスト比率が約 30%
- [ ] CompositionBlock / ElementExclusion では test_size が無視される
- [ ] GUI の「テスト比率」スライダーが範囲 0.1〜0.5、デフォルト 0.2

---

## T-C: Nested-CV デフォルト OFF

**確認すべき点:**
- [ ] `model_sel_check` の `value=False`（デフォルト OFF）
- [ ] チェックを ON にして実行すると Nested-CV が動作する
- [ ] OFF のままでは通常の実験のみ実行（高速）

---

## T-D: 全WF比較 UI

**確認すべき点:**
- [ ] 🔬 Individual Run タブに「⚡ 全WF比較」ボタンがある
- [ ] ボタンクリックで全 WF（LIN, LASSO, ARD, RF, XGB, ENS）が実行される
- [ ] 比較サマリーに `最優秀WF` と RMSE が表示される
- [ ] WF別 RMSE バーチャートが表示される

---

## T-E: OOD Feature Discovery

**確認すべき点:**
- [ ] OOD & Individual タブに「🔭 OOD Feature Discovery」サブタブがある
- [ ] 「CSV 読み込み」ボタンで列名がチェックボックスに反映される
- [ ] 追加特徴量なし（CSV未指定）でもベースライン評価が実行される
- [ ] 探索結果テーブルに `candidate_feature`, `baseline_rmse`, `ood_rmse`, `improvement` が含まれる
- [ ] `improvement > 0` の特徴量が存在する場合に「最良特徴量」が表示される
- [ ] OOD 結果がない状態（未実行）で実行するとエラーメッセージが表示される

---

## 自動テスト（pytest / スクリプト）

### 実行コマンド

```bash
cd extrapolation_discovery_platform_PR148
PYTHONPATH=. python3 -m pytest tests/ -v          # pytest がある場合
PYTHONPATH=. python3 tests/test_pipeline.py        # 直接実行
```

または以下のコマンドで本ドキュメント記載の T1〜T9 を一括実行：

```bash
PYTHONPATH=. python3 << 'EOF'
# テストスクリプト（RUNBOOK.md の「API使用例」と対になる）
# tests/test_pipeline.py を参照
EOF
```

---

## T1: Stage1 再現性（同一入力 → 同一結果）

**確認コマンド:**
```python
from extrapolation_discovery_platform.pipeline import stage1_preprocess
prep1 = stage1_preprocess(X, y, comp, ["FS_ALL"], ["WF-LIN"], seeds=[42],
                           active_policies=["CompositionBlock"])
prep2 = stage1_preprocess(X, y, comp, ["FS_ALL"], ["WF-LIN"], seeds=[42],
                           active_policies=["CompositionBlock"])
assert prep1.effective_cols["FS_ALL"] == prep2.effective_cols["FS_ALL"]
```

**確認すべき点:**
- [ ] `prep.success == True`
- [ ] 2回実行した `effective_cols["FS_ALL"]` が完全一致（同一列・同一順序）
- [ ] `fold_plan` に `"CompositionBlock"` キーが存在する
- [ ] `fold_plan["CompositionBlock"]` の各要素が `(ndarray, ndarray)` のタプル
- [ ] `mc_reports` に FS 名のキーが存在する
- [ ] 有効列数が 0 より大きい（`len(effective_cols["FS_ALL"]) > 0`）

---

## T2: Stage2 再現性（同一条件 → 同一 RMSE）

**確認コマンド:**
```python
from extrapolation_discovery_platform.pipeline import stage2_train
tr1 = stage2_train(prep1, X, y, "WF-LIN", "CompositionBlock", "FS_ALL", quick=True, seed=42)
tr2 = stage2_train(prep2, X, y, "WF-LIN", "CompositionBlock", "FS_ALL", quick=True, seed=42)
assert abs(tr1.rmse_test_mean - tr2.rmse_test_mean) < 1e-8
```

**確認すべき点:**
- [ ] `tr.success == True`
- [ ] `tr.runs` の長さが `n_folds` と一致（通常 5）
- [ ] `tr.rmse_test_mean` が `float("nan")` でない（学習が成功している）
- [ ] `tr.r2_test_mean` が -∞ より大きい
- [ ] 2回実行した `rmse_test_mean` の差が 1e-8 未満（完全一致）
- [ ] `TrainResult` に `ood_result` フィールドが **存在しない**（OOD 分離の確認）

---

## T3: Stage3 独立性（OOD が RunResult に混入しない）

**確認コマンド:**
```python
from extrapolation_discovery_platform.pipeline import stage3_detect_ood
import dataclasses
from extrapolation_discovery_platform.workflows import RunResult

ood = stage3_detect_ood(X, prep1.effective_cols["FS_ALL"], prep1.fold_plan)
rr_fields = {f.name for f in dataclasses.fields(RunResult)}
assert "ood_result" not in rr_fields
assert ood.success
```

**確認すべき点:**
- [ ] `ood.success == True`
- [ ] `ood.ood_result` が `None` でない
- [ ] `ood.ood_result.composite_scores` の長さが `n_test_samples` と一致
- [ ] `ood.primary_train_idx` と `ood.primary_test_idx` が `None` でない
- [ ] `RunResult` のフィールド一覧に `ood_result` が **存在しない**
- [ ] Stage2 の結果（`TrainResult.runs`）に OOD 関連フィールドが **存在しない**

---

## T4: runner.py が pipeline.py に委譲している

**確認コマンド:**
```python
import inspect
from extrapolation_discovery_platform.runner import ExperimentRunner
src = inspect.getsource(ExperimentRunner.run)
assert "stage1_preprocess" in src
assert "stage3_detect_ood"  in src
assert "_phase3_precompute_folds" not in src  # 旧実装が残っていない
assert "_phase6_ood"              not in src  # 旧実装が残っていない
```

**確認すべき点:**
- [ ] `stage1_preprocess` が `run()` 内で呼ばれている
- [ ] `stage3_detect_ood` が `run()` 内で呼ばれている
- [ ] `_phase3_precompute_folds` メソッドが **削除されている**（ソース検索で見つからない）
- [ ] `_phase6_ood` メソッドが **削除されている**（ソース検索で見つからない）
- [ ] `runner._effective_cols` に Stage1 の結果が格納されている（実行後）
- [ ] `runner._ood_split_indices` に OOD の primary インデックスが格納されている（実行後）

---

## T5: individual_runner.py が pipeline.py に委譲している

**確認コマンド:**
```python
import inspect
from extrapolation_discovery_platform.individual_runner import run_individual
src = inspect.getsource(run_individual)
assert "stage1_preprocess" in src
assert "stage2_train"      in src
assert "stage3_detect_ood" in src
```

**確認すべき点:**
- [ ] `stage1_preprocess` が `run_individual()` 内で呼ばれている
- [ ] `stage2_train` が `run_individual()` 内で呼ばれている
- [ ] `stage3_detect_ood` が `run_individual()` 内で呼ばれている
- [ ] 旧来の Step1〜11 の独自前処理コードが **削除されている**
- [ ] `IndividualRunResult` に `ood_result` フィールドがある（Stage3 の出力を格納するため）
- [ ] `IndividualRunResult.runs` に `RunResult[]` が格納されている

---

## T6: 一括計算と個別計算の結果一致

**確認コマンド:**
```python
from extrapolation_discovery_platform.runner import ExperimentRunner
from extrapolation_discovery_platform.individual_runner import run_individual
import numpy as np

runner = ExperimentRunner(seeds=[42], quick=True)
runs_bulk, _, _ = runner.run(comp, X, y,
    selected_workflows=["WF-LIN"],
    selected_feature_sets=["FS_ALL"],
    selected_split_policies=["CompositionBlock"])
ec = runner._effective_cols.get("FS_ALL")
res_ind = run_individual("WF-LIN", "FS_ALL", "CompositionBlock",
    features_df=X, target=y, compositions_df=comp,
    seed=42, n_folds=5, quick=True, precomputed_columns=ec)

bulk_rmse = np.mean([r.rmse_test for r in runs_bulk
    if r.workflow=="WF-LIN" and r.split_policy=="CompositionBlock" and r.rmse_test>0])
diff_pct = abs(bulk_rmse - res_ind.rmse_test_mean) / (bulk_rmse + 1e-8) * 100
assert diff_pct < 1.0
```

**確認すべき点:**
- [ ] `precomputed_columns=ec`（runner の effective_cols）を渡した場合、RMSE 誤差が **1% 未満**
- [ ] `precomputed_columns=None`（なし）の場合は Stage1 を再実行し独立計算になる（誤差が大きい可能性あり — 許容）
- [ ] `res_ind.n_folds_executed == 5`（指定した fold 数が実行されている）
- [ ] `res_ind.ood_result` が `None` でない（Stage3 が正常実行されている）

---

## T7: RandomCV デフォルト無効化

**確認コマンド:**
```python
runner = ExperimentRunner(seeds=[42], quick=True)
runs, _, _ = runner.run(comp, X, y)
policies = {r.split_policy for r in runs}
assert "RandomCV" not in policies

# 明示的に有効化すると含まれる
runner2 = ExperimentRunner(seeds=[42], quick=True)
runs2, _, _ = runner2.run(comp, X, y,
    selected_split_policies=["CompositionBlock", "RandomCV"])
policies2 = {r.split_policy for r in runs2}
assert "RandomCV" in policies2
```

**確認すべき点:**
- [ ] `selected_split_policies` 未指定時に `"RandomCV"` が **含まれない**
- [ ] `selected_split_policies=["CompositionBlock", "RandomCV"]` 指定時に **含まれる**
- [ ] GUI の「分割ポリシー設定」アコーディオンで RandomCV チェックボックスが **デフォルト OFF**
- [ ] GUI で RandomCV を ON にして実行すると結果に RandomCV が含まれる

---

## T7-2: n_folds の設定が正しく伝達される

**確認コマンド:**
```python
from extrapolation_discovery_platform.pipeline import stage1_preprocess
from extrapolation_discovery_platform.runner import ExperimentRunner

# n_folds=3 で分割数が 3 になるか
prep = stage1_preprocess(X, y, comp, ["FS_BASE"], ["WF-LIN"],
                          seeds=[42], active_policies=["CompositionBlock"], n_folds=3)
assert len(prep.fold_plan["CompositionBlock"]) == 3

# ExperimentRunner に伝達されるか
runner = ExperimentRunner(seeds=[42], quick=True, n_folds=3)
assert runner._n_folds == 3
runs, _, _ = runner.run(comp, X, y, selected_split_policies=["CompositionBlock"])
folds = {r.fold for r in runs if r.split_policy == "CompositionBlock"}
assert max(folds) == 2  # 0,1,2 の 3-fold
```

**確認すべき点:**
- [ ] `stage1_preprocess(n_folds=3)` で `fold_plan["CompositionBlock"]` の長さが 3
- [ ] `ExperimentRunner(n_folds=3)` で `_n_folds == 3` が保存される
- [ ] 実行後の `RunResult.fold` の最大値が `n_folds - 1`
- [ ] デフォルト (`n_folds` 未指定) で 5-fold になる
- [ ] GUI の「分割数 (n_folds)」スライダーが **2〜10** の範囲でデフォルト **5**

---

## T8: WF-ENS ≠ WF-XGB

**確認コマンド:**
```python
from extrapolation_discovery_platform.workflows import WorkflowXGB, WorkflowENS
kw = dict(feature_set="FS_BASE", split_policy="CompositionBlock", fold=0)
r_xgb = WorkflowXGB(quick=True).run(X_tr, y_tr, X_te, y_te, seed=42, **kw)
r_ens  = WorkflowENS(quick=True).run(X_tr, y_tr, X_te, y_te, seed=42, **kw)
assert abs(r_xgb.rmse_test - r_ens.rmse_test) > 0.01
```

**確認すべき点:**
- [ ] `WF-XGB` と `WF-ENS` の `rmse_test` が **異なる**（同一値は修正前のバグ）
- [ ] `WorkflowENS` の `base_workflow` が `"ridge"` になっている
- [ ] ダッシュボードのヒートマップで WF-ENS と WF-XGB の色が異なる

---

## T9: _run_job が _IR_FACTORIES に委譲している

**確認コマンド:**
```python
import inspect
from extrapolation_discovery_platform.runner import _run_job
src = inspect.getsource(_run_job)
assert "_IR_FACTORIES" in src
assert "_BUILTIN_FACTORIES" not in src
```

**確認すべき点:**
- [ ] `_run_job` が `_IR_FACTORIES`（individual_runner の工場辞書）を使っている
- [ ] 旧来の `_BUILTIN_FACTORIES` が **存在しない**
- [ ] `_run_job` で WF-LIN を実行した結果と `_IR_FACTORIES["WF-LIN"](True,True).run()` の結果が**完全一致**

---

## GUI 手動確認チェックリスト

### 起動確認
- [ ] アプリが `http://localhost:7860` で起動する
- [ ] タイトルバーに `PR#148` が表示される
- [ ] 以下のタブが存在する:
  - [x] Config & Run
  - [x] Data Summary
  - [x] Dashboard
  - [x] Results
  - [x] OOD Map
  - [x] 🔬 Individual Run
  - [x] Literature Search
  - [x] Report
  - [x] Model Info

### Config & Run タブ
- [ ] サンプルデータ生成ボタンが動作する
- [ ] CSV アップロードが動作する
- [ ] ワークフロー選択チェックボックスが 6 種類（LIN, LASSO, ARD, RF, XGB, ENS）ある
- [ ] 特徴量セット選択チェックボックスが 6 種類（BASE, THERMO, SIZE, ELECTRON, ALL, MAGPIE）ある
- [ ] 「分割ポリシー設定」アコーディオンが存在する
  - [ ] CompositionBlock が **デフォルト ON**
  - [ ] ElementExclusion が **デフォルト ON**
  - [ ] RandomCV が **デフォルト OFF**（警告文が表示されている）
- [ ] [Run Analysis] ボタンが動作する
- [ ] 進捗バーが更新される

### Dashboard タブ
- [ ] 実行後に KPI（Runs, Best FS, Score, OOD count）が表示される
- [ ] ヒートマップが表示される
- [ ] Validity ランキングプロットが表示される
- [ ] WF フィルタドロップダウンが動作する

### Results タブ
- [ ] アルゴリズム別パリティグリッドが表示される
- [ ] 性能比較バーチャートが表示される
- [ ] Train vs Test パリティが表示される
- [ ] 結果テーブルに全 fold の RunResult が表示される
- [ ] WF / FS / SP のフィルタドロップダウンが動作する

### OOD Map タブ
- [ ] OOD マップが表示される（scatter plot）
- [ ] OOD サマリーテキストが表示される
- [ ] OOD 候補テーブルが表示される

### 🔬 Individual Run タブ
- [ ] WF・FS・Split Policy のドロップダウンが表示される
- [ ] Split Policy のデフォルトが `CompositionBlock`
- [ ] [実行] ボタンが動作する
- [ ] パリティプロットが表示される
- [ ] OOD マップが表示される
- [ ] fold 別メトリクステーブルが表示される
- [ ] 一括計算と同一条件で実行した場合、RMSE が概ね一致する（1% 以内）

### サイドバー（モバイル表示）
- [ ] サイドバー非表示ボタンが動作する（サイドバーが隠れる）
- [ ] ページをリロードしてもサイドバー制御が機能する

---

## パフォーマンス確認

| 条件 | 期待する動作 |
|---|---|
| n=100, WF=1, FS=1, folds=5 | Stage1+2+3 が 60 秒以内に完了 |
| n=100, WF=6, FS=6, folds=5 | 全計算が 300 秒（quick=True）以内 |
| Stage3 OOD のみ | 5 秒以内（フォールド数に関わらず） |

---

## 回帰確認（PR#147 との比較）

以下は PR#147 から PR#148 で **意図的に変更した** 挙動です。
テスト時に「バグ」と混同しないよう注意してください。

| 変更点 | PR#147 の挙動 | PR#148 の挙動 | 理由 |
|---|---|---|---|
| デフォルト split policy | RandomCV を含む | CompositionBlock + ElementExclusion のみ | データリーク防止 |
| WF-ENS の結果 | WF-XGB と同一 | WF-XGB と異なる | base_workflow を ridge に修正 |
| OOD の保存先 | RunResult に混入（なし）/ IndividualRunResult | Stage3 の OODStageResult に分離 | 設計の明確化 |
| 個別計算の前処理 | 独自実装（Step1〜11） | pipeline.py の Stage1〜3 に委譲 | 一括計算との結果一致 |

---

## トラブルシューティング

### テスト失敗: `T6: diff > 1%`

`precomputed_columns` を渡しているか確認してください。
渡していない場合、Stage1 が再実行されるため特徴量選択の乱数が影響し、
一括計算と若干異なる結果になる場合があります。

```python
# 正しい呼び出し方
ec = runner._effective_cols.get("FS_ALL")  # 一括計算後に取得
result = run_individual(..., precomputed_columns=ec)
```

### テスト失敗: `OOD n_ood = 0`

少数サンプルのテストデータ（n=80 程度）では OOD サンプルが検出されない場合があります。
これは正常な動作です。本番データ（n > 200）では適切に検出されます。

### テスト失敗: `_phase3_precompute_folds` が存在する

pipeline.py への委譲が完了していません。`runner.py` を確認してください。

```bash
grep -n "_phase3_precompute_folds\|_phase6_ood" runner.py
# → 何も表示されないことを確認（def がない）
```
