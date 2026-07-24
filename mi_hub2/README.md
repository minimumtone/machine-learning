# mi_hub2 — MI 統合環境(runcell / OptiMat / MLflow / Feast / AutoML) + 研究エージェント

「Jupyter(runcell)を操縦席、MLflow を記録層、parquet + provenance を連携層」とする
疎結合アーキテクチャの実装一式。全モジュールはコンテナ内でエンドツーエンド検証済み
(TC-Python 実接続部と OptiMat 実イメージ部を除く)。

```
[知識]   既存 GraphRAG + ExtractBib(ローカル。Bohrium は設計参考のみ)
[操縦席] JupyterLab + runcell 0.2.0 + pygwalker 0.5.0.1
   ├→ [計算]   OptiMat Alloys(Docker) → optimat_bridge で living DB を吸い上げ
   ├→ [計算]   TC-Python パイプライン  → notebooks/02 の定型 8 行で記録
   ├→ [記録]   MLflow(sqlite)         → tracking.track() / mi_hub.run_id タグ
   ├→ [特徴量] Feast(registry+offline のみの最小構成)
   ├→ [AutoML] FLAML                    → automl.fit_baseline()
   └→ [事例]   case_template(1 事例 = 1 ディレクトリ規約。実行者非依存)
[研究エージェント] mi_hub.agent(状態駆動の仮説・反証検証ループ + Human-in-the-loop)
```

## 構成

| パス | 内容 | フェーズ |
|---|---|---|
| `setup/phase1_setup.sh` | conda 環境 + runcell/pygwalker/MLflow/mi_hub 一括導入 | 1 |
| `src/mi_hub/datastore.py` | parquet + provenance(run_id/created_at/source/code_ver)共通層 | 1 |
| `src/mi_hub/tracking.py` | MLflow ヘルパー(track/log_table/log_metrics/runs) | 1 |
| `notebooks/02_ternary_ingest_demo.py` | TC-Python 三元断面 → 記録の定型(ドライラン可) | 1 |
| `setup/phase2_optimat/` | OptiMat の docker-compose + living DB マウント手順 | 2 |
| `src/mi_hub/optimat_bridge.py` | living DB(SQLite/JSON)スナップショット取得 | 2 |
| `case_template/` | 事例テンプレート(README/run.py/env.yml。runcell/手動どちらでも実行可) | 3 |
| `feast_repo/` | Feast 最小構成(hea_features の FeatureView 例) | 4 |
| `src/mi_hub/automl.py` | FLAML ベースライン + MLflow 一元記録 | 4 |
| `notebooks/01_eda_and_automl_demo.py` | pygwalker EDA + FLAML デモ | 1/4 |
| `src/mi_hub/agent/` | 状態駆動研究エージェント(下記参照) | 5 |
| `tests/` | 受入試験(指示書 §19 準拠) | 5 |

## 研究エージェント(Phase 5, 指示書「研究エージェント機能追加改訂版」準拠)

Devin/OpenHands 型の状態駆動実行ループを Human-in-the-loop 制御下で実装:

```
Goal → Observe → Plan → Human Check → Act → Observe Result → Evaluate → Replan
```

| モジュール | 内容 | 指示書 |
|---|---|---|
| `agent/states.py` | エージェント/タスク/仮説/エラー状態 | §6, §9.1 |
| `agent/models.py` | Goal/Plan/Task/Hypothesis/Budget/StopConditions 等 | §4, §8, §15 |
| `agent/roles.py` | Evidence/Hypothesis/ModelSelection/VerificationPlanning/Execution/Evaluation/SafetyApproval の論理エージェント | §5 |
| `agent/loop.py` | Research Manager(実行ループ・計画版管理・停止条件・HITL・永続化) | §4, §5.1, §8, §12 |
| `agent/errors.py` | エラー分類・自動修正(単位変換等)・再試行上限 | §9 |
| `agent/tools.py` | t2X/MCP ゲートウェイのモック(MInt レジストリ・文献検索・推論) | §3 |
| `agent/llm.py` | LLM 補助(構造化・仮説候補・要約のみ。判定には不使用) | §16 |
| `agent/api.py` | FastAPI 研究エージェント API | §14 |
| `agent/ui_streamlit.py` | チャット UI + Agent 状態ペイン | §13 |

### 起動

```bash
pip install -e ".[agent]"            # または: pip install pydantic fastapi uvicorn streamlit
PYTHONPATH=src streamlit run src/mi_hub/agent/ui_streamlit.py   # チャットUI
PYTHONPATH=src uvicorn mi_hub.agent.api:app --port 8800          # REST API
PYTHONPATH=src python -m pytest tests/ -q                        # 受入試験
```

- セッション状態は `$MI_HUB_DATA/agent_sessions/*.json` に永続化され、再開可能。
- `OPENAI_API_KEY` 設定時は目標構造化・仮説候補生成に LLM を使用(未設定時は決定論的フォールバック)。
- 高コスト操作(モデル一括実行・DFT 提案等)は承認されるまで実行されない(§11)。
- 反証条件の変更・仮説の正式採用/反証は人間のみが行える(§5.8, §10)。
- MVP 制限: 自動反復 5 回・モデル実行 30 件・DFT/実験は提案のみ(§18)。
- 実 MInt/GraphRAG 接続時は `agent/tools.py` の `ToolGateway` を同一インターフェイスで差し替える。

## クイックスタート(Phase 1)

```bash
bash setup/phase1_setup.sh hub
export MI_HUB_DATA=$HOME/mi_hub_data
export MI_HUB_MLFLOW=sqlite:///$HOME/mi_hub_data/mlflow.db
conda activate hub
python notebooks/02_ternary_ingest_demo.py     # 記録層の動作確認
mlflow ui --backend-store-uri $MI_HUB_MLFLOW -p 5000
jupyter lab                                     # runcell から自然言語で操作
```

既存 TC-Python パイプラインへの組み込みは `notebooks/02` の
「計算 → 記録」8 行を計算ループの外側に被せるだけ。

## 全パイプライン共通の定型

```python
from mi_hub import datastore as ds, tracking as tr

rid = ds.new_run_id()
with tr.track("<experiment>", run_id=rid, params={...}):
    df = <計算本体>
    ds.save(df, "<kind>", run_id=rid, source="<tc_python|optimat|dft|...>")
    tr.log_table(df)
    tr.log_metrics({...})
```

## 既知の注意点

- **runcell のデータガバナンス**: クラウド LLM 前提のため、未公開データを
  ノートブックに載せる前に外部送信範囲とローカルエンドポイント設定可否を確認。
- **FLAML × MLflow**: FLAML 2.3+ の自動ロギングは skops 直列化で失敗するため
  `mlflow_logging=False` で無効化済み(記録は mi_hub 側で一元化)。
- **OptiMat**: イメージ名・コンテナ内 DB パスは配布元の最新 README に合わせて
  `setup/phase2_optimat/docker-compose.yml` を書き換えること。
- **Feast**: 依存が重いので Phase 1 環境と分けるか、導入時に pin を確認。

## 検証済み事項(2026-07-06, Python 3.12 / mlflow 3.14 / flaml 2.x)

- datastore 保存・結合ロード・catalog
- tracking(sqlite バックエンド、artifact、mi_hub.run_id タグ)
- 三元断面デモのドライラン一式
- FLAML 回帰ベースライン → モデル/予測/指標の MLflow 記録
- case_template の run.py スケルトン実行
