# 外挿発見基盤 モック完成工程表
# Extrapolation Discovery Platform - Implementation Roadmap

> HEAは例題。プラットフォームは汎用設計とする。

---

## Phase 1: 基盤モジュール完成（実装容易・依存少）

| # | タスク | 難易度 | 依存 | 成果物 |
|---|--------|--------|------|--------|
| 1-1 | `runner.py` 実験オーケストレータ | 低 | features/dataset/splitters/workflows/ood/evaluation (済) | runner.py |
| 1-2 | `literature_graph/schemas.py` データモデル定義 | 低 | なし（pure dataclass） | schemas.py |
| 1-3 | `literature_graph/workflow_text.py` テンプレテキスト生成 | 低 | schemas.py | workflow_text.py |
| 1-4 | 文献シードデータ作成 (20-50 papers) | 低 | schemas.py | papers.jsonl, workflows.jsonl, edges.jsonl |

## Phase 2: 検索・推薦エンジン（FAISS + フィルタ）

| # | タスク | 難易度 | 依存 | 成果物 |
|---|--------|--------|------|--------|
| 2-1 | `literature_graph/vector_index.py` VectorIndex抽象 + FAISSバックエンド | 中 | schemas.py, workflow_text.py | vector_index.py |
| 2-2 | `literature_graph/search.py` 2段構え検索（embedding → 構造化フィルタ） | 中 | vector_index.py, schemas.py | search.py |
| 2-3 | `literature_graph/feature_recommender.py` 文献由来FeatureSet生成 | 中 | search.py, features.py | feature_recommender.py |

## Phase 3: 統合・レポート・デモ

| # | タスク | 難易度 | 依存 | 成果物 |
|---|--------|--------|------|--------|
| 3-1 | `report.py` に文献近傍WF証拠・特徴量推薦セクション追加 | 低 | search.py, feature_recommender.py | report.py (更新) |
| 3-2 | `search_demo.md` 例クエリ5つ + 結果 + 推薦FeatureSet | 低 | search.py | search_demo.md |
| 3-3 | `__init__.py` 汎用化（HEA固有名を抽象化） | 低 | 全モジュール | __init__.py (更新) |

## Phase 4: 品質保証

| # | タスク | 難易度 | 依存 | 成果物 |
|---|--------|--------|------|--------|
| 4-1 | lint / type check / import検証 | 低 | 全Phase | - |
| 4-2 | 簡易動作確認スクリプト | 低 | 全Phase | - |

## Phase 5: PR作成・CI

| # | タスク | 難易度 | 依存 | 成果物 |
|---|--------|--------|------|--------|
| 5-1 | git add / commit / push / PR作成 | - | Phase 4 | PR |
| 5-2 | CI確認・修正 | - | 5-1 | - |

---

## 設計原則

1. **汎用性**: HEAは`domain="HEA"`パラメータで注入。コア機能は材料系一般に適用可能。
2. **ロバスト性**: 全モジュールにlogging, 型ヒント, 例外ハンドリング。
3. **契約(Contract)固定**: schemas.pyのデータモデルはMVP→本番で互換を保つ。
4. **段階的拡張**: FAISS→Milvus, JSONL→Neo4j はI/F変更不要で差替え可能。
5. **著作権**: 論文本文は保存しない。書誌+構造化メタデータ+自作要約のみ。

---

## 受け入れ基準 (Definition of Done)

- [ ] 135 Runが再現可能
- [ ] 妥当性スコアが算出される
- [ ] OODクラスタが可視化される
- [ ] 少なくとも1つの「外挿領域候補」が提案される
- [ ] 特徴量の上位5つが妥当性順に並ぶ
- [ ] `composition only + yield_strength + N<300` クエリで妥当な論文WFが上位に出る
- [ ] 推奨FeatureSet案が最大5特徴追加で生成される
- [ ] レポートに「文献近傍WF」が自動添付される
