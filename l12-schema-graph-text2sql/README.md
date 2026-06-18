# L1₂ Schema-Graph-Assisted Text-to-SQL

> **デリバリZIPをお持ちの方は [README_DELIVERY.md](README_DELIVERY.md) を先にお読みください。**
> 本READMEは完全リポジトリ（Docker・DB・テスト含む）向けです。

L1₂型金属間化合物探索のためのスキーマグラフ支援型Text-to-SQLシステム。

## Overview

自然言語クエリからPostgreSQLのSELECT文を自動生成し、L1₂型金属間化合物（Ni₃Al型γ'相候補、A₃B型規則化FCC化合物）を探索するシステムです。

ERスキーマをNetworkXグラフ化し、関連テーブル・カラム・JOIN経路を制約としてLLMに与えることで、不正JOIN・存在しないカラム生成・multi-hop query失敗を低減します。

```
Natural Language Query → 材料用語正規化 → 条件抽出 → テーブル・カラム推定
→ スキーマグラフJOIN経路探索 → 制約付きSQL生成 → SQL安全検査（SQLGuard 14種検証）
→ PostgreSQL実行 → 結果表示
```

## Quick Start

> **Note**: デリバリZIPを受け取った方は `README_DELIVERY.md` を参照してください。
> 以下はリポジトリのクローンを前提とした完全環境構築手順です。
> ZIPにはリポジトリの主要ファイルが同梱されています。詳細は `README_DELIVERY.md` を参照。

### 前提条件

- Python 3.11+
- Docker / Docker Compose
- OpenAI API key（Proposed手法の実行に必要。Rule-based fallbackはAPI key不要）

### 1. 依存パッケージのインストール

```bash
cd l12-schema-graph-text2sql
pip install -e ".[dev]"
```

### 2. PostgreSQL起動（スキーマ + データ自動投入）

```bash
cd docker
docker compose up -d
cd ..
```

これにより `db/extended_schema.sql`（30テーブル）と `db/insert_data.sql`（データ投入）が
自動適用され、material_entry等にデータが投入されます。

### 3. 環境変数の設定

```bash
cp .env.example .env
# .env を編集し OPENAI_API_KEY を設定
```

### 4. Text-to-SQL実行

```bash
python -c "
from llm.sql_generator import pipeline
result = pipeline('Niを含む安定なL1₂型化合物を形成エネルギーが低い順に出して')
print(result['sql'])
"
```

### 5. FastAPI起動（オプション）

```bash
uvicorn api.main:app --reload
# POST /query with {"query": "L1₂構造を持つ化合物を一覧にして"}
```

### 6. テスト実行

```bash
pytest tests/ -v
# 126テスト全パスを確認
```

### 7. 評価パイプライン実行（オプション）

```bash
# 100クエリ×5手法の完全評価（OpenAI API key必要、10-15分程度）
python scripts/run_full_evaluation.py
```

## Project Structure

```
l12-schema-graph-text2sql/
├── docker/              # Docker Compose設定（docker-compose.yml）
│   └── docker-compose.yml
├── db/                  # スキーマ定義・データ投入
│   ├── extended_schema.sql  # 30テーブルスキーマ（Docker起動時に自動適用）
│   ├── insert_data.sql      # データ投入SQL（Docker起動時に自動適用）
│   └── sample_queries.sql
├── ingestion/           # データ生成・正規化
│   ├── generate_extended_data.py  # 拡張データ生成
│   └── data_normalizer.py        # データ正規化
├── graph/               # Schema Graph構築・走査
│   ├── schema_parser.py        # FK関係抽出（information_schema）
│   ├── graph_builder.py        # NetworkXグラフ構築
│   ├── traversal_engine.py     # Steiner木近似走査
│   └── join_path_generator.py  # JOIN条件生成
├── llm/                 # LLM連携・条件抽出
│   ├── entity_extractor.py     # 材料用語抽出（元素、構造、安定性等）
│   ├── schema_linker.py        # テーブル・カラムマッピング
│   ├── condition_mapper.py     # SQL WHERE句生成
│   ├── sql_generator.py        # 制約付きSQL生成パイプライン
│   ├── few_shot_store.py       # Few-shot例の蓄積・検索
│   └── material_terms.yaml     # 材料用語辞書（L1₂, B2, γ'等）
├── safety/              # SQL安全検査（SQLGuard 14種検証）
│   ├── sql_validator.py      # 13種の個別検査 + 統合検証
│   ├── sql_guard.py          # ガードエントリポイント
│   └── allowed_schema.yaml   # 許可テーブル・カラム定義
├── evaluation/          # 評価パイプライン
│   ├── evaluation_dataset.jsonl # 100クエリ（Easy/Medium/Hard/VeryHard）
│   ├── gold_sql/        # 正解SQL 212件（著者100件 + 独立100件 + VH追加12件）
│   ├── expected_results/ # 正解実行結果JSON（212件）
│   ├── metrics.py       # 評価指標（構文妥当率、実行精度等）
│   ├── run_proposed.py  # Proposed手法実行
│   ├── proposed_result.csv      # 代表ラン (= Run 2, 70.6%)
│   ├── proposed_result_run1.csv # Run 1 (72.7%)
│   ├── proposed_result_run2.csv # Run 2 (70.6%)
│   ├── proposed_result_run3.csv # Run 3 (69.4%)
│   └── baseline_result.csv     # ベースライン4手法結果
├── scripts/             # 評価・分析スクリプト
│   ├── run_full_evaluation.py      # 5手法完全評価
│   ├── run_proposed_only.py        # Proposed手法のみ再評価
│   ├── run_expert_evaluation.py    # 独立設計100件評価
│   ├── compute_paper_figures.py    # 論文数値JSON生成
│   └── validate_paper_numbers.py   # TeX数値検証
├── api/                 # FastAPI アプリケーション
│   └── main.py
├── tests/               # ユニットテスト（126件）
├── paper/               # LaTeX原稿
├── pyproject.toml       # Python依存パッケージ定義
└── .env.example         # 環境変数テンプレート
```

## Evaluation

100件の評価クエリ（Gold SQL参照テーブル数による再分類: Easy 27, Medium 28, Hard 22, Very Hard 23）で以下の5手法を比較:

| Method | LLMに渡す情報 | 構文妥当率 | 実行成功率 | 実行精度 | テーブル幻覚率 | JOIN幻覚 |
|--------|--------------|-----------|-----------|---------|--------------|---------|
| B1: LLM-only | 何も渡さない | 98% | 98% | 64.6% | 0% | 16件 |
| B2: Full Schema | 全テーブル一覧 | 94% | 94% | 68.7% | 0% | 18件 |
| B3: Rule-based | 辞書ルール（LLM不使用） | 100% | 100% | 52.8% | 0% | 0件 |
| B4: FK-list | FK関係リストのみ | 98% | 98% | 66.4% | 0% | 21件 |
| **P: Proposed** | **Steiner木で選んだサブグラフ** | **100%** | **100%** | **70.6%** (3回平均70.9%±1.7pp) | **0%** | **3件** |

### 3-run 統計の経緯

| ドラフト | ラン構成 | 平均±σ | 備考 |
|---|---|---|---|
| v1 | 69.3, 70.6, 69.4 | 69.8%±0.7pp | Run 1 (69.3%) のCSVが梱包ミスでRun 2の複製になっていた |
| v2 (現行) | 72.7, 70.6, 69.4 | 70.9%±1.7pp | 69.3ランのCSVは復元不可のため新規Run 1を独立再評価 |

v1の69.3%ランは生CSVが消失しており復元不可能。現3ランは全て独立実行（MD5一意確認済み）。
代表ラン = Run 2 (70.6%, 中央値ラン) を `proposed_result.csv` として使用。

### JOIN方向バグ修正の評価影響

v4でgraph層のJOIN方向バグ（`_edge_source`による逆方向走査時のカラム入れ替え）を修正。
結果CSVは修正前コードで生成されたものだが、評価100クエリへの影響はなし:
非対称カラム名のJOINは2件（`material_defect–element`, `application_domain`自己参照）のみで、
評価クエリでこれらのテーブルに触れるものは0件。残りは全て`entry_id=entry_id`型の対称JOIN。

### ベースラインCSVの注記

`baseline_result.csv` は `condition_mapper`/`entity_extractor` の辞書拡張
（elastic_tensor, thermal_property, magnetic_property対応）前のコードで生成。
辞書拡張後にB3 (Rule-based) を再ランすると52.8%から変動する可能性あり。

## Key Features

- **Schema Graph走査**: NetworkXによるFK関係のグラフ化、Steiner木近似による最小JOINパス探索
- **材料用語辞書**: L1₂, B2, γ', Cu₃Au型, CsCl型などの日英バイリンガル同義語辞書
- **制約付きSQL生成**: 許可テーブル・カラム・JOINのみ使用可能
- **SQLGuard 14種検証**: ブラックリスト、SELECT-only、複文検出、危険関数、テーブル/カラムホワイトリスト、JOIN整合性、LIMIT自動注入、CTE検査、型安全、トートロジー検出、サブクエリ深度制限、システムテーブル検出
- **Rule-based fallback**: API keyなしでも動作する決定的SQL生成
- **B2対応**: CsCl型（B2）、NaCl型、NiAs型、BiF3型にも対応可能な設計

## Seed Data

デフォルト: 120件のL1₂型化合物mock data（既知11件を含む）:
Ni₃Al, Ni₃Ga, Ni₃Ge, Co₃Ti, Al₃Sc, Al₃Ti, Pt₃Al, Ir₃Nb, Co₃Al, Co₃W, Co₃Ta

OQMD拡張データ投入で最大1,470件（L12 392 + B2 636 + NaCl 355 + NiAs 74 + BiF3 13）。

## Environment Variables

| 変数名 | 説明 | デフォルト |
|--------|------|-----------|
| POSTGRES_USER | PostgreSQLユーザー | l12_user |
| POSTGRES_PASSWORD | PostgreSQLパスワード | l12_password |
| POSTGRES_DB | データベース名 | l12_materials |
| POSTGRES_HOST | ホスト | localhost |
| POSTGRES_PORT | ポート | 5432 |
| OPENAI_API_KEY | OpenAI APIキー | （要設定） |
| LLM_MODEL | 使用するLLMモデル | gpt-5.5 |
| SQL_ROW_LIMIT | SQL結果の最大行数 | 100 |
| SQL_TIMEOUT_SECONDS | SQL実行タイムアウト | 10 |
