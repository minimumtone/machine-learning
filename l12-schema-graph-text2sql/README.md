# L1₂ Schema-Graph-Assisted Text-to-SQL

L1₂型金属間化合物探索のためのスキーマグラフ支援型Text-to-SQLシステム。

## Overview

自然言語クエリからPostgreSQLのSELECT文を自動生成し、L1₂型金属間化合物（Ni₃Al型γ'相候補、A₃B型規則化FCC化合物）を探索するシステムです。

ERスキーマをNetworkXグラフ化し、関連テーブル・カラム・JOIN経路を制約としてLLMに与えることで、不正JOIN・存在しないカラム生成・multi-hop query失敗を低減します。

```
Natural Language Query → 材料用語正規化 → 条件抽出 → テーブル・カラム推定
→ スキーマグラフJOIN経路探索 → 制約付きSQL生成 → SQL安全検査（SQLGuard 8層）
→ PostgreSQL実行 → 結果表示
```

## Quick Start

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
# 125テスト全パスを確認
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
├── safety/              # SQL安全検査（SQLGuard 8層）
│   └── sql_validator.py
├── evaluation/          # 評価パイプライン
│   ├── evaluation_dataset.jsonl # 100クエリ（Easy/Medium/Hard/VeryHard）
│   ├── gold_sql/        # 正解SQL 100件
│   ├── expected_results/ # 正解実行結果JSON
│   ├── metrics.py       # 評価指標（構文妥当率、実行精度等）
│   ├── run_proposed.py  # Proposed手法実行
│   └── *.csv            # 各手法の評価結果
├── scripts/             # 評価・分析スクリプト
│   ├── run_full_evaluation.py      # 5手法完全評価
│   └── generate_gold_sql_and_results.py  # Gold SQL生成
├── api/                 # FastAPI アプリケーション
│   └── main.py
├── tests/               # ユニットテスト（125件）
├── paper/               # LaTeX原稿
├── pyproject.toml       # Python依存パッケージ定義
└── .env.example         # 環境変数テンプレート
```

## Evaluation

100件の評価クエリ（Easy 27, Medium 28, Hard 22, Very Hard 23）で以下の5手法を比較:

| Method | LLMに渡す情報 | 構文妥当率 | 実行成功率 | 実行精度 | テーブル幻覚率 | JOIN幻覚 |
|--------|--------------|-----------|-----------|---------|--------------|---------|
| B1: LLM-only | 何も渡さない | 98% | 98% | 64.6% | 0% | 16件 |
| B2: Full Schema | 全テーブル一覧 | 94% | 94% | 68.7% | 0% | 18件 |
| B3: Rule-based | 辞書ルール（LLM不使用） | 100% | 93% | 36.9% | 0% | 3件 |
| B4: FK-list | FK関係リストのみ | 98% | 98% | 66.4% | 0% | 21件 |
| **P: Proposed** | **Steiner木で選んだサブグラフ** | **100%** | **100%** | **70.6%** (3回平均69.8%±0.7pp) | **0%** | **3件** |

## Key Features

- **Schema Graph走査**: NetworkXによるFK関係のグラフ化、Steiner木近似による最小JOINパス探索
- **材料用語辞書**: L1₂, B2, γ', Cu₃Au型, CsCl型などの日英バイリンガル同義語辞書
- **制約付きSQL生成**: 許可テーブル・カラム・JOINのみ使用可能
- **SQLGuard 8層検証**: ブラックリスト、SELECT-only、複文検出、危険関数、テーブル/カラムホワイトリスト、JOIN整合性、LIMIT自動注入
- **Rule-based fallback**: API keyなしでも動作する決定的SQL生成
- **B2対応**: CsCl型（B2）、NaCl型、NiAs型、BiF3型にも対応可能な設計

## Seed Data

デフォルト: 120件のL1₂型化合物mock data（既知11件を含む）:
Ni₃Al, Ni₃Ga, Ni₃Ge, Co₃Ti, Al₃Sc, Al₃Ti, Pt₃Al, Ir₃Nb, Co₃Al, Co₃W, Co₃Ta

OQMD拡張データ投入で最大1,471件（L12 393 + B2 636 + NaCl 355 + NiAs 74 + BiF3 13）まで拡張可能。

## Environment Variables

| 変数名 | 説明 | デフォルト |
|--------|------|-----------|
| POSTGRES_USER | PostgreSQLユーザー | l12_user |
| POSTGRES_PASSWORD | PostgreSQLパスワード | l12_password |
| POSTGRES_DB | データベース名 | l12_materials |
| POSTGRES_HOST | ホスト | localhost |
| POSTGRES_PORT | ポート | 5432 |
| OPENAI_API_KEY | OpenAI APIキー | （要設定） |
| LLM_MODEL | 使用するLLMモデル | gpt-4o-mini |
| SQL_ROW_LIMIT | SQL結果の最大行数 | 100 |
| SQL_TIMEOUT_SECONDS | SQL実行タイムアウト | 10 |
