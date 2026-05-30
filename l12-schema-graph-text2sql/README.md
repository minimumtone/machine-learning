# L1₂ Schema-Graph-Assisted Text-to-SQL

L1₂型金属間化合物探索のためのスキーマグラフ支援型Text-to-SQLシステム。

## Overview

自然言語クエリからPostgreSQLのSELECT文を自動生成し、L1₂型金属間化合物（Ni₃Al型γ'相候補、A₃B型規則化FCC化合物）を探索するシステムです。

ERスキーマをNetworkXグラフ化し、関連テーブル・カラム・JOIN経路を制約としてLLMに与えることで、不正JOIN・存在しないカラム生成・multi-hop query失敗を低減します。

```
Natural Language Query → 材料用語正規化 → 条件抽出 → テーブル・カラム推定
→ スキーマグラフJOIN経路探索 → 制約付きSQL生成 → SQL安全検査
→ PostgreSQL実行 → 結果表示
```

## Quick Start

### 1. PostgreSQL起動

```bash
cd docker
docker compose up -d
```

### 2. Seed dataの生成・投入

```bash
# Generate seed CSVs (120 compounds)
python ingestion/generate_seed_data.py

# Load into PostgreSQL
python ingestion/load_seed_data.py
```

### 3. Text-to-SQL実行

```bash
# Copy and edit .env
cp .env.example .env

# Run the pipeline
python -c "
from llm.sql_generator import pipeline
result = pipeline('Niを含む安定なL1₂型化合物を形成エネルギーが低い順に出して')
print(result['sql'])
"
```

### 4. FastAPI起動

```bash
uvicorn api.main:app --reload
# POST /query with {"query": "L1₂構造を持つ化合物を一覧にして"}
```

### 5. テスト実行

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Project Structure

```
l12-schema-graph-text2sql/
├── docker/              # Docker Compose for PostgreSQL
├── db/                  # Schema, seed data, sample queries
├── ingestion/           # Data generation and loading
├── graph/               # Schema parser, graph builder, traversal
├── llm/                 # Entity extraction, schema linking, SQL generation
├── safety/              # SQL validation and safe execution
├── evaluation/          # Evaluation dataset, baselines, metrics
├── notebooks/           # Analysis notebooks
├── api/                 # FastAPI application
└── tests/               # Unit tests
```

## Evaluation

100件の評価クエリ（Easy 20, Medium 30, Hard 30, Very Hard 20）で以下を比較:

| Method | Description |
|--------|-------------|
| Baseline 1 | LLM only (no schema info) |
| Baseline 2 | LLM + full schema prompt |
| Baseline 3 | LLM + embedding-based retrieval |
| Baseline 4 | LLM + FK list |
| **Proposed** | **LLM + schema linking + graph traversal** |

## Key Features

- **スキーマグラフ**: NetworkXによるER関係のグラフ化とJOIN経路自動生成
- **材料用語辞書**: L1₂, γ', Cu₃Au型などの正規化辞書
- **制約付きSQL生成**: 許可テーブル・カラム・JOINのみ使用可能
- **SQL安全検査**: sqlglotによるパース検証、禁止操作検出
- **Rule-based fallback**: API keyなしでも動作する決定的SQL生成

## Seed Data

120件のL1₂型化合物mock data（11件の既知化合物を含む）:
Ni₃Al, Ni₃Ga, Ni₃Ge, Co₃Ti, Al₃Sc, Al₃Ti, Pt₃Al, Ir₃Nb, Co₃Al, Co₃W, Co₃Ta

## Environment Variables

See `.env.example` for configuration options.
