# L1₂ Schema-Graph-Assisted Text-to-SQL — 配布ZIP向けガイド

> **このドキュメントは配布ZIP同梱用です。**
> 本ZIPはリポジトリの完全検証可能パッケージです。

## ZIP内容物

本ZIPはリポジトリのほぼ完全なコピーであり、以下が含まれます:

### 論文・評価データ
- **論文PDF** (`paper/t2sql_materials_paper.pdf`)
- **論文TeXソース** (`paper/t2sql_materials_paper.tex`)
- **論文数値JSON** (`paper/paper_figures.json` — Single Source of Truth)
- **評価結果CSV** (`evaluation/proposed_result*.csv`, `baseline_result.csv`)
- **分析用注釈付CSV** (`evaluation/proposed_result_annotated.csv` — `n_tables`/`ntables_difficulty`/`original_difficulty`列付き。代表ランCSVはrun2とバイト同一を維持)
- **Gold SQL** (`evaluation/gold_sql/` 212件：著者設計100件 + 独立設計元100件 + 新規VH12件)
- **期待結果JSON** (`evaluation/expected_results/` 212件：著者設計100件 + 独立設計元100件 + 新規VH12件)
- **独立評価結果** (`evaluation/expert_evaluation_results.json`)
- **材料分析CSV** (`evaluation/known_l12_recovery.csv`, `stable_l12_candidates.csv`, `gamma_prime_candidate_ranking.csv`, `ni3al_lattice_matched_candidates.csv`)
- **プロトタイプ分布** (`evaluation/prototype_distribution.csv` — L12=392件等の集計根拠)

### ソースコード
- **スキーマグラフ** (`graph/` — schema_parser, graph_builder, traversal_engine, join_path_generator)
- **LLM連携** (`llm/` — entity_extractor, schema_linker, condition_mapper, sql_generator, few_shot_store)
- **SQL安全検査** (`safety/` — sql_validator.py 14種検証, sql_guard.py, allowed_schema.yaml)
- **評価スクリプト** (`scripts/run_full_evaluation.py`, `scripts/run_proposed_only.py`, `scripts/compute_paper_figures.py`, `scripts/validate_paper_numbers.py`)
- **データ生成** (`ingestion/` — generate_extended_data.py, data_normalizer.py)
- **FastAPI** (`api/main.py`)

### インフラ・テスト
- **Docker設定** (`docker/docker-compose.yml`)
- **DB定義** (`db/extended_schema.sql`, `db/insert_data.sql` — 1,470件)
- **テスト** (`tests/` — 134件)
- **依存定義** (`pyproject.toml`)
- **環境変数テンプレート** (`.env.example`)
- **実験スクリプト** (`experiments/`)

## 検証可能な項目（ZIP単体）

### 1. CSV数値の再計算

```python
import csv

with open("evaluation/proposed_result.csv") as f:
    rows = list(csv.DictReader(f))

# 代表ラン (Run 2) の精度
acc = sum(float(r["execution_accuracy"]) for r in rows) / len(rows)
print(f"Proposed accuracy: {acc:.4f}")  # → 0.7059 (70.6%)
```

### 2. 3-run統計の検証

```python
import csv, statistics

files = {
    "run1": "evaluation/proposed_result_run1.csv",
    "run2": "evaluation/proposed_result_run2.csv",
    "run3": "evaluation/proposed_result_run3.csv",
}
accs = []
for name, path in files.items():
    with open(path) as f:
        rows = list(csv.DictReader(f))
    acc = sum(float(r["execution_accuracy"]) for r in rows) / len(rows)
    accs.append(acc * 100)
    print(f"{name}: {acc:.1%}")

print(f"Mean: {statistics.mean(accs):.1f}% ± {statistics.stdev(accs):.1f}pp")
# → Run1: 72.7%, Run2: 70.6%, Run3: 69.4%, Mean: 70.9% ± 1.7pp
```

### 3. Gold SQLテーブル数による難易度分類

```python
import re, os

for sql_file in sorted(os.listdir("evaluation/gold_sql")):
    with open(f"evaluation/gold_sql/{sql_file}") as f:
        sql = f.read()
    tables = set(re.findall(r'\bFROM\s+(\w+)|\bJOIN\s+(\w+)', sql, re.IGNORECASE))
    n = len({t for pair in tables for t in pair if t})
    # n_tables → difficulty: 1-2=Easy, 3=Medium, 4=Hard, 5+=Very Hard
```

### 4. paper_figures.json の再生成

```bash
# expert_evaluation_results.json がない場合は独立評価セクションをスキップ
python scripts/compute_paper_figures.py
```

## 完全再現

完全な評価再現は、このZIP展開版またはリポジトリクローンのどちらでも可能です:

```bash
# ZIP展開版の場合
cd l12-schema-graph-text2sql
pip install -e ".[dev]"
cd docker && docker compose up -d && cd ..
cp .env.example .env  # OPENAI_API_KEY を設定
python scripts/run_full_evaluation.py  # 100クエリ×5手法 (10-15分)
```

## 3-run統計の経緯

| ドラフト | ラン構成 | 平均±σ | 備考 |
|---|---|---|---|
| v1 | 69.3, 70.6, 69.4 | 69.8%±0.7pp | Run 1 (69.3%) のCSVが梱包ミスでRun 2の複製に |
| v2 (現行) | 72.7, 70.6, 69.4 | 70.9%±1.7pp | 69.3ランのCSV復元不可→新規Run 1を独立再評価 |

代表ラン = Run 2 (70.6%, 中央値ラン) を `proposed_result.csv` として使用。

## 独立評価の再採点経緯

| 版 | 平均実行精度 | 二値正答率 | 根拠 |
|---|---|---|---|
| v10以前 | 79.3% | 77.0% | 旧DB（insert_data.sql変更前）で評価。expected_resultsとJSONが74/100件不一致 |
| v11 | 76.6% | 67.0% | 現DB（insert_data.sql最終版）で再採点。expected_results全100件一致 |
| v14（現行） | **62.5%** | **53.3%** | 統一難易度基準で60件調和セットに再構成（元プール48件＋新規VH12件） |

v11→v14の変化は生成SQLの変更ではなく、評価セットの再構成による：
- 元の100件プールからEasy/Medium/Hardを統一複雑度スコアで48件選定
- 5-7テーブルJOINを必要とするVery Hard 12件を新規設計・追加
- 結果として難易度分布がEasy 12 / Medium 18 / Hard 18 / Very Hard 12に均等化
- 新規VH12件の平均精度は19.4%であり、全体精度を引き下げている

v10→v11の変化はDB変更による：insert_data.sqlがv10以前の評価後に
3回変更され（commit `82a661e`, `1bd9f2e`, `41a1861`）、DB状態が変化したため
同じSQLでも期待結果の行数が変わり、22件の正誤が反転した（16件↓6件↑）。
著者設計100件のexpected_resultsは現DBと完全一致しており影響なし。

## JOIN方向バグ修正の影響確認

v4でgraph層のJOIN方向バグ（`_edge_source`による逆方向走査時のカラム入れ替え）を修正したが、
結果CSVは修正前コードで生成されたもの。検証の結果、評価100クエリへの影響はなし:
非対称カラム名のJOINは2件のみ（`material_defect–element`, `application_domain`自己参照）で、
評価クエリでこれらのテーブルに触れるものは0件。残りは全て`entry_id=entry_id`型の対称JOINで結果同一。

## pipeline() の簡易実行について

`llm.sql_generator.pipeline()` を `join_list=None` で呼ぶと、5テーブルのフォールバックJOINセットで動作します。
論文が主張する30テーブルSchema Graph走査を再現するには、DBからFK情報を取得して `join_list` を明示的に渡す必要があります:

```python
from graph.join_path_generator import get_allowed_join_list
join_list = get_allowed_join_list(db_connection)  # requires live DB
result = pipeline(query, join_list=join_list)
```

**LIMIT値について**: rule-based fallbackはLIMITを生成しません。評価時は `normalize_limit()` により
LIMITなしのSQLに `LIMIT 10000` が統一的に付加されます（全手法共通）。
API経由の場合は `sql_validator.check_limit()` が `LIMIT 10000` を自動付加します。

## LaTeX再コンパイルに必要な環境

論文PDFの再コンパイルには以下が必要です:

- LuaLaTeX (TeX Live 2024以降推奨)
- 日本語フォント: IPAexMincho, IPAexGothic (`apt install fonts-ipaexfont` 等)
- パッケージ: `fontspec`, `luatexja`, `booktabs`, `amsmath`, `hyperref` 等

フォントが未インストールの環境では `fontspec Error: The font "IPAexMincho" cannot be found` で停止します。
PDF自体は同梱済みのため、閲覧のみであれば再コンパイル不要です。

## ベースラインCSVに関する注記

`baseline_result.csv` は `condition_mapper`/`entity_extractor` の辞書拡張（elastic_tensor,
thermal_property, magnetic_property対応）前のコードで生成されたもの。
辞書拡張後にB3 (Rule-based) を再ランすると52.8%から変動する可能性がある。

## 150クエリ参考実験について

`experiments/results/` の150クエリ実験結果（`extended_schema_experiment.json` 等）は
**論文の主実験（100クエリ評価）とは別の参考実験**です。

- **使用モデル**: gpt-4o-mini（主実験はgpt-5.5）
- **環境**: 30テーブル・150クエリ（3条件比較）
- **位置づけ**: Graph Traversalの効果検証用の補助実験。論文の主張（70.6%等）の根拠ではない
