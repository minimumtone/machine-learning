# L1₂ Schema-Graph-Assisted Text-to-SQL — デリバリZIP向けガイド

> **このドキュメントはデリバリZIP同梱用です。**
> 完全リポジトリ（Docker・DB・テスト含む）での再現手順は `README.md` を参照してください。

## ZIP内容物

本ZIPには以下が含まれます:

- **論文PDF** (`paper/t2sql_materials_paper.pdf`)
- **評価結果CSV** (`evaluation/proposed_result*.csv`, `baseline_result.csv`)
- **分析用注釈付CSV** (`evaluation/proposed_result_annotated.csv` — `n_tables`/`ntables_difficulty`/`original_difficulty`列付き。代表ランCSVはrun2とバイト同一を維持)
- **Gold SQL** (`evaluation/gold_sql/` 100件)
- **期待結果JSON** (`evaluation/expected_results/` 100件)
- **材料分析CSV** (`evaluation/known_l12_recovery.csv`, `stable_l12_candidates.csv`, `gamma_prime_candidate_ranking.csv`, `ni3al_lattice_matched_candidates.csv`)
- **プロトタイプ分布** (`evaluation/prototype_distribution.csv` — L12=392件等の集計根拠)
- **ソースコード** (`graph/`, `llm/`, `safety/`, `evaluation/metrics.py`)
- **評価スクリプト** (`scripts/run_full_evaluation.py`, `scripts/run_proposed_only.py`, `scripts/compute_paper_figures.py`)
- **安全検査定義** (`safety/allowed_schema.yaml`)
- **論文数値JSON** (`paper/paper_figures.json` — Single Source of Truth)

## ZIP に含まれないもの

以下はリポジトリ内にのみ存在し、ZIPには同梱されていません:

| ファイル/ディレクトリ | 用途 |
|---|---|
| `pyproject.toml` | Python依存パッケージ定義 |
| `docker/` | Docker Compose設定 |
| `db/` | スキーマSQL・データ投入SQL |
| `tests/` | ユニットテスト125件 |
| `api/` | FastAPIアプリケーション |
| `.env.example` | 環境変数テンプレート |
| `experiments/results/` | アブレーション実験結果 |
| `VERIFICATION_GUIDE.md` | 検証手順書 |

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

## 完全再現（リポジトリ必要）

完全な評価再現にはリポジトリのクローンが必要です:

```bash
git clone <repository_url>
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

## JOIN方向バグ修正の影響確認

v4でgraph層のJOIN方向バグ（`_edge_source`による逆方向走査時のカラム入れ替え）を修正したが、
結果CSVは修正前コードで生成されたもの。検証の結果、評価100クエリへの影響はなし:
非対称カラム名のJOINは2件のみ（`material_defect–element`, `application_domain`自己参照）で、
評価クエリでこれらのテーブルに触れるものは0件。残りは全て`entry_id=entry_id`型の対称JOINで結果同一。

## ベースラインCSVに関する注記

`baseline_result.csv` は `condition_mapper`/`entity_extractor` の辞書拡張（elastic_tensor,
thermal_property, magnetic_property対応）前のコードで生成されたもの。
辞書拡張後にB3 (Rule-based) を再ランすると36.9%から変動する可能性がある。
