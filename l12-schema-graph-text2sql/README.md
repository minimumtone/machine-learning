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

これにより `db/001_schema.sql`（33テーブル）→ `db/002_reference_data.sql`（マスタ）→
`db/003_material_data.sql`（材料データ）→ `db/004_views.sql`（ビュー）→
`db/005_roles.sql`（読み取り専用ロール）→ `db/006_integrity_checks.sql`（整合性検査、再実行可能）→
`db/007_initialization_marker.sql`（初期化完了マーカー）が順に自動適用されます。

注意：`db/005_roles.sql` は所有者ロール `l12_user` を名指しで参照するため、`POSTGRES_USER` はデフォルトの `l12_user` から変更しないでください（変更するとロード時に明示的なエラーで停止します）。

注意：`db/006_integrity_checks.sql` が失敗したDB（001〜005のみ適用された途中状態）を検証用DBとして使用しないでください。006 はアサーションのみの再実行可能ファイルで、任意の時点で再検証に使えます。006 の全アサーション通過後に適用される `db/007_initialization_marker.sql` が `schema_initialization_status` テーブルに `version='007'` の行を作成するため、使用前に `SELECT 1 FROM schema_initialization_status WHERE version='007';` で初期化完了を確認できます。このマーカーは「初期化完了マーカー」であり、現在の整合性状態の保証ではありません（初期化後に書き換えれば壊せます）。本DBは初期化完了後は不変（immutable）な検証用フィクスチャとして扱い、初期化完了後のエンティティデータのINSERT/UPDATE/DELETEはサポートしません。利用は読み取り専用ロール `l12_reader` で行い、migration owner（`l12_user`）以外にwrite権限を与えないでください。また、propertyディクショナリ（`property_definition`）の変更と各propertyテーブルへの書き込みを並行して行うことは想定していません。

3値BOOLEANの扱い：`density_of_states.is_metallic` のみ意図的にNULL可（NULL=金属性未判定）です。gold SQL では「金属」を `is_metallic = TRUE`（判定済みのTRUEのみ）として扱い、NULL（未判定）は「金属」にも「非金属」にも含めない規約に統一しています。他のBOOLEAN列はすべて NOT NULL です（`phase_stability.is_stable` は `energy_above_hull NOT NULL` により生成列も常に2値）。

設計上の意図的な簡略化：`calculation` は (entry, calculation_type, method, functional) ごとに1件のみ保持します。カットオフ・k点メッシュ・擬ポテンシャル・U値などの数値パラメータ軸は本検証用DBでは持たず、汎用の計算アーカイブとしては UNIQUE が強すぎる点を明示しておきます。

エネルギー規約（reference_set）：`phase_stability.formation_energy_per_atom` は「その材料の `reference_set`（`reference_energy_set` マスタへのFK）が定める元素参照状態に対する生成エネルギー」です。純元素側の `pure_element_reference.delta_e` は OQMD の delta_e（生成エネルギー、eV/atom）であり、全DFTエネルギーでも参照エネルギー値そのものでもありません。`formation_enthalpy` ビューの `enthalpy_vs_element_ground_states` は同一 `reference_set` 内で `formation_energy - Σ xᵢ·delta_eᵢ` を計算し、「フィットされた参照状態基準」を「収録純元素基底状態基準」へ付け替えた値です（同一規約内では参照エネルギーが厳密に相殺するため二重補正にはなりません）。異なる `reference_set` 間の混用はビューのJOIN条件（`per.reference_set = ps.reference_set`）と006のset単位被覆検査で構造的に防がれます。なお本フィクスチャで材料（`phase_stability`）が使うエネルギー規約はパッケージ固有の共通規約 `L12-FIXTURE-PBE-v1` の1件のみです。これとは別に、`pure_element_reference` にはテスト専用規約 `L12-FIXTURE-DIVERGENCE-TEST-v1`（全元素の delta_e を +0.05 eV/atom シフトした複製。`fixture_source_reference_set` に未登録のため材料側からは使用不可）を収録しています。これは `reference_set` 条件を欠いたJOINが偶然正しい結果を返さないようにするための発散検出フィクスチャで、`tests/test_db_integrity.py` が実際に差が生じることを検証します。命名の根拠：化合物の生成エネルギーは実データベースからの取り込み値ではなく、本パッケージの生成器（`ingestion/generate_extended_data.py`。既知L1₂化合物はキュレーション値、その他は範囲内乱数）が合成した値であり、`pure_element_reference` に収録した OQMD DFT-PBE の純元素 delta_e（実データ）を元素参照状態として「宣言」したものです。変換式は存在せず（値の出所が合成であるため）、外部DB間のエネルギー補正も行っていません。したがって `OQMD-PBE` / `MP-PBE` などの実DB規約名を名乗ることは誤解を招くため、フィクスチャ固有名を採用しています。`material_entry.source_db`（OQMD / Materials Project / AFLOW）は合成上の出所ラベルにすぎず、エネルギー値の出所を意味しません。許容される (source_db, reference_set) の組は `fixture_source_reference_set` マップに宣言され、006が全ロード行の組がマップに存在することを検査します（マルチ規約データを載せる場合は `reference_energy_set` とマップに行を追加し、同じ機構がset別に機能します）。

数値・単一truth・不変条件（第6次レビュー対応）：(1) 物理量カラムには有限値CHECK（NaN / ±Infinity 拒否）を付与しています（生成エネルギー・E_hull・delta_e・組成分率・転用スキーマの delta_e / hull_distance など）。(2) `phase_diagram_entry.is_on_hull` は `hull_distance <= 0.001` から導出される生成列で、`phase_stability.is_stable` と同一の運用定義の単一truthです。(3) EAV 3表（calculated/measured/element property）の `value` は NOT NULL で、「値が未知」は行の不存在で表現します。(4) `property_definition.value_type` は本フィクスチャで実際に使用する `'float'` のみに限定しています（整数propertyは未使用のため。整数対応を追加する場合は小数値を拒否するtrigger検証が必要です）。(5) `property_definition.canonical_unit` のマスタ側UPDATEは、不整合な子行が存在する場合trigger（`prevent_invalid_canonical_unit_change`）で拒否されます。(6) `reference_energy_set` の規約フィールド（method/functional/source/fit_name）は、そのsetがロード済みエネルギーから参照された後はtrigger（`prevent_referenced_convention_change`）で変更不可です。

実験測定の未知条件の制限：`experimental_measurement` の `UNIQUE NULLS NOT DISTINCT` により、NULL の測定条件（reference/method/温度/圧力）は独立した測定を表しません。同一材料につき「条件未知の測定」は1件しか表現できず、独立した実測値を共存させるには実際の測定条件を記録する必要があります。

転用スキーマ（`db/transfer_schema.sql`）の安定性truth：転用評価DBでは `oqmd_formation_energies.on_hull` は `hull_distance <= 0.001` から導出される生成列であり、両者が矛盾する行は存在できません。転用gold SQLの安定判定は `on_hull = true`（すなわち `hull_distance <= 0.001`、本体スキーマと同一の運用定義）をtruthとします。

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
├── db/                  # スキーマ定義・データ投入（001→006の順に自動適用）
│   ├── 001_schema.sql          # 33テーブルスキーマ（FK/UNIQUE/CHECK制約付き）
│   ├── 002_reference_data.sql  # マスタ・参照データ（元素・プロトタイプ・辞書等）
│   ├── 003_material_data.sql   # 材料エントリデータ（1,470化合物+89純元素）
│   ├── 004_views.sql           # 派生ビュー（formation_enthalpy）
│   ├── 005_roles.sql           # 読み取り専用ロール（l12_reader）
│   ├── 006_integrity_checks.sql # ロード後整合性検査（組成合計=1等、再実行可能）
│   ├── 007_initialization_marker.sql # 初期化完了マーカー
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
│   ├── gold_sql/        # 正解SQL 264件（本評価244件 + 転用20件）
│   ├── expected_results/ # 正解実行結果JSON（264件）
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
- **ハイブリッドReranker**: 性能重視の3箇所再ランキング — SQL候補選択（GPT-5.5 LLM）、Few-shot例取得（Cross-Encoder ms-marco-MiniLM, ローカル<50ms）、Schema linkingテーブル並び替え（GPT-5.5 LLM）。84クエリA/Bテストで+4.9pp改善（81.5%→86.4%）
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
| RERANK_MODEL | Reranker用LLMモデル | gpt-5.5 |
| SQL_ROW_LIMIT | SQL結果の最大行数 | 100 |
| SQL_TIMEOUT_SECONDS | SQL実行タイムアウト | 10 |
