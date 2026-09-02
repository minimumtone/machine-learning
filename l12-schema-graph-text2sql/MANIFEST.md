# 評価成果物マニフェスト（evaluation/ → 論文の表・図）

`evaluation/` 直下の JSON/JSONL 成果物と、日本語原稿（stam-m_ja.tex）で
それらを使用する表・図・節の対応表。表番号ラベルは TeX の `\label{}` 名で示す。
「入力」はデータセット・監査等の入力資材、「保存run」はLLM評価の保存出力、
「派生」は保存runから決定的に再計算される集約・統計を指す。

## データセット（入力）

| ファイル | 内容 | 使用箇所 |
|---|---|---|
| `main_evaluation_dataset.jsonl` | 主評価245クエリ | 表 tab:query_design・主評価全般 |
| `evaluation_dataset.jsonl` | アブレーション用100クエリ（主評価245の部分集合） | 表 tab:ablation ほか |
| `expert_evaluation_dataset.jsonl` | 独立設計100クエリ（主評価245の部分集合） | §4.3.1・表 tab:sup_expert_summary |
| `evaluation_dataset_en.jsonl` | 日英paired評価用の英訳100クエリ | 表 tab:language_eval |
| `independent_en_dataset.jsonl` | 非翻訳英語25クエリ | 表 tab:language_eval |
| `cte_evaluation_dataset.jsonl` | CTE新規10クエリ | 表 tab:cte_results |
| `cte15_dataset.jsonl` | CTE 15クエリ（新規10＋主評価5） | 表 tab:cte_results |
| `prototype_evaluation_dataset.jsonl` | 転移バリアントA（プロトタイプ拡張）20クエリ | 表 tab:transfer_eval |
| `transfer_evaluation_dataset.jsonl` | 転移バリアントB（未知スキーマ）20クエリ | 表 tab:transfer_eval |
| `transfer_obfuscated_evaluation_dataset.jsonl` | 転移バリアントC（難読化）20クエリ | 表 tab:transfer_eval |
| `mp_transfer_evaluation_dataset.jsonl` | 転移バリアントD（Materials Project）15クエリ | 表 tab:transfer_eval |
| `query_catalog.json`（+ `query_catalog.csv`） | canonical 300クエリカタログ | 補足資料のクエリ一覧整合 |

## 保存run（LLM評価出力）

| ファイル | 内容 | 使用箇所 |
|---|---|---|
| `main_eval_with_sql.json` | 主評価245の canonical run（86.1%、生成SQL付き） | 本文主結果・表 tab:multiaxis 系・採点監査の入力 |
| `ablation_run_1.json` … `ablation_run_5.json` | アブレーション7条件×100クエリ×5ラン | 表 tab:ablation（5ラン平均）・図 ablation_bar |
| `ablation_results.json` | 単一ランのアブレーション成果物（クエリ単位行） | CTEサブセット集計・エラー分析の入力（表1の値そのものではない。ファイル内 `_note` 参照） |
| `independent_eval_results.json` | 独立設計100クエリの別評価run（81.5%） | §4.3.1 |
| `language_paired_ja_run1..3.json` / `language_paired_en_run1..3.json` | 日英paired 100クエリ×各3ラン | 表 tab:language_eval |
| `independent_en_run1..3.json` | 非翻訳英語25クエリ×3ラン（96.5±2.3%） | 表 tab:language_eval |
| `llm_only_results.json` | LLM-onlyベースライン100クエリ | 表 tab:llm_only |
| `cte_eval_results.json` | CTE 15クエリ評価（72.7%） | 表 tab:cte_results |
| `prototype_eval_results.json` | 転移A評価 | 表 tab:transfer_eval |
| `transfer_eval_results.json` | 転移B評価 | 表 tab:transfer_eval |
| `transfer_obfuscated_eval_results.json` | 転移C評価 | 表 tab:transfer_eval |
| `mp_transfer_eval_results.json` | 転移D評価 | 表 tab:transfer_eval |
| `reranker_eval_results.json` | リランカーA/B比較20クエリ（92.2% vs 77.7%） | リランカー比較節 |
| `model_comparison_results.json` | モデル間比較 | 表 tab:model_comp・図 model_comparison |
| `multiaxis_results.json` | 多軸指標評価 | 表 tab:multiaxis・図 multiaxis_radar |
| `fewshot_sensitivity_results.json` | few-shot k感度分析 | 表 tab:fewshot_k・図 fewshot_sensitivity |
| `dict_sensitivity_results.json` | 辞書サイズ感度分析 | 表 tab:dict_size・図 dict_sensitivity |

## 派生・統計・監査（決定的に再計算可能）

| ファイル | 内容 | 使用箇所 |
|---|---|---|
| `ablation_multirun_stats.json` | 5ラン平均±SD | 表 tab:ablation |
| `ablation_significance_v2.json` | Wilcoxon（SciPy method='exact'）＋Holm補正p値 | 表 tab:ablation |
| `significance_recomputed.json` | 統計再計算のprovenance付き成果物 | 表 tab:ablation のp値検証 |
| `language_paired_stats.json` | 日英paired統計（符号置換p=0.193・bootstrap CI） | 表 tab:language_eval・§4.3.2 |
| `language_eval_summary.json` | 言語評価の難易度別集計 | 表 tab:language_eval |
| `scoring_audit.json` | 採点方式監査（historical/exact/lenient/ordered/strict） | 表 tab:scoring_audit（strict 25.5%・n=241 は本文言及） |
| `error_analysis_counts.json` | 失敗カテゴリ計数 | 表 tab:error_analysis・図 error_distribution |
| `failure_analysis.json` | 主評価の失敗39件の分析 | エラー分析節 |
| `distinct_audit.json` | DISTINCT使用監査（歴史的監査成果物） | 本文表・図では未使用（履歴） |
| `gold_change_manifest_r22a.json` | R22A gold修正の変更manifest | 本文表・図では未使用（履歴） |

## ディレクトリ

| ディレクトリ | 内容 |
|---|---|
| `gold_sql/` / `expected_results/` | 主評価245（＋転移A 20）のgold SQLと期待結果 |
| `gold_sql_independent_en/` / `expected_results_independent_en/` | 非翻訳英語25クエリのgold SQLと期待結果 |
| `gold_sql_obfuscated/` / `expected_results_obfuscated/` | 転移バリアントC（難読化）のgold SQLと期待結果 |
| `gold_sql_mp/` / `expected_results_mp_transfer/` | 転移バリアントD（Materials Project）のgold SQLと期待結果 |
| `generated_sql/` | 保存runから抽出した生成SQL |

図（`ablation_bar` / `fewshot_sensitivity` / `dict_sensitivity` /
`multiaxis_radar` / `model_comparison` / `error_distribution`）はいずれも
SSOT（`paper_data.json`）のみから `generate_figures.py` で再生成される。
