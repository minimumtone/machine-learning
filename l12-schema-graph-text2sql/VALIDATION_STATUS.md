# Validation / Audit Status (2026-06-02)

## 1. データ生成経緯の確認（致命的 A-2 対応）

- `ingestion/generate_extended_data.py` により 1,470 化合物エントリが生成されている。
- 11 種の既知 L1$_2$ 化合物はハードコードされた参照値を使用。
- 純元素データ 89 元素は `scripts/fetch_oqmd_pure_elements.py` で OQMD API から取得（`pure_element_reference`）。
- それ以外の格子定数、形成エネルギー、機械的・磁気的・熱的・合成・欠陥・粒界エネルギー等はすべて合成データ。
- この旨を `npj-compt.mater.tex` §3.1、§9、Conclusion、Acknowledgments で明示した。
- §9 「材料設計仮説」5 項目は削除し、「合成データ上の機能デモ」として再定義した。

## 2. 数値 SSoT の再生成

- `scripts/compute_all_figures.py` を実行し `paper/paper_data.json` を最新化。
- 主要数値（独立評価 76.9%、転用 A--D、CTE 15 件 61.6%）が JSON と一致することを確認。
- 指摘された旧数値（80.5/95.0/97.3/72.5/32.3/-3.9pp など）は本文から除去済み。

## 3. F1 定義の明確化（A-3 対応）

- Table 8 (`tab:multiaxis`) のキャプションに「F1 はクエリ単位の F1 平均であり、ティアー平均 P/R の調和平均ではない」と追記。
- `evaluation/metrics.py` では F1 = 2PR/(P+R) を各クエリで計算後平均している。

## 4. 主指標を recall だけにしない対応（B 対応）

- Abstract に execution recall 84.7%、precision 77.3%、F1 74.8% を明記。
- 表 8 には recall/precision/F1 を同掲。

## 5. 残っている既知の齟齬・未解決項目

### 5.1 テーブル間の数値不一致
- `tab:known_l12` の Ni$_3$Ga (3.66 / 0.010)、Ni$_3$Ge (3.61 / 0.040) と `tab:sup_lattice_match` の Ni$_3$Ge (3.583 / 0.075)、Ni$_3$Ga (3.598 / 0.030) が一致しない。
- 原因：`generate_extended_data.py` が既知化合物の後に同じ組成のランダム生成エントリを追加しているため、同一組成で複数の `lattice_a`/`E_hull` が存在する。
- 対応案：
  1. `generate_extended_data.py` を修正して既知化合物と重複する組成を生成しないようにし DB を再生成（全評価再実行が必要）。
  2. または、両テーブルで重複する組成（Ni$_3$Ga, Ni$_3$Ge）を除外する。

### 5.2 辞書サイズの表記
- 本文全体では 497 語と説明しているが、辞書サイズ感度実験 (Table 10) は 61 エントリをフルサイズとしている。
- これは「感度分析用の 61 語サブセット」と説明が必要。

### 5.3 CTE ablation とメイン ablation の混同
- `tab:cte_results` は 5 件の CTE クエリについての ablation。
- `tab:ablation` は 100 件の一般クエリについての ablation。
- 本文では同一の ablation と扱わないよう、キャプションで区別を明記すべき。

### 5.4 ベースライン比較
- DIN-SQL / DAIL-SQL / SchemaGraphSQL との同一 DB 比較は未実施。
- 実装・実行には追加で評価スクリプトと API 呼び出しが必要。

### 5.5 参考文献
- `[15] MKNA`, `[16] OptiMat`, `[24] pinax`, `[12] SchemaGraphSQL` などの巻・頁・年情報を要確認。

### 5.6 日本語テキスト
- `npj-compt.mater.tex` は LuaLaTeX 下で日本語を含んでいる。npj 投稿時は `luatexja` をコメントアウトし、日本語を除去して pdfLaTeX コンパイルが必要。

### 5.7 スコープ・投稿先
- 合成データ中心かつ材料科学的知見がないため、npj Comput. Mater. は極めて厳しい。
- Digital Discovery (RSC) または Computational Materials Science / Journal of Chemical Information and Modeling への再構成が現実的。

## 6. 検証コマンド結果

- `pytest tests/ -q`: 134 passed
- `ruff check .`: all checks passed
- `python -m pyright`: 0 errors
- `lualatex npj-compt.mater.tex` ×2 pass: 48 ページ、エラーなし
