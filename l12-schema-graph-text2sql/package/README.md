# L12 Text-to-SQL 投稿パッケージ（原稿本文 / Supplementary / MDR登録用スクリプト・データ）

査読者が「原稿を読む」「補足資料を読む」「掲載数値と実装を検証する」を別々に行えるように、
資材を次の3区分に分けている。3区分はいずれも同一のgitコミット（`03_mdr_scripts_data/sql_package/GIT_COMMIT`）から
`scripts/build_submission_package.py` で機械的に組み立てており、全ファイルのSHA-256を `SHA256SUMS.txt` に列挙している。
ビルダーは、同梱ファイルが `GIT_COMMIT` のコミットと一致しない（変更済み・未追跡）場合、PDF が原稿ソース（TeX・bib・cls・bst・図PNG）より古い場合、
凍結リスト `SHA256SUMS_ja.txt` が不一致の場合、静的検証が失敗する場合はパッケージを作らない。
`--allow-dirty` で明示的に許可した場合のみ、コミットと異なるファイルの一覧を `WORKTREE_DIRTY.txt` としてZIP直下に同梱する（本パッケージには存在しない＝作業木はコミットと一致）。

| 区分 | 目的 | 想定する提出先 |
|---|---|---|
| `01_manuscript/` | 原稿本文（LaTeXソース・PDF・図・参考文献・カバーレター草稿） | 投稿誌（本文） |
| `02_supplementary/` | Supplementary Material（補足資料PDFと、補足資料が参照する機械可読の表） | 投稿誌（補足ファイル） |
| `03_mdr_scripts_data/` | 再現・検証用のスクリプトとデータ一式（DDL・fixture・gold SQL・保存済みモデル出力・評価器・SSOT・検証ログ） | NIMS MDR（データリポジトリ）に登録し、Data availability からDOIで参照 |

## 1. 各区分の内容

### 1.1 `01_manuscript/` — 原稿本文

| ファイル | 内容 |
|---|---|
| `stam-m_ja.tex` | 日本語原稿の単一ソース（gitタグ `ja-final` で凍結）。本文（§1–5、謝辞、可用性、貢献）に続けて `\appendix` 以降に補足資料 S1–S10 を含む。 |
| `stam-m_ja.pdf` | 上記をLuaLaTeX＋BibTeXでビルドした完全版PDF（30ページ；本文＋参考文献 p.1–16、補足資料 p.17–30）。 |
| `stam-m_ja_main.pdf` | 完全版PDFから本文＋参考文献ページ（p.1–16）のみを抽出したもの（`pypdf` によるページ抽出。内容は完全版と同一）。 |
| `references.bib`, `interact.cls`, `tfnlm.bst` | 参考文献データベースと Taylor & Francis（STAM）用クラス／bstファイル。 |
| `figures/*.png` | 本文で使う図6点（`\includegraphics` 対象）。`paper_data.json`（SSOT）から `paper_scripts/generate_figures.py` で決定的に再生成できる。 |
| `cover_letter_ja.md` | 投稿カバーレターの日本語草稿（`[ ]` 箇所は投稿前に記入）。 |

補足資料の本文は現在、単一TeXの `\appendix` 以降に置いてある（本文と補足の相互参照 `\ref` を1回のビルドで解決するため）。
投稿時に補足を別PDFとして提出する場合は `02_supplementary/stam-m_ja_supplementary.pdf` を用いる。

### 1.2 `02_supplementary/` — Supplementary Material

| ファイル | 内容 |
|---|---|
| `stam-m_ja_supplementary.pdf` | 完全版PDFの補足資料ページ（p.17–30；S1 ユニットテスト分類、S2 生成SQL例、S3 LLM設定、S4 100クエリ詳細抜粋、S5 条件間差分、S6 独立クエリ詳細、S7 方法の詳細（Algorithm S1・条件抽出器・辞書・SQLGuard 15チェック・OQMDレイアウト対応）、S8 結果の詳細（$\gamma'$候補全10行、格子定数27エントリ、感度分析、失敗内訳、安全性、採点監査）、S9 考察の補足、S10 再現ワークフロー）。 |
| `tables/per_query_results.csv` | アブレーション100クエリのクエリ単位結果（full条件・第1ラン、100行）。旧表S3の全行に相当し、S4 には10行抜粋のみ掲載。 |
| `tables/per_query_by_condition.csv` | 100クエリ×7条件の実行再現率（第1ラン）。旧表S4（条件間差分34件）の全数に相当。 |
| `tables/per_query_by_condition_mean5.csv` | 同、5ラン平均。 |
| `tables/per_query_tables_provenance.json` | 上記3表の生成元（保存run・SHA-256・生成スクリプト）。 |

`tables/` の正本は `03_mdr_scripts_data/sql_package/evaluation/` にあり、ここに置いたものはそのコピーである（SHA-256は `SHA256SUMS.txt` で一致を確認できる）。
`scripts/build_per_query_tables.py` で保存runから決定的に再生成できる。

### 1.3 `03_mdr_scripts_data/` — MDR登録用スクリプト・データ（自己完結）

この区分だけで、論文の全掲載数値・図・統計検定を保存済み評価出力から再生成し、DBを再構築して gold SQL と期待結果を検証できる。
`01_manuscript/` `02_supplementary/` を持たない利用者（MDRからダウンロードした人）が単独で使う前提で構成している。

| ディレクトリ | 内容 | 詳細ドキュメント |
|---|---|---|
| `sql_package/db/` | PostgreSQL 15 用 DDL（`001_schema.sql`〜`007_initialization_marker.sql`）、検証RDBのfixture（L1$_2$型化合物1,470行＋OQMD由来の純元素参照エネルギー89元素＝`material_entry` 1,559行）、一般化試験用の転用スキーマ、Materials Project スナップショット | `sql_package/README_SQL.md` |
| `sql_package/docker/` | PostgreSQL 15 コンテナ定義（初期化SQLを自動投入） | 同上 |
| `sql_package/evaluation/` | 評価データセット（主評価245・アブレーション100・独立設計100・CTE・一般化A–D・日英paired・非翻訳英語25）、gold SQL、期待結果、**保存済みモデル出力**（アブレーション5ラン、言語評価各3ラン、独立・CTE・一般化・LLM-only 等）、生成SQL 245本、採点監査、統計検定結果、provenance | `sql_package/MANIFEST.md`（成果物→論文の表・図の対応表） |
| `sql_package/llm/`, `graph/`, `safety/`, `ingestion/` | パイプライン本体（SQL生成・スキーマリンク・辞書・リランカー・プロンプト）、FKグラフとJOIN経路探索、SQLGuard、データ生成器 | `sql_package/README_SQL.md` |
| `sql_package/scripts/` | 評価・再集計・監査・DB構築スクリプト（下記 §3） | 同上 |
| `sql_package/tests/` | ユニットテスト166件（`pytest`；DB依存15件は `FULL_DB_TEST=1` で実行） | 同上 |
| `paper/` | 凍結版原稿ソース（`stam-m_ja.tex`・bib・cls・bst・図6点）、**SSOT `paper_data.json`**（論文の全掲載数値と図の元データ）、凍結基準 `SHA256SUMS_ja.txt` / `ja_numbers.tsv` / `glossary.tsv`、`jp_reranker_vh_results.json` | 本README §2 |
| `paper_scripts/` | `compute_all_figures.py`（保存run・DB→SSOT再生成）、`generate_figures.py`（SSOT→図）、`verify_ssot.py`（SSOTと成果物・実装の整合）、`verify_paper_numbers.py`（TeXの全数値がSSOTに由来するかの監査）、`requirements-paper.txt` | 本README §2 |
| `verification_logs/` | 配布前に実施したDB込み完全検証・数値監査・ビルドのログ（v13〜v24） | — |

`paper/` に原稿ソースを重複して置く理由は、`verify_paper_numbers.py` の監査対象（原稿中の数値がSSOTに由来するか）が原稿そのものだからである。
`01_manuscript/` の同名ファイルとバイト同一であることは `SHA256SUMS.txt` で確認できる（PDFは `03_mdr_scripts_data/` には含めない）。

## 2. 査読者向け検証手順

必要なもの: Python 3.11以上（検証済み 3.12）。DB込み検証には Docker（または PostgreSQL 15）。図の再生成には matplotlib。
LuaLaTeX は PDF を再ビルドする場合のみ。**LLM API キーは以下のどの手順にも不要**（§3 の再推論を除く）。

### 2.1 レベル0: 数値の出所を確認する（DB不要・依存なし）

```bash
cd 03_mdr_scripts_data
python3 paper_scripts/verify_paper_numbers.py        # 既定: paper/*.tex を監査 → "TeX files audited: 1 (stam-m_ja.tex)"、gating 0、exit 0
python3 paper_scripts/verify_ssot.py                 # SSOT と evaluation/ 成果物・実装（SQLGuard チェック数等）の整合 5/5
(cd paper && sha256sum -c SHA256SUMS_ja.txt)         # 凍結13ファイル（ja-final）の一致
```

`verify_paper_numbers.py` は原稿中の全数値トークンを `paper/paper_data.json` と突き合わせ、p値など監査対象の数値がSSOTに無ければ exit 1 とする。
`paper/ja_numbers.tsv` はその全件レポート（tex_in_json / tex_not_in_json / json_missing_in_tex）である。

### 2.2 レベル1: 保存済み評価出力から掲載数値・統計・図を再生成する（DB不要）

```bash
cd 03_mdr_scripts_data/sql_package
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-repro.txt
python scripts/verify_all.py --static-only           # PASS=8: provenance SHA-256、生成SQLの静的検査、成果物整合
python -m pytest -q                                  # 151 passed（DB依存15件は skip；FULL_DB_TEST=1 で166）
python scripts/build_per_query_tables.py             # → evaluation/per_query_results.csv ほか（02_supplementary/tables の正本）
python scripts/recompute_significance.py --apply     # アブレーション p 値（符号置換検定・Holm）を保存5ランから再計算
python scripts/compute_language_stats.py             # 日英 paired の符号置換検定・ブートストラップCI
python scripts/summarize_language_eval.py            # 言語評価の集計
python scripts/derive_main_artifacts.py              # 主評価245の再採点派生物（86.1% / exact 10.6%）
python scripts/audit_scoring.py                      # 採点方式の監査（表 tab:scoring_audit）
python scripts/build_query_catalog.py                # canonical 300 クエリカタログ
python scripts/compute_unified_difficulty.py         # 非翻訳英語25問の post-hoc 難易度
```

いずれも決定的で、再実行後に変わるのは各JSONの provenance（`generated_at` / `git_commit`）のみである。
`paper_data.json`（SSOT）自体の再生成は DB を要する（2.3）。図はSSOTから DB なしで再生成できる:

```bash
cd 03_mdr_scripts_data
pip install -r paper_scripts/requirements-paper.txt
python3 paper_scripts/generate_figures.py            # → paper/figures/*.png（同梱PNGとバイト同一になる）
```

### 2.3 レベル2: DB を再構築して gold SQL・期待結果・SSOT を検証する（Docker）

```bash
cd 03_mdr_scripts_data/sql_package
(cd docker && POSTGRES_PASSWORD=l12_password GIT_COMMIT=$(cat ../GIT_COMMIT) docker compose up -d)
export POSTGRES_HOST=127.0.0.1 POSTGRES_PASSWORD=l12_password
python scripts/build_transfer_db.py && python scripts/build_obfuscated_transfer_db.py && python scripts/build_mp_transfer_db.py
export L12_DSN=postgresql://l12_user:l12_password@127.0.0.1:5432/l12_materials
export TRANSFER_DSN=postgresql://l12_user:l12_password@127.0.0.1:5432/oqmd_transfer
export OBF_TRANSFER_DSN=postgresql://l12_user:l12_password@127.0.0.1:5432/oqmd_transfer_obfuscated
export MP_DSN=postgresql://l12_user:l12_password@127.0.0.1:5432/mp_transfer
python scripts/verify_all.py --warnings-as-errors    # PASS=18 WARN=0 FAIL=0: gold SQL 全件実行・期待結果一致・順序契約・語彙監査・整合検査
FULL_DB_TEST=1 python -m pytest -q                   # 166 passed
cd .. && FULL_DB_TEST=1 PYTHONPATH=sql_package python3 paper_scripts/compute_all_figures.py   # → paper/paper_data.json 再生成
```

再生成した `paper_data.json` は `_meta.generated_at` / `_meta.git_commit` 以外が同梱版と一致し、その後 `generate_figures.py` で図6点がバイト同一に再生成される。
Windows（PowerShell）での同等手順は `FINAL_CHECKLIST.md` にある。

### 2.4 レベル3: LLM 評価を再実行する（要 OpenAI API キー・有料）

`scripts/eval_*.py`（アブレーション `eval_ablation_multirun.py`、言語 `eval_language_paired.py` / `eval_independent_en.py`、独立 `eval_independent.py`、CTE、一般化、LLM-only、多軸）は
`OPENAI_API_KEY` と 2.3 のDBを要し、非決定的である。論文の掲載数値は**保存済み出力**（`evaluation/*_run*.json` 等）から算出したものであり、
再実行値はモデルの確率的挙動により変動する。本パッケージの配布前検証では有料の全面再評価は行っておらず、日英1問ずつの疎通確認のみ実施した（`verification_logs/`）。

## 3. データの由来に関する注意

- 検証RDBのうち、生成エンタルピー計算の参照状態に使う**純元素の参照エネルギー（89元素）は OQMD の DFT 計算値**をそのまま用いている。
- それ以外の化合物データ（L1$_2$/B2/NaCl/NiAs/BiF$_3$ 型の全エネルギー・生成エネルギー・格子定数・弾性率、キュリー温度、粒界エネルギー、合成方法、文献参照など）は
  OQMD のデータ形式と値域を参考に `sql_package/ingestion/` の生成器で作成した**合成検証データ**であり、実在化合物の計算結果ではない。
  材料工学評価（§3.7・S8.1–8.2）の候補ランキングは、SQL生成の正否を判定するための題材であり、実材料の設計提案ではない。
- 一般化試験Dの Materials Project スナップショット（`db/mp_transfer_snapshot.json.gz`）は取得時点・IDの変換規則を `README_SQL.md` に記す。

## 4. 版履歴（v24）

- v24d: 外部レビュー6件（`no_dict`／`no_graph`／最短JOIN経路／n-best／SQLGuard／共通列Recall）に対する、原稿と現行実装の齟齬の解消。実装・評価値・SSOT・図PNG・保存済み出力は v24c と同一で、変更は `stam-m_ja.tex` の記述のみ（LLM の再実行なし）。PDF は 29→30 ページ（本文＋参考文献 p.1–16、補足 p.17–30；本文の説明追加による）。
  (1) `no_dict`：材料ドメイン辞書全体ではなく、スキーマ写像辞書（条件→テーブル・カラム対応、61エントリ）のみを無効化し、用語辞書 `material_terms.yaml` と条件抽出器は有効、と定義を限定。辞書感度評価の 0% も同一条件（50/25/10% は両辞書を同時削減）であることを本文・補足・表の行ラベルに明記。両辞書を完全に除去した条件は未測定と制限事項に追記。
  (2) `no_graph`：`table_graph` だけでなく `join_list`・`all_columns` も渡さないため、SQL生成器の固定フォールバック文脈（6テーブル・25カラム・5 JOIN条件）に置き換わる複合条件であることを明記し、FKグラフ単独の効果とは解釈しないと記述。
  (3) 最短JOIN経路（方針変更）：旧設計では貪欲最短経路の ON 条件を生成制約としていたが、現行のパイプライン統合以降は最短経路を「必要テーブルの連結可能性の確認」にのみ用い、プロンプトには必要テーブルに接する FK エッジ全体（両方向）を許容JOIN条件として渡す（例：4テーブルで38本、最短経路は3本）。節題を「FKグラフによるJOIN制約」に変更し、図1キャプション・補足 Algorithm S1 の役割記述・SchemaGraphSQL との比較を整合。
  (4) n-best：GPT-5.5 を n 回（既定3）独立に呼び出して候補を得る（1回の呼び出しで複数候補を返させるのではない）、ルールベース候補の追加、機械的スコア 0.6：LLM 意味スコア 0.4 の重み付き和による選択、`no_nbest`（n=1）では候補採点・ルールベース候補・リランカー選択を含む多候補経路全体を省略、と記述を実装に合わせた。
  (5) SQLGuard：検査結果を棄却／警告／書き換えの3区分に分け、15項目のうち 12 が棄却、非FK JOIN と算術型検査の 2 が警告（実行は妨げない）、LIMIT の自動追加・上限化の 1 が書き換え、と本文・補足リストの各項目に区分を付記。「いずれかの検査に通らないSQLは棄却」「SELECT * 禁止」等の現行実装にない記述を削除。
  (6) 共通列投影：Recall だけでなく Precision・F1・緩い完全一致（common-column exact）も gold と生成結果の共通列への投影で定義されるため、必要な回答列の欠落を検出しないことを明記し、欠落を検出する指標（厳密な完全結果集合一致・require-all 列）への参照を付けた。
  あわせて `no_reranker` が SQL 候補の GPT-5.5 リランキングとスキーマリンクのテーブル並べ替えの双方を無効化する（少数ショット例選択のクロスエンコーダは有効）ことを明記。
  `ja-final` を再移動し `SHA256SUMS_ja.txt` / `ja_numbers.tsv` を更新（詳細は `verification_logs/v24_three_part_package_ja_revision.log` §12）。
- v24c: 著者レビューによる原稿の記述整理と図1の描き直し（数値・SSOT・図PNG・保存済み出力は v24b と同一、29ページ・本文 p.1–15 / 補足 p.16–29 も同一）。
  (1) 序論の SuperCon / PoLyInfo（RDF/SPARQL）の記述を「RDB変換が必要」という差分から「用語と項目間関係を明示してから問い合わせ可能にする同じ課題設定（知識をデータ側に置くか問い合わせ生成側に置くか）」へ改め、§4 関連比較の本文・表（SPARQL は問い合わせ言語で NL 変換は対象外と脚注）と今後の課題 (8) を整合。
  (2) 「無機材料RDBの標準評価ベンチマークは存在しない」を、物性予測ベンチマーク LLM4Mat-Bench と知識グラフ／API プロトタイプ（Zimmermann et al. 2025, `references.bib` +1）を引いたうえで「共通の質問文・正解SQL・期待結果で評価するベンチマークはない」に限定。
  (3) 序論の「貢献4項目」リストを削除（内容は §2・§3・§4・§5 と重複）し、目的の1文に集約。
  (4) 制限事項「あいまいな表現」を、閾値がユーザーに依存するため統一評価が困難で本評価では事前に閾値を固定した、本来はユーザーが判断・指定できる設計とすべきで本研究はそれを提供していない、と明記（今後の課題 (7)）。
  (5) CTE の説明を §2.1.1 の初出1回に集約し、以後は「多段計算クエリ」と呼ぶ（本文・補足の CTE 出現 39 → 9、残りは SQLGuard の構文名など技術的に必要な箇所）。§2.2 冒頭の難易度定義の重複1文を削除、表2（アブレーション）のキャプションを短縮して検定・CI の定義を本文へ移動。
  (6) 図1を DDL（`db/001_schema.sql`）と一致する FK 参照グラフに描き直し（36 テーブル中 30 を表示、省略 6 テーブルをキャプションに列挙、色凡例、実線＝直接FK／破線＝`calculation` 経由、識別子の途中改行なし、多重度は示さないと明記）。旧図にあった `band_structure` 等から `material_entry` への直接FK（DDLに存在しない）と「2経路」の記述を削除。
  (7) 補足の図 S3・S4 を1段に、S5 を別段に配置。S10 から配布予定と一致しないファイル名の記載を削除。
  `ja-final` を再移動し `SHA256SUMS_ja.txt` / `ja_numbers.tsv` を更新（詳細は `verification_logs/v24_three_part_package_ja_revision.log` §11）。
- v24b: 著者レビューによる原稿2箇所の修正（数値・SSOT・図・保存済み出力・ページ割りは v24 と同一）。
  (1) Abstract に「86.1\% は期待される行の回収率であり、SQLの完全一致率ではない」の注記を戻し、一般化試験（スキーマの異なる3種のRDB）と独立設計クエリ（実装に関与しない共同著者による100問）を数値なしで具体化。
  (2) $E_\text{hull}$ の平易な説明を「その組成における最安定状態（他の相への分解を含む）のエネルギーからの差」に修正（凸包の定義に合わせた）。
  `ja-final` を再移動し `SHA256SUMS_ja.txt` / `ja_numbers.tsv` を更新（詳細は `verification_logs/v24_three_part_package_ja_revision.log` §10）。
- v24: 日本語原稿の改稿（Abstract 短縮、DFT計算由来データであることの早期明示、平易化）。数値・SSOT・図・保存済み出力は v23e と同一。
  PDF 30→29ページ。関連研究として LLM4Mat-Bench（Rubungo et al., MLST 2025）を序論と LLM-only ベースライン節に各1文で追加（`references.bib` +1）。
  `ja-final` を `119230bb` へ移動し `SHA256SUMS_ja.txt` / `ja_numbers.tsv` を更新。
  パッケージを本README の3区分へ再構成し、`scripts/build_submission_package.py` で機械的に組み立てるようにした（v23e までの `paper/`・`paper_scripts/`・`sql_package/`・`verification_logs/` は `03_mdr_scripts_data/` 直下にそのまま入る）。
- v23〜v23e の変更履歴（原稿削減 38→30ページ、アブレーション p 値の符号置換検定への統一、`verify_all.py` provenance 検査の強化、旧英語版の削除）は
  `03_mdr_scripts_data/verification_logs/v23_ja_trim_signperm_unification.log` を参照。

## 5. 既知の未完了事項

- 英語版原稿は未作成。凍結版 `stam-m_ja.tex` を複製して訳出する（手順は `FINAL_CHECKLIST.md`）。
  英訳後の合格条件は「`paper_scripts/verify_paper_numbers.py` の既定実行が exit 0 **かつ** `TeX files audited: 2 (stam-m.tex, stam-m_ja.tex)` を出力すること」である
  （現在は英語版が無いため 1 本のみを監査して exit 0 になる。exit 0 だけでは英語版の同期は保証されない）。
- `paper/paper_data.json` の `_meta.git_commit` は SSOT を最後に再生成した時点のコミットであり、`GIT_COMMIT`（パッケージのコミット）と一致しないことがある（データは同一）。
