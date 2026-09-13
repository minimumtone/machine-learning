# 03_mdr_scripts_data — MDR登録用スクリプト・データ（自己完結）

このディレクトリ単独で、論文の全掲載数値・統計検定・図を保存済み評価出力から再生成し、
検証RDBを再構築して gold SQL と期待結果を検証できる。手順はルート `README.md` §2（レベル0〜3）を参照。

| パス | 内容 | ドキュメント |
|---|---|---|
| `sql_package/` | DDL・fixture・Docker定義、評価データセット・gold SQL・期待結果・保存済みモデル出力・生成SQL・監査結果、パイプライン実装、評価/再集計/監査スクリプト、ユニットテスト166件 | `sql_package/README_SQL.md`（環境構築・DB検証手順）、`sql_package/MANIFEST.md`（成果物→論文の表・図の対応） |
| `sql_package/GIT_COMMIT` | 本パッケージを組み立てたリポジトリのコミット | — |
| `paper/` | 凍結版原稿ソース（`stam-m_ja.tex`・bib・cls・bst・図6点、PDFは含まない）、SSOT `paper_data.json`、凍結基準 `SHA256SUMS_ja.txt`・`ja_numbers.tsv`・`glossary.tsv`、`jp_reranker_vh_results.json` | ルート `README.md` §1.3 |
| `paper_scripts/` | `verify_paper_numbers.py`（原稿の数値監査）、`verify_ssot.py`（SSOT整合）、`compute_all_figures.py`（SSOT再生成；DB要）、`generate_figures.py`（図再生成）、`requirements-paper.txt` | ルート `README.md` §2 |
| `verification_logs/` | 配布前検証ログ（DB込み `verify_all` 実行ログ、各版の変更・検証記録） | — |

`paper/` の原稿ソースは `../01_manuscript/` とバイト同一（ルート `SHA256SUMS.txt`）。
`verify_paper_numbers.py` の監査対象が原稿そのものであるため、ここにも同梱している。

LLM API キーが必要なのは `sql_package/scripts/eval_*.py` による再推論のみで、上記の再生成・検証には不要である。
