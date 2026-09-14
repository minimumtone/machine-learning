# 02_supplementary — Supplementary Material

- `stam-m_ja_supplementary.pdf`: 完全版 PDF（`../01_manuscript/stam-m_ja.pdf`）の補足資料ページ（p.17–30）。
  S1 ユニットテスト分類 / S2 生成SQL例 / S3 LLM設定 / S4 100クエリ詳細（10行抜粋） / S5 条件間差分 / S6 独立クエリ詳細 /
  S7 方法の詳細（Algorithm S1、条件抽出器、MeCab辞書、SQLGuard 15チェック、OQMDレイアウト対応） /
  S8 結果の詳細（$\gamma'$候補全10行、格子定数27エントリ、感度分析、失敗内訳、安全性、採点監査） / S9 考察の補足 / S10 再現ワークフロー。
  補足資料のソースは `../01_manuscript/stam-m_ja.tex` の `\appendix` 以降にあり、別ファイルには分けていない（本文との `\ref` を1回のビルドで解決するため）。
- `tables/per_query_results.csv`: S4 の全100行（アブレーション100クエリ、full条件・第1ラン；質問文・難易度・実行再現率・exact・レイテンシ・合否）。
- `tables/per_query_by_condition.csv`: S5 の全数（100クエリ×7条件の実行再現率、第1ラン）。
- `tables/per_query_by_condition_mean5.csv`: 同、5ラン平均。
- `tables/per_query_tables_provenance.json`: 3表の入力（保存run 5本のSHA-256）・生成スクリプト・出力SHA-256。

`tables/` は `../03_mdr_scripts_data/sql_package/evaluation/` の同名ファイルのコピーであり、
`python scripts/build_per_query_tables.py`（`../03_mdr_scripts_data/sql_package/` で実行）により保存済み評価出力から決定的に再生成できる。
