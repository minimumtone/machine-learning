#!/usr/bin/env python3
"""Translate Japanese text in main_en.tex to English."""

TRANSLATIONS = {
    # Section headers
    r"\section{緒言}": r"\section{Introduction}",
    r"\section{提案手法}": r"\section{Proposed Method}",
    r"\section{データベースと評価設計}": r"\section{Database and Evaluation Design}",
    r"\section{Ablation Study}": r"\section{Ablation Study}",
    r"\section{感度分析}": r"\section{Sensitivity Analysis}",
    r"\section{多軸評価指標}": r"\section{Multi-axis Evaluation Metrics}",
    r"\section{エラー分析}": r"\section{Error Analysis}",
    r"\section{SQLインジェクションと安全性}": r"\section{SQL Injection and Safety}",
    r"\section{材料工学的評価}": r"\section{Materials Engineering Evaluation}",
    r"\section{考察}": r"\section{Discussion}",
    r"\section{結論}": r"\section{Conclusion}",
    # Subsection headers
    r"\subsection{無機材料研究におけるデータ探索の障壁}": r"\subsection{Barriers to Data Exploration in Inorganic Materials Research}",
    r"\subsection{自然言語データベースインターフェースの意義}": r"\subsection{Significance of Natural Language Database Interfaces}",
    r"\subsection{Text-to-SQL技術と材料分野での現状}": r"\subsection{Text-to-SQL Technology and Current State in Materials Science}",
    r"\subsection{本研究の目的と貢献}": r"\subsection{Objectives and Contributions}",
    r"\subsection{パイプライン概要}": r"\subsection{Pipeline Overview}",
    r"\subsection{Few-shot例検索}": r"\subsection{Few-shot Example Retrieval}",
    r"\subsection{スキーマリンキング}": r"\subsection{Schema Linking}",
    r"\subsection{SQL生成と$n$-best候補}": r"\subsection{SQL Generation and $n$-best Candidates}",
    r"\subsection{Steiner木JOIN制約}": r"\subsection{Steiner Tree JOIN Constraint}",
    r"\subsection{条件抽出器}": r"\subsection{Condition Extractor}",
    r"\subsection{材料ドメイン辞書}": r"\subsection{Materials Domain Dictionary}",
    r"\subsection{SQLGuard}": r"\subsection{SQLGuard}",
    r"\subsection{ハイブリッドリランカー}": r"\subsection{Hybrid Reranker}",
    r"\subsection{リペアループ}": r"\subsection{Repair Loop}",
    r"\subsection{検証データベース}": r"\subsection{Validation Database}",
    r"\subsection{評価クエリ}": r"\subsection{Evaluation Queries}",
    r"\subsection{評価指標}": r"\subsection{Evaluation Metrics}",
    r"\subsection{ablation条件別エラーパターン}": r"\subsection{Error Patterns by Ablation Condition}",
    r"\subsection{Very Hard失敗の内訳分析}": r"\subsection{Breakdown Analysis of Very Hard Failures}",
    r"\subsection{既知L1$_2$化合物の再発見}": r"\subsection{Rediscovery of Known L1$_2$ Compounds}",
    r"\subsection{$\gamma'$相候補ランキング}": r"\subsection{$\gamma'$ Phase Candidate Ranking}",
    r"\subsection{Ni$_3$Al近傍格子定数候補}": r"\subsection{Ni$_3$Al Neighborhood Lattice Constant Candidates}",
    r"\subsection{安定L1$_2$候補の抽出}": r"\subsection{Extraction of Stable L1$_2$ Candidates}",
    r"\subsection{材料設計仮説の生成}": r"\subsection{Generation of Materials Design Hypotheses}",
    r"\subsection{無機材料RDBにおけるスキーマ障壁と自然言語化の意義}": r"\subsection{Schema Barriers in Inorganic Materials RDBs and the Significance of NL Access}",
    r"\subsection{Few-shot例の寄与}": r"\subsection{Contribution of Few-shot Examples}",
    r"\subsection{材料ドメイン辞書の寄与}": r"\subsection{Contribution of the Materials Domain Dictionary}",
    r"\subsection{リランカーの寄与}": r"\subsection{Contribution of the Reranker}",
    r"\subsection{SQLGuard・$n$-best・Steiner木の寄与}": r"\subsection{Contribution of SQLGuard, $n$-best, and Steiner Tree}",
    r"\subsection{関連システムとの比較}": r"\subsection{Comparison with Related Systems}",
    r"\subsection{LLM-onlyおよびスキーマ提示のみとの定量比較}": r"\subsection{Quantitative Comparison with LLM-only and Schema-only Approaches}",
    r"\subsection{多段計算クエリへの対応}": r"\subsection{Handling Multi-step Calculation Queries}",
    r"\subsection{独立設計クエリによる評価（難易度調和比較）}": r"\subsection{Evaluation with Independently Designed Queries (Difficulty-Harmonized)}",
    r"\subsection{カラム間比較クエリへの対応}": r"\subsection{Handling Cross-column Comparison Queries}",
    r"\subsection{RDBスキーマ設計とText-to-SQLコスト}": r"\subsection{RDB Schema Design and Text-to-SQL Cost}",
    r"\subsection{CTE多段計算クエリの評価}": r"\subsection{Evaluation of CTE Multi-step Calculation Queries}",
    r"\subsection{同種無機材料RDBへの展開可能性}": r"\subsection{Extensibility to Similar Inorganic Materials RDBs}",
    r"\subsection{制約と今後の課題}": r"\subsection{Limitations and Future Work}",
    r"\subsection{今後の展望}": r"\subsection{Future Prospects}",
    r"\subsection{Few-shot例数の影響}": r"\subsection{Effect of the Number of Few-shot Examples}",
    r"\subsection{ドメイン辞書サイズの影響}": r"\subsection{Effect of Domain Dictionary Size}",
    r"\subsection{LLMモデルの影響}": r"\subsection{Effect of LLM Model}",
    # Appendix sections
    r"\section{ユニットテストカテゴリ詳細}": r"\section{Unit Test Category Details}",
    r"\section{回帰テストケース一覧}": r"\section{Regression Test Cases}",
    r"\section{生成SQL例}": r"\section{Generated SQL Examples}",
    r"\section{SQLインジェクション・安全性テスト詳細}": r"\section{SQL Injection and Safety Test Details}",
    r"\section{LLMモード設定}": r"\section{LLM Mode Configuration}",
    r"\section{評価クエリ100件の詳細結果}": r"\section{Detailed Results for 100 Evaluation Queries}",
    r"\section{Ablation条件間の精度差異クエリ}": r"\section{Accuracy Difference Queries Between Ablation Conditions}",
    r"\section{独立設計クエリプールの詳細結果}": r"\section{Detailed Results for Independent Query Pool}",
    # Subsection labels in appendix
    r"\subsection{「Niを含む安定なL1$_2$化合物を出して」}": r'\subsection{``List stable L1$_2$ compounds containing Ni\'\'}',
    r"\subsection{「NiとAlの両方を含むL1$_2$化合物を出して」}": r'\subsection{``List L1$_2$ compounds containing both Ni and Al\'\'}',
}

# Table header translations (common patterns)
TABLE_HEADER_TRANSLATIONS = {
    "条件": "Condition",
    "全体": "Overall",
    "有意性": "Significance",
    "項目": "Item",
    "件数": "Count",
    "指標": "Metric",
    "デフォルト": "Default",
    "カスタム辞書": "Custom Dict.",
    "単一トークン化率": "Single-token rate",
    "改善語数": "Improved terms",
    "悪化語数": "Degraded terms",
    "パターン種別": "Pattern type",
    "入力例": "Input example",
    "出力": "Output",
    "記号比較": "Symbol comparison",
    "日本語後置": "Japanese postfix",
    "日本語より": "Japanese comparative",
    "符号判定": "Sign determination",
    "範囲指定": "Range specification",
    "形成エネルギーが0.5以上": r"formation energy $\geq$ 0.5",
    "格子定数が3.0より大きい": r"lattice const. > 3.0",
    "形成エネルギーが負": "negative formation energy",
    "難易度": "Difficulty",
    "テーブル数": "Tables",
    "代表的内容": "Representative Content",
    "単一条件検索": "Single-condition search",
    "構造+組成+安定性横断": "Structure+composition+stability",
    "複合条件・集約": "Compound conditions/aggregation",
    "材料設計クエリ": "Materials design queries",
    "レイテンシ (s)": "Latency (s)",
    "失敗モード": "Failure mode",
    "代表例": "Representative case",
    "WHERE条件組合せミス": "WHERE condition combination error",
    "集約・サブクエリ誤用": "Aggregation/subquery misuse",
    "多元素AND失敗": "Multi-element AND failure",
    "型不一致による意味エラー": "Semantic error from type mismatch",
    "複合WHERE（3条件以上）": "Compound WHERE (3+ conditions)",
    "SELF-JOIN展開": "SELF-JOIN expansion",
    "リペアで実行成功、結果不一致": "Execution ok via repair, result mismatch",
    "辞書サイズ": "Dict. size",
    "モデル": "Model",
}

def main():
    """Apply Japanese→English string replacements to paper/main_en.tex."""
    with open("paper/main_en.tex", "r") as f:
        content = f.read()

    # Apply section/subsection translations
    for jp, en in TRANSLATIONS.items():
        content = content.replace(jp, en)

    # Apply table header translations (longest-first to avoid partial matches)
    for jp, en in sorted(TABLE_HEADER_TRANSLATIONS.items(), key=lambda kv: len(kv[0]), reverse=True):
        content = content.replace(jp, en)

    with open("paper/main_en.tex", "w") as f:
        f.write(content)

    print("Done. Applied translations.")

if __name__ == "__main__":
    main()
