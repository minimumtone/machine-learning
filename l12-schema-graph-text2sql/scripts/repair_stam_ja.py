#!/usr/bin/env python3
"""Repair stam-m_ja.tex: insert frontmatter and remove duplicate structural markers."""
import sys

in_path, out_path = sys.argv[1], sys.argv[2]

with open(in_path, "r", encoding="utf-8") as f:
    text = f.read()

frontmatter = r'''
\begin{document}

\title{無機材料リレーショナルデータベースの自然言語アクセス：\\
スキーマグラフ制約型Text-to-SQLによる設計原理と実証}

\author{
\name{Satoshi Minamoto\textsuperscript{a}\thanks{CONTACT Satoshi Minamoto. Email: minamoto.satoshi@nims.go.jp} and Chie Suematsu\textsuperscript{a}}
\affil{\textsuperscript{a}Materials Data Platform, National Institute for Materials Science (NIMS), 1-1 Namiki, Tsukuba 305-0044, Ibaraki, Japan}
}

\maketitle

\begin{abstract}
無機材料データベースはSQLの知識を前提としており、材料研究者にとってデータ活用の障壁となっている。
本研究では、無機材料RDBへの自然言語アクセスを実現するText-to-SQLパイプラインの設計原理を提示し、その有効性を実証する。
提案手法は、few-shot例検索、LLMによるSQL生成、外部キーグラフに基づくSteiner木JOIN制約、材料ドメイン辞書、AST解析ベースのSQL安全性検証、LLMリランカーの6要素から構成される。

L1$_2$型金属間化合物を中心とする31テーブルのRDBに対し、100件の評価クエリで7条件のablation study（5独立ラン、計3{,}500回評価）を実施した。
few-shot例と材料ドメイン辞書がそれぞれ$-$7.4\,pp（$p<0.001$）、$-$7.3\,pp（$p<0.001$）と統計的に有意な寄与を示した。
多軸評価では実行再現率84.7\%、適合率77.3\%、F1値74.8\%、構文妥当率100\%、JOIN一致率93.2\%を達成した。
全評価は5独立ランの平均$\pm$標準偏差で報告し、Wilcoxon符号順位検定による有意性を確認した。
さらに、プロトタイプ拡張（A）90.0\%、OQMD風命名変更（B）90.5\%、ランダム化名前転用（C）90.0\%、実在するMaterials Project風スキーマ（D）100.0\%の4種類のゼロ適応転用試験に成功し、パイプラインがスキーマ非依存に汎化することを実証した。
独立設計100件の外部検証（76.9\%）とCTE多段計算15パターンの評価により、設計原理の外的妥当性と適用限界を定量化した。
\end{abstract}

\begin{keywords}
Text-to-SQL \; 無機材料データベース \; 自然言語インターフェース \; スキーマグラフ \; ドメイン特化辞書 \; 材料インフォマティクス
\end{keywords}

% =====================================================================
'''

# Locate the end of lstset block
lines = text.splitlines(keepends=False)
# Find the line that is the closing '}' of \lstset, then the following "% ===” line, then insert frontmatter.
insert_idx = None
for i, line in enumerate(lines):
    if line.strip() == '}' and i > 0 and 'lstset' in lines[i-1]:
        # look for % ==== line after it
        for j in range(i+1, min(i+5, len(lines))):
            if lines[j].startswith('% ====') or lines[j].startswith('% ----'):
                insert_idx = j + 1
                break
        break

if insert_idx is None:
    # fallback: find first \section{はじめに} duplicate
    for i, line in enumerate(lines):
        if line.startswith('\\section{') and i + 1 < len(lines) and lines[i+1] == line:
            insert_idx = i
            break

# Remove duplicate structural markers (section, appendix, bibliographystyle, end document)
clean = []
skip_prefixes = ('\\section{', '\\appendix', '\\bibliographystyle{', '\\end{document}')
for line in lines:
    # if line is empty or doesn't match any of these skip patterns, keep normally
    to_skip = any(line == last and (line.startswith('\\section{') or line == '\\appendix' or line.startswith('\\bibliographystyle{') or line == '\\end{document}') for last in clean)
    if to_skip:
        continue
    clean.append(line)

# Insert frontmatter at insert_idx
if insert_idx is not None:
    # Adjust index after clean (same or earlier). Find the position of the original marker in clean.
    target = None
    for i, line in enumerate(clean):
        if line.startswith('\\section{') and i + 1 < len(clean) and clean[i+1] == line:
            target = i
            break
    if target is not None:
        # remove the duplicate line (second one)
        if clean[target] == clean[target+1]:
            clean.pop(target+1)
        # insert before the single section
        clean = clean[:target] + [frontmatter.rstrip()] + clean[target:]
    else:
        clean = clean[:insert_idx] + [frontmatter.rstrip()] + clean[insert_idx:]

with open(out_path, "w", encoding="utf-8") as f:
    f.write('\n'.join(clean) + '\n')

print(f"Repaired {out_path}")
