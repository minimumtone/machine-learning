# 01_manuscript — 原稿本文

- `stam-m_ja.tex`: 日本語原稿の単一ソース（gitタグ `ja-final` で凍結。SHA-256 は `../03_mdr_scripts_data/paper/SHA256SUMS_ja.txt`）。本文の後、`\appendix` 以降に補足資料 S1–S10 を含む。
- `stam-m_ja.pdf`: LuaLaTeX＋BibTeX による完全版（本文＋参考文献 p.1–15、補足資料 p.16–29）。
- `stam-m_ja_main.pdf`: 完全版から本文＋参考文献ページのみを抽出したもの。
- `references.bib`, `interact.cls`, `tfnlm.bst`: 参考文献と STAM（Taylor & Francis）用クラス／bst。
- `figures/`: 本文の図6点（PNG）。`../03_mdr_scripts_data/paper_scripts/generate_figures.py` で SSOT から再生成できる。
- `cover_letter_ja.md`: カバーレターの日本語草稿（`[ ]` は投稿前に記入）。

再ビルド: `lualatex stam-m_ja && bibtex stam-m_ja && lualatex stam-m_ja && lualatex stam-m_ja`（日本語フォントは LuaTeX-ja が解決する）。

原稿中の数値がすべて SSOT（`../03_mdr_scripts_data/paper/paper_data.json`）に由来することは
`python3 ../03_mdr_scripts_data/paper_scripts/verify_paper_numbers.py` で確認できる（ルート `README.md` §2.1）。
