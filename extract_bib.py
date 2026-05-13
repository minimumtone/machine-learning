#!/usr/bin/env python3
"""
extract_bib.py — PDFから論文メタデータをAIで抽出しBibTeXに保存するツール

機能:
  - PDF からテキストを抽出 (pymupdf4llm)
  - OpenAI API でタイトル・著者・DOI に加え、
    materials / theory / methods / summary_ja を抽出
  - JabRef 互換の .bib ファイルへ出力
    (カスタムフィールド materials, theory, methods + comment に日本語要約)

使い方:
  python extract_bib.py paper1.pdf paper2.pdf ...
  python extract_bib.py --input-dir ./papers/
  python extract_bib.py paper.pdf --output my_library.bib
"""

import argparse
import json
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# PDF → テキスト
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: str, max_chars: int = 80_000) -> str:
    """PDF ファイルからテキストを抽出する。pymupdf4llm > PyMuPDF の順にフォールバック。"""
    try:
        import pymupdf4llm
        text = pymupdf4llm.to_markdown(pdf_path)
    except ImportError:
        import fitz
        doc = fitz.open(pdf_path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()

    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n... (truncated)"
    return text


# ---------------------------------------------------------------------------
# OpenAI メタデータ抽出
# ---------------------------------------------------------------------------

SYSTEM_MSG = (
    "あなたは材料科学・工学の専門家です。論文テキストから以下の情報を抽出し"
    "JSONで返してください:\n"
    "- title: タイトル\n"
    "- authors: 著者（カンマ区切りの文字列）\n"
    "- year: 出版年（整数、不明なら null）\n"
    "- doi: DOI（不明なら null）\n"
    "- journal: ジャーナル名（不明なら null）\n"
    "- materials: 対象物質（例: 'SrTiO3', 'High-entropy alloy'）\n"
    "- theory: 理論・モデル（例: 'Density Functional Theory', 'Dislocation Dynamics'）\n"
    "- methods: 実験・解析手法（例: 'TEM', 'Machine Learning regression'）\n"
    "- summary_ja: 論文の要点を日本語で2〜3文にまとめたもの\n"
    "\n"
    "回答はJSONオブジェクトのみ（マークダウンのコードブロック不要）。"
)


def get_metadata_via_ai(text: str, model: str = "gpt-4o") -> Dict:
    """OpenAI API を使って論文テキストからメタデータを抽出する。"""
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "環境変数 OPENAI_API_KEY が設定されていません。\n"
            "  export OPENAI_API_KEY='sk-...'"
        )

    client = OpenAI(api_key=api_key, timeout=120)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": f"以下の論文テキストを解析してください:\n\n{text}"},
        ],
        temperature=0.0,
        max_tokens=2048,
    )
    raw = response.choices[0].message.content or ""
    return _parse_json(raw)


def _parse_json(text: str) -> Dict:
    """LLM 応答から JSON を取り出す。"""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise ValueError(f"JSON の解析に失敗しました: {text[:300]}...")


# ---------------------------------------------------------------------------
# BibTeX 生成キー
# ---------------------------------------------------------------------------

def _make_cite_key(meta: Dict, pdf_path: str) -> str:
    """著者姓 + 年 のキーを生成する。不明なら PDF ファイル名から生成。"""
    authors = meta.get("authors", "")
    year = meta.get("year")

    first_author = ""
    if authors:
        first = authors.split(",")[0].strip()
        parts = first.split()
        if parts:
            first_author = parts[-1].lower()
            first_author = unicodedata.normalize("NFKD", first_author)
            first_author = re.sub(r"[^a-z]", "", first_author)

    if not first_author:
        first_author = Path(pdf_path).stem.lower()
        first_author = re.sub(r"[^a-z0-9]", "", first_author)[:20]

    year_str = str(year) if year else "nd"
    return f"{first_author}{year_str}"


# ---------------------------------------------------------------------------
# BibTeX 書き出し
# ---------------------------------------------------------------------------

def _escape_bib(value: str) -> str:
    """BibTeX 値のエスケープ。"""
    return value.replace("{", "\\{").replace("}", "\\}")


def _format_bib_entry(key: str, meta: Dict) -> str:
    """1 エントリ分の BibTeX 文字列を組み立てる。"""
    lines = [f"@article{{{key},"]

    field_map = [
        ("title", meta.get("title", "")),
        ("author", meta.get("authors", "")),
        ("year", str(meta.get("year", "")) if meta.get("year") else ""),
        ("journal", meta.get("journal", "") or ""),
        ("doi", meta.get("doi", "") or ""),
        ("materials", meta.get("materials", "") or ""),
        ("theory", meta.get("theory", "") or ""),
        ("methods", meta.get("methods", "") or ""),
        ("comment", meta.get("summary_ja", "") or ""),
    ]

    for fname, fval in field_map:
        if fval:
            lines.append(f"  {fname} = {{{_escape_bib(str(fval))}}},")

    lines.append("}")
    return "\n".join(lines)


def add_to_bib(bib_path: str, key: str, meta: Dict) -> None:
    """既存の .bib ファイルにエントリを追記（重複キー時はスキップ）。"""
    path = Path(bib_path)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""

    if re.search(rf"@\w+\{{\s*{re.escape(key)}\s*,", existing):
        print(f"  [skip] キー '{key}' は既に存在します。")
        return

    entry = _format_bib_entry(key, meta)
    with open(path, "a", encoding="utf-8") as f:
        if existing and not existing.endswith("\n"):
            f.write("\n")
        f.write("\n" + entry + "\n")

    print(f"  [add]  {key} → {bib_path}")


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def process_pdf(pdf_path: str, bib_path: str, model: str) -> Optional[Dict]:
    """PDF 1 ファイルを処理して bib に追加。"""
    print(f"\n--- {pdf_path} ---")

    if not Path(pdf_path).exists():
        print(f"  [error] ファイルが見つかりません: {pdf_path}")
        return None

    print("  テキスト抽出中 ...")
    text = extract_text_from_pdf(pdf_path)
    if len(text.strip()) < 100:
        print("  [warn] テキストが短すぎます。スキャン PDF の可能性があります。")
        return None

    print(f"  AI 解析中 (model={model}) ...")
    meta = get_metadata_via_ai(text, model=model)

    print(f"  title   : {meta.get('title', '???')}")
    print(f"  authors : {meta.get('authors', '???')}")
    print(f"  doi     : {meta.get('doi', '???')}")
    print(f"  materials: {meta.get('materials', '???')}")
    print(f"  theory  : {meta.get('theory', '???')}")
    print(f"  methods : {meta.get('methods', '???')}")
    print(f"  summary : {(meta.get('summary_ja', '') or '')[:80]}...")

    key = _make_cite_key(meta, pdf_path)
    add_to_bib(bib_path, key, meta)
    return meta


def collect_pdfs(paths: List[str], input_dir: Optional[str]) -> List[str]:
    """CLI 引数とディレクトリ指定から PDF パスを収集する。"""
    pdfs: List[str] = list(paths)
    if input_dir:
        dir_path = Path(input_dir)
        if dir_path.is_dir():
            pdfs.extend(str(p) for p in sorted(dir_path.glob("*.pdf")))
        else:
            print(f"[warn] ディレクトリが見つかりません: {input_dir}")
    return pdfs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PDF 論文から AI でメタデータを抽出し BibTeX に保存",
    )
    parser.add_argument("pdfs", nargs="*", help="処理する PDF ファイル")
    parser.add_argument(
        "--input-dir", "-d",
        help="PDF ファイルを含むディレクトリ",
    )
    parser.add_argument(
        "--output", "-o",
        default="library.bib",
        help="出力 .bib ファイル (default: library.bib)",
    )
    parser.add_argument(
        "--model", "-m",
        default="gpt-4o",
        help="OpenAI モデル名 (default: gpt-4o)",
    )
    args = parser.parse_args()

    pdfs = collect_pdfs(args.pdfs, args.input_dir)
    if not pdfs:
        parser.print_help()
        print("\n[error] PDF ファイルを指定してください。")
        sys.exit(1)

    print(f"対象 PDF: {len(pdfs)} 件  →  出力: {args.output}")

    results: List[Dict] = []
    for pdf in pdfs:
        meta = process_pdf(pdf, args.output, args.model)
        if meta:
            results.append(meta)

    print(f"\n=== 完了: {len(results)}/{len(pdfs)} 件を処理しました ===")


if __name__ == "__main__":
    main()
