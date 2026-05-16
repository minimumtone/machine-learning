#!/usr/bin/env python3
"""
extract_bib.py — PDFから論文メタデータをAIで抽出しBibTeXに保存するツール (v2)

機能:
  - PDF からテキストを抽出 (pymupdf4llm / PyMuPDF)
  - OpenAI API または LM Studio (ローカルLLM) でメタデータ抽出
    (title, authors, doi, materials, theory, methods, summary_ja)
  - SHA-256 ハッシュによるチェックポイント管理
    → PDF の置き場所が変わっても処理済みファイルを再処理しない
  - DOI ベース重複検出
  - 並行処理 (ThreadPoolExecutor) で 1万ファイル規模に対応
  - JabRef 互換 .bib ファイルへ出力

使い方:
  python extract_bib.py -d ./papers/
  python extract_bib.py -d ./papers/ --workers 8 --provider lmstudio
  python extract_bib.py paper1.pdf paper2.pdf -o my_library.bib
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

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
# SHA-256 ハッシュ (パス非依存の識別子)
# ---------------------------------------------------------------------------

def file_sha256(path: str, chunk_size: int = 1 << 20) -> str:
    """ファイルの SHA-256 ハッシュを返す。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# チェックポイント管理
# ---------------------------------------------------------------------------

class Checkpoint:
    """SHA-256 → メタデータ のマッピングを JSON で永続化。スレッドセーフ。"""

    def __init__(self, path: str = ".extract_bib_checkpoint.json"):
        self.path = Path(path)
        self._data: Dict[str, Dict] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
                log.info("チェックポイント読み込み: %d 件", len(self._data))
            except (json.JSONDecodeError, OSError) as e:
                log.warning("チェックポイント読み込み失敗 (%s), 新規作成", e)
                self._data = {}

    def save(self) -> None:
        with self._lock:
            snapshot = dict(self._data)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(snapshot, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    def is_done(self, sha: str) -> bool:
        with self._lock:
            return sha in self._data

    def get(self, sha: str) -> Optional[Dict]:
        with self._lock:
            return self._data.get(sha)

    def put(self, sha: str, meta: Dict) -> None:
        with self._lock:
            self._data[sha] = meta

    def all_entries(self) -> List[Dict]:
        with self._lock:
            return list(self._data.values())

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


# ---------------------------------------------------------------------------
# LLM クライアント (OpenAI / LM Studio 共通)
# ---------------------------------------------------------------------------

SYSTEM_MSG = (
    "あなたは材料科学・工学の専門家です。論文テキストから以下の情報を抽出し"
    "JSONで返してください:\n"
    "- title: タイトル\n"
    "- authors: 著者（カンマ区切りの文字列）\n"
    "- year: 出版年（整数、不明なら null）\n"
    "- doi: DOI（不明なら null）\n"
    "- journal: ジャーナル名（不明なら null）\n"
    "- materials: 対象物質のリスト（カンマ区切り。例: 'SrTiO3, BaTiO3, High-entropy alloy'）\n"
    "- study_type: 研究種別（'experimental', 'theoretical', 'computational', 'review' のいずれか、"
    "または 'experimental+computational' のように複合）\n"
    "- theory: 物理理論・モデル（カンマ区切り。例: 'Density Functional Theory, "
    "Thermodynamic CALPHAD, Phase field model'）\n"
    "- exp_methods: 実験手法（カンマ区切り。例: 'TEM, XRD, dilatometry, nanoindentation'。"
    "実験がなければ null）\n"
    "- math_methods: 数理的手法（カンマ区切り。例: 'Bayesian inference, Monte Carlo simulation, "
    "finite element method, genetic algorithm, regression analysis'。該当なければ null）\n"
    "- ml_methods: 機械学習手法（カンマ区切り。例: 'Neural Network, Random Forest, "
    "Gaussian Process, XGBoost, deep learning'。該当なければ null）\n"
    "- properties: 対象物性・測定量（カンマ区切り。例: 'Ms temperature, lattice constant, "
    "elastic modulus, phase diagram'）\n"
    "- keywords: 検索用キーワード（カンマ区切り、5〜10個。材料名・手法・物性・分野を含む）\n"
    "- summary_ja: 論文の要点を日本語で2〜3文にまとめたもの\n"
    "\n"
    "回答はJSONオブジェクトのみ（マークダウンのコードブロック不要）。"
)

DEFAULT_PROVIDERS: Dict[str, Dict[str, Any]] = {
    "openai": {
        "base_url": None,
        "model": "gpt-5.4-nano",
        "api_key_env": "OPENAI_API_KEY",
        "api_key_required": True,
    },
    "lmstudio": {
        "base_url": "http://localhost:1234/v1",
        "model": "local-model",
        "api_key_env": None,
        "api_key_required": False,
    },
}


def _build_client(
    provider: str = "openai",
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> tuple:
    """OpenAI 互換クライアントと使用モデル名を返す。"""
    from openai import OpenAI

    cfg = DEFAULT_PROVIDERS.get(provider, DEFAULT_PROVIDERS["openai"])

    resolved_url = base_url or cfg["base_url"]
    resolved_model = model or cfg["model"]

    if api_key:
        resolved_key = api_key
    elif cfg["api_key_env"]:
        resolved_key = os.getenv(cfg["api_key_env"], "")
    else:
        resolved_key = "not-needed"

    if cfg["api_key_required"] and not resolved_key:
        raise RuntimeError(
            f"環境変数 {cfg['api_key_env']} が設定されていません。"
        )

    kwargs: Dict[str, Any] = {"api_key": resolved_key, "timeout": 180}
    if resolved_url:
        kwargs["base_url"] = resolved_url

    return OpenAI(**kwargs), resolved_model


def _estimate_tokens(text: str) -> int:
    """テキストのトークン数を推定する。

    学術論文（化学式・数式・特殊記号多め）は BPE トークナイザーで
    1文字=1トークンになるケースが多いため、保守的に推定する。
    """
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    non_ascii = len(text) - ascii_chars
    return int(ascii_chars / 1.5 + non_ascii / 1.0)


def _split_text_by_tokens(text: str, max_tokens: int = 5000) -> List[str]:
    """テキストを max_tokens 以下のチャンクに分割する。段落境界で分割を試みる。"""
    if _estimate_tokens(text) <= max_tokens:
        return [text]

    paragraphs = re.split(r"\n\s*\n", text)
    chunks: List[str] = []
    current_chunk = ""

    for para in paragraphs:
        candidate = (current_chunk + "\n\n" + para).strip() if current_chunk else para
        if _estimate_tokens(candidate) <= max_tokens:
            current_chunk = candidate
        else:
            if current_chunk:
                chunks.append(current_chunk)
            if _estimate_tokens(para) > max_tokens:
                sentences = re.split(r"(?<=[.。!?])\s+", para)
                current_chunk = ""
                for sent in sentences:
                    candidate = (current_chunk + " " + sent).strip() if current_chunk else sent
                    if _estimate_tokens(candidate) <= max_tokens:
                        current_chunk = candidate
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = sent
            else:
                current_chunk = para

    if current_chunk:
        chunks.append(current_chunk)

    return chunks if chunks else [text[:max_tokens * 4]]


def _merge_metadata(results: List[Dict]) -> Dict:
    """複数チャンクの解析結果をマージする。最初のチャンクを優先（title/authors等）。"""
    if len(results) == 1:
        return results[0]

    merged: Dict[str, Any] = {}
    list_fields = {"materials", "theory", "exp_methods", "math_methods", "ml_methods", "properties", "keywords"}
    first_priority = {"title", "authors", "year", "doi", "journal", "study_type"}

    for key in first_priority:
        for r in results:
            val = r.get(key)
            if val:
                merged[key] = val
                break

    for key in list_fields:
        seen: set = set()
        parts: List[str] = []
        for r in results:
            val = r.get(key, "")
            if not val:
                continue
            for item in re.split(r"[;,]", val):
                item = item.strip()
                if item and item.lower() not in seen:
                    seen.add(item.lower())
                    parts.append(item)
        merged[key] = ", ".join(parts) if parts else ""

    summaries = [r.get("summary_ja", "") for r in results if r.get("summary_ja")]
    if summaries:
        merged["summary_ja"] = " ".join(summaries)

    return merged


def _call_llm_single(
    client: Any,
    resolved_model: str,
    text: str,
    max_retries: int = 3,
    provider: str = "openai",
) -> Dict:
    """単一チャンクをLLMに送信してメタデータを取得する。リトライ付き。"""
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=resolved_model,
                messages=[
                    {"role": "system", "content": SYSTEM_MSG},
                    {"role": "user", "content": f"以下の論文テキストを解析してください:\n\n{text}"},
                ],
                temperature=0.0,
            )
            if not response or not response.choices:
                raise ValueError("LLM から空のレスポンスが返されました（choices が空）")
            msg = response.choices[0].message
            if msg is None:
                raise ValueError("LLM レスポンスの message が None です")
            raw = msg.content
            if not raw:
                raise ValueError("LLM レスポンスの content が空です")
            return _parse_json(raw)
        except Exception as e:
            if attempt < max_retries:
                wait = 2 ** attempt
                log.warning("API エラー (attempt %d/%d): %s — %ds 後にリトライ", attempt, max_retries, e, wait)
                time.sleep(wait)
            else:
                raise


def get_metadata_via_ai(
    text: str,
    provider: str = "openai",
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    max_retries: int = 3,
    max_tokens_per_chunk: int = 2500,
) -> Dict:
    """チャンク分割で LLM API にメタデータ抽出を依頼。ローカル LLM のコンテキスト長にも対応。"""
    client, resolved_model = _build_client(provider, base_url, model, api_key)

    est = _estimate_tokens(text)
    chunks = _split_text_by_tokens(text, max_tokens=max_tokens_per_chunk)
    log.info("テキスト分割: %d チャンク（推定 %d トークン）", len(chunks), est)

    chunk_results = []
    for i, chunk in enumerate(chunks):
        try:
            result = _call_llm_single(client, resolved_model, chunk, max_retries, provider)
            chunk_results.append(result)
        except Exception as e:
            err_str = str(e).lower()
            is_context_err = "context length" in err_str or "n_ctx" in err_str or "choices が空" in str(e)
            if is_context_err and _estimate_tokens(chunk) > 500:
                half = len(chunk) // 2
                log.warning("チャンク %d がコンテキスト超過、半分に縮小してリトライ", i + 1)
                try:
                    result = _call_llm_single(client, resolved_model, chunk[:half], max_retries, provider)
                    chunk_results.append(result)
                    continue
                except Exception:
                    pass
                quarter = len(chunk) // 4
                log.warning("チャンク %d を 1/4 に縮小してリトライ", i + 1)
                try:
                    result = _call_llm_single(client, resolved_model, chunk[:quarter], max_retries, provider)
                    chunk_results.append(result)
                    continue
                except Exception:
                    pass
            if len(chunks) > 1:
                log.warning("チャンク %d/%d をスキップ: %s", i + 1, len(chunks), e)
                continue
            else:
                raise

    if not chunk_results:
        raise ValueError("すべてのチャンクで LLM の応答を取得できませんでした")

    return _merge_metadata(chunk_results)


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
# BibTeX 生成
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


def _escape_bib(value: str) -> str:
    """BibTeX 値のエスケープ。"""
    return value.replace("{", "\\{").replace("}", "\\}")


def _sanitize_filename(name: str, max_len: int = 100) -> str:
    """ファイル名に使えない文字を除去し、長さを制限する。"""
    forbidden = r'[\\/:*?"<>|]'
    name = re.sub(forbidden, "", name)
    name = re.sub(r"\s+", " ", name).strip()
    if len(name) > max_len:
        name = name[:max_len].rstrip()
    return name


def _rename_pdf(pdf_path: str, meta: Dict) -> str:
    """解析結果を元に PDF を 著者_タイトル.pdf にリネームする。新しいパスを返す。"""
    authors = meta.get("authors", "") or ""
    title = meta.get("title", "") or ""
    if not authors and not title:
        return pdf_path

    first_author = authors.split(",")[0].strip()
    if " " in first_author:
        first_author = first_author.split()[-1]
    first_author = _sanitize_filename(first_author, max_len=30)
    title_clean = _sanitize_filename(title, max_len=80)

    if first_author and title_clean:
        new_name = f"{first_author}_{title_clean}.pdf"
    elif title_clean:
        new_name = f"{title_clean}.pdf"
    else:
        new_name = f"{first_author}.pdf"

    parent = os.path.dirname(pdf_path)
    new_path = os.path.join(parent, new_name)

    if new_path != pdf_path and os.path.exists(new_path):
        base, ext = os.path.splitext(new_name)
        counter = 2
        while os.path.exists(os.path.join(parent, f"{base}_{counter}{ext}")):
            counter += 1
        new_path = os.path.join(parent, f"{base}_{counter}{ext}")

    if new_path != pdf_path:
        os.rename(pdf_path, new_path)
        log.info("PDF リネーム: %s → %s", os.path.basename(pdf_path), os.path.basename(new_path))

    return new_path


def _format_bib_entry(key: str, meta: Dict) -> str:
    """1 エントリ分の BibTeX 文字列を組み立てる。"""
    lines = [f"@article{{{key},"]

    source = meta.get("_source_file", "")
    file_basename = os.path.basename(source) if source else ""

    field_map = [
        ("title", meta.get("title", "")),
        ("author", meta.get("authors", "")),
        ("year", str(meta.get("year", "")) if meta.get("year") else ""),
        ("journal", meta.get("journal", "") or ""),
        ("doi", meta.get("doi", "") or ""),
        ("file", file_basename),
        ("materials", meta.get("materials", "") or ""),
        ("studytype", meta.get("study_type", "") or ""),
        ("theory", meta.get("theory", "") or ""),
        ("expmethods", meta.get("exp_methods", "") or ""),
        ("mathmethods", meta.get("math_methods", "") or ""),
        ("mlmethods", meta.get("ml_methods", "") or ""),
        ("properties", meta.get("properties", "") or ""),
        ("keywords", meta.get("keywords", "") or ""),
        ("comment", meta.get("summary_ja", "") or ""),
    ]

    for fname, fval in field_map:
        if fval:
            lines.append(f"  {fname} = {{{_escape_bib(str(fval))}}},")

    lines.append("}")
    return "\n".join(lines)


def write_bib(bib_path: str, entries: List[Dict]) -> None:
    """全エントリを .bib ファイルに書き出す（DOI ベース重複除去付き）。"""
    path = Path(bib_path)

    seen_keys: set = set()
    seen_dois: set = set()
    bib_blocks: List[str] = []

    if path.exists():
        existing = path.read_text(encoding="utf-8")
        for m in re.finditer(r"@\w+\{\s*([^,]+)\s*,", existing):
            seen_keys.add(m.group(1).strip())
        for m in re.finditer(r"doi\s*=\s*\{([^}]+)\}", existing, re.IGNORECASE):
            seen_dois.add(m.group(1).strip().lower())

    new_count = 0
    for meta in entries:
        doi = (meta.get("doi") or "").strip().lower()
        if doi and doi in seen_dois:
            log.info("  [skip/doi] DOI 重複: %s", doi)
            continue

        key = _make_cite_key(meta, meta.get("_source_file", "unknown"))
        base_key = key
        counter = 2
        while key in seen_keys:
            key = f"{base_key}_{counter}"
            counter += 1

        seen_keys.add(key)
        if doi:
            seen_dois.add(doi)
        bib_blocks.append(_format_bib_entry(key, meta))
        new_count += 1

    if bib_blocks:
        with open(path, "a", encoding="utf-8") as f:
            existing_text = path.read_text(encoding="utf-8") if path.stat().st_size > 0 else ""
            if existing_text and not existing_text.endswith("\n"):
                f.write("\n")
            f.write("\n" + "\n\n".join(bib_blocks) + "\n")

    log.info("BibTeX 出力: %d 件追加 → %s", new_count, bib_path)


# ---------------------------------------------------------------------------
# PDF 収集
# ---------------------------------------------------------------------------

def collect_pdfs(paths: List[str], input_dir: Optional[str], recursive: bool = True) -> List[str]:
    """CLI 引数とディレクトリ指定から PDF パスを収集する。"""
    pdfs: List[str] = list(paths)
    if input_dir:
        dir_path = Path(input_dir)
        if dir_path.is_dir():
            glob_pattern = "**/*.pdf" if recursive else "*.pdf"
            pdfs.extend(str(p) for p in sorted(dir_path.glob(glob_pattern)))
        else:
            log.warning("ディレクトリが見つかりません: %s", input_dir)
    return pdfs


# ---------------------------------------------------------------------------
# 1ファイル処理 (ワーカー関数)
# ---------------------------------------------------------------------------

def _process_one(
    pdf_path: str,
    checkpoint: Checkpoint,
    provider: str,
    model: Optional[str],
    base_url: Optional[str],
    api_key: Optional[str],
    rename: bool = False,
) -> Optional[Dict]:
    """PDF 1 ファイルを処理。チェックポイント済みならスキップ。

    Returns (meta, is_new): is_new=False ならキャッシュからの取得。
    """
    sha = file_sha256(pdf_path)

    if checkpoint.is_done(sha):
        log.info("  [skip/hash] %s (処理済み)", Path(pdf_path).name)
        return checkpoint.get(sha), False

    text = extract_text_from_pdf(pdf_path)
    if len(text.strip()) < 100:
        log.warning("  [skip/short] %s テキスト不足", Path(pdf_path).name)
        return None, False

    meta = get_metadata_via_ai(
        text, provider=provider, model=model, base_url=base_url, api_key=api_key,
    )
    if rename:
        new_path = _rename_pdf(pdf_path, meta)
        meta["_source_file"] = new_path
    else:
        meta["_source_file"] = pdf_path
    meta["_sha256"] = sha

    checkpoint.put(sha, meta)
    return meta, True


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def run_batch(
    pdfs: List[str],
    bib_path: str = "library.bib",
    checkpoint_path: str = ".extract_bib_checkpoint.json",
    provider: str = "openai",
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    workers: int = 4,
    save_every: int = 20,
    on_progress: Optional[Any] = None,
    rename: bool = False,
) -> List[Dict]:
    """PDF リストをバッチ処理してチェックポイントに保存し、BibTeX を出力。

    on_progress: callable(done, total, pdf_path, meta_or_none) — UI 向けコールバック
    """
    checkpoint = Checkpoint(checkpoint_path)
    results: List[Dict] = []
    done_count = 0
    total = len(pdfs)

    log.info("対象 PDF: %d 件 (workers=%d, provider=%s)", total, workers, provider)

    def _worker(pdf_path: str) -> tuple:
        try:
            meta, is_new = _process_one(pdf_path, checkpoint, provider, model, base_url, api_key, rename=rename)
            return pdf_path, meta, is_new, None
        except Exception as e:
            return pdf_path, None, False, str(e)

    new_entries: List[Dict] = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker, p): p for p in pdfs}
        for future in as_completed(futures):
            pdf_path, meta, is_new, error = future.result()
            done_count += 1

            if error:
                log.error("  [error] %s: %s", Path(pdf_path).name, error)
            elif meta:
                results.append(meta)
                if is_new:
                    new_entries.append(meta)
                log.info(
                    "  [%d/%d] %s → %s",
                    done_count, total,
                    Path(pdf_path).name,
                    meta.get("title", "???")[:60],
                )

            if on_progress:
                on_progress(done_count, total, pdf_path, meta)

            if done_count % save_every == 0:
                checkpoint.save()

    checkpoint.save()

    if new_entries:
        write_bib(bib_path, new_entries)
    log.info("=== 完了: %d/%d 件を処理 (新規 %d 件) ===", len(results), total, len(new_entries))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PDF 論文から AI でメタデータを抽出し BibTeX に保存 (v2: 大規模対応)",
    )
    parser.add_argument("pdfs", nargs="*", help="処理する PDF ファイル")
    parser.add_argument("--input-dir", "-d", help="PDF ファイルを含むディレクトリ")
    parser.add_argument("--output", "-o", default="library.bib", help="出力 .bib ファイル")
    parser.add_argument("--checkpoint", default=".extract_bib_checkpoint.json", help="チェックポイントファイル")
    parser.add_argument("--provider", "-p", choices=["openai", "lmstudio"], default="openai")
    parser.add_argument("--model", "-m", default=None, help="モデル名 (省略時: プロバイダデフォルト)")
    parser.add_argument("--base-url", default=None, help="API ベース URL (LM Studio 等)")
    parser.add_argument("--workers", "-w", type=int, default=4, help="並行ワーカー数")
    parser.add_argument("--no-recursive", action="store_true", help="サブディレクトリを再帰探索しない")
    parser.add_argument("--rename", action="store_true", help="PDF を 著者_タイトル.pdf にリネーム")
    args = parser.parse_args()

    pdfs = collect_pdfs(args.pdfs, args.input_dir, recursive=not args.no_recursive)
    if not pdfs:
        parser.print_help()
        print("\n[error] PDF ファイルを指定してください。")
        sys.exit(1)

    run_batch(
        pdfs=pdfs,
        bib_path=args.output,
        checkpoint_path=args.checkpoint,
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        workers=args.workers,
        rename=args.rename,
    )


if __name__ == "__main__":
    main()
