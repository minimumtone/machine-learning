#!/usr/bin/env python3
"""
extract_bib_app.py — Streamlit UI for AI-powered PDF → BibTeX extraction (v2)

機能:
  - ディレクトリ指定で配下の全 PDF を一括処理 (1万件スケール対応)
  - SHA-256 チェックポイントで処理済みスキップ (ファイル移動に対応)
  - OpenAI / LM Studio (ローカルLLM) 切り替え
  - 蓄積メタデータの検索・フィルタ機能 (材料検索エンジン)
  - BibTeX ダウンロード / ディスク保存

起動:
  streamlit run extract_bib_app.py
"""

import hashlib
import json
import logging
import os
import re
import threading
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

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

    def all_entries_with_sha(self) -> List[tuple]:
        """(sha, meta) のリストを返す。"""
        with self._lock:
            return list(self._data.items())

    def remove(self, sha: str) -> None:
        with self._lock:
            self._data.pop(sha, None)

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
    "- materials: 対象物質（例: 'SrTiO3', 'High-entropy alloy'）\n"
    "- theory: 理論・モデル（例: 'Density Functional Theory', 'Dislocation Dynamics'）\n"
    "- methods: 実験・解析手法（例: 'TEM', 'Machine Learning regression'）\n"
    "- summary_ja: 論文の要点を日本語で2〜3文にまとめたもの\n"
    "\n"
    "回答はJSONオブジェクトのみ（マークダウンのコードブロック不要）。"
)

DEFAULT_PROVIDERS: Dict[str, Dict[str, Any]] = {
    "openai": {
        "base_url": None,
        "model": "gpt-4o-mini",
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


def get_metadata_via_ai(
    text: str,
    provider: str = "openai",
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    max_retries: int = 3,
) -> Dict:
    """LLM API でメタデータを抽出する。リトライ付き。"""
    client, resolved_model = _build_client(provider, base_url, model, api_key)

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=resolved_model,
                messages=[
                    {"role": "system", "content": SYSTEM_MSG},
                    {"role": "user", "content": f"以下の論文テキストを解析してください:\n\n{text}"},
                ],
                temperature=0.0,
                max_tokens=2048,
            )
            raw = response.choices[0].message.content or ""
            return _parse_json(raw)
        except Exception as e:
            if attempt < max_retries:
                wait = 2 ** attempt
                log.warning("API エラー (attempt %d/%d): %s — %ds 後にリトライ", attempt, max_retries, e, wait)
                time.sleep(wait)
            else:
                raise


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

# ---------------------------------------------------------------------------
# ページ設定
# ---------------------------------------------------------------------------

st.set_page_config(page_title="PDF → BibTeX Extractor", page_icon="📖", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;500;700&display=swap');
    html, body, .stMarkdown, .stText, .stDataFrame,
    h1, h2, h3, h4, h5, h6, p, label, input, button, textarea, select,
    .stRadio div, .stSelectbox div, .stMultiSelect div, .stTextInput div,
    .stTabs div, .stExpander div, .stAlert div, .stCaption, .stSubheader {
        font-family: 'Noto Sans JP', sans-serif !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# サイドバー: LLM 設定
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("LLM 設定")
    provider = st.radio("プロバイダ", ["OpenAI", "LM Studio (ローカル)"], horizontal=True)

    if provider == "OpenAI":
        provider_key = "openai"
        api_key = st.text_input(
            "OpenAI API Key",
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password",
        )
        model = st.selectbox("モデル", ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"])
        base_url = None
    else:
        provider_key = "lmstudio"
        api_key = None
        base_url = st.text_input("LM Studio URL", value="http://localhost:1234/v1")
        model = st.text_input(
            "モデル名",
            value="",
            placeholder="例: bonsai-8b, gemma-3-4b など",
            help="LM Studio で読み込んでいるモデル名を入力してください",
        )

    st.divider()
    st.header("出力設定")
    output_name = st.text_input("BibTeX ファイル名", value="library.bib")
    checkpoint_path = st.text_input("チェックポイント", value=".extract_bib_checkpoint.json")

# ---------------------------------------------------------------------------
# タブ構成
# ---------------------------------------------------------------------------

tab_extract, tab_search, tab_cleanup = st.tabs(["📥 PDF 一括抽出", "🔍 文献検索", "🧹 クリーンアップ"])

# ===================================================================
# タブ 1: PDF 一括抽出
# ===================================================================

with tab_extract:
    st.header("PDF → BibTeX 一括抽出")
    st.markdown("ディレクトリを指定すると、配下の **全 PDF** を AI 解析し BibTeX を生成します。")

    dir_path = st.text_input(
        "PDF ディレクトリのパス",
        placeholder="/home/user/papers",
        help="サブディレクトリ内も再帰的に探索します",
        key="extract_dir",
    )
    recursive = st.checkbox("サブディレクトリも探索", value=True)
    force_reprocess = st.checkbox(
        "チェックポイントを無視して再処理",
        value=False,
        help="以前の処理結果を無視して全件を再解析します（モデル切り替え時など）",
    )

    col_scan, col_clear = st.columns(2)
    with col_scan:
        scan_clicked = st.button("PDF をスキャン", type="primary", use_container_width=True)
    with col_clear:
        if st.button("結果をクリア", use_container_width=True):
            for k in ["pdf_list", "results", "bib_text"]:
                st.session_state.pop(k, None)
            st.rerun()

    # --- スキャン ---
    if scan_clicked and dir_path:
        target = Path(dir_path)
        if not target.is_dir():
            st.error(f"ディレクトリが見つかりません: {dir_path}")
        else:
            pattern = "**/*.pdf" if recursive else "*.pdf"
            pdfs = sorted(str(p) for p in target.glob(pattern))
            if not pdfs:
                st.warning("PDF ファイルが見つかりませんでした。")
            else:
                st.session_state["pdf_list"] = pdfs
                for k in ["results", "bib_text"]:
                    st.session_state.pop(k, None)
                st.success(f"{len(pdfs)} 件の PDF を検出しました。")

    # --- 検出一覧 & 実行 ---
    if "pdf_list" in st.session_state:
        pdf_list: List[str] = st.session_state["pdf_list"]

        with st.expander(f"検出 PDF 一覧 ({len(pdf_list)} 件)", expanded=False):
            for i, p in enumerate(pdf_list, 1):
                st.text(f"{i}. {p}")

        if st.button(
            f"全 {len(pdf_list)} 件を解析して BibTeX 生成",
            type="primary",
            use_container_width=True,
        ):
            if provider_key == "openai" and not api_key:
                st.error("OpenAI API Key を入力してください。")
            elif provider_key == "lmstudio" and not model:
                st.error("LM Studio のモデル名を入力してください。（例: bonsai-8b）")
            else:
                if api_key:
                    os.environ["OPENAI_API_KEY"] = api_key

                ckpt = Checkpoint(checkpoint_path)
                results: List[Dict] = []
                errors: List[Dict] = []
                skipped = 0
                total = len(pdf_list)

                progress_bar = st.progress(0)
                progress_text = st.empty()
                status_area = st.empty()
                stats_area = st.empty()
                time_area = st.empty()

                t_start = time.time()

                for idx, pdf_path in enumerate(pdf_list):
                    fname = Path(pdf_path).name
                    done = idx + 1
                    pct = done / total
                    progress_bar.progress(pct)
                    progress_text.markdown(
                        f"**進捗: {done} / {total} 件 ({pct:.0%})**"
                    )

                    elapsed = time.time() - t_start
                    if idx > 0:
                        speed = idx / elapsed
                        remaining = (total - done) / speed
                        elapsed_str = time.strftime("%M:%S", time.gmtime(elapsed))
                        remaining_str = time.strftime("%M:%S", time.gmtime(remaining))
                        time_area.caption(
                            f"経過: {elapsed_str}  |  残り推定: {remaining_str}  |  "
                            f"速度: {speed:.1f} 件/秒"
                        )

                    try:
                        sha = file_sha256(pdf_path)

                        if not force_reprocess and ckpt.is_done(sha):
                            cached = ckpt.get(sha)
                            if cached:
                                results.append(cached)
                            skipped += 1
                            stats_area.markdown(
                                f"成功 **{len(results)}**  |  "
                                f"スキップ **{skipped}**  |  "
                                f"エラー **{len(errors)}**"
                            )
                            continue

                        status_area.info(f"テキスト抽出中: {fname}")
                        text = extract_text_from_pdf(pdf_path)
                        if len(text.strip()) < 100:
                            errors.append({"_file": pdf_path, "_error": "テキスト不足"})
                            stats_area.markdown(
                                f"成功 **{len(results)}**  |  "
                                f"スキップ **{skipped}**  |  "
                                f"エラー **{len(errors)}**"
                            )
                            continue

                        status_area.info(f"AI 解析中: {fname}")
                        meta = get_metadata_via_ai(
                            text,
                            provider=provider_key,
                            model=model,
                            base_url=base_url,
                            api_key=api_key,
                        )
                        meta["_source_file"] = pdf_path
                        meta["_sha256"] = sha
                        meta["_cite_key"] = _make_cite_key(meta, pdf_path)

                        ckpt.put(sha, meta)
                        results.append(meta)

                        if done % 20 == 0:
                            ckpt.save()

                    except Exception as e:
                        errors.append({"_file": pdf_path, "_error": str(e)})

                    stats_area.markdown(
                        f"成功 **{len(results)}**  |  "
                        f"スキップ **{skipped}**  |  "
                        f"エラー **{len(errors)}**"
                    )

                ckpt.save()
                progress_bar.progress(1.0)
                progress_text.markdown("**完了!**")
                elapsed_total = time.time() - t_start
                time_area.caption(
                    f"合計時間: {time.strftime('%M:%S', time.gmtime(elapsed_total))}"
                )
                status_area.empty()

                if skipped > 0 and skipped == total:
                    st.warning(
                        f"⚠️ 全 {total} 件がチェックポイント済みのためスキップされました。\n\n"
                        "別のモデル（LM Studio 等）で再解析したい場合は、"
                        "「**チェックポイントを無視して再処理**」にチェックを入れて再実行してください。"
                    )
                elif skipped > 0:
                    st.info(
                        f"ℹ️ {skipped} 件はチェックポイント済み（前回の結果を使用）、"
                        f"{total - skipped - len(errors)} 件を新規解析しました。"
                    )

                # BibTeX 組み立て
                bib_entries: List[str] = []
                used_keys: set = set()
                for r in results:
                    key = r.get("_cite_key") or _make_cite_key(r, r.get("_source_file", ""))
                    base_key = key
                    counter = 2
                    while key in used_keys:
                        key = f"{base_key}_{counter}"
                        counter += 1
                    used_keys.add(key)
                    bib_entries.append(_format_bib_entry(key, r))

                bib_text = "\n\n".join(bib_entries) + "\n" if bib_entries else ""
                st.session_state["results"] = results
                st.session_state["bib_text"] = bib_text
                st.session_state["errors"] = errors
                st.session_state["skipped"] = skipped

    # --- 結果表示 ---
    if "results" in st.session_state:
        results = st.session_state["results"]
        bib_text = st.session_state.get("bib_text", "")
        errors = st.session_state.get("errors", [])
        skipped = st.session_state.get("skipped", 0)

        st.subheader(f"結果: {len(results)} 件成功 / {skipped} 件スキップ / {len(errors)} 件エラー")

        if results:
            display_data = []
            for r in results:
                display_data.append({
                    "Key": r.get("_cite_key", ""),
                    "Title": (r.get("title") or "")[:80],
                    "Year": r.get("year", ""),
                    "Materials": r.get("materials", ""),
                    "Theory": r.get("theory", ""),
                    "Methods": r.get("methods", ""),
                })
            st.dataframe(display_data, use_container_width=True, height=400)

            # --- 統計ダッシュボード ---
            st.subheader("📊 統計ダッシュボード")

            from collections import Counter

            col_s1, col_s2 = st.columns(2)

            with col_s1:
                # 年別論文数
                years = [r.get("year") for r in results if r.get("year")]
                if years:
                    year_counts = Counter(years)
                    year_sorted = sorted(year_counts.items())
                    st.markdown("**年別論文数**")
                    st.bar_chart(
                        {str(y): c for y, c in year_sorted},
                    )

            with col_s2:
                # Materials 頻度
                mats = [r.get("materials", "") for r in results if r.get("materials")]
                if mats:
                    mat_counts = Counter(mats).most_common(10)
                    st.markdown("**Materials（上位10件）**")
                    st.bar_chart(
                        {m[:30]: c for m, c in mat_counts},
                    )

            col_s3, col_s4 = st.columns(2)

            with col_s3:
                # Theory 頻度
                theories = [r.get("theory", "") for r in results if r.get("theory")]
                if theories:
                    theory_counts = Counter(theories).most_common(10)
                    st.markdown("**Theory（上位10件）**")
                    st.bar_chart(
                        {t[:30]: c for t, c in theory_counts},
                    )

            with col_s4:
                # Methods 頻度
                methods = [r.get("methods", "") for r in results if r.get("methods")]
                if methods:
                    method_counts = Counter(methods).most_common(10)
                    st.markdown("**Methods（上位10件）**")
                    st.bar_chart(
                        {m[:30]: c for m, c in method_counts},
                    )

            # --- エントリ詳細カード ---
            st.subheader("📄 エントリ詳細")
            for r in results:
                title_short = (r.get("title") or "(no title)")[:80]
                with st.expander(f"📄 {title_short}"):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f"**Title:** {r.get('title', '')}")
                        st.markdown(f"**Authors:** {r.get('authors', '')}")
                        st.markdown(f"**Year:** {r.get('year', '')}")
                        st.markdown(f"**Journal:** {r.get('journal', '')}")
                        doi = r.get("doi", "")
                        if doi:
                            st.markdown(f"**DOI:** [{doi}](https://doi.org/{doi})")
                    with c2:
                        st.markdown(f"**Materials:** {r.get('materials', '')}")
                        st.markdown(f"**Theory:** {r.get('theory', '')}")
                        st.markdown(f"**Methods:** {r.get('methods', '')}")
                    summary = r.get("summary_ja", "")
                    if summary:
                        st.markdown(f"**日本語要約:** {summary}")

        if errors:
            with st.expander(f"エラー ({len(errors)} 件)", expanded=False):
                for r in errors:
                    st.error(f"{Path(r['_file']).name}: {r['_error']}")

        if bib_text:
            st.subheader("BibTeX 出力")
            with st.expander("BibTeX プレビュー", expanded=False):
                st.code(bib_text[:10000], language="bibtex")
                if len(bib_text) > 10000:
                    st.caption(f"（全 {len(bib_text)} 文字中、先頭 10,000 文字を表示）")

            col_dl, col_save = st.columns(2)
            with col_dl:
                st.download_button(
                    label=f"{output_name} をダウンロード",
                    data=bib_text,
                    file_name=output_name,
                    mime="text/plain",
                    use_container_width=True,
                )
            with col_save:
                save_dir = st.text_input("保存先ディレクトリ", key="save_dir")
                if st.button("ディスクに保存", use_container_width=True) and save_dir:
                    save_path = Path(save_dir) / output_name
                    try:
                        save_path.parent.mkdir(parents=True, exist_ok=True)
                        save_path.write_text(bib_text, encoding="utf-8")
                        st.success(f"保存しました: {save_path}")
                    except Exception as e:
                        st.error(f"保存失敗: {e}")

# ===================================================================
# タブ 2: 文献検索
# ===================================================================

with tab_search:
    st.header("🔍 文献検索エンジン")
    st.markdown(
        "チェックポイントに蓄積されたメタデータから文献を検索します。"
    )

    ckpt_search = Checkpoint(checkpoint_path)
    all_entries = ckpt_search.all_entries()

    if not all_entries:
        st.info("まだメタデータがありません。「PDF 一括抽出」タブで PDF を処理してください。")
    else:
        st.caption(f"登録済み文献: {len(all_entries)} 件")

        # --- 検索入力 ---
        query = st.text_input("キーワード検索", placeholder="例: SrTiO3, DFT, Machine Learning...")
        search_mode = st.radio("検索モード", ["OR (いずれか含む)", "AND (すべて含む)"], horizontal=True)

        # --- フィルタ ---
        col_f1, col_f2, col_f3 = st.columns(3)

        all_materials = sorted({e.get("materials", "") for e in all_entries if e.get("materials")})
        all_theories = sorted({e.get("theory", "") for e in all_entries if e.get("theory")})
        all_methods = sorted({e.get("methods", "") for e in all_entries if e.get("methods")})

        with col_f1:
            filter_materials = st.multiselect("Materials フィルタ", all_materials)
        with col_f2:
            filter_theory = st.multiselect("Theory フィルタ", all_theories)
        with col_f3:
            filter_methods = st.multiselect("Methods フィルタ", all_methods)

        # --- 検索実行 ---
        def _matches_query(entry: Dict, keywords: List[str], mode: str) -> bool:
            if not keywords:
                return True
            searchable = " ".join(
                str(entry.get(f, ""))
                for f in ["title", "authors", "materials", "theory", "methods", "summary_ja", "journal", "doi"]
            ).lower()
            if mode.startswith("AND"):
                return all(kw.lower() in searchable for kw in keywords)
            return any(kw.lower() in searchable for kw in keywords)

        def _matches_filters(entry: Dict) -> bool:
            if filter_materials and entry.get("materials", "") not in filter_materials:
                return False
            if filter_theory and entry.get("theory", "") not in filter_theory:
                return False
            if filter_methods and entry.get("methods", "") not in filter_methods:
                return False
            return True

        keywords = [k.strip() for k in query.split(",") if k.strip()] if query else []
        filtered = [
            e for e in all_entries
            if _matches_query(e, keywords, search_mode) and _matches_filters(e)
        ]

        st.subheader(f"検索結果: {len(filtered)} 件")

        if filtered:
            # テーブル表示
            table_data = []
            for e in filtered:
                table_data.append({
                    "Title": (e.get("title") or "")[:80],
                    "Authors": (e.get("authors") or "")[:40],
                    "Year": e.get("year", ""),
                    "Materials": e.get("materials", ""),
                    "Theory": e.get("theory", ""),
                    "Methods": e.get("methods", ""),
                    "DOI": e.get("doi", ""),
                })
            st.dataframe(table_data, use_container_width=True, height=400)

            # 選択エントリの詳細
            st.subheader("エントリ詳細")
            for e in filtered:
                title_short = (e.get("title") or "(no title)")[:80]
                with st.expander(f"📄 {title_short}"):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f"**Title:** {e.get('title', '')}")
                        st.markdown(f"**Authors:** {e.get('authors', '')}")
                        st.markdown(f"**Year:** {e.get('year', '')}")
                        st.markdown(f"**Journal:** {e.get('journal', '')}")
                        doi = e.get("doi", "")
                        if doi:
                            st.markdown(f"**DOI:** [{doi}](https://doi.org/{doi})")
                    with c2:
                        st.markdown(f"**Materials:** {e.get('materials', '')}")
                        st.markdown(f"**Theory:** {e.get('theory', '')}")
                        st.markdown(f"**Methods:** {e.get('methods', '')}")
                    st.markdown(f"**日本語要約:** {e.get('summary_ja', '')}")

            # 検索結果の BibTeX エクスポート
            st.subheader("検索結果のエクスポート")
            export_entries: List[str] = []
            used_keys: set = set()
            for e in filtered:
                key = _make_cite_key(e, e.get("_source_file", ""))
                base_key = key
                counter = 2
                while key in used_keys:
                    key = f"{base_key}_{counter}"
                    counter += 1
                used_keys.add(key)
                export_entries.append(_format_bib_entry(key, e))

            export_bib = "\n\n".join(export_entries) + "\n"
            st.download_button(
                label=f"検索結果を BibTeX でダウンロード ({len(filtered)} 件)",
                data=export_bib,
                file_name=f"search_results_{len(filtered)}.bib",
                mime="text/plain",
                use_container_width=True,
            )

# ===================================================================
# タブ 3: クリーンアップ
# ===================================================================

with tab_cleanup:
    st.header("🧹 クリーンアップ — 不要エントリの検出・削除")
    st.markdown(
        "PDF ディレクトリをスキャンし、**ファイルが存在しなくなった論文**を"
        "チェックポイント / BibTeX から個別に削除できます。"
    )

    cleanup_dir = st.text_input(
        "PDF ディレクトリのパス",
        placeholder="/home/user/papers",
        help="現在の PDF 置き場を指定してください",
        key="cleanup_dir",
    )
    cleanup_recursive = st.checkbox("サブディレクトリも探索", value=True, key="cleanup_recursive")

    if st.button("孤立エントリを検出", type="primary", use_container_width=True):
        if not cleanup_dir or not Path(cleanup_dir).is_dir():
            st.error("有効なディレクトリを指定してください。")
        else:
            ckpt = Checkpoint(checkpoint_path)
            if len(ckpt) == 0:
                st.info("チェックポイントにエントリがありません。")
            else:
                # 現在のディレクトリ内の PDF の SHA-256 を計算
                pattern = "**/*.pdf" if cleanup_recursive else "*.pdf"
                current_pdfs = sorted(str(p) for p in Path(cleanup_dir).glob(pattern))
                st.caption(f"ディレクトリ内の PDF: {len(current_pdfs)} 件")

                with st.spinner("PDF のハッシュを計算中..."):
                    current_shas = set()
                    for p in current_pdfs:
                        try:
                            current_shas.add(file_sha256(p))
                        except Exception:
                            pass

                # チェックポイントのエントリと照合
                orphans = []
                for sha, meta in ckpt.all_entries_with_sha():
                    if sha not in current_shas:
                        orphans.append((sha, meta))

                if not orphans:
                    st.success("✅ 孤立エントリはありません。すべてのエントリに対応する PDF が存在します。")
                else:
                    st.warning(
                        f"⚠️ {len(orphans)} 件の孤立エントリを検出しました。"
                        "対応する PDF がディレクトリに見つかりません。"
                    )
                    st.session_state["orphans"] = orphans
                    st.session_state["cleanup_checkpoint_path"] = checkpoint_path

    # --- 孤立エントリの個別確認 ---
    if "orphans" in st.session_state and st.session_state["orphans"]:
        orphans = st.session_state["orphans"]
        ckpt_path = st.session_state.get("cleanup_checkpoint_path", checkpoint_path)

        st.subheader(f"孤立エントリ一覧（{len(orphans)} 件）")

        # 全選択/全解除
        col_all, col_none = st.columns(2)
        with col_all:
            select_all = st.button("すべて選択", use_container_width=True)
        with col_none:
            deselect_all = st.button("すべて解除", use_container_width=True)

        selections = {}
        for i, (sha, meta) in enumerate(orphans):
            title = (meta.get("title") or "(no title)")[:60]
            authors = (meta.get("authors") or "")[:40]
            source = meta.get("_source_file", "不明")
            label = f"**{title}**  —  {authors}  (元ファイル: {Path(source).name if source else '不明'})"

            default_checked = select_all if select_all else (not deselect_all)
            selections[sha] = st.checkbox(
                label,
                value=default_checked,
                key=f"orphan_{i}_{sha[:8]}",
            )

            with st.expander(f"詳細: {title}", expanded=False):
                st.markdown(f"- **Materials:** {meta.get('materials', '')}")
                st.markdown(f"- **Theory:** {meta.get('theory', '')}")
                st.markdown(f"- **Methods:** {meta.get('methods', '')}")
                st.markdown(f"- **日本語要約:** {meta.get('summary_ja', '')}")
                st.caption(f"SHA-256: {sha}")

        selected_shas = [sha for sha, checked in selections.items() if checked]
        st.markdown(f"**{len(selected_shas)} / {len(orphans)} 件を削除対象として選択中**")

        if st.button(
            f"選択した {len(selected_shas)} 件をチェックポイントから削除",
            type="primary",
            use_container_width=True,
            disabled=(len(selected_shas) == 0),
        ):
            ckpt = Checkpoint(ckpt_path)
            for sha in selected_shas:
                ckpt.remove(sha)
            ckpt.save()

            remaining = [(s, m) for s, m in orphans if s not in selected_shas]
            if remaining:
                st.session_state["orphans"] = remaining
            else:
                st.session_state.pop("orphans", None)

            st.success(f"✅ {len(selected_shas)} 件をチェックポイントから削除しました。")
            st.rerun()
