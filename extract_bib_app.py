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

import os
from pathlib import Path
from typing import Dict, List

import streamlit as st

from extract_bib import (
    Checkpoint,
    _format_bib_entry,
    _make_cite_key,
    extract_text_from_pdf,
    file_sha256,
    get_metadata_via_ai,
)

# ---------------------------------------------------------------------------
# ページ設定
# ---------------------------------------------------------------------------

st.set_page_config(page_title="PDF → BibTeX Extractor", page_icon="📖", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;500;700&display=swap');
    html, body, [class*="st-"], .stMarkdown, .stText, .stDataFrame,
    h1, h2, h3, h4, h5, h6, p, span, div, label, input, button, textarea, select {
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
        model = st.selectbox("モデル", ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"])
        base_url = None
    else:
        provider_key = "lmstudio"
        api_key = None
        base_url = st.text_input("LM Studio URL", value="http://localhost:1234/v1")
        model = st.text_input("モデル名", value="local-model")

    st.divider()
    st.header("出力設定")
    output_name = st.text_input("BibTeX ファイル名", value="library.bib")
    checkpoint_path = st.text_input("チェックポイント", value=".extract_bib_checkpoint.json")

# ---------------------------------------------------------------------------
# タブ構成
# ---------------------------------------------------------------------------

tab_extract, tab_search = st.tabs(["📥 PDF 一括抽出", "🔍 文献検索"])

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
            else:
                if api_key:
                    os.environ["OPENAI_API_KEY"] = api_key

                ckpt = Checkpoint(checkpoint_path)
                results: List[Dict] = []
                errors: List[Dict] = []
                skipped = 0

                progress = st.progress(0, text="準備中...")
                status_area = st.empty()
                stats_area = st.empty()

                for idx, pdf_path in enumerate(pdf_list):
                    fname = Path(pdf_path).name
                    progress.progress(
                        idx / len(pdf_list),
                        text=f"({idx + 1}/{len(pdf_list)}) {fname}",
                    )

                    try:
                        sha = file_sha256(pdf_path)

                        if ckpt.is_done(sha):
                            cached = ckpt.get(sha)
                            if cached:
                                results.append(cached)
                            skipped += 1
                            continue

                        status_area.info(f"テキスト抽出中: {fname}")
                        text = extract_text_from_pdf(pdf_path)
                        if len(text.strip()) < 100:
                            errors.append({"_file": pdf_path, "_error": "テキスト不足"})
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

                        if (idx + 1) % 20 == 0:
                            ckpt.save()

                    except Exception as e:
                        errors.append({"_file": pdf_path, "_error": str(e)})

                    stats_area.markdown(
                        f"✅ {len(results)} 件成功  |  ⏭️ {skipped} 件スキップ  |  ❌ {len(errors)} 件エラー"
                    )

                ckpt.save()
                progress.progress(1.0, text="完了!")
                status_area.empty()

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
