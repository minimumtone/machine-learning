"""Streamlit UI for the L1_2 schema-graph Text-to-SQL pipeline.

Run with:
    streamlit run ui/app.py

Environment variables (same as the evaluation scripts):
    OPENAI_API_KEY, POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB,
    POSTGRES_USER, POSTGRES_PASSWORD
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import psycopg
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph.graph_builder import build_table_graph  # noqa: E402
from graph.schema_parser import get_foreign_keys  # noqa: E402
from llm.sql_generator import (  # noqa: E402
    build_schema_context_from_db,
    pipeline,
)
from safety.sql_guard import (  # noqa: E402
    execute_sql,
    get_readonly_connection_string,
)
from safety.sql_validator import validate_sql  # noqa: E402

st.set_page_config(page_title="L1\u2082 Text-to-SQL", layout="wide")


@st.cache_resource
def load_schema_context() -> dict:
    """Build the schema graph context once per server process."""
    with psycopg.connect(get_readonly_connection_string()) as conn:
        ctx = build_schema_context_from_db(conn)
        fks = get_foreign_keys(conn)
    ctx["table_graph"] = build_table_graph(fks)
    return ctx


def main() -> None:
    st.title("L1\u2082 \u30b9\u30ad\u30fc\u30de\u30b0\u30e9\u30d5 Text-to-SQL")
    st.caption(
        "\u81ea\u7136\u8a00\u8a9e\u306e\u8cea\u554f\u2192SQL\u751f\u6210\u2192"
        "SQLGuard\u691c\u67fb\u2192read-only\u5b9f\u884c\u306e\u30c7\u30e2UI"
    )

    with st.sidebar:
        st.header("\u8a2d\u5b9a")
        n_best = st.slider("n-best \u5019\u88dc\u6570", 1, 5, 3)
        auto_execute = st.checkbox("\u691c\u67fb\u5f8c\u306b\u81ea\u52d5\u5b9f\u884c", value=True)
        row_limit = st.number_input(
            "\u8868\u793a\u884c\u6570\u4e0a\u9650", min_value=10, max_value=1000, value=100
        )

    try:
        ctx = load_schema_context()
        st.sidebar.success(
            f"\u30b9\u30ad\u30fc\u30de\u63a5\u7d9a\u6e08: "
            f"{len(ctx['all_tables'])} \u30c6\u30fc\u30d6\u30eb / "
            f"{len(ctx['join_list'])} FK"
        )
    except Exception as exc:  # pragma: no cover - UI feedback path
        st.error(f"DB\u63a5\u7d9a\u306b\u5931\u6557\u3057\u307e\u3057\u305f: {exc}")
        st.stop()

    examples = [
        "L12\u578b\u5316\u5408\u7269\u306e\u683c\u5b50\u5b9a\u6570\u3092\u4e00\u89a7\u306b\u3057\u3066\u3002",
        "Ni\u3092\u542b\u3080\u5b89\u5b9a\u306a\u5316\u5408\u7269\u3092\u62bd\u51fa\u3057\u3066\u3002",
        "\u4f53\u7a4d\u5f3e\u6027\u7387\u304c150 GPa\u4ee5\u4e0a\u306e\u5316\u5408\u7269\u3092\u8868\u793a\u3057\u3066\u3002",
    ]
    def _apply_example() -> None:
        chosen = st.session_state["example_select"]
        if chosen != "(\u81ea\u7531\u5165\u529b)":
            st.session_state["question_input"] = chosen

    st.selectbox(
        "\u8cea\u554f\u4f8b",
        ["(\u81ea\u7531\u5165\u529b)"] + examples,
        key="example_select",
        on_change=_apply_example,
    )
    question = st.text_area(
        "\u8cea\u554f\uff08\u65e5\u672c\u8a9e\uff09", key="question_input", height=80
    )

    if st.button("SQL\u3092\u751f\u6210", type="primary") and question.strip():
        t0 = time.perf_counter()
        with st.spinner("SQL\u751f\u6210\u4e2d\u2026"):
            result = pipeline(
                question.strip(),
                join_list=ctx["join_list"],
                all_columns=ctx["all_columns"],
                n_best=n_best,
                execute_fn=execute_sql if n_best > 1 else None,
                table_graph=ctx["table_graph"],
            )
        gen_sec = time.perf_counter() - t0
        st.session_state["gen"] = {
            "question": question.strip(),
            "result": result,
            "gen_sec": gen_sec,
            "exec_result": None,
            "auto_exec_done": False,
        }

    gen = st.session_state.get("gen")
    if gen is None:
        return
    if gen["question"] != question.strip():
        # Question changed since generation: stale results are dropped.
        del st.session_state["gen"]
        return

    result = gen["result"]
    if result.get("mode") == "rejected":
        st.warning(f"\u5165\u529b\u304c\u62d2\u5426\u3055\u308c\u307e\u3057\u305f: {result.get('reason')}")
        return

    sql = result.get("sql", "")
    st.subheader("\u751f\u6210SQL")
    st.code(sql, language="sql")
    st.caption(f"\u751f\u6210\u6642\u9593: {gen['gen_sec']:.1f} \u79d2")

    validation = validate_sql(sql)
    if validation["valid"]:
        st.success("SQLGuard\u691c\u67fb: OK")
    else:
        st.error(f"SQLGuard\u691c\u67fb: NG \u2014 {validation['errors']}")
        return

    run_now = (auto_execute and not gen["auto_exec_done"]) or st.button(
        "\u5b9f\u884c\uff08read-only\uff09"
    )
    if run_now:
        with st.spinner("\u5b9f\u884c\u4e2d\uff08read-only\uff09\u2026"):
            gen["exec_result"] = execute_sql(sql)
        gen["auto_exec_done"] = True

    exec_result = gen["exec_result"]
    if exec_result is None:
        return
    if not exec_result.get("success"):
        st.error(f"\u5b9f\u884c\u30a8\u30e9\u30fc: {exec_result.get('errors')}")
        return
    rows = exec_result.get("rows", [])
    cols = exec_result.get("columns", [])
    st.subheader(f"\u5b9f\u884c\u7d50\u679c\uff08{len(rows)} \u884c\uff09")
    if rows:
        st.dataframe(
            pd.DataFrame(rows, columns=cols).head(int(row_limit)),
            use_container_width=True,
        )
    else:
        diag = exec_result.get("empty_diagnosis")
        st.info(f"0\u4ef6\u3067\u3057\u305f\u3002\u8a3a\u65ad: {diag}")


main()
