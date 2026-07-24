"""チャットUI + Agentタブ（指示書 §13）。

起動:  streamlit run src/mi_hub/agent/ui_streamlit.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from mi_hub.agent import llm
from mi_hub.agent.loop import ResearchManager
from mi_hub.agent.models import SessionState
from mi_hub.agent.states import HypothesisState

st.set_page_config(page_title="MI-HUB2 研究エージェント", layout="wide")

# 日本語グリフを優先（CJKフォールバックで簡体字字形になるのを防ぐ）
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;500;700&display=swap');
    html, body, [class*="css"], [data-testid="stAppViewContainer"] * {
        font-family: "Noto Sans JP", "Hiragino Kaku Gothic ProN", "Hiragino Sans",
                     "Yu Gothic UI", "Yu Gothic", "Meiryo", sans-serif !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_manager() -> ResearchManager:
    return ResearchManager()


m = get_manager()

st.title("MI-HUB2 pinax2.0-pilot 研究エージェント")

# --- セッション選択 / 作成 ---
with st.sidebar:
    st.header("研究セッション")
    sessions = m.store.list_sessions()
    selected = st.selectbox("既存セッション", ["(新規)"] + sessions)
    goal_text = st.text_area(
        "研究目標",
        "Ni-Al B2相において、Alアンチサイトが相安定性低下の主要因であるか、"
        "登録済みモデル群を用いて検証する。",
    )
    if st.button("セッション開始", type="primary"):
        state = m.create_session(goal_text)
        m.generate_plan(state)
        st.session_state["sid"] = state.session_id
        st.rerun()
    if selected != "(新規)":
        st.session_state["sid"] = selected

sid = st.session_state.get("sid")
if not sid:
    st.info("左のサイドバーから研究セッションを開始してください。")
    st.stop()

state: SessionState | None = m.store.load(sid)
if state is None:
    st.error(f"セッションが見つかりません: {sid}")
    st.stop()

chat_col, agent_col = st.columns([3, 2])

with chat_col:
    st.subheader("チャット")
    for msg in state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    user_msg = st.chat_input("メッセージを入力（例: 実行を続けて / 一時停止 / 終了）")
    if user_msg:
        state.chat_history.append({"role": "user", "content": user_msg})
        if "一時停止" in user_msg:
            m.pause(state)
            reply = "セッションを一時停止しました。"
        elif "再開" in user_msg:
            m.resume(state)
            reply = "セッションを再開しました。"
        elif "終了" in user_msg:
            m.complete(state)
            reply = "セッションを終了しました。"
        elif any(k in user_msg for k in ("実行", "続けて", "進めて", "自動")):
            result = m.run_auto(state)
            n = len(result["executed"])
            reply = (
                f"{n} 件のタスクを実行しました。現在の状態: {result['agent_state']}"
                + (f"（{result['stop_reason']}）" if result["stop_reason"] else "")
            )
            if state.agent_state.value == "awaiting_approval":
                reply += "\n\n承認待ちの操作があります。Agentタブで承認してください。"
        else:
            # 自由対話（LLM）。実行や承認は行わず、説明・案内のみ。
            obs = m.observe(state)
            context = {
                "goal": state.goal.model_dump() if state.goal else None,
                "agent_state": state.agent_state.value,
                "observation": obs.model_dump(),
                "plan": [
                    {"task": t.task_id, "agent": t.agent, "action": t.action,
                     "status": t.status.value, "description": t.description}
                    for t in (state.plan.tasks if state.plan else [])
                ],
                "hypotheses": [
                    {"id": h.hypothesis_id, "statement": h.statement,
                     "status": h.status.value,
                     "falsification_conditions": h.falsification_conditions}
                    for h in state.hypotheses
                ],
                "pending_approvals": [
                    {"id": a.approval_id, "description": a.description}
                    for a in state.approvals if a.status == "pending"
                ],
                "errors": [e.message for e in state.errors if not e.resolved],
                "stop_reason": state.stop_reason,
            }
            reply = llm.chat_reply(context, state.chat_history, user_msg)
            if reply is None:
                reply = (
                    f"現在の状態: {state.agent_state.value}、"
                    f"進捗 {obs.goal_progress:.0%}、証拠 {obs.evidence_count} 件。"
                    + (f"停止理由: {state.stop_reason}。" if state.stop_reason else "")
                    + "\n\n「実行を続けて」でタスクを進められます。"
                    "承認・仮説判定はAgentペインの各タブから操作してください。"
                    "（自由対話には OPENAI_API_KEY の設定が必要です）"
                )
        state.chat_history.append({"role": "assistant", "content": reply})
        m.store.save(state)
        st.rerun()

with agent_col:
    st.subheader("Agent")
    obs = m.observe(state)  # 読み取り専用（agent_state は変更・保存しない）
    c1, c2, c3 = st.columns(3)
    c1.metric("状態", state.agent_state.value)
    c2.metric("進捗", f"{obs.goal_progress:.0%}")
    c3.metric("残り反復", obs.iterations_remaining)
    c1.metric("適用可能モデル", f"{obs.applicable_models}/{obs.available_models}")
    c2.metric("残り実行予算", obs.budget_remaining_runs)
    c3.metric("証拠数", obs.evidence_count)
    if state.stop_reason:
        st.warning(f"停止理由: {state.stop_reason}")

    tabs = st.tabs(["計画", "仮説", "承認", "証拠", "エラー", "履歴"])

    with tabs[0]:
        if state.plan:
            st.caption(
                f"plan {state.plan.plan_id} v{state.plan.version}"
                f"（変更理由: {state.plan.reason_for_change}）"
            )
            df = pd.DataFrame(
                [
                    {
                        "task": t.task_id,
                        "agent": t.agent,
                        "action": t.action,
                        "状態": t.status.value,
                        "依存": ",".join(t.depends_on),
                        "承認要": "要" if t.requires_approval else "-",
                        "説明": t.description,
                    }
                    for t in state.plan.tasks
                ]
            )
            st.dataframe(df, use_container_width=True, hide_index=True)
            for a in m.next_actions(state):
                label = f"実行: {a['action']} ({a['task_id']})"
                if st.button(label, key=f"run-{a['task_id']}"):
                    m.execute_task(state, a["task_id"])
                    st.rerun()
        else:
            if st.button("計画を生成"):
                m.generate_plan(state)
                st.rerun()

    with tabs[1]:
        for h in state.hypotheses:
            with st.expander(f"{h.hypothesis_id}: {h.statement[:40]}"):
                st.write("状態:", h.status.value)
                st.write("反証条件:", h.falsification_conditions)
                st.write("反証条件承認済:", h.falsification_approved)
                cols = st.columns(2)
                if cols[0].button("検証承認", key=f"appr-{h.hypothesis_id}"):
                    m.set_hypothesis_status(
                        state, h.hypothesis_id, HypothesisState.APPROVED_FOR_TESTING
                    )
                    st.rerun()
                if cols[1].button("却下", key=f"rej-{h.hypothesis_id}"):
                    m.set_hypothesis_status(
                        state, h.hypothesis_id, HypothesisState.REJECTED_BY_HUMAN
                    )
                    st.rerun()

    with tabs[2]:
        pending = [a for a in state.approvals if a.status == "pending"]
        if not pending:
            st.caption("承認待ちはありません。")
        for a in pending:
            st.write(f"{a.approval_id}: {a.description}（task: {a.task_id}）")
            cols = st.columns(2)
            if cols[0].button("承認", key=f"ok-{a.approval_id}"):
                m.resolve_approval(state, a.approval_id, True)
                st.rerun()
            if cols[1].button("却下", key=f"ng-{a.approval_id}"):
                m.resolve_approval(state, a.approval_id, False)
                st.rerun()

    with tabs[3]:
        for e in state.evidence:
            st.write(f"- **{e.evidence_id}** [{e.evidence_type}] {e.claim}")

    with tabs[4]:
        for err in state.errors:
            st.write(
                f"- {err.error_type.value}: {err.message}"
                f"（{'解決済' if err.resolved else '未解決'}）"
            )

    with tabs[5]:
        for c in state.plan_history:
            st.write(
                f"- v{c.version} by {c.created_by}: {c.reason_for_change} "
                f"(+{len(c.added_tasks)} / -{len(c.removed_tasks)})"
            )
