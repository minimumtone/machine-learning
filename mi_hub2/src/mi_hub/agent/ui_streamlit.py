"""チャットUI + Agentタブ（指示書 §13）。

起動:  streamlit run src/mi_hub/agent/ui_streamlit.py
"""

from __future__ import annotations

import os
import time

import pandas as pd
import streamlit as st

from mi_hub.agent import llm
from mi_hub.agent.graphrag import GraphRAGProvider, build_default_provider
from mi_hub.agent.loop import ResearchManager
from mi_hub.agent.models import ApprovalRequest, Evidence, SessionState
from mi_hub.agent.states import HypothesisState

APPROVAL_KIND_LABELS = {
    "task_execution": "タスク実行",
    "script_execution": "スクリプト実行",
}


def execute_script_approval(manager: ResearchManager, s: SessionState,
                            approval: ApprovalRequest) -> str:
    """承認済みスクリプトを実行し、結果を証拠と監査ログに記録して応答文を返す。"""
    script = approval.payload.get("script", "")
    workdir = manager.session_workspace(s)
    res = manager.gateway.run_script(script, workdir=workdir)
    s.evidence.append(Evidence(
        source_type="script_execution",
        claim=f"スクリプト実行（exit code {res['exit_code']}）: {approval.description}",
        conditions={
            "approval_id": approval.approval_id,
            "stdout": res["stdout"][-2000:],
            "workdir": workdir,
            "generated_files": res["generated_files"],
        },
        evidence_type="computation",
        limitations=["サンドボックス実行結果。再現性はスクリプト本文を参照"],
    ))
    s.audit("human", "script_executed", approval_id=approval.approval_id,
            exit_code=res["exit_code"])
    reply = f"スクリプトを実行しました（exit code {res['exit_code']}）。"
    if res["stdout"].strip():
        reply += f"\n\n出力:\n```\n{res['stdout'].strip()[-3000:]}\n```"
    if res["stderr"].strip():
        reply += f"\n\nstderr:\n```\n{res['stderr'].strip()[-2000:]}\n```"
    if res["generated_files"]:
        files = "\n".join(f"- {workdir}/{f}" for f in res["generated_files"])
        reply += f"\n\n生成ファイル:\n{files}"
    reply += "\n\n実行結果は証拠タブに記録しました。"
    return reply


def pending_approval_summary(s: SessionState) -> str:
    lines = []
    for a in s.approvals:
        if a.status == "pending":
            kind = APPROVAL_KIND_LABELS.get(a.kind, a.kind)
            lines.append(f"- [{kind}] {a.description}（{a.approval_id}）")
    return "\n".join(lines)

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
    manager = ResearchManager()
    graphrag_dir = os.path.join(str(manager.store.base_dir), "graphrag")
    manager.gateway.register_knowledge_provider(build_default_provider(graphrag_dir))
    return manager


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
    if st.session_state.get("sid"):
        cur = m.store.load(st.session_state["sid"])
        if cur is not None and st.button("事例レポートを書き出す"):
            report_path = m.export_case_report(cur)
            st.success(f"書き出しました: {report_path}")
            with open(report_path, encoding="utf-8") as rf:
                st.download_button(
                    "レポートをダウンロード", rf.read(),
                    file_name=f"case_report_{cur.session_id}.md",
                )

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
        pending = [a for a in state.approvals if a.status == "pending"]
        intent_info = llm.classify_intent(user_msg, bool(pending))
        intent = intent_info["intent"]
        if intent == "pause":
            m.pause(state)
            reply = "セッションを一時停止しました。"
        elif intent == "resume":
            m.resume(state)
            reply = "セッションを再開しました。"
        elif intent == "complete":
            m.complete(state)
            reply = "セッションを終了しました。"
        elif intent == "run":
            result = m.run_auto(state)
            n = len(result["executed"])
            reply = (
                f"{n} 件のタスクを実行しました。現在の状態: {result['agent_state']}"
                + (f"（{result['stop_reason']}）" if result["stop_reason"] else "")
            )
            if state.agent_state.value == "awaiting_approval":
                reply += (
                    "\n\n**承認待ちの操作:**\n" + pending_approval_summary(state)
                    + "\n\nチャットで「承認」/「却下」と入力するか、承認タブから操作できます。"
                )
            if state.evaluations and state.evaluations[-1].data_gaps:
                gaps = "\n".join(f"- {g}" for g in state.evaluations[-1].data_gaps)
                reply += f"\n\n**追加的に必要なデータ:**\n{gaps}"
        elif intent in ("approve", "reject"):
            if not pending:
                reply = "現在、承認待ちの操作はありません。"
            elif len(pending) > 1:
                reply = (
                    "承認待ちが複数あります。承認タブから個別に操作してください。\n\n"
                    + pending_approval_summary(state)
                )
            else:
                a = pending[0]
                approve = intent == "approve"
                m.resolve_approval(state, a.approval_id, approve)
                kind = APPROVAL_KIND_LABELS.get(a.kind, a.kind)
                if approve and a.kind == "script_execution":
                    reply = f"[{kind}] {a.description} を承認しました。\n\n"
                    reply += execute_script_approval(m, state, a)
                elif approve:
                    reply = (
                        f"[{kind}] {a.description} を承認しました。"
                        "\n\n「実行を続けて」で承認済みタスクを実行できます。"
                    )
                else:
                    reply = f"[{kind}] {a.description} を却下しました。"
        elif intent == "script" and intent_info.get("script"):
            script = intent_info["script"]
            req = ApprovalRequest(
                kind="script_execution",
                description=(intent_info.get("reason") or user_msg)[:80],
                payload={"script": script},
            )
            state.approvals.append(req)
            state.audit("human_chat", "script_proposed", approval_id=req.approval_id)
            reply = (
                "以下のスクリプトを提案します（未実行）。内容を確認の上、"
                "チャットで「承認」と入力するか承認タブから承認すると実行されます。\n\n"
                f"```bash\n{script}\n```"
            )
        else:
            # 自由対話（LLM）。実行や承認は行わず、議論・説明・案内のみ。
            obs = m.observe(state)
            context = {
                "agent_state": state.agent_state.value,
                "observation": obs.model_dump(),
                "plan": [
                    {"task": t.task_id, "agent": t.agent, "action": t.action,
                     "status": t.status.value, "description": t.description}
                    for t in (state.plan.tasks if state.plan else [])
                ],
                "pending_approvals": [
                    {"id": a.approval_id, "kind": a.kind, "description": a.description}
                    for a in state.approvals if a.status == "pending"
                ],
                "stop_reason": state.stop_reason,
                **m.memory_context(state),
            }
            reply = llm.chat_reply(context, state.chat_history[:-1], user_msg)
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

    pending_now = [a for a in state.approvals if a.status == "pending"]
    if pending_now:
        st.error(
            f"承認待ち {len(pending_now)} 件 — 承認タブで内容を確認してください：\n\n"
            + pending_approval_summary(state)
        )

    if state.evaluations:
        with st.expander("記憶（短期: 直近評価 / 長期: 全体履歴）"):
            mem = m.memory_context(state)
            st.markdown("**短期記憶（直近の計算の妥当性）**")
            last = mem["short_term_memory"]["last_evaluation"]
            if last:
                st.write(f"- 結果: {last['step_result']} / 品質: {last['result_quality']}")
                st.write(f"- 進捗: {last['goal_progress_before']:.0%} → {last['goal_progress_after']:.0%}")
            st.markdown("**長期記憶（全体としての妥当性の材料）**")
            st.write(f"- 評価履歴: {len(mem['long_term_memory']['evaluation_history'])} 件")
            st.write(f"- 証拠: {len(mem['long_term_memory']['evidence'])} 件 / "
                     f"計画改訂: {mem['long_term_memory']['plan_versions']} 回")

    if state.evaluations and state.evaluations[-1].data_gaps:
        with st.container(border=True):
            st.markdown("**追加的に必要なデータ**")
            for gap in state.evaluations[-1].data_gaps:
                st.write(f"- {gap}")

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

        graphrag = next(
            (p for p in m.gateway.knowledge_providers
             if isinstance(p, GraphRAGProvider)), None,
        )
        if graphrag is not None:
            st.markdown("---")
            st.markdown("**GraphRAG 新規仮説候補**（取込済み文献の知識グラフから）")
            if st.button("研究目標から仮説候補を生成", key="graphrag-hyp"):
                query = state.goal.statement if state.goal else ""
                st.session_state["graphrag_props"] = graphrag.propose_hypotheses(query)
            for p in st.session_state.get("graphrag_props", []):
                with st.container(border=True):
                    st.write(p["statement"])
                    st.caption(
                        "根拠文献: "
                        + " / ".join(d["title"] for d in p["supporting_docs"])
                        + f" — {p['note']}"
                    )
            if st.button("利用ログから辞書を更新", key="graphrag-dict"):
                added = graphrag.update_from_logs()
                if added:
                    st.success(f"辞書に追加: {', '.join(added)}")
                else:
                    st.info("追加すべき頻出未知語はありません。")

    with tabs[2]:
        pending = [a for a in state.approvals if a.status == "pending"]
        if not pending:
            st.caption("承認待ちはありません。")
        for a in pending:
            with st.container(border=True):
                kind = APPROVAL_KIND_LABELS.get(a.kind, a.kind)
                st.markdown(f"**[{kind}]** {a.description}")
                st.caption(
                    f"{a.approval_id}"
                    + (f" / task: {a.task_id}" if a.task_id else "")
                    + f" / 要求: {time.strftime('%H:%M:%S', time.localtime(a.requested_at))}"
                )
                if a.kind == "script_execution" and a.payload.get("script"):
                    st.code(a.payload["script"], language="bash")
                cols = st.columns(2)
                if cols[0].button("承認", key=f"ok-{a.approval_id}", type="primary"):
                    m.resolve_approval(state, a.approval_id, True)
                    if a.kind == "script_execution":
                        reply = execute_script_approval(m, state, a)
                        state.chat_history.append({"role": "assistant", "content": reply})
                    m.store.save(state)
                    st.rerun()
                if cols[1].button("却下", key=f"ng-{a.approval_id}"):
                    m.resolve_approval(state, a.approval_id, False)
                    st.rerun()
        resolved = [a for a in state.approvals if a.status != "pending"]
        if resolved:
            st.markdown("**承認履歴**")
            for a in reversed(resolved):
                kind = APPROVAL_KIND_LABELS.get(a.kind, a.kind)
                mark = "承認" if a.status == "approved" else "却下"
                when = (time.strftime("%m/%d %H:%M", time.localtime(a.resolved_at))
                        if a.resolved_at else "-")
                st.write(f"- [{mark}] [{kind}] {a.description}（{when} / {a.resolved_by}）")

    with tabs[3]:
        for e in state.evidence:
            st.write(f"- **{e.evidence_id}** [{e.evidence_type}] {e.claim}")
            workdir = e.conditions.get("workdir")
            files = e.conditions.get("generated_files") or []
            if e.conditions.get("stdout"):
                with st.expander(f"実行出力（{e.evidence_id}）"):
                    st.code(str(e.conditions["stdout"]))
            if workdir and files:
                for fname in files:
                    fpath = os.path.join(str(workdir), str(fname))
                    if str(fname).lower().endswith((".png", ".jpg", ".jpeg", ".svg")) \
                            and os.path.exists(fpath):
                        st.image(fpath, caption=fname)
                    else:
                        st.write(f"  - 成果物: {fpath}")

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
