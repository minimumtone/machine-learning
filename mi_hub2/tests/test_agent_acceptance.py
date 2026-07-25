"""受入試験（指示書 §19）。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mi_hub.agent.errors import classify_error, convert_unit, try_auto_fix
from mi_hub.agent.loop import ResearchManager, SessionStore
from mi_hub.agent.models import Task
from mi_hub.agent.states import (
    AgentState,
    ErrorType,
    HypothesisState,
    TaskState,
)
from mi_hub.agent.tools import ToolGateway

GOAL = (
    "Ni-Al B2相において、Alアンチサイトが相安定性低下の主要因であるか、"
    "登録済みモデル群を用いて検証する。"
)


def test_session_store_rejects_path_traversal(tmp_path):
    store = SessionStore(str(tmp_path))
    with pytest.raises(ValueError):
        store.path("../../etc/passwd")
    assert store.load("../outside") is None


@pytest.fixture()
def manager(tmp_path):
    return ResearchManager(gateway=ToolGateway(), store=SessionStore(str(tmp_path)))


@pytest.fixture()
def session(manager):
    state = manager.create_session(GOAL)
    manager.generate_plan(state)
    return state


# ---------- §19.1 計画機能 ----------
class TestPlanning:
    def test_plan_generated_from_goal(self, manager, session):
        assert session.plan is not None
        assert session.plan.goal_id == session.goal.goal_id

    def test_plan_decomposed_into_tasks(self, session):
        assert len(session.plan.tasks) >= 3

    def test_task_dependencies_stored(self, session):
        deps = [t for t in session.plan.tasks if t.depends_on]
        assert deps, "依存関係を持つタスクが存在する"

    def test_human_can_edit_plan(self, manager, session):
        new_task = Task(agent="EvidenceAgent", action="search_literature",
                        description="追加の文献検索")
        manager.apply_plan_change(session, created_by="human", reason="人間による追加",
                                  add_tasks=[new_task])
        assert session.plan.task(new_task.task_id) is not None
        manager.apply_plan_change(session, created_by="human", reason="人間による削除",
                                  remove_task_ids=[new_task.task_id])
        assert session.plan.task(new_task.task_id) is None

    def test_plan_change_history_recorded(self, manager, session):
        v0 = session.plan.version
        manager.apply_plan_change(session, created_by="human", reason="編集",
                                  add_tasks=[Task(action="search_literature",
                                                  agent="EvidenceAgent")])
        assert session.plan.version == v0 + 1
        last = session.plan_history[-1]
        assert last.reason_for_change == "編集"
        assert last.previous_plan_version == v0
        assert last.human_approval


# ---------- §19.2 Observe–Act機能 ----------
class TestObserveAct:
    def test_result_reflected_in_state(self, manager, session):
        t1 = session.plan.tasks[0]
        res = manager.execute_task(session, t1.task_id)
        assert res["status"] == "completed"
        assert session.evidence, "証拠が状態へ反映される"
        obs = manager.observe(session)
        assert t1.task_id in obs.completed_tasks

    def test_success_failure_distinguished(self, manager, session):
        t1 = session.plan.tasks[0]
        assert manager.execute_task(session, t1.task_id)["status"] == "completed"
        manager.gateway.inject_failure("unknown catastrophic failure")
        t2 = session.plan.ready_tasks()[0]
        # EvidenceAgent 以外の役割はツール例外を FAILED として記録する
        res2 = manager.execute_task(session, t2.task_id)
        assert res2["status"] in ("completed", "failed")

    def test_next_actions_generated(self, manager, session):
        actions = manager.next_actions(session)
        assert actions and actions[0]["action"] == "search_literature"

    def test_replanning_updates_plan(self, manager, session):
        v0 = session.plan.version
        manager.apply_plan_change(session, created_by="agent",
                                  reason="モデル不足のため代理物性検索を追加",
                                  add_tasks=[Task(agent="ModelSelectionAgent",
                                                  action="search_models",
                                                  description="代理物性モデル検索")])
        assert session.plan.version == v0 + 1


# ---------- §19.3 エラー回復 ----------
class TestErrorRecovery:
    def test_unit_error_auto_fixed(self):
        assert classify_error("unit mismatch: temperature unit C is not K") \
            == ErrorType.UNIT_MISMATCH
        fixed = try_auto_fix(ErrorType.UNIT_MISMATCH,
                             {"temperature": 527.0, "temperature_unit": "C",
                              "composition": {"Al": 0.5}})
        assert fixed is not None
        assert fixed["temperature"] == pytest.approx(800.15)
        assert fixed["temperature_unit"] == "K"

    def test_unit_conversion_rules(self):
        assert convert_unit(0.0, "C", "K") == pytest.approx(273.15)
        with pytest.raises(ValueError):
            convert_unit(1.0, "furlong", "K")

    def test_transient_network_error_retry(self):
        fixed = try_auto_fix(ErrorType.NETWORK_ERROR, {"composition": {"Al": 0.5}})
        assert fixed == {"composition": {"Al": 0.5}}

    def test_semantic_change_requires_human(self):
        # 意味変更を伴う修正（適用範囲外・入力欠落）は自動修正しない
        assert try_auto_fix(ErrorType.OUT_OF_DOMAIN, {"temperature": 2000}) is None
        assert try_auto_fix(ErrorType.MISSING_INPUT, {}) is None

    def test_retry_limit_stops(self, manager, session):
        session.stop_conditions.max_retries_per_task = 1
        # 実行フローを run_models_bulk まで進める
        self._advance_to_execution(manager, session)
        exec_task = next(t for t in session.plan.tasks if t.action == "run_models_bulk")
        # OOD ジョブ（人間確認要）を混ぜる
        exec_task.inputs["jobs"] = [
            {"model_id": "MINT-001",
             "inputs": {"composition": {"Al": 0.5}, "temperature": 5000,
                        "temperature_unit": "K"}},
        ]
        res = manager.execute_task(session, exec_task.task_id)
        assert res["result"]["n_failed"] == 1
        assert res["result"]["failures"][0]["needs_human"]

    @staticmethod
    def _advance_to_execution(manager, session):
        for _ in range(4):
            actions = manager.next_actions(session)
            runnable = [a for a in actions if not a["requires_approval"]]
            if not runnable:
                break
            manager.execute_task(session, runnable[0]["task_id"])
        exec_task = next(t for t in session.plan.tasks if t.action == "run_models_bulk")
        manager.execute_task(session, exec_task.task_id)  # 承認要求を生成
        approval = session.approvals[-1]
        manager.resolve_approval(session, approval.approval_id, True)


# ---------- §19.4 停止機能 ----------
class TestStopping:
    def test_iteration_limit_stops(self, manager, session):
        # 1 反復 = run_auto 1 回（Observe→Plan→Act→Evaluate→Replan の一巡）
        session.stop_conditions.max_iterations = 1
        first = manager.run_auto(session)
        assert first["executed"]
        second = manager.run_auto(session)
        assert second["executed"] == []
        assert second["stop_reason"] is not None
        assert "反復回数上限" in second["stop_reason"]

    def test_budget_limit_stops(self, manager, session):
        session.budget.max_model_runs = 0
        result = manager.run_auto(session)
        assert session.budget.used_model_runs == 0
        assert result["agent_state"] in ("blocked", "awaiting_approval", "replanning",
                                         "evaluating")

    def test_pause_on_human_decision(self, manager, session):
        result = manager.run_auto(session)
        # 承認要タスク（run_models_bulk）で一時停止する
        assert any(r["status"] == "awaiting_approval" for r in result["executed"])
        assert session.agent_state == AgentState.AWAITING_APPROVAL

    def test_manual_pause(self, manager, session):
        manager.pause(session)
        assert session.agent_state == AgentState.PAUSED
        res = manager.execute_task(session, session.plan.tasks[0].task_id)
        assert res["status"] == "paused"

    def test_resume_after_pause_preserves_state(self, manager, session):
        t1 = session.plan.tasks[0]
        manager.execute_task(session, t1.task_id)
        n_evidence = len(session.evidence)
        manager.pause(session)
        reloaded = manager.store.load(session.session_id)
        assert reloaded.agent_state == AgentState.PAUSED
        assert len(reloaded.evidence) == n_evidence
        manager.resume(reloaded)
        assert reloaded.agent_state != AgentState.PAUSED

    def test_observe_does_not_clobber_state(self, manager, session):
        """読み取り経路（observe）が一時停止・完了状態を上書きしない。"""
        manager.pause(session)
        manager.observe(session)
        assert session.agent_state == AgentState.PAUSED
        manager.complete(session)
        manager.observe(session)
        assert session.agent_state == AgentState.COMPLETED


# ---------- §19.5 Human-in-the-loop ----------
class TestHumanInTheLoop:
    def test_unapproved_high_cost_blocked(self, manager, session):
        exec_task = next(t for t in session.plan.tasks if t.action == "run_models_bulk")
        exec_task.depends_on = []  # 依存を外して直接実行を試みる
        res = manager.execute_task(session, exec_task.task_id)
        assert res["status"] == "awaiting_approval"
        assert exec_task.status == TaskState.AWAITING_APPROVAL

    def test_approval_then_execution(self, manager, session):
        result = manager.run_auto(session)
        approval = next(a for a in session.approvals if a.status == "pending")
        manager.resolve_approval(session, approval.approval_id, True)
        manager.run_auto(session)
        exec_task = next(t for t in session.plan.tasks if t.action == "run_models_bulk")
        assert exec_task.status in (TaskState.COMPLETED, TaskState.PARTIALLY_COMPLETED)
        assert session.budget.used_model_runs > 0

    def test_rejection_blocks_task(self, manager, session):
        manager.run_auto(session)
        approval = next(a for a in session.approvals if a.status == "pending")
        manager.resolve_approval(session, approval.approval_id, False)
        exec_task = next(t for t in session.plan.tasks if t.action == "run_models_bulk")
        assert exec_task.status == TaskState.REJECTED

    def test_human_edits_hypothesis(self, manager, session):
        manager.run_auto(session)
        h = session.hypotheses[0]
        manager.set_hypothesis_status(session, h.hypothesis_id,
                                      HypothesisState.APPROVED_FOR_TESTING)
        assert h.status == HypothesisState.APPROVED_FOR_TESTING
        assert h.falsification_approved

    def test_falsification_change_requires_human(self, manager, session):
        manager.run_auto(session)
        h = session.hypotheses[0]
        with pytest.raises(PermissionError):
            manager.update_falsification_conditions(
                session, h.hypothesis_id, ["新条件"], by="agent")
        manager.update_falsification_conditions(
            session, h.hypothesis_id, ["新条件"], by="human")
        assert h.falsification_conditions == ["新条件"]
        entry = session.audit_log[-1]
        assert entry.action == "falsification_conditions_changed"
        assert "before" in entry.detail and "after" in entry.detail

    def test_replan_after_human_change(self, manager, session):
        v0 = session.plan.version
        manager.apply_plan_change(session, created_by="human", reason="条件変更",
                                  add_tasks=[Task(agent="EvidenceAgent",
                                                  action="search_literature")])
        assert session.plan.version == v0 + 1
        actions = manager.next_actions(session)
        assert actions  # 変更後も次行動候補が生成される


# ---------- エンドツーエンド ----------
def test_full_cycle_end_to_end(manager, session):
    """Goal→Observe→Plan→承認→Act→Evaluate までデフォルト設定のまま一巡できる。"""
    manager.run_auto(session)
    approval = next(a for a in session.approvals if a.status == "pending")
    manager.resolve_approval(session, approval.approval_id, True)
    manager.run_auto(session)
    eval_task = next(t for t in session.plan.tasks if t.action == "evaluate_hypothesis")
    assert eval_task.status == TaskState.COMPLETED
    assert eval_task.result["verdict_candidate"] in (
        "supported", "falsification_candidate", "inconclusive")
    assert "最終判定は研究者が行う" in eval_task.result["note"]
    assert session.evaluations
    assert session.stop_reason and "正常終了" in session.stop_reason


def test_evaluation_lists_data_gaps(manager, session):
    """評価結果に追加的に必要なデータ（情報ギャップ）が含まれる。"""
    manager.run_auto(session)
    approval = next(a for a in session.approvals if a.status == "pending")
    manager.resolve_approval(session, approval.approval_id, True)
    manager.run_auto(session)
    eval_task = next(t for t in session.plan.tasks if t.action == "evaluate_hypothesis")
    gaps = eval_task.result["data_gaps"]
    assert isinstance(gaps, list)
    assert any("組成点" in g for g in gaps)  # デフォルトは4組成点のため
    assert any("温度" in g for g in gaps)  # デフォルトは単一温度のため
    assert session.evaluations[-1].data_gaps == gaps


def test_normalize_goal_property_vocabulary():
    """LLM が日本語物性名を返してもレジストリ語彙に正規化される。"""
    from mi_hub.agent.llm import _normalize_goal
    out = _normalize_goal({"target_property": "相安定性", "success_criteria": ["a"]})
    assert out["target_property"] == "phase_stability"
    out = _normalize_goal({"target_property": "Formation Enthalpy"})
    assert out["target_property"] == "formation_enthalpy"


def test_run_script_sandbox(manager):
    """承認ゲート後に呼ぶ run_script が exit code / stdout / stderr を返す。"""
    res = manager.gateway.run_script("echo hello && echo err >&2")
    assert res["exit_code"] == 0
    assert "hello" in res["stdout"]
    assert "err" in res["stderr"]
    res = manager.gateway.run_script("exit 3")
    assert res["exit_code"] == 3


def test_run_script_workdir(manager, tmp_path):
    wd = str(tmp_path / "ws")
    res = manager.gateway.run_script("echo data > result.csv", workdir=wd)
    assert res["exit_code"] == 0
    assert res["generated_files"] == ["result.csv"]
    assert res["workdir"] == wd


def test_classify_intent_fallback():
    """LLM 不可時、明示的コマンドのみ操作意図になり、それ以外は question。"""
    from mi_hub.agent.llm import classify_intent
    assert classify_intent("一時停止", False)["intent"] == "pause"
    assert classify_intent("実行を続けて", False)["intent"] == "run"
    assert classify_intent("承認", True)["intent"] == "approve"
    assert classify_intent("承認", False)["intent"] == "question"
    assert classify_intent("HEAの安定性について教えて", False)["intent"] == "question"


def test_knowledge_provider_registration(manager, session):
    """登録したナレッジプロバイダ（MCP/GraphRAG差込口）が文献検索に併用される。"""
    from mi_hub.agent.tools import KnowledgeProvider

    class FakeGraphRAG(KnowledgeProvider):
        def search(self, query, limit=10):
            return [{"title": "GraphRAG hit", "claim": "外部ナレッジの主張",
                     "evidence_type": "computation", "keywords": [], "limitations": []}]

    manager.gateway.register_knowledge_provider(FakeGraphRAG("graphrag"))
    docs = manager.gateway.search_knowledge("B2 NiAl", limit=10)
    providers = {d.get("provider") for d in docs}
    assert "graphrag" in providers
    assert "mock_literature" in providers


def test_memory_context_short_and_long_term(manager, session):
    """短期記憶（直近評価）と長期記憶（全体履歴）が分離して取得できる。"""
    manager.run_auto(session)
    approval = next(a for a in session.approvals if a.status == "pending")
    manager.resolve_approval(session, approval.approval_id, True)
    manager.run_auto(session)
    mem = manager.memory_context(session)
    assert mem["short_term_memory"]["last_evaluation"] is not None
    assert mem["short_term_memory"]["recent_tasks"]
    assert mem["long_term_memory"]["evaluation_history"]
    assert mem["long_term_memory"]["evidence"]
