"""Phase 1: 仮説の構造化・Falsifier・3値判定の受入テスト。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mi_hub.agent.judgement import judge_hypothesis
from mi_hub.agent.loop import ResearchManager, SessionStore
from mi_hub.agent.models import Hypothesis, SessionState
from mi_hub.agent.states import HypothesisState, TaskState
from mi_hub.agent.tools import ToolGateway

GOAL = (
    "Ni-Al B2相において、Alアンチサイトが相安定性低下の主要因であるか、"
    "登録済みモデル群を用いて検証する。"
)


@pytest.fixture()
def manager(tmp_path):
    return ResearchManager(gateway=ToolGateway(), store=SessionStore(str(tmp_path)))


@pytest.fixture()
def session(manager):
    state = manager.create_session(GOAL)
    manager.generate_plan(state)
    return state


def _run_full_cycle(manager, session):
    manager.run_auto(session)
    approval = next(a for a in session.approvals if a.status == "pending")
    manager.resolve_approval(session, approval.approval_id, True)
    manager.run_auto(session)


# ---------- 仮説の構造化 ----------
class TestStructuredHypothesis:
    def test_hypothesis_has_structured_fields(self, manager, session):
        manager.run_auto(session)
        h = session.hypotheses[0]
        assert h.falsification_conditions, "反証条件が必須"
        assert h.source_evidence, "根拠証拠IDが紐付く"
        assert isinstance(h.applicability, dict)

    def test_backward_compat_old_session_json(self, tmp_path):
        """新フィールドが無い旧セッションJSONも読み込める。"""
        old = {
            "session_id": "SESSION-old00001",
            "hypotheses": [{"hypothesis_id": "H-old", "statement": "旧仮説"}],
        }
        state = SessionState.model_validate(old)
        h = state.hypotheses[0]
        assert h.mechanism == ""
        assert h.counter_evidence == []
        assert h.judgement is None


# ---------- Falsifier ----------
class TestFalsifier:
    def test_plan_contains_falsifier_task(self, session):
        assert any(t.agent == "FalsifierAgent" for t in session.plan.tasks)

    def test_falsifier_collects_counter_evidence(self, manager, session):
        manager.run_auto(session)
        t = next(t for t in session.plan.tasks if t.agent == "FalsifierAgent")
        assert t.status == TaskState.COMPLETED
        assert t.result["hypotheses_reviewed"]
        h = session.hypotheses[0]
        assert h.counter_evidence, "反対・条件依存の証拠が収集される"
        assert h.alternative_mechanisms, "別機構の候補が提示される"

    def test_falsifier_respects_approved_conditions(self, manager, session):
        """研究者承認済みの反証条件は Falsifier が書き換えない。"""
        manager.run_auto(session)
        h = session.hypotheses[0]
        manager.update_falsification_conditions(session, h.hypothesis_id,
                                                ["固定条件"], by="human")
        h.falsification_approved = True
        before = list(h.falsification_conditions)
        from mi_hub.agent.models import Task
        from mi_hub.agent.roles import FalsifierAgent

        FalsifierAgent().run(session, manager.gateway, Task(agent="FalsifierAgent"))
        assert h.falsification_conditions == before


# ---------- 3値判定（ルール評価） ----------
def _h(direction="positive"):
    return Hypothesis(statement="テスト仮説",
                      applicability={"expected_direction": direction})


class TestJudgementRules:
    def test_supported(self):
        j = judge_hypothesis(_h(), slope=1.0, mean_uncertainty=0.01,
                             n_points=5, n_independent_groups=2)
        assert j.verdict == "supported"
        assert all(c.passed for c in j.criteria)

    def test_refuted_on_opposite_direction(self):
        j = judge_hypothesis(_h(), slope=-1.0, mean_uncertainty=0.01,
                             n_points=5, n_independent_groups=2)
        assert j.verdict == "refuted"

    def test_inconclusive_when_not_significant(self):
        j = judge_hypothesis(_h(), slope=0.01, mean_uncertainty=1.0,
                             n_points=5, n_independent_groups=2)
        assert j.verdict == "inconclusive"
        assert "効果の有意性" in j.rationale

    def test_inconclusive_without_reproduction(self):
        j = judge_hypothesis(_h(), slope=1.0, mean_uncertainty=0.01,
                             n_points=5, n_independent_groups=1)
        assert j.verdict == "inconclusive"

    def test_negative_direction_hypothesis(self):
        j = judge_hypothesis(_h("negative"), slope=-1.0, mean_uncertainty=0.01,
                             n_points=5, n_independent_groups=2)
        assert j.verdict == "supported"

    def test_rationale_defers_to_human(self):
        j = judge_hypothesis(_h(), slope=1.0, mean_uncertainty=0.01,
                             n_points=5, n_independent_groups=2)
        assert "研究者" in j.rationale


# ---------- 判定案の生成と確定（Human-in-the-loop） ----------
class TestJudgementFlow:
    def test_evaluation_attaches_judgement(self, manager, session):
        _run_full_cycle(manager, session)
        main = next(h for h in session.hypotheses if h.counter_to is None)
        assert main.judgement is not None
        assert main.judgement.verdict in ("supported", "refuted", "inconclusive")
        assert main.judgement.decided_by == "rule"
        assert not main.judgement.confirmed_by_human

    def test_judgement_confirmation_asked_in_chat(self, manager, session):
        _run_full_cycle(manager, session)
        assert any("【判定案】" in msg["content"] and "確定しますか" in msg["content"]
                   for msg in session.chat_history if msg["role"] == "assistant")

    def test_confirm_judgement_sets_status(self, manager, session):
        _run_full_cycle(manager, session)
        main = next(h for h in session.hypotheses if h.counter_to is None)
        manager.confirm_judgement(session, main.hypothesis_id, accept=True)
        expected = {"supported": HypothesisState.SUPPORTED,
                    "refuted": HypothesisState.FALSIFIED,
                    "inconclusive": HypothesisState.INCONCLUSIVE}
        assert main.status == expected[main.judgement.verdict]
        assert main.judgement.confirmed_by_human
        assert session.audit_log[-1].action == "judgement_confirmed"

    def test_defer_judgement_keeps_status(self, manager, session):
        _run_full_cycle(manager, session)
        main = next(h for h in session.hypotheses if h.counter_to is None)
        before = main.status
        manager.confirm_judgement(session, main.hypothesis_id, accept=False)
        assert main.status == before
        assert not main.judgement.confirmed_by_human

    def test_confirm_requires_human(self, manager, session):
        _run_full_cycle(manager, session)
        main = next(h for h in session.hypotheses if h.counter_to is None)
        with pytest.raises(PermissionError):
            manager.confirm_judgement(session, main.hypothesis_id,
                                      accept=True, by="agent")

    def test_reevaluation_keeps_confirmed_judgement(self, manager, session):
        """研究者が確定した判定は再評価で上書きされない。"""
        _run_full_cycle(manager, session)
        main = next(h for h in session.hypotheses if h.counter_to is None)
        manager.confirm_judgement(session, main.hypothesis_id, accept=True)
        confirmed = main.judgement
        from mi_hub.agent.models import Task
        from mi_hub.agent.roles import EvaluationAgent

        exec_task = next(t for t in session.plan.tasks
                         if t.action == "run_models_bulk")
        task = Task(agent="EvaluationAgent", action="evaluate_hypothesis",
                    inputs={"results": exec_task.result["results"]})
        res = EvaluationAgent().run(session, manager.gateway, task)
        assert main.judgement is confirmed
        assert main.judgement.confirmed_by_human
        assert main.hypothesis_id not in res["judgements"]

    def test_falsification_review_string_fields(self, monkeypatch):
        """LLM が文字列で返しても一文字に分解されない。"""
        from mi_hub.agent import llm

        monkeypatch.setattr(llm, "_chat_json", lambda *a, **k: {
            "counter_queries": ["q1"],
            "falsification_conditions": "独立系列が逆傾向を示す",
            "alternative_mechanisms": "別機構",
        })
        out = llm.falsification_review("goal", "仮説", [])
        assert out["falsification_conditions"] == ["独立系列が逆傾向を示す"]
        assert out["alternative_mechanisms"] == ["別機構"]

    def test_confirm_without_judgement_raises(self, manager, session):
        manager.run_auto(session)
        h = session.hypotheses[0]
        assert h.judgement is None
        with pytest.raises(ValueError):
            manager.confirm_judgement(session, h.hypothesis_id, accept=True)
