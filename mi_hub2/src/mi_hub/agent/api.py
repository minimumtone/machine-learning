"""研究エージェント API（指示書 §14）。

起動:  uvicorn mi_hub.agent.api:app --port 8800
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .loop import ResearchManager
from .models import SessionState, Task
from .states import HypothesisState

app = FastAPI(title="mi_hub research agent", version="0.1.0")
_manager = ResearchManager()


def get_manager() -> ResearchManager:
    return _manager


def _load(session_id: str) -> SessionState:
    state = get_manager().store.load(session_id)
    if state is None:
        raise HTTPException(404, f"session not found: {session_id}")
    return state


class GoalRequest(BaseModel):
    statement: str
    target_material: str | None = None
    target_phase: str | None = None
    target_property: str | None = None


class PlanPatch(BaseModel):
    reason: str = ""
    add_tasks: list[dict[str, Any]] = Field(default_factory=list)
    remove_task_ids: list[str] = Field(default_factory=list)
    reorder_task_ids: list[str] = Field(default_factory=list)


class ActionRequest(BaseModel):
    session_id: str
    task_id: str


class ResultRequest(BaseModel):
    session_id: str
    result: dict[str, Any] = Field(default_factory=dict)


class ApprovalRequestBody(BaseModel):
    session_id: str
    approve: bool
    by: str = "human"


class PlanCalculationRequest(BaseModel):
    session_id: str
    hypothesis_id: str | None = None
    hypothesis_text: str | None = None


class SQSRequest(BaseModel):
    session_id: str
    elements: list[str]
    concentrations: dict[str, float] | None = None
    prototype: str = "fcc"
    a0: float = 3.6
    max_size: int = 16
    n_steps: int = 10000


class CalculationJobRequest(BaseModel):
    session_id: str
    code: str
    elements: list[str]
    params: dict[str, Any] = Field(default_factory=dict)
    description: str = ""
    estimated_node_hours: float = 1.0


class AnalysisProposalRequest(BaseModel):
    session_id: str
    purpose: str
    script: str | None = None
    job_id: str | None = None


class AnalysisRunRequest(BaseModel):
    session_id: str
    max_fix_attempts: int = 2


class HypothesisPatch(BaseModel):
    session_id: str
    status: str | None = None
    falsification_conditions: list[str] | None = None
    by: str = "human"


# §14.1 研究目標作成
@app.post("/api/agent/goals")
def create_goal(req: GoalRequest) -> dict[str, Any]:
    m = get_manager()
    state = m.create_session(
        req.statement,
        target_material=req.target_material,
        target_phase=req.target_phase,
        target_property=req.target_property,
    )
    return {"session_id": state.session_id, "goal": state.goal.model_dump()}


@app.get("/api/agent/sessions")
def list_sessions() -> dict[str, Any]:
    return {"sessions": get_manager().store.list_sessions()}


# §14.2 現在状態取得
@app.get("/api/agent/sessions/{session_id}/state")
def get_state(session_id: str) -> dict[str, Any]:
    m = get_manager()
    state = _load(session_id)
    obs = m.observe(state)  # 読み取り専用（agent_state は変更・保存しない）
    return {
        "session_id": state.session_id,
        "agent_state": state.agent_state.value,
        "goal": state.goal.model_dump() if state.goal else None,
        "observation": obs.model_dump(),
        "plan": state.plan.model_dump() if state.plan else None,
        "hypotheses": [h.model_dump() for h in state.hypotheses],
        "evidence": [e.model_dump() for e in state.evidence],
        "approvals": [a.model_dump() for a in state.approvals],
        "errors": [e.model_dump() for e in state.errors],
        "budget": state.budget.model_dump(),
        "stop_conditions": state.stop_conditions.model_dump(),
        "stop_reason": state.stop_reason,
        "data_gaps": (state.evaluations[-1].data_gaps if state.evaluations else []),
        "plan_history": [c.model_dump() for c in state.plan_history],
        "memory": m.memory_context(state),
    }


# §14.3 計画生成
@app.post("/api/agent/sessions/{session_id}/plans")
def create_plan(session_id: str) -> dict[str, Any]:
    m = get_manager()
    state = _load(session_id)
    plan = m.generate_plan(state)
    return plan.model_dump()


# §14.4 計画更新
@app.patch("/api/agent/plans/{plan_id}")
def patch_plan(plan_id: str, req: PlanPatch, session_id: str) -> dict[str, Any]:
    m = get_manager()
    state = _load(session_id)
    if state.plan is None or state.plan.plan_id != plan_id:
        raise HTTPException(404, f"plan not found: {plan_id}")
    add = [Task.model_validate(t) for t in req.add_tasks]
    plan = m.apply_plan_change(
        state, created_by="human", reason=req.reason, add_tasks=add,
        remove_task_ids=req.remove_task_ids or None,
        reorder_task_ids=req.reorder_task_ids or None,
    )
    return plan.model_dump()


# §14.5 次行動候補取得
@app.get("/api/agent/sessions/{session_id}/next-actions")
def next_actions(session_id: str) -> dict[str, Any]:
    m = get_manager()
    state = _load(session_id)
    actions = m.next_actions(state)
    m.store.save(state)
    return {"next_actions": actions}


# §14.6 行動実行
@app.post("/api/agent/actions")
def execute_action(req: ActionRequest) -> dict[str, Any]:
    m = get_manager()
    state = _load(req.session_id)
    try:
        return m.execute_task(state, req.task_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.post("/api/agent/sessions/{session_id}/run-auto")
def run_auto(session_id: str) -> dict[str, Any]:
    m = get_manager()
    state = _load(session_id)
    return m.run_auto(state)


# §14.7 実行結果登録（外部計算結果の登録）
@app.post("/api/agent/actions/{action_id}/results")
def register_result(action_id: str, req: ResultRequest) -> dict[str, Any]:
    m = get_manager()
    state = _load(req.session_id)
    if state.plan is None:
        raise HTTPException(404, "plan not found")
    task = state.plan.task(action_id)
    if task is None:
        raise HTTPException(404, f"task not found: {action_id}")
    task.result = req.result
    from .states import TaskState

    task.status = TaskState.COMPLETED
    state.audit("external", "result_registered", task_id=action_id)
    m.store.save(state)
    return {"task_id": action_id, "status": task.status.value}


# §14.8 再計画
@app.post("/api/agent/sessions/{session_id}/replan")
def replan(session_id: str, req: PlanPatch) -> dict[str, Any]:
    m = get_manager()
    state = _load(session_id)
    if state.plan is None:
        plan = m.generate_plan(state)
        return plan.model_dump()
    add = [Task.model_validate(t) for t in req.add_tasks]
    plan = m.apply_plan_change(
        state, created_by="agent", reason=req.reason or "replanning",
        add_tasks=add, remove_task_ids=req.remove_task_ids or None,
        reorder_task_ids=req.reorder_task_ids or None,
    )
    return plan.model_dump()


# §14.9 一時停止
@app.post("/api/agent/sessions/{session_id}/pause")
def pause(session_id: str) -> dict[str, Any]:
    m = get_manager()
    state = _load(session_id)
    m.pause(state)
    return {"agent_state": state.agent_state.value}


# §14.10 再開
@app.post("/api/agent/sessions/{session_id}/resume")
def resume(session_id: str) -> dict[str, Any]:
    m = get_manager()
    state = _load(session_id)
    m.resume(state)
    return {"agent_state": state.agent_state.value}


# §14.11 終了
@app.post("/api/agent/sessions/{session_id}/complete")
def complete(session_id: str) -> dict[str, Any]:
    m = get_manager()
    state = _load(session_id)
    m.complete(state)
    return {"agent_state": state.agent_state.value}


# 計算コード選択（仮説 → コード推薦）
@app.post("/api/agent/calculations/plan")
def plan_calculation(req: PlanCalculationRequest) -> dict[str, Any]:
    m = get_manager()
    state = _load(req.session_id)
    try:
        return m.plan_calculation(state, hypothesis_id=req.hypothesis_id,
                                  hypothesis_text=req.hypothesis_text)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


# SQS 構造生成（icet）
@app.post("/api/agent/structures/sqs")
def generate_sqs(req: SQSRequest) -> dict[str, Any]:
    m = get_manager()
    state = _load(req.session_id)
    try:
        return m.generate_sqs(
            state, req.elements, concentrations=req.concentrations,
            prototype=req.prototype, a0=req.a0,
            max_size=req.max_size, n_steps=req.n_steps)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(400, str(exc)) from exc


# 大サイクル総括（推論→仮説→検証→反証事例→まとめ）
@app.post("/api/agent/sessions/{session_id}/synthesis")
def synthesize_cycle(session_id: str) -> dict[str, Any]:
    m = get_manager()
    state = _load(session_id)
    return m.synthesize_cycle(state)


# 計算ジョブ提案（入力生成 → 承認付き提案）
@app.post("/api/agent/calculations/jobs")
def propose_calculation_job(req: CalculationJobRequest) -> dict[str, Any]:
    m = get_manager()
    state = _load(req.session_id)
    try:
        job = m.propose_calculation_job(
            state, req.code, req.elements, params=req.params,
            description=req.description,
            estimated_node_hours=req.estimated_node_hours)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return job.model_dump()


# 解析提案（スクリプト生成 → 承認要求）
@app.post("/api/agent/analyses")
def propose_analysis(req: AnalysisProposalRequest) -> dict[str, Any]:
    m = get_manager()
    state = _load(req.session_id)
    try:
        approval = m.propose_analysis(state, req.purpose, script=req.script,
                                      job_id=req.job_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return approval.model_dump()


# 承認済み解析の実行（エラー時は自動修正して再試行、結果を返却）
@app.post("/api/agent/analyses/{approval_id}/run")
def run_analysis(approval_id: str, req: AnalysisRunRequest) -> dict[str, Any]:
    m = get_manager()
    state = _load(req.session_id)
    try:
        return m.run_approved_analysis(state, approval_id,
                                       max_fix_attempts=req.max_fix_attempts)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


# 承認解決（Human-in-the-loop §10）
@app.post("/api/agent/approvals/{approval_id}")
def resolve_approval(approval_id: str, req: ApprovalRequestBody) -> dict[str, Any]:
    m = get_manager()
    state = _load(req.session_id)
    try:
        return m.resolve_approval(state, approval_id, req.approve, by=req.by)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


# 仮説の修正（Human-in-the-loop §10）
@app.patch("/api/agent/hypotheses/{hypothesis_id}")
def patch_hypothesis(hypothesis_id: str, req: HypothesisPatch) -> dict[str, Any]:
    m = get_manager()
    state = _load(req.session_id)
    try:
        if req.status:
            m.set_hypothesis_status(state, hypothesis_id, HypothesisState(req.status),
                                    by=req.by)
        if req.falsification_conditions is not None:
            m.update_falsification_conditions(state, hypothesis_id,
                                              req.falsification_conditions, by=req.by)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    h = state.hypothesis(hypothesis_id)
    return h.model_dump() if h else {}
