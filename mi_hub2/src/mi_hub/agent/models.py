"""データモデル（指示書 §4, §8, §15）。"""

from __future__ import annotations

import time
import uuid
from typing import Any

from pydantic import BaseModel, Field

from .states import AgentState, ErrorType, HypothesisState, TaskState


def _uid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


class ResearchGoal(BaseModel):
    goal_id: str = Field(default_factory=lambda: _uid("GOAL"))
    statement: str
    target_material: str | None = None
    target_phase: str | None = None
    target_property: str | None = None
    success_criteria: list[str] = Field(default_factory=list)


class Evidence(BaseModel):
    evidence_id: str = Field(default_factory=lambda: _uid("E"))
    source_type: str = "journal_article"
    claim: str = ""
    conditions: dict[str, Any] = Field(default_factory=dict)
    evidence_type: str = "experiment"
    independence_group: str | None = None
    limitations: list[str] = Field(default_factory=list)


class Hypothesis(BaseModel):
    hypothesis_id: str = Field(default_factory=lambda: _uid("H"))
    statement: str
    counter_to: str | None = None  # 対立仮説の場合、主仮説ID
    supporting_predictions: list[str] = Field(default_factory=list)
    falsification_conditions: list[str] = Field(default_factory=list)
    falsification_approved: bool = False  # 反証条件は研究者承認後に固定（§5.3）
    hold_conditions: list[str] = Field(default_factory=list)
    required_inputs: list[str] = Field(default_factory=list)
    required_outputs: list[str] = Field(default_factory=list)
    applicability: dict[str, Any] = Field(default_factory=dict)
    status: HypothesisState = HypothesisState.DRAFT


class Task(BaseModel):
    task_id: str = Field(default_factory=lambda: _uid("TASK"))
    agent: str = "ResearchManagerAgent"
    action: str = ""
    description: str = ""
    status: TaskState = TaskState.PROPOSED
    depends_on: list[str] = Field(default_factory=list)
    requires_approval: bool = False
    approval_id: str | None = None
    inputs: dict[str, Any] = Field(default_factory=dict)
    result_ids: list[str] = Field(default_factory=list)
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    retry_count: int = 0
    estimated_cost: float = 0.0
    tool: str | None = None
    reason: str = ""


class Plan(BaseModel):
    plan_id: str = Field(default_factory=lambda: _uid("PLAN"))
    goal_id: str = ""
    version: int = 1
    created_at: float = Field(default_factory=time.time)
    created_by: str = "agent"
    reason_for_change: str = "initial plan"
    previous_plan_version: int | None = None
    tasks: list[Task] = Field(default_factory=list)

    def task(self, task_id: str) -> Task | None:
        for t in self.tasks:
            if t.task_id == task_id:
                return t
        return None

    def ready_tasks(self) -> list[Task]:
        """依存タスクが完了し実行可能なタスクを返す。"""
        done = {
            t.task_id
            for t in self.tasks
            if t.status in (TaskState.COMPLETED, TaskState.PARTIALLY_COMPLETED, TaskState.SKIPPED)
        }
        out = []
        for t in self.tasks:
            if t.status in (
                TaskState.PROPOSED,
                TaskState.PENDING,
                TaskState.READY,
                TaskState.AWAITING_APPROVAL,
            ) and all(d in done for d in t.depends_on):
                out.append(t)
        return out


class PlanChange(BaseModel):
    """計画変更履歴（§8.1）。"""

    plan_id: str
    version: int
    created_at: float = Field(default_factory=time.time)
    created_by: str = "agent"
    reason_for_change: str = ""
    previous_plan_version: int | None = None
    added_tasks: list[str] = Field(default_factory=list)
    removed_tasks: list[str] = Field(default_factory=list)
    reordered_tasks: list[str] = Field(default_factory=list)
    human_approval: bool = False


class ApprovalRequest(BaseModel):
    approval_id: str = Field(default_factory=lambda: _uid("APPROVAL"))
    task_id: str | None = None
    kind: str = "task_execution"
    description: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)  # 例: script_execution の script 本文
    status: str = "pending"  # pending / approved / rejected
    requested_at: float = Field(default_factory=time.time)
    resolved_at: float | None = None
    resolved_by: str | None = None


class Budget(BaseModel):
    max_model_runs: int = 30  # MVP制限（§18）
    used_model_runs: int = 0
    max_gpu_hours: float = 10.0
    used_gpu_hours: float = 0.0

    def exceeded(self) -> bool:
        return (
            self.used_model_runs >= self.max_model_runs
            or self.used_gpu_hours >= self.max_gpu_hours
        )


class StopConditions(BaseModel):
    max_iterations: int = 5  # MVP: 自動反復回数は3〜5回（§18）
    current_iteration: int = 0
    minimum_information_gain: float = 0.01
    max_retries_per_task: int = 3


class StepEvaluation(BaseModel):
    step_result: str = "completed"
    goal_progress_before: float = 0.0
    goal_progress_after: float = 0.0
    evidence_added: list[str] = Field(default_factory=list)
    hypotheses_affected: list[str] = Field(default_factory=list)
    new_conflicts: list[str] = Field(default_factory=list)
    result_quality: str = "acceptable"
    requires_replanning: bool = False
    data_gaps: list[str] = Field(default_factory=list)  # 追加的に必要なデータ


class ErrorRecord(BaseModel):
    error_id: str = Field(default_factory=lambda: _uid("ERR"))
    task_id: str | None = None
    error_type: ErrorType = ErrorType.UNKNOWN_ERROR
    message: str = ""
    auto_recoverable: bool = False
    resolved: bool = False
    created_at: float = Field(default_factory=time.time)


class AuditLogEntry(BaseModel):
    timestamp: float = Field(default_factory=time.time)
    actor: str = ""
    action: str = ""
    detail: dict[str, Any] = Field(default_factory=dict)


class Observation(BaseModel):
    """現在状態の観察結果（§4.1 Step 2）。"""

    goal_progress: float = 0.0
    active_hypotheses: list[str] = Field(default_factory=list)
    available_models: int = 0
    applicable_models: int = 0
    completed_tasks: list[str] = Field(default_factory=list)
    failed_tasks: list[str] = Field(default_factory=list)
    pending_approvals: list[str] = Field(default_factory=list)
    unresolved_issues: list[str] = Field(default_factory=list)
    evidence_count: int = 0
    budget_remaining_runs: int = 0
    iterations_remaining: int = 0


class SessionState(BaseModel):
    """研究セッションの作業記憶（§7.1、§15）。"""

    session_id: str = Field(default_factory=lambda: _uid("SESSION"))
    goal: ResearchGoal | None = None
    agent_state: AgentState = AgentState.IDLE
    plan: Plan | None = None
    plan_history: list[PlanChange] = Field(default_factory=list)
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    approvals: list[ApprovalRequest] = Field(default_factory=list)
    errors: list[ErrorRecord] = Field(default_factory=list)
    evaluations: list[StepEvaluation] = Field(default_factory=list)
    audit_log: list[AuditLogEntry] = Field(default_factory=list)
    budget: Budget = Field(default_factory=Budget)
    stop_conditions: StopConditions = Field(default_factory=StopConditions)
    last_observation: Observation | None = None
    next_action_candidates: list[str] = Field(default_factory=list)
    stop_reason: str | None = None
    chat_history: list[dict[str, str]] = Field(default_factory=list)

    def hypothesis(self, hid: str) -> Hypothesis | None:
        for h in self.hypotheses:
            if h.hypothesis_id == hid:
                return h
        return None

    def approval(self, aid: str) -> ApprovalRequest | None:
        for a in self.approvals:
            if a.approval_id == aid:
                return a
        return None

    def audit(self, actor: str, action: str, **detail: Any) -> None:
        self.audit_log.append(AuditLogEntry(actor=actor, action=action, detail=detail))
