"""Research Manager Agent — 状態駆動実行ループ（指示書 §4, §5.1, §8, §12）。

Goal → Observe → Plan → Human Check → Act → Observe Result → Evaluate → Replan
の反復を、承認範囲・予算・反復回数の制約下で実行する。
"""

from __future__ import annotations

import json
import os
import platform
import re
from pathlib import Path
from typing import Any, ClassVar

from . import llm
from .graphrag import GraphRAGProvider
from .models import (
    ApprovalRequest,
    Evidence,
    Hypothesis,
    JobRecord,
    Observation,
    Plan,
    PlanChange,
    ResearchGoal,
    SessionState,
    Task,
)
from .roles import (
    APPROVAL_REQUIRED_ACTIONS,
    EvaluationAgent,
    EvidenceAgent,
    ExecutionAgent,
    FalsifierAgent,
    HypothesisAgent,
    ModelSelectionAgent,
    SafetyApprovalAgent,
    VerificationPlanningAgent,
)
from .scheduler import (
    SchedulerError,
    SchedulerGateway,
    resolve_scheduler_from_env,
)
from .states import AgentState, HypothesisState, TaskState
from .tools import ToolGateway

_SESSION_ID_RE = re.compile(r"[A-Za-z0-9_-]+")

_DEFAULT_DIR = os.path.join(
    os.environ.get("MI_HUB_DATA", os.path.expanduser("~/mi_hub_data")), "agent_sessions"
)


class SessionStore:
    """作業記憶の永続化（§7.1: セッション再開可能にする）。"""

    def __init__(self, base_dir: str | None = None):
        self.base_dir = Path(base_dir or _DEFAULT_DIR)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def path(self, session_id: str) -> Path:
        if not _SESSION_ID_RE.fullmatch(session_id):
            raise ValueError(f"invalid session_id: {session_id!r}")
        return self.base_dir / f"{session_id}.json"

    def save(self, state: SessionState) -> None:
        self.path(state.session_id).write_text(
            json.dumps(state.model_dump(), ensure_ascii=False, indent=1, default=str),
            encoding="utf-8",
        )

    def load(self, session_id: str) -> SessionState | None:
        try:
            p = self.path(session_id)
        except ValueError:
            return None
        if not p.exists():
            return None
        return SessionState.model_validate(json.loads(p.read_text(encoding="utf-8")))

    def list_sessions(self) -> list[str]:
        return sorted(p.stem for p in self.base_dir.glob("*.json"))


class ResearchManager:
    """研究セッション全体を統括する（§5.1）。"""

    def __init__(self, gateway: ToolGateway | None = None,
                 store: SessionStore | None = None,
                 scheduler: SchedulerGateway | None = None):
        self.gateway = gateway or ToolGateway()
        self.store = store or SessionStore()
        self.scheduler = scheduler or resolve_scheduler_from_env()
        self.safety = SafetyApprovalAgent()
        self._roles = {
            "EvidenceAgent": EvidenceAgent(),
            "HypothesisAgent": HypothesisAgent(),
            "FalsifierAgent": FalsifierAgent(),
            "ModelSelectionAgent": ModelSelectionAgent(),
            "VerificationPlanningAgent": VerificationPlanningAgent(),
            "ExecutionAgent": ExecutionAgent(),
            "EvaluationAgent": EvaluationAgent(),
        }

    # ---------- Goal ----------
    def create_session(self, goal_statement: str, **goal_kwargs: Any) -> SessionState:
        structured = llm.structure_goal(goal_statement)
        structured.update({k: v for k, v in goal_kwargs.items() if v is not None})
        goal = ResearchGoal(statement=goal_statement, **{
            k: structured.get(k) for k in
            ("target_material", "target_phase", "target_property", "success_criteria")
            if structured.get(k) is not None
        })
        state = SessionState(goal=goal, agent_state=AgentState.IDLE)
        # 会話は研究者の疑問・依頼（研究目標）から始める
        state.chat_history.append({"role": "user", "content": goal_statement})
        state.chat_history.append({
            "role": "assistant",
            "content": "研究目標を受け付けました。計画を生成し、承認をいただきながら検証を進めます。",
        })
        state.audit("ResearchManager", "session_created", goal_id=goal.goal_id)
        self.store.save(state)
        return state

    # ---------- Observe (§4.1 Step 2) ----------
    def observe(self, state: SessionState) -> Observation:
        """現在状態の観察。副作用は last_observation の更新のみ（agent_state は変更しない）。"""
        plan = state.plan
        completed = [t.task_id for t in plan.tasks if t.status == TaskState.COMPLETED] if plan else []
        failed = [t.task_id for t in plan.tasks if t.status == TaskState.FAILED] if plan else []
        total = len(plan.tasks) if plan and plan.tasks else 0
        obs = Observation(
            goal_progress=(len(completed) / total) if total else 0.0,
            active_hypotheses=[h.hypothesis_id for h in state.hypotheses
                               if h.status not in (HypothesisState.ARCHIVED,
                                                   HypothesisState.REJECTED_BY_HUMAN)],
            available_models=len(self.gateway.models),
            applicable_models=len(self._applicable_models(state)),
            completed_tasks=completed,
            failed_tasks=failed,
            pending_approvals=[a.approval_id for a in state.approvals if a.status == "pending"],
            unresolved_issues=[e.message for e in state.errors if not e.resolved],
            evidence_count=len(state.evidence),
            budget_remaining_runs=max(0, state.budget.max_model_runs - state.budget.used_model_runs),
            iterations_remaining=max(0, state.stop_conditions.max_iterations
                                     - state.stop_conditions.current_iteration),
        )
        state.last_observation = obs
        return obs

    def _applicable_models(self, state: SessionState) -> list[Any]:
        goal = state.goal
        if not goal or not goal.target_property:
            return []
        elements = []
        if goal.target_material:
            elements = [e for e in goal.target_material.replace("-", " ").split() if e]
        try:
            return self.gateway.search_models(goal.target_property, elements,
                                              goal.target_phase)
        except Exception:
            return []  # 観察はベストエフォート（失敗はタスク実行側で扱う）

    # ---------- Plan (§4.1 Step 3) ----------
    def generate_plan(self, state: SessionState) -> Plan:
        state.agent_state = AgentState.PLANNING
        goal = state.goal
        t1 = Task(agent="EvidenceAgent", action="search_literature",
                  description="関連論文・証拠を検索する",
                  inputs={"query": goal.statement if goal else ""})
        t2 = Task(agent="HypothesisAgent", action="generate_hypotheses",
                  description="主仮説と対立仮説を生成する", depends_on=[t1.task_id])
        t2b = Task(agent="FalsifierAgent", action="search_counter_evidence",
                   description="仮説を否定しうる証拠・条件依存性・別機構を探索する（反証優先）",
                   depends_on=[t2.task_id])
        t3 = Task(agent="ModelSelectionAgent", action="search_models",
                  description="適用可能なMIntモデルを検索する", depends_on=[t2b.task_id])
        t4 = Task(agent="VerificationPlanningAgent", action="plan_verification",
                  description="検証ジョブ（入力条件）を生成する", depends_on=[t3.task_id])
        t5 = Task(agent="ExecutionAgent", action="run_models_bulk",
                  description="適用可能なモデルを一括実行する", depends_on=[t4.task_id],
                  requires_approval=True)
        t6 = Task(agent="EvaluationAgent", action="evaluate_hypothesis",
                  description="傾きと不確実性から仮説の判定材料を整理する",
                  depends_on=[t5.task_id])
        plan = Plan(goal_id=goal.goal_id if goal else "", tasks=[t1, t2, t2b, t3, t4, t5, t6])
        state.plan = plan
        state.plan_history.append(PlanChange(
            plan_id=plan.plan_id, version=plan.version,
            reason_for_change="initial plan",
            added_tasks=[t.task_id for t in plan.tasks],
        ))
        state.audit("ResearchManager", "plan_generated", plan_id=plan.plan_id,
                    version=plan.version)
        self.store.save(state)
        return plan

    def apply_plan_change(self, state: SessionState, *, created_by: str = "human",
                          reason: str = "", add_tasks: list[Task] | None = None,
                          remove_task_ids: list[str] | None = None,
                          reorder_task_ids: list[str] | None = None) -> Plan:
        """計画の編集（人間介入 §10 / 再計画 §8）。版管理される。"""
        plan = state.plan
        if plan is None:
            raise ValueError("計画が存在しません")
        prev_version = plan.version
        added, removed = [], []
        if remove_task_ids:
            plan.tasks = [t for t in plan.tasks if t.task_id not in remove_task_ids]
            removed = list(remove_task_ids)
        for t in add_tasks or []:
            plan.tasks.append(t)
            added.append(t.task_id)
        reordered = []
        if reorder_task_ids:
            by_id = {t.task_id: t for t in plan.tasks}
            ordered = [by_id[i] for i in reorder_task_ids if i in by_id]
            rest = [t for t in plan.tasks if t.task_id not in set(reorder_task_ids)]
            plan.tasks = ordered + rest
            reordered = list(reorder_task_ids)
        plan.version += 1
        plan.reason_for_change = reason
        plan.previous_plan_version = prev_version
        plan.created_by = created_by
        state.plan_history.append(PlanChange(
            plan_id=plan.plan_id, version=plan.version, created_by=created_by,
            reason_for_change=reason, previous_plan_version=prev_version,
            added_tasks=added, removed_tasks=removed, reordered_tasks=reordered,
            human_approval=(created_by == "human"),
        ))
        state.audit(created_by, "plan_changed", version=plan.version, reason=reason)
        self.store.save(state)
        return plan

    # ---------- Next actions (§14.5) ----------
    def next_actions(self, state: SessionState) -> list[dict[str, Any]]:
        plan = state.plan
        if plan is None:
            return [{"action": "generate_plan", "reason": "計画が未生成"}]
        out = []
        for t in plan.ready_tasks():
            needs = t.requires_approval or t.action in APPROVAL_REQUIRED_ACTIONS
            approved = False
            if t.approval_id:
                a = state.approval(t.approval_id)
                approved = bool(a and a.status == "approved")
            out.append({
                "task_id": t.task_id, "action": t.action, "agent": t.agent,
                "description": t.description,
                "requires_approval": needs, "approved": approved,
            })
        state.next_action_candidates = [o["task_id"] for o in out]
        return out

    # ---------- Act (§4.1 Step 4-6) ----------
    def execute_task(self, state: SessionState, task_id: str) -> dict[str, Any]:
        plan = state.plan
        if plan is None:
            raise ValueError("計画が存在しません")
        task = plan.task(task_id)
        if task is None:
            raise ValueError(f"タスクが存在しません: {task_id}")
        if state.agent_state == AgentState.PAUSED:
            return {"status": "paused", "detail": "セッションは一時停止中です"}
        ok, reason = self.safety.check(state, task)
        if not ok:
            if "承認" in reason and task.approval_id is None:
                self.safety.request_approval(state, task)
                state.agent_state = AgentState.AWAITING_APPROVAL
                self.store.save(state)
                return {"status": "awaiting_approval", "approval_id": task.approval_id,
                        "detail": reason}
            state.agent_state = AgentState.BLOCKED
            self.store.save(state)
            return {"status": "blocked", "detail": reason}

        # 依存タスクの結果を入力へ引き継ぐ
        self._wire_inputs(state, task)
        state.agent_state = AgentState.EXECUTING
        task.status = TaskState.RUNNING
        role = self._roles.get(task.agent)
        try:
            result = role.run(state, self.gateway, task) if role else {}
            task.result = result
            failures = result.get("failures") or []
            successes = result.get("results")
            if failures and successes:
                task.status = TaskState.PARTIALLY_COMPLETED
            elif failures and not successes and task.agent == "ExecutionAgent":
                task.status = TaskState.FAILED
            else:
                task.status = TaskState.COMPLETED
            state.audit(task.agent, "task_executed", task_id=task.task_id,
                        status=task.status.value)
            status = "completed" if task.status != TaskState.FAILED else "failed"
            if task.status != TaskState.FAILED:
                self._add_science_comment(state, task)
                self._ask_judgement_confirmation(state, task)
        except Exception as exc:  # ツール層以外の予期しない失敗
            from .errors import record_error

            err = record_error(task.task_id, str(exc))
            state.errors.append(err)
            task.status = TaskState.FAILED
            task.error = {"message": str(exc), "error_type": err.error_type.value}
            status = "failed"
        state.agent_state = AgentState.EVALUATING
        self._after_step(state)
        self.store.save(state)
        return {"status": status, "task_id": task.task_id,
                "result": task.result, "error": task.error}

    _COMMENT_TARGETS: ClassVar[dict[str, str]] = {
        "HypothesisAgent": "仮説",
        "FalsifierAgent": "反証検討",
        "EvaluationAgent": "評価結果",
        "ExecutionAgent": "計算結果",
    }

    def _add_science_comment(self, state: SessionState, task: Task) -> None:
        """仮説・計算結果への専門的コメントを研究者へ提示する（LLM 不可時は何もしない）。"""
        kind = self._COMMENT_TARGETS.get(task.agent)
        if not kind or not task.result:
            return
        goal = state.goal.statement if state.goal else ""
        payload = {"task": task.description, "result": task.result,
                   "hypotheses": [h.statement for h in state.hypotheses]}
        comment = llm.science_comment(goal, kind, payload)
        if comment:
            state.chat_history.append({
                "role": "assistant",
                "content": f"【エージェント所見（{kind}）】\n{comment}",
            })
            state.audit("ResearchManager", "science_comment",
                        task_id=task.task_id, kind=kind)

    def _ask_judgement_confirmation(self, state: SessionState, task: Task) -> None:
        """評価完了時、ルール評価による判定案の確定を研究者へ明示的に問いかける。"""
        if task.action != "evaluate_hypothesis" or not task.result:
            return
        judgements = task.result.get("judgements") or {}
        labels = {"supported": "支持（Supported）",
                  "refuted": "反証（Refuted）",
                  "inconclusive": "保留（Inconclusive）"}
        for hid, j in judgements.items():
            h = state.hypothesis(hid)
            if h is None:
                continue
            verdict = labels.get(j.get("verdict", ""), j.get("verdict", ""))
            state.chat_history.append({
                "role": "assistant",
                "content": (
                    f"【判定案】仮説「{h.statement}」のルール評価による判定案は"
                    f"「{verdict}」です。\n根拠: {j.get('rationale', '')}\n"
                    "この判定を確定しますか？（確定する場合は仮説タブの"
                    "「判定を確定」、保留のまま検証を続ける場合は「保留のまま続行」を"
                    "選択してください。最終判定は研究者に委ねられます）"
                ),
            })
            state.audit("ResearchManager", "judgement_proposed",
                        hypothesis_id=hid, verdict=j.get("verdict"))

    def _wire_inputs(self, state: SessionState, task: Task) -> None:
        plan = state.plan
        assert plan is not None
        for dep_id in task.depends_on:
            dep = plan.task(dep_id)
            if not dep or not dep.result:
                continue
            if task.action == "plan_verification" and "recommended_models" in dep.result:
                task.inputs.setdefault("model_ids", dep.result["recommended_models"])
            if task.action == "run_models_bulk" and "jobs" in dep.result:
                task.inputs.setdefault("jobs", dep.result["jobs"])
            if task.action == "evaluate_hypothesis" and "results" in dep.result:
                task.inputs.setdefault("results", dep.result["results"])

    def _after_step(self, state: SessionState) -> None:
        """Observe Result → Evaluate → Replan 判定（§4.1 Step 5-7）。"""
        self.observe(state)
        stop = self.check_stop_conditions(state)
        if stop:
            state.stop_reason = stop
            if "承認待ち" in stop or "人間判断" in stop:
                state.agent_state = AgentState.AWAITING_APPROVAL
            elif "完了" in stop:
                state.agent_state = AgentState.COMPLETED
            else:
                state.agent_state = AgentState.BLOCKED
        else:
            state.agent_state = AgentState.REPLANNING

    # ---------- Stop conditions (§12) ----------
    def check_stop_conditions(self, state: SessionState) -> str | None:
        sc = state.stop_conditions
        plan = state.plan
        if plan and plan.tasks and all(
            t.status in (TaskState.COMPLETED, TaskState.SKIPPED,
                         TaskState.PARTIALLY_COMPLETED)
            for t in plan.tasks
        ):
            return "正常終了: 承認済み検証計画が完了"
        if sc.current_iteration >= sc.max_iterations:
            return "資源制約: 反復回数上限に到達"
        if state.budget.exceeded():
            return "資源制約: 計算予算上限に到達"
        if any(a.status == "pending" for a in state.approvals):
            return None  # 承認待ちは execute 時に個別処理
        return None

    # ---------- Workspace ----------
    def session_workspace(self, state: SessionState) -> str:
        """セッション専用の作業ディレクトリ（スクリプト実行の成果物置き場）を返す。"""
        d = self.store.base_dir / "workspaces" / state.session_id
        d.mkdir(parents=True, exist_ok=True)
        return str(d)

    # ---------- 外部ジョブ（Slurm等、承認必須） ----------
    def propose_job(self, state: SessionState, name: str, script: str,
                    kind: str = "dft", description: str = "",
                    estimated_node_hours: float = 1.0) -> JobRecord:
        """外部ジョブを提案し、承認要求を登録する（投入は承認後のみ）。"""
        job = JobRecord(
            scheduler=self.scheduler.name, name=name, kind=kind, script=script,
            workdir=os.path.join(self.session_workspace(state), "jobs", name),
            estimated_node_hours=estimated_node_hours,
        )
        req = ApprovalRequest(
            kind="job_submission",
            description=description or f"外部ジョブ投入: {name}（{kind}、"
                        f"推定 {estimated_node_hours:.1f} ノード時間）",
            payload={"job_id": job.job_id, "script": script,
                     "estimated_node_hours": estimated_node_hours},
        )
        job.approval_id = req.approval_id
        state.jobs.append(job)
        state.approvals.append(req)
        state.audit("ResearchManager", "job_proposed", job_id=job.job_id,
                    approval_id=req.approval_id, name=name, kind=kind)
        state.chat_history.append({
            "role": "assistant",
            "content": f"「{req.description}」という提案がありますが、実行しますか？"
                       "（承認するまで投入されません。実行する場合は「承認」、"
                       "しない場合は「却下」と入力してください。承認タブからも操作可）",
        })
        self.store.save(state)
        return job

    def submit_approved_job(self, state: SessionState, job_id: str) -> JobRecord:
        """承認済みジョブをスケジューラへ投入する。"""
        import time as _time

        job = state.job(job_id)
        if job is None:
            raise ValueError(f"ジョブが存在しません: {job_id}")
        if job.state != "proposed":
            raise ValueError(f"ジョブは投入済みです: {job_id}（{job.state}）")
        approval = state.approval(job.approval_id) if job.approval_id else None
        if approval is None or approval.status != "approved":
            raise ValueError(f"未承認のジョブは投入できません: {job_id}")
        remaining = state.budget.max_node_hours - state.budget.used_node_hours
        if job.estimated_node_hours > remaining:
            raise ValueError(
                f"ノード時間予算不足: 残 {remaining:.1f}h < 推定 {job.estimated_node_hours:.1f}h"
            )
        job.scheduler_job_id = self.scheduler.submit(job.script, job.workdir, job.name)
        # 投入直後にまず永続化する（直後の状態確認が失敗しても二重投入を防ぐ）
        job.state = "pending"
        job.submitted_at = _time.time()
        state.budget.used_node_hours += job.estimated_node_hours
        state.audit("ResearchManager", "job_submitted", job_id=job.job_id,
                    scheduler_job_id=job.scheduler_job_id, state=job.state)
        self.store.save(state)
        try:
            job.state = self.scheduler.status(job.scheduler_job_id)
        except SchedulerError as exc:
            state.audit("ResearchManager", "job_poll_failed",
                        job_id=job.job_id, error=str(exc))
            self.store.save(state)
            return job
        if job.state in ("completed", "failed", "cancelled"):
            self._finalize_job(state, job)
        self.store.save(state)
        return job

    def poll_jobs(self, state: SessionState) -> list[JobRecord]:
        """未完了ジョブの状態をポーリングし、完了したものを証拠化する。"""
        updated = []
        for job in state.jobs:
            if job.state not in ("pending", "running") or not job.scheduler_job_id:
                continue
            try:
                new_state = self.scheduler.status(job.scheduler_job_id)
            except SchedulerError as exc:
                state.audit("ResearchManager", "job_poll_failed",
                            job_id=job.job_id, error=str(exc))
                continue
            if new_state != job.state:
                job.state = new_state
                if new_state in ("completed", "failed", "cancelled"):
                    self._finalize_job(state, job)
                updated.append(job)
        if updated:
            self.store.save(state)
        return updated

    def _finalize_job(self, state: SessionState, job: JobRecord) -> None:
        import time as _time

        job.finished_at = _time.time()
        generated = sorted(os.listdir(job.workdir)) if os.path.isdir(job.workdir) else []
        state.evidence.append(Evidence(
            source_type="external_job",
            claim=f"外部ジョブ {job.name}（{job.kind}）が {job.state} で終了",
            conditions={"job_id": job.job_id,
                        "scheduler": job.scheduler,
                        "scheduler_job_id": job.scheduler_job_id,
                        "workdir": job.workdir,
                        "generated_files": generated},
            evidence_type="computation",
            limitations=["外部計算の終了状態。科学的妥当性は出力の検証が必要"],
        ))
        state.audit("ResearchManager", "job_finished", job_id=job.job_id,
                    state=job.state)

    # ---------- Case report (事例の蓄積) ----------
    def export_case_report(self, state: SessionState) -> str:
        """セッションを事例レポート（Markdown）として作業ディレクトリに書き出す。

        会話・提案スクリプト・実行環境・結果・情報ギャップを1ファイルに集約し、
        今後の機能改修や類似研究の参照に使う。書き出し先のパスを返す。
        """
        lines: list[str] = [f"# 事例レポート: {state.session_id}", ""]
        if state.goal:
            lines += ["## 研究目標", state.goal.statement, ""]
            if state.goal.success_criteria:
                lines += ["成功基準:"] + [
                    f"- {c}" for c in state.goal.success_criteria
                ] + [""]
        lines += [
            "## 実行環境",
            f"- python: {platform.python_version()} / {platform.platform()}",
            f"- agent_state: {state.agent_state.value}"
            + (f" / 停止理由: {state.stop_reason}" if state.stop_reason else ""),
            "",
        ]
        if state.hypotheses:
            lines += ["## 仮説"] + [
                f"- [{h.status.value}] {h.statement}" for h in state.hypotheses
            ] + [""]
        if state.chat_history:
            lines += ["## 会話ログ"]
            for msg in state.chat_history:
                lines += [f"### {msg['role']}", msg["content"], ""]
        script_approvals = [a for a in state.approvals if a.kind == "script_execution"]
        if script_approvals:
            lines += ["## 提案・実行スクリプト"]
            for a in script_approvals:
                lines += [
                    f"### {a.approval_id}（{a.status}）",
                    a.description,
                    "```bash",
                    a.payload.get("script", ""),
                    "```",
                    "",
                ]
        if state.evidence:
            lines += ["## 証拠・実行結果"]
            for e in state.evidence:
                lines += [f"### {e.evidence_id} [{e.evidence_type}]", e.claim]
                if e.conditions.get("stdout"):
                    lines += ["```", str(e.conditions["stdout"]), "```"]
                for f in e.conditions.get("generated_files") or []:
                    if str(f).lower().endswith((".png", ".jpg", ".jpeg", ".svg")):
                        lines += [f"![{f}]({f})"]  # レポートと同じ作業ディレクトリ
                    else:
                        lines += [f"- 成果物: {e.conditions.get('workdir', '')}/{f}"]
                lines += [""]
        gaps = state.evaluations[-1].data_gaps if state.evaluations else []
        if gaps:
            lines += ["## 追加的に必要なデータ"] + [f"- {g}" for g in gaps] + [""]
        if state.errors:
            lines += ["## エラー"] + [
                f"- {e.error_type.value}: {e.message}"
                f"（{'解決済' if e.resolved else '未解決'}）"
                for e in state.errors
            ] + [""]
        path = os.path.join(self.session_workspace(state), "case_report.md")
        report_text = "\n".join(lines)
        with open(path, "w", encoding="utf-8") as f:
            f.write(report_text)
        state.audit("ResearchManager", "case_report_exported", path=path)
        # 事例ナレッジの自動還元: GraphRAG が登録済みなら事例として取り込む
        for provider in self.gateway.knowledge_providers:
            if isinstance(provider, GraphRAGProvider):
                provider.ingest_case_report(
                    state.session_id,
                    state.goal.statement if state.goal else state.session_id,
                    report_text)
                state.audit("ResearchManager", "case_knowledge_ingested",
                            provider=provider.name)
        self.store.save(state)
        return path

    # ---------- Memory ----------
    def memory_context(self, state: SessionState) -> dict[str, Any]:
        """短期記憶（直近の計算・評価）と長期記憶（セッション全体）を分けて返す。

        短期: 直近ステップの妥当性評価に使う。長期: 蓄積された証拠・評価履歴から
        全体としての妥当性を判断する材料に使う。判断は研究者が行う。
        """
        recent_tasks = [
            {"task": t.task_id, "action": t.action, "status": t.status.value,
             "description": t.description}
            for t in (state.plan.tasks if state.plan else [])
            if t.status in (TaskState.COMPLETED, TaskState.FAILED)
        ][-3:]
        short_term = {
            "last_evaluation": state.evaluations[-1].model_dump() if state.evaluations else None,
            "recent_tasks": recent_tasks,
            "unresolved_errors": [e.message for e in state.errors if not e.resolved],
        }
        long_term = {
            "goal": state.goal.model_dump() if state.goal else None,
            "hypotheses": [
                {"id": h.hypothesis_id, "statement": h.statement, "status": h.status.value,
                 "falsification_conditions": h.falsification_conditions}
                for h in state.hypotheses
            ],
            "evidence": [
                {"id": e.evidence_id, "type": e.evidence_type, "claim": e.claim}
                for e in state.evidence
            ],
            "evaluation_history": [
                {"result": ev.step_result, "quality": ev.result_quality,
                 "progress_after": ev.goal_progress_after, "data_gaps": ev.data_gaps}
                for ev in state.evaluations
            ],
            "plan_versions": len(state.plan_history),
        }
        return {"short_term_memory": short_term, "long_term_memory": long_term}

    # ---------- Human-in-the-loop (§10) ----------
    def resolve_approval(self, state: SessionState, approval_id: str,
                         approve: bool, by: str = "human") -> dict[str, Any]:
        import time as _time

        req = state.approval(approval_id)
        if req is None:
            raise ValueError(f"承認要求が存在しません: {approval_id}")
        if req.status != "pending":
            raise ValueError(
                f"承認要求は既に解決済みです: {approval_id}（{req.status}）"
            )
        req.status = "approved" if approve else "rejected"
        req.resolved_at = _time.time()
        req.resolved_by = by
        state.audit(by, "approval_resolved", approval_id=approval_id, approved=approve)
        if not approve and req.task_id and state.plan:
            task = state.plan.task(req.task_id)
            if task:
                task.status = TaskState.REJECTED
        elif approve and req.task_id and state.plan:
            task = state.plan.task(req.task_id)
            if task and task.status == TaskState.AWAITING_APPROVAL:
                task.status = TaskState.READY
        self.store.save(state)
        return {"approval_id": approval_id, "status": req.status}

    def set_hypothesis_status(self, state: SessionState, hypothesis_id: str,
                              status: HypothesisState, by: str = "human") -> None:
        """仮説の正式採用・反証等は人間のみが確定できる。"""
        h = state.hypothesis(hypothesis_id)
        if h is None:
            raise ValueError(f"仮説が存在しません: {hypothesis_id}")
        h.status = status
        if status == HypothesisState.APPROVED_FOR_TESTING:
            h.falsification_approved = True  # 反証条件の固定
        state.audit(by, "hypothesis_status_changed",
                    hypothesis_id=hypothesis_id, status=status.value)
        self.store.save(state)

    def confirm_judgement(self, state: SessionState, hypothesis_id: str,
                          accept: bool, by: str = "human") -> Hypothesis:
        """ルール評価による判定案を研究者が確定（または保留のまま継続）する。

        確定時は verdict に応じて仮説状態を SUPPORTED / FALSIFIED / INCONCLUSIVE へ遷移する。
        """
        if by != "human":
            raise PermissionError("判定の確定には研究者の承認が必要です")
        h = state.hypothesis(hypothesis_id)
        if h is None:
            raise ValueError(f"仮説が存在しません: {hypothesis_id}")
        if h.judgement is None:
            raise ValueError(f"判定案が存在しません: {hypothesis_id}")
        if accept:
            mapping = {
                "supported": HypothesisState.SUPPORTED,
                "refuted": HypothesisState.FALSIFIED,
                "inconclusive": HypothesisState.INCONCLUSIVE,
            }
            h.status = mapping.get(h.judgement.verdict, HypothesisState.INCONCLUSIVE)
            h.judgement.confirmed_by_human = True
        state.audit(by, "judgement_confirmed" if accept else "judgement_deferred",
                    hypothesis_id=hypothesis_id, verdict=h.judgement.verdict,
                    status=h.status.value)
        self.store.save(state)
        return h

    def update_falsification_conditions(self, state: SessionState, hypothesis_id: str,
                                        conditions: list[str], by: str = "human") -> None:
        """反証条件の変更は人間承認必須（§4.1 Step 7）。変更前後を保存する。"""
        if by != "human":
            raise PermissionError("反証条件の変更には研究者の承認が必要です")
        h = state.hypothesis(hypothesis_id)
        if h is None:
            raise ValueError(f"仮説が存在しません: {hypothesis_id}")
        before = list(h.falsification_conditions)
        h.falsification_conditions = conditions
        state.audit(by, "falsification_conditions_changed",
                    hypothesis_id=hypothesis_id, before=before, after=conditions)
        self.store.save(state)

    # ---------- Pause / Resume / Complete (§14.9-11) ----------
    def pause(self, state: SessionState) -> None:
        state.agent_state = AgentState.PAUSED
        state.audit("human", "session_paused")
        self.store.save(state)

    def resume(self, state: SessionState) -> None:
        state.agent_state = AgentState.REPLANNING
        state.stop_reason = None
        state.audit("human", "session_resumed")
        self.store.save(state)

    def complete(self, state: SessionState) -> None:
        state.agent_state = AgentState.COMPLETED
        state.stop_reason = "研究者が終了を承認"
        state.audit("human", "session_completed")
        self.store.save(state)

    # ---------- Auto loop ----------
    def run_auto(self, state: SessionState, max_steps: int | None = None) -> dict[str, Any]:
        """承認不要タスクを自動で連続実行する。承認要タスクで一時停止する。

        1 回の呼出しが Observe→Plan→Act→Evaluate→Replan の 1 反復に相当する（§18）。
        """
        sc = state.stop_conditions
        if sc.current_iteration >= sc.max_iterations:
            state.stop_reason = "資源制約: 反復回数上限に到達"
            state.agent_state = AgentState.BLOCKED
            self.store.save(state)
            return {"executed": [], "agent_state": state.agent_state.value,
                    "stop_reason": state.stop_reason}
        executed = []
        steps = 0
        limit = max_steps or (len(state.plan.tasks) if state.plan else 10)
        while steps < limit:
            if state.agent_state in (AgentState.PAUSED, AgentState.COMPLETED,
                                     AgentState.CANCELLED, AgentState.FAILED):
                break
            actions = self.next_actions(state)
            runnable = [a for a in actions
                        if not a["requires_approval"] or a["approved"]]
            if not runnable:
                if actions:
                    # 承認待ちを生成して停止
                    res = self.execute_task(state, actions[0]["task_id"])
                    executed.append(res)
                break
            res = self.execute_task(state, runnable[0]["task_id"])
            executed.append(res)
            steps += 1
            if res["status"] in ("awaiting_approval", "blocked", "paused"):
                break
            if state.stop_reason:
                break
        if executed:
            sc.current_iteration += 1
            self.store.save(state)
        return {"executed": executed, "agent_state": state.agent_state.value,
                "stop_reason": state.stop_reason}
