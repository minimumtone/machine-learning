"""専門エージェント群（指示書 §5）。

MVP では物理的なマルチエージェント並列実行は行わず、論理的役割として分離する（§18）。
各役割は SessionState と ToolGateway を受け取り、決定論的に動作する。
LLM は候補生成にのみ使用される（llm モジュール経由）。
"""

from __future__ import annotations

from typing import Any

from . import llm
from .errors import record_error, requires_human_review, try_auto_fix
from .judgement import judge_hypothesis
from .models import (
    ApprovalRequest,
    Evidence,
    Hypothesis,
    SessionState,
    StepEvaluation,
    Task,
)
from .states import HypothesisState, TaskState
from .tools import ToolError, ToolGateway

# 人間承認が必要な操作（§11.2）
APPROVAL_REQUIRED_ACTIONS = {
    "run_models_bulk",
    "run_high_cost_model",
    "submit_dft_job",
    "calphad_large_search",
    "external_api_call",
    "send_confidential_data",
    "register_model",
    "update_model",
    "adopt_hypothesis",
    "falsify_hypothesis",
    "register_knowledge_graph",
    "operate_experiment_equipment",
    "change_verification_conditions",
    "change_falsification_conditions",
}

# 自動実行可能な操作（§11.1）
AUTO_ALLOWED_ACTIONS = {
    "search_literature",
    "search_models",
    "get_model_metadata",
    "run_low_cost_model",
    "validate_inputs",
    "normalize_units",
    "make_plots",
    "aggregate_results",
    "analyze_error",
    "generate_next_actions",
    "reuse_results",
    "generate_hypotheses",
    "search_counter_evidence",
    "evaluate_hypothesis",
}


class SafetyApprovalAgent:
    """権限・コスト・承認状態の確認と禁止操作の遮断（§5.8）。

    この判定は決定論的ルールで行い、他エージェントは上書きできない。
    """

    def check(self, state: SessionState, task: Task) -> tuple[bool, str]:
        """(実行可, 理由) を返す。"""
        if state.budget.exceeded():
            return False, "計算予算上限に到達"
        if task.action in APPROVAL_REQUIRED_ACTIONS or task.requires_approval:
            approval = state.approval(task.approval_id) if task.approval_id else None
            if approval is None or approval.status != "approved":
                return False, "人間承認が必要（未承認）"
        return True, "ok"

    def request_approval(self, state: SessionState, task: Task, kind: str = "task_execution") -> ApprovalRequest:
        req = ApprovalRequest(task_id=task.task_id, kind=kind, description=task.description)
        state.approvals.append(req)
        task.approval_id = req.approval_id
        task.status = TaskState.AWAITING_APPROVAL
        state.audit("SafetyApprovalAgent", "approval_requested",
                    approval_id=req.approval_id, task_id=task.task_id)
        return req


class EvidenceAgent:
    """文献・データベース検索と証拠抽出（§5.2）。"""

    def run(self, state: SessionState, gateway: ToolGateway, task: Task) -> dict[str, Any]:
        query = task.inputs.get("query") or (state.goal.statement if state.goal else "")
        docs = gateway.search_knowledge(query)
        added = []
        for doc in docs:
            ev = Evidence(
                source_type=doc.get("source_type", "journal_article"),
                claim=doc.get("claim", ""),
                conditions=doc.get("conditions", {}),
                evidence_type=doc.get("evidence_type", "experiment"),
                limitations=doc.get("limitations", []),
            )
            state.evidence.append(ev)
            added.append(ev.evidence_id)
        task.result_ids = added
        return {"evidence_ids": added, "count": len(added)}


class HypothesisAgent:
    """主仮説・対立仮説・反証条件案の生成（§5.3）。

    反証条件は研究者承認後に固定される。
    """

    def run(self, state: SessionState, gateway: ToolGateway, task: Task) -> dict[str, Any]:
        goal = state.goal.statement if state.goal else ""
        claims = [e.claim for e in state.evidence]
        candidates = llm.generate_hypotheses(goal, claims)
        added = []
        main_id: str | None = None
        source_ids = [e.evidence_id for e in state.evidence]
        for c in candidates:
            scope = c.get("scope")
            h = Hypothesis(
                statement=c.get("statement", ""),
                counter_to=main_id if c.get("is_counter") else None,
                mechanism=str(c.get("mechanism", "") or ""),
                supporting_predictions=c.get("supporting_predictions", []),
                falsification_conditions=c.get("falsification_conditions", []),
                applicability=scope if isinstance(scope, dict) else {},
                source_evidence=source_ids,
                status=HypothesisState.PROPOSED,
            )
            state.hypotheses.append(h)
            if not c.get("is_counter"):
                main_id = h.hypothesis_id
            added.append(h.hypothesis_id)
        task.result_ids = added
        return {"hypothesis_ids": added}


class FalsifierAgent:
    """反証担当（Falsifier）: 確証バイアスを防ぐ。

    仮説を否定しうる文献・条件依存性・別機構を能動的に探し、
    反証条件の候補を補強する（反証条件の確定は研究者承認による）。
    """

    _TARGET_STATES = (
        HypothesisState.PROPOSED,
        HypothesisState.HUMAN_REVIEWED,
        HypothesisState.APPROVED_FOR_TESTING,
    )

    def run(self, state: SessionState, gateway: ToolGateway, task: Task) -> dict[str, Any]:
        goal = state.goal.statement if state.goal else ""
        claims = [e.claim for e in state.evidence]
        counter_ids: list[str] = []
        added_conditions: dict[str, list[str]] = {}
        reviewed: list[str] = []
        for h in state.hypotheses:
            if h.status not in self._TARGET_STATES:
                continue
            review = llm.falsification_review(goal, h.statement, claims)
            reviewed.append(h.hypothesis_id)
            for query in review["counter_queries"]:
                for doc in gateway.search_knowledge(query, limit=3):
                    ev = Evidence(
                        source_type=doc.get("source_type", "journal_article"),
                        claim=doc.get("claim", ""),
                        conditions=doc.get("conditions", {}),
                        evidence_type=doc.get("evidence_type", "experiment"),
                        limitations=list(doc.get("limitations", []))
                        + ["反証検討（Falsifier）の検索結果。仮説への支持/反対は研究者が判断"],
                    )
                    state.evidence.append(ev)
                    h.counter_evidence.append(ev.evidence_id)
                    counter_ids.append(ev.evidence_id)
            new_conds = [c for c in review["falsification_conditions"]
                         if c not in h.falsification_conditions]
            if new_conds and not h.falsification_approved:
                h.falsification_conditions.extend(new_conds)
                added_conditions[h.hypothesis_id] = new_conds
            for m in review["alternative_mechanisms"]:
                if m not in h.alternative_mechanisms:
                    h.alternative_mechanisms.append(m)
        task.result_ids = counter_ids
        return {
            "hypotheses_reviewed": reviewed,
            "counter_evidence_ids": counter_ids,
            "added_falsification_conditions": added_conditions,
            "note": "反証条件の確定・仮説の採否は研究者が行う",
        }


class ModelSelectionAgent:
    """MInt 登録モデルの検索と適用範囲確認（§5.4）。"""

    def run(self, state: SessionState, gateway: ToolGateway, task: Task) -> dict[str, Any]:
        goal = state.goal
        prop = task.inputs.get("target_property") or (goal.target_property if goal else "")
        elements = task.inputs.get("elements")
        if not elements and goal and goal.target_material:
            elements = [e for e in goal.target_material.replace("-", " ").split() if e]
        phase = task.inputs.get("phase") or (goal.target_phase if goal else None)
        models = gateway.search_models(prop or "", elements or [], phase)
        recommended = [m.model_id for m in models]
        excluded = [
            {"model_id": m.model_id, "reason": "対象物性が不一致"}
            for m in gateway.models
            if m.model_id not in recommended
        ]
        groups = {m.independence_group for m in models}
        result = {
            "recommended_models": recommended,
            "excluded_models": excluded,
            "independent_series": len(groups),
            "missing_model_types": [] if recommended else [prop],
        }
        task.result_ids = recommended
        return result


class VerificationPlanningAgent:
    """検証タスクの分解と入力条件生成（§5.5）。"""

    def run(self, state: SessionState, gateway: ToolGateway, task: Task) -> dict[str, Any]:
        model_ids = task.inputs.get("model_ids", [])
        compositions = task.inputs.get(
            "compositions",
            [{"Ni": 1 - x, "Al": x} for x in (0.48, 0.50, 0.52, 0.54)],
        )
        temperature = task.inputs.get("temperature", 800.0)
        jobs = [
            {"model_id": mid, "inputs": {"composition": c, "temperature": temperature,
                                         "temperature_unit": "K"}}
            for mid in model_ids
            for c in compositions
        ]
        return {"jobs": jobs, "estimated_cost": len(jobs)}


class ExecutionAgent:
    """承認済みタスクの実行・再試行・エラー構造化（§5.6）。"""

    def run(self, state: SessionState, gateway: ToolGateway, task: Task) -> dict[str, Any]:
        jobs = task.inputs.get("jobs", [])
        results: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        max_retries = state.stop_conditions.max_retries_per_task
        for job in jobs:
            if state.budget.used_model_runs >= state.budget.max_model_runs:
                failures.append({"job": job, "error": "budget exceeded"})
                break
            inputs = dict(job["inputs"])
            attempt = 0
            while True:
                try:
                    res = gateway.run_model(job["model_id"], inputs)
                    state.budget.used_model_runs += 1
                    results.append({"job": job, "result": res})
                    break
                except ToolError as exc:
                    err = record_error(task.task_id, exc.message)
                    state.errors.append(err)
                    if requires_human_review(err.error_type):
                        failures.append({"job": job, "error": exc.message,
                                         "needs_human": True})
                        break
                    fixed = try_auto_fix(err.error_type, inputs)
                    attempt += 1
                    if fixed is None or attempt > max_retries:
                        failures.append({"job": job, "error": exc.message,
                                         "needs_human": fixed is None})
                        break
                    inputs = fixed
                    err.resolved = True
                    state.audit("ExecutionAgent", "auto_fix_retry",
                                task_id=task.task_id, attempt=attempt,
                                error_type=err.error_type.value)
        task.retry_count = max(task.retry_count, 0)
        return {"results": results, "failures": failures,
                "n_success": len(results), "n_failed": len(failures)}


class EvaluationAgent:
    """モデル出力比較と仮説支持候補の整理（§5.7）。

    最終的な科学的結論は自動確定しない（判定材料の整理まで）。
    """

    def run(self, state: SessionState, gateway: ToolGateway, task: Task) -> dict[str, Any]:
        exec_results = task.inputs.get("results", [])
        by_comp: dict[float, list[float]] = {}
        for r in exec_results:
            x_al = float(r["job"]["inputs"]["composition"].get("Al", 0.5))
            by_comp.setdefault(x_al, []).append(float(r["result"]["prediction"]))
        xs = sorted(by_comp)
        means = [gateway.analyze(by_comp[x])["mean"] for x in xs]
        slope = 0.0
        if len(xs) >= 2:
            n = len(xs)
            mx, my = sum(xs) / n, sum(means) / n
            denom = sum((x - mx) ** 2 for x in xs)
            if denom > 0:
                slope = sum((x - mx) * (y - my) for x, y in zip(xs, means)) / denom
        stds = [gateway.analyze(by_comp[x])["std"] for x in xs]
        mean_std = sum(stds) / len(stds) if stds else 0.0
        verdict_candidate = "inconclusive"
        if xs and abs(slope) > 3 * (mean_std + 1e-9):
            verdict_candidate = "supported" if slope > 0 else "falsification_candidate"
        data_gaps = self._data_gaps(state, gateway, exec_results, xs, slope, mean_std)
        for h in state.hypotheses:
            if h.status == HypothesisState.APPROVED_FOR_TESTING:
                h.status = HypothesisState.UNDER_EVALUATION
        groups = set()
        for r in exec_results:
            info = gateway.get_model(r["job"]["model_id"])
            if info is not None:
                groups.add(info.independence_group)
        judgements: dict[str, dict[str, Any]] = {}
        if exec_results:
            for h in state.hypotheses:
                if h.counter_to is not None or h.status in (
                    HypothesisState.ARCHIVED, HypothesisState.REJECTED_BY_HUMAN,
                ):
                    continue
                j = judge_hypothesis(
                    h, slope=slope, mean_uncertainty=mean_std,
                    n_points=len(xs), n_independent_groups=len(groups),
                )
                h.judgement = j
                judgements[h.hypothesis_id] = j.model_dump()
        evaluation = StepEvaluation(
            step_result="completed",
            goal_progress_before=state.last_observation.goal_progress if state.last_observation else 0.0,
            goal_progress_after=min(1.0, (state.last_observation.goal_progress if state.last_observation else 0.0) + 0.2),
            hypotheses_affected=[h.hypothesis_id for h in state.hypotheses],
            result_quality="acceptable" if exec_results else "insufficient",
            requires_replanning=not exec_results,
            data_gaps=data_gaps,
        )
        state.evaluations.append(evaluation)
        return {
            "slope": slope,
            "mean_uncertainty": mean_std,
            "verdict_candidate": verdict_candidate,
            "judgements": judgements,
            "data_gaps": data_gaps,
            "note": "最終判定は研究者が行う（Human-in-the-loop）",
            "evaluation": evaluation.model_dump(),
        }

    def _data_gaps(
        self,
        state: SessionState,
        gateway: ToolGateway,
        exec_results: list[dict[str, Any]],
        xs: list[float],
        slope: float,
        mean_std: float,
    ) -> list[str]:
        """判定材料を強化するために追加的に必要なデータを決定論的に列挙する。"""
        gaps: list[str] = []
        if not exec_results:
            gaps.append("モデル予測結果（承認済みジョブの実行結果）")
            return gaps
        if len(xs) < 5:
            gaps.append(
                f"追加の組成点（現在 {len(xs)} 点。傾き推定の信頼性向上に 5 点以上を推奨）"
            )
        groups = set()
        for r in exec_results:
            info = gateway.get_model(r["job"]["model_id"])
            if info is not None:
                groups.add(info.independence_group)
        if len(groups) < 2:
            gaps.append("独立なモデル系列（別の independence_group）による再予測")
        if xs and abs(slope) <= 3 * (mean_std + 1e-9):
            gaps.append("不確実性低減のための追加サンプリング（傾きが不確実性に埋もれている）")
        if not any(e.evidence_type == "experiment" for e in state.evidence):
            gaps.append("実験参照値（相安定性・欠陥形成エネルギーの実測データ）")
        temps = {r["job"]["inputs"].get("temperature") for r in exec_results}
        if len(temps) < 2:
            gaps.append("温度依存性の確認（複数温度での予測）")
        return gaps
