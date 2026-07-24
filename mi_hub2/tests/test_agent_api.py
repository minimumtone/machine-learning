"""研究エージェント API（§14）のテスト。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mi_hub.agent import api
from mi_hub.agent.loop import ResearchManager, SessionStore
from mi_hub.agent.tools import ToolGateway


@pytest.fixture()
def client(tmp_path, monkeypatch):
    manager = ResearchManager(gateway=ToolGateway(), store=SessionStore(str(tmp_path)))
    monkeypatch.setattr(api, "_manager", manager)
    return TestClient(api.app)


def _create(client) -> str:
    r = client.post("/api/agent/goals", json={
        "statement": "Ni-Al B2相のAlアンチサイト仮説を検証する"})
    assert r.status_code == 200
    return r.json()["session_id"]


def test_goal_and_state(client):
    sid = _create(client)
    r = client.get(f"/api/agent/sessions/{sid}/state")
    assert r.status_code == 200
    body = r.json()
    assert body["agent_state"] == "observing"
    assert body["goal"]["target_phase"] == "B2"


def test_plan_lifecycle(client):
    sid = _create(client)
    plan = client.post(f"/api/agent/sessions/{sid}/plans").json()
    assert len(plan["tasks"]) >= 3
    r = client.patch(f"/api/agent/plans/{plan['plan_id']}?session_id={sid}", json={
        "reason": "人間による編集",
        "add_tasks": [{"agent": "EvidenceAgent", "action": "search_literature",
                       "description": "追加検索"}],
    })
    assert r.status_code == 200
    assert r.json()["version"] == plan["version"] + 1


def test_next_actions_and_execute(client):
    sid = _create(client)
    client.post(f"/api/agent/sessions/{sid}/plans")
    actions = client.get(f"/api/agent/sessions/{sid}/next-actions").json()["next_actions"]
    assert actions
    r = client.post("/api/agent/actions", json={
        "session_id": sid, "task_id": actions[0]["task_id"]})
    assert r.status_code == 200
    assert r.json()["status"] == "completed"


def test_approval_flow(client):
    sid = _create(client)
    client.post(f"/api/agent/sessions/{sid}/plans")
    client.post(f"/api/agent/sessions/{sid}/run-auto")
    state = client.get(f"/api/agent/sessions/{sid}/state").json()
    pending = [a for a in state["approvals"] if a["status"] == "pending"]
    assert pending
    r = client.post(f"/api/agent/approvals/{pending[0]['approval_id']}", json={
        "session_id": sid, "approve": True})
    assert r.json()["status"] == "approved"
    client.post(f"/api/agent/sessions/{sid}/run-auto")
    state = client.get(f"/api/agent/sessions/{sid}/state").json()
    exec_task = next(t for t in state["plan"]["tasks"]
                     if t["action"] == "run_models_bulk")
    assert exec_task["status"] in ("completed", "partially_completed")


def test_pause_resume_complete(client):
    sid = _create(client)
    client.post(f"/api/agent/sessions/{sid}/plans")
    assert client.post(f"/api/agent/sessions/{sid}/pause").json()["agent_state"] == "paused"
    assert client.post(f"/api/agent/sessions/{sid}/resume").json()["agent_state"] != "paused"
    assert client.post(f"/api/agent/sessions/{sid}/complete").json()["agent_state"] == "completed"


def test_hypothesis_patch_permissions(client):
    sid = _create(client)
    client.post(f"/api/agent/sessions/{sid}/plans")
    client.post(f"/api/agent/sessions/{sid}/run-auto")
    state = client.get(f"/api/agent/sessions/{sid}/state").json()
    hid = state["hypotheses"][0]["hypothesis_id"]
    r = client.patch(f"/api/agent/hypotheses/{hid}", json={
        "session_id": sid, "falsification_conditions": ["x"], "by": "agent"})
    assert r.status_code == 403
    r = client.patch(f"/api/agent/hypotheses/{hid}", json={
        "session_id": sid, "falsification_conditions": ["x"], "by": "human"})
    assert r.status_code == 200
    assert r.json()["falsification_conditions"] == ["x"]
