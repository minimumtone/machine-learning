"""外部ジョブ投入（Slurm/モック）・DFT入力生成・OQMD・事例還元の試験。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mi_hub.agent.dft import format_poscar, format_poscar_b2, write_vasp_inputs
from mi_hub.agent.graphrag import build_default_provider
from mi_hub.agent.loop import ResearchManager, SessionStore
from mi_hub.agent.oqmd import OQMDProvider
from mi_hub.agent.scheduler import (
    LocalMockScheduler,
    SchedulerError,
    SlurmScheduler,
    build_scheduler,
    estimate_node_hours,
    make_sbatch_script,
)
from mi_hub.agent.tools import ToolError, ToolGateway


@pytest.fixture()
def manager(tmp_path):
    return ResearchManager(gateway=ToolGateway(),
                           store=SessionStore(str(tmp_path)),
                           scheduler=LocalMockScheduler())


@pytest.fixture()
def session(manager):
    return manager.create_session("Al-Mn B2構造の生成エネルギーをDFTで検証する")


class TestScheduler:
    def test_sbatch_script_format(self):
        script = make_sbatch_script("srun vasp_std", "b2_almn",
                                    partition="regular", nodes=2,
                                    time_limit="02:00:00",
                                    modules=["vasp/6.4"])
        assert "#SBATCH --job-name=b2_almn" in script
        assert "#SBATCH --nodes=2" in script
        assert "#SBATCH --partition=regular" in script
        assert "module load vasp/6.4" in script
        assert script.rstrip().endswith("srun vasp_std")

    def test_estimate_node_hours(self):
        assert estimate_node_hours(2, "02:30:00") == pytest.approx(5.0)
        assert estimate_node_hours(1, "1-00:00:00") == pytest.approx(24.0)
        assert estimate_node_hours(1, "30:00") == pytest.approx(0.5)

    def test_local_mock_lifecycle(self, tmp_path):
        sched = LocalMockScheduler()
        jid = sched.submit("echo hello > result.txt", str(tmp_path / "wd"), "job1")
        assert sched.status(jid) == "completed"
        assert (tmp_path / "wd" / "result.txt").read_text().strip() == "hello"
        jid2 = sched.submit("exit 3", str(tmp_path / "wd"), "job2")
        assert sched.status(jid2) == "failed"
        with pytest.raises(SchedulerError):
            sched.status("unknown-id")

    def test_slurm_status_mapping(self, monkeypatch):
        sched = SlurmScheduler()
        outputs = {}

        def fake_run(args):
            return subprocess.CompletedProcess(args, 0,
                                               stdout=outputs[args[0]], stderr="")

        monkeypatch.setattr(sched, "_run", fake_run)
        outputs.update({"squeue": "RUNNING\n"})
        assert sched.status("123") == "running"
        outputs.update({"squeue": "", "sacct": "COMPLETED\n"})
        assert sched.status("123") == "completed"
        outputs.update({"sacct": "CANCELLED by 1000\n"})
        assert sched.status("123") == "cancelled"
        outputs.update({"sacct": ""})
        with pytest.raises(SchedulerError):
            sched.status("123")

    def test_build_scheduler(self):
        assert build_scheduler("local_mock").name == "local_mock"
        assert build_scheduler("slurm", ssh_host="hpc").ssh_host == "hpc"
        with pytest.raises(ValueError):
            build_scheduler("pbs")


class TestJobWorkflow:
    def test_unapproved_job_cannot_be_submitted(self, manager, session):
        job = manager.propose_job(session, "b2_almn", "echo dft",
                                  estimated_node_hours=1.0)
        assert job.state == "proposed"
        with pytest.raises(ValueError, match="未承認"):
            manager.submit_approved_job(session, job.job_id)

    def test_approved_job_runs_and_records_evidence(self, manager, session):
        job = manager.propose_job(session, "b2_almn",
                                  "echo -8.21 > energy.txt",
                                  estimated_node_hours=0.5)
        manager.resolve_approval(session, job.approval_id, True)
        job = manager.submit_approved_job(session, job.job_id)
        assert job.state == "completed"
        assert session.budget.used_node_hours == pytest.approx(0.5)
        ev = [e for e in session.evidence if e.source_type == "external_job"]
        assert ev and "energy.txt" in ev[0].conditions["generated_files"]
        # 再投入は拒否される
        with pytest.raises(ValueError, match="投入済み"):
            manager.submit_approved_job(session, job.job_id)

    def test_budget_limit_blocks_submission(self, manager, session):
        session.budget.max_node_hours = 1.0
        job = manager.propose_job(session, "big", "echo x",
                                  estimated_node_hours=5.0)
        manager.resolve_approval(session, job.approval_id, True)
        with pytest.raises(ValueError, match="予算不足"):
            manager.submit_approved_job(session, job.job_id)

    def test_session_persists_jobs(self, manager, session):
        manager.propose_job(session, "b2_almn", "echo dft")
        loaded = manager.store.load(session.session_id)
        assert loaded.jobs and loaded.jobs[0].name == "b2_almn"


class TestDFTInputs:
    def test_write_vasp_inputs(self, tmp_path):
        poscar = format_poscar_b2("Al", "Mn", 2.95)
        files = write_vasp_inputs(str(tmp_path / "b2"), poscar)
        assert files == ["INCAR", "KPOINTS", "POSCAR"]
        text = (tmp_path / "b2" / "POSCAR").read_text()
        assert "Al Mn" in text and "0.5 0.5 0.5" in text
        assert "ENCUT = 520" in (tmp_path / "b2" / "INCAR").read_text()

    def test_format_poscar_validates(self):
        with pytest.raises(ValueError):
            format_poscar("bad", 4.0, [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                          ["Al"], [2], [[0, 0, 0]])


class TestOQMD:
    def test_search_extracts_composition_and_maps_schema(self, monkeypatch):
        def fake_get(url, params=None, timeout=None):
            assert params["composition"] == "Al2O3"
            return httpx.Response(
                200,
                json={"data": [{"name": "Al2O3", "entry_id": 14842,
                                "delta_e": -3.2536, "stability": 0.0115,
                                "spacegroup": "R-3c", "band_gap": 5.8,
                                "prototype": "corundum"}]},
                request=httpx.Request("GET", url),
            )

        monkeypatch.setattr(httpx, "get", fake_get)
        docs = OQMDProvider().search("Al2O3 の生成エネルギーを知りたい")
        assert docs and docs[0]["source_type"] == "database"
        assert "-3.2536 eV/atom" in docs[0]["claim"]
        assert docs[0]["conditions"]["entry_id"] == 14842

    def test_invalid_composition_rejected(self):
        with pytest.raises(ToolError, match="不正な組成"):
            OQMDProvider().get_formation_energies("../etc")

    def test_api_error_becomes_tool_error(self, monkeypatch):
        def fake_get(url, params=None, timeout=None):
            raise httpx.ConnectError("proxy required")

        monkeypatch.setattr(httpx, "get", fake_get)
        with pytest.raises(ToolError, match="OQMD API"):
            OQMDProvider().get_formation_energies("Al2O3")


class TestCaseKnowledge:
    def test_case_report_auto_ingested_into_graphrag(self, manager, session,
                                                     tmp_path):
        provider = build_default_provider(str(tmp_path / "graphrag"))
        manager.gateway.register_knowledge_provider(provider)
        session.chat_history.append(
            {"role": "user", "content": "中距離秩序の安定性をMLIPで評価した"})
        manager.export_case_report(session)
        doc_id = f"case:{session.session_id}"
        assert doc_id in provider.docs
        hits = provider.search("中距離秩序 の過去事例")
        assert any(h.get("doc_id") == doc_id for h in hits)
        assert any(a.action == "case_knowledge_ingested"
                   for a in session.audit_log)
