"""計算コード選択（codes / scriptgen / plan_calculation）の試験。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mi_hub.agent.codes import (
    CODE_CATALOG,
    CalcRequirements,
    catalog_summary,
    format_recommendation,
    recommend_codes,
)
from mi_hub.agent.loop import ResearchManager, SessionStore
from mi_hub.agent.scheduler import LocalMockScheduler
from mi_hub.agent.scriptgen import generate_inputs
from mi_hub.agent.tools import ToolGateway


@pytest.fixture()
def manager(tmp_path):
    return ResearchManager(gateway=ToolGateway(),
                           store=SessionStore(str(tmp_path)),
                           scheduler=LocalMockScheduler())


@pytest.fixture()
def session(manager):
    return manager.create_session("Al-Mn B2構造の生成エネルギーを検証する")


class TestRecommendCodes:
    def test_benchmark_small_system_prefers_vasp(self):
        req = CalcRequirements(properties=["formation_enthalpy"],
                               elements=["Al", "Mn"], n_atoms=2,
                               accuracy="benchmark")
        recs = recommend_codes(req)
        assert recs[0].code == "vasp"
        assert any("直接計算可能" in r for r in recs[0].reasons)

    def test_screening_prefers_low_cost(self):
        req = CalcRequirements(properties=["formation_enthalpy"],
                               n_atoms=100, accuracy="screening")
        recs = recommend_codes(req)
        assert recs[0].code in ("mlip", "pycalphad")

    def test_diffusion_md_prefers_dynamics_codes(self):
        req = CalcRequirements(properties=["diffusivity"], n_atoms=5000,
                               dynamics=True, temperature_dependent=True)
        recs = recommend_codes(req)
        assert {r.code for r in recs} <= {"mlip", "lammps"}
        assert recs[0].code == "lammps"  # 5000原子はMLIPも可だがMDに強い

    def test_phase_diagram_prefers_pycalphad(self):
        req = CalcRequirements(properties=["phase_diagram"],
                               phase_diagram=True, temperature_dependent=True)
        recs = recommend_codes(req)
        assert recs[0].code == "pycalphad"

    def test_oversize_penalized(self):
        req = CalcRequirements(properties=["formation_enthalpy"],
                               n_atoms=100000, accuracy="benchmark")
        recs = recommend_codes(req)
        vasp = next(r for r in recs if r.code == "vasp")
        assert any("超過" in c for c in vasp.cautions)

    def test_catalog_and_format(self):
        assert {c.code for c in CODE_CATALOG} == {"vasp", "mlip", "lammps",
                                                  "pycalphad"}
        summary = catalog_summary()
        assert all("limitations" in s for s in summary)
        req = CalcRequirements(properties=["diffusivity"], dynamics=True)
        text = format_recommendation(req, recommend_codes(req))
        assert "計算コードの推薦" in text and "根拠" in text


class TestScriptGen:
    def test_vasp_inputs(self, tmp_path):
        out = generate_inputs("vasp", str(tmp_path), elements=["Al", "Mn"],
                              params={"a0": 2.9})
        assert set(out["files"]) == {"INCAR", "KPOINTS", "POSCAR"}
        assert "srun vasp_std" in out["sbatch"]

    def test_mlip_script(self, tmp_path):
        out = generate_inputs("mlip", str(tmp_path),
                              elements=["Fe", "Ni", "Cr"],
                              params={"temperature": 800})
        script = (tmp_path / "run_mlip.py").read_text()
        assert "CHGNetCalculator" in script and "Langevin" in script
        assert out["command"] == "python3 run_mlip.py"

    def test_lammps_input(self, tmp_path):
        out = generate_inputs("lammps", str(tmp_path), elements=["Al", "Cu"],
                              params={"temperature": 600, "n_steps": 500})
        text = (tmp_path / "in.lammps").read_text()
        assert "pair_style eam/alloy" in text
        assert "velocity all create 600.0" in text
        assert "run 500" in text
        assert out["files"] == ["in.lammps"]

    def test_pycalphad_requires_tdb(self, tmp_path):
        with pytest.raises(ValueError, match="tdb_file"):
            generate_inputs("pycalphad", str(tmp_path), elements=["Fe", "V"])
        out = generate_inputs("pycalphad", str(tmp_path),
                              elements=["Fe", "V"],
                              params={"tdb_file": "fev.tdb"})
        script = (tmp_path / "run_calphad.py").read_text()
        assert "equilibrium" in script and "fev.tdb" in script
        assert out["files"] == ["run_calphad.py"]

    def test_unknown_code(self, tmp_path):
        with pytest.raises(ValueError, match="未対応"):
            generate_inputs("abinit", str(tmp_path), elements=["Si"])


class TestPlanCalculation:
    def test_plan_calculation_records_recommendation(self, manager, session):
        out = manager.plan_calculation(
            session, hypothesis_text="Al-Mn B2 の生成エンタルピーは負である")
        assert out["recommendations"]
        assert any("計算コードの推薦" in m["content"]
                   for m in session.chat_history)
        assert any(e.action == "calculation_planned"
                   for e in session.audit_log)

    def test_plan_calculation_requires_hypothesis(self, manager, session):
        with pytest.raises(ValueError):
            manager.plan_calculation(session)
        with pytest.raises(ValueError):
            manager.plan_calculation(session, hypothesis_id="H-notexist")

    def test_propose_calculation_job_creates_approval(self, manager, session):
        job = manager.propose_calculation_job(
            session, "vasp", ["Al", "Mn"], params={"a0": 2.9},
            estimated_node_hours=2.0)
        assert job.kind == "vasp"
        assert job.state == "proposed"
        approval = session.approval(job.approval_id)
        assert approval is not None and approval.status == "pending"
        assert set(job.detail["generated_files"]) == {"INCAR", "KPOINTS",
                                                      "POSCAR"}
        # 未承認では投入できない
        with pytest.raises(ValueError, match="未承認"):
            manager.submit_approved_job(session, job.job_id)


class TestAnalysisPipeline:
    def test_propose_requires_script_without_llm(self, manager, session,
                                                 monkeypatch):
        monkeypatch.setattr("mi_hub.agent.llm.generate_analysis_script",
                            lambda *a, **k: None)
        with pytest.raises(ValueError, match="解析スクリプト"):
            manager.propose_analysis(session, "格子定数の統計解析")

    def test_run_requires_approval(self, manager, session):
        req = manager.propose_analysis(session, "テスト解析",
                                       script="echo ok")
        assert req.kind == "analysis_execution"
        with pytest.raises(ValueError, match="未承認"):
            manager.run_approved_analysis(session, req.approval_id)

    def test_success_returns_results_to_chat(self, manager, session):
        req = manager.propose_analysis(
            session, "テスト解析",
            script="echo '平均値: 1.23' && echo done > result.txt")
        manager.resolve_approval(session, req.approval_id, approve=True)
        out = manager.run_approved_analysis(session, req.approval_id)
        assert out["ok"] is True
        assert "result.txt" in out["result"]["generated_files"]
        assert any("【解析結果】" in m["content"] and "平均値: 1.23" in m["content"]
                   for m in session.chat_history)
        assert any(e.source_type == "analysis" for e in session.evidence)

    def test_auto_fix_retries_on_error(self, manager, session, monkeypatch):
        req = manager.propose_analysis(session, "失敗する解析",
                                       script="exit 1")
        manager.resolve_approval(session, req.approval_id, approve=True)
        monkeypatch.setattr("mi_hub.agent.llm.fix_analysis_script",
                            lambda script, out, err: "echo fixed")
        out = manager.run_approved_analysis(session, req.approval_id)
        assert out["ok"] is True
        assert len(out["attempts"]) == 2
        assert out["final_script"] == "echo fixed"
        assert any(e.action == "analysis_script_fixed"
                   for e in session.audit_log)

    def test_fix_failure_reports_error(self, manager, session, monkeypatch):
        req = manager.propose_analysis(session, "修正不能な解析",
                                       script="exit 2")
        manager.resolve_approval(session, req.approval_id, approve=True)
        monkeypatch.setattr("mi_hub.agent.llm.fix_analysis_script",
                            lambda *a: None)
        out = manager.run_approved_analysis(session, req.approval_id)
        assert out["ok"] is False
        assert any("自動修正でも解決できませんでした" in m["content"]
                   for m in session.chat_history)

    def test_approval_single_use(self, manager, session):
        req = manager.propose_analysis(session, "一回限り", script="echo once")
        manager.resolve_approval(session, req.approval_id, approve=True)
        assert manager.run_approved_analysis(session, req.approval_id)["ok"]
        with pytest.raises(ValueError, match="実行済み"):
            manager.run_approved_analysis(session, req.approval_id)


class TestReviewFixes:
    def test_unknown_property_falls_back(self):
        req = CalcRequirements(properties=["formation enthalpy of B2"])
        recs = recommend_codes(req)
        assert recs
        assert all("カタログ語彙と一致せず" in r.cautions[0] for r in recs)

    def test_lammps_missing_files_block_submission(self, manager, session):
        job = manager.propose_calculation_job(session, "lammps", ["Al", "Cu"])
        assert set(job.detail["missing_files"]) == {"data.lammps",
                                                    "potential.eam.alloy"}
        approval = session.approval(job.approval_id)
        assert approval is not None and "要配置ファイル" in approval.description
        manager.resolve_approval(session, job.approval_id, approve=True)
        with pytest.raises(ValueError, match="未配置"):
            manager.submit_approved_job(session, job.job_id)
        for f in ("data.lammps", "potential.eam.alloy"):
            Path(job.workdir, f).write_text("dummy")
        submitted = manager.submit_approved_job(session, job.job_id)
        assert submitted.state in ("pending", "running", "completed", "failed")
