"""t2X／MCPゲートウェイのモック実装（指示書 §3）。

実運用では MInt モデルレジストリ・推論サービス・GraphRAG・論文検索等の
MCP エンドポイントへ差し替える。ここでは MVP 用に決定論的なモックを提供し、
インターフェイス（ToolGateway）を固定する。
"""

from __future__ import annotations

import hashlib
import os
import random
import subprocess
import tempfile
from typing import Any

from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    model_id: str
    name: str
    target_property: str
    elements: list[str]
    phase: str | None = None
    composition_range: dict[str, list[float]] = Field(default_factory=dict)
    temperature_range_K: list[float] = Field(default_factory=lambda: [300.0, 1500.0])
    cost: str = "low"  # low / high
    maturity: str = "validated"
    independence_group: str = "GROUP-A"


# モックの MInt モデルレジストリ
MOCK_MODELS: list[ModelInfo] = [
    ModelInfo(model_id="MINT-001", name="B2生成エンタルピーGBRT", target_property="formation_enthalpy",
              elements=["Ni", "Al"], phase="B2", temperature_range_K=[300, 1200], cost="low",
              independence_group="GROUP-A"),
    ModelInfo(model_id="MINT-002", name="B2生成エンタルピーNN", target_property="formation_enthalpy",
              elements=["Ni", "Al"], phase="B2", temperature_range_K=[300, 1000], cost="low",
              independence_group="GROUP-B"),
    ModelInfo(model_id="MINT-003", name="欠陥形成エネルギー回帰", target_property="defect_formation_energy",
              elements=["Ni", "Al"], phase="B2", cost="low", independence_group="GROUP-A"),
    ModelInfo(model_id="MINT-004", name="CALPHAD代理モデル", target_property="phase_stability",
              elements=["Ni", "Al", "Cr"], phase=None, cost="high", independence_group="GROUP-C"),
    ModelInfo(model_id="MINT-005", name="格子定数XGBoost", target_property="lattice_constant",
              elements=["Ni", "Al", "Co", "Cr", "Fe"], phase="FCC", cost="low",
              independence_group="GROUP-A"),
]

# モックの文献データベース
MOCK_LITERATURE: list[dict[str, Any]] = [
    {"title": "Point defects and order in B2 NiAl", "source_type": "journal_article",
     "claim": "Al過剰側ではB2規則度が低下する", "evidence_type": "experiment",
     "conditions": {"temperature_K": 1200, "composition_range": {"Al": [0.50, 0.58]}},
     "keywords": ["antisite", "B2", "NiAl", "アンチサイト", "規則度"],
     "limitations": ["欠陥種を直接測定していない"]},
    {"title": "First-principles study of antisite defects in NiAl", "source_type": "journal_article",
     "claim": "Alアンチサイトの形成エネルギーはNi空孔より高い", "evidence_type": "computation",
     "conditions": {"temperature_K": 0}, "keywords": ["antisite", "DFT", "NiAl", "vacancy"],
     "limitations": ["0Kでの計算のみ"]},
    {"title": "Thermodynamic assessment of the Ni-Al system", "source_type": "journal_article",
     "claim": "B2相はNi過剰側で広い安定領域を持つ", "evidence_type": "assessment",
     "conditions": {"composition_range": {"Al": [0.40, 0.55]}},
     "keywords": ["CALPHAD", "NiAl", "phase diagram", "B2", "安定"],
     "limitations": ["欠陥機構は含まない"]},
]


class ToolError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class KnowledgeProvider:
    """外部ナレッジ源（MCPサーバ・GraphRAG・社内DB等）の差し込み口。

    search() を実装したオブジェクトを ToolGateway.register_knowledge_provider()
    で登録すると、文献検索時にモックレジストリと併せて照会される。
    MCP 接続の場合は search() 内で MCP クライアント呼出しを実装する。
    """

    name: str = "knowledge"

    def __init__(self, name: str = "knowledge"):
        self.name = name

    def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """query に関連する文書のリストを返す。

        各文書は search_literature と同じスキーマ
        （title / source_type / claim / evidence_type / conditions /
        keywords / limitations）に従うこと。
        """
        raise NotImplementedError


class ToolGateway:
    """MCPゲートウェイのインターフェイス兼モック実装。"""

    def __init__(self, fail_next: str | None = None):
        self.models = list(MOCK_MODELS)
        self.literature = list(MOCK_LITERATURE)
        self._fail_next = fail_next  # テスト用: 次回呼出しを指定エラーで失敗させる
        self.call_log: list[dict[str, Any]] = []
        self.knowledge_providers: list[KnowledgeProvider] = []

    def register_knowledge_provider(self, provider: KnowledgeProvider) -> None:
        """外部ナレッジ源（MCP/GraphRAG等）を登録する。"""
        self.knowledge_providers.append(provider)

    def search_knowledge(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """モック文献DBと登録済みナレッジプロバイダを横断検索する。"""
        results = self.search_literature(query, limit)
        for r in results:
            r.setdefault("provider", "mock_literature")
        for p in self.knowledge_providers:
            try:
                for doc in p.search(query, limit):
                    doc.setdefault("provider", p.name)
                    results.append(doc)
            except (ToolError, NotImplementedError, ConnectionError, TimeoutError, OSError, ValueError) as exc:
                self.call_log.append({"tool": "search_knowledge", "provider": p.name,
                                      "error": str(exc)})
        return results[:limit]

    def _maybe_fail(self, tool: str) -> None:
        if self._fail_next:
            msg = self._fail_next
            self._fail_next = None
            raise ToolError(msg)

    def inject_failure(self, message: str) -> None:
        self._fail_next = message

    # --- t2LiteratureQuery ---
    def search_literature(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        self._maybe_fail("search_literature")
        self.call_log.append({"tool": "search_literature", "query": query})
        terms = [w for w in query.replace("、", " ").split() if w]
        hits = []
        for doc in self.literature:
            text = doc["title"] + " " + doc["claim"] + " " + " ".join(doc["keywords"])
            score = sum(1 for t in terms if t.lower() in text.lower())
            if score > 0:
                hits.append((score, doc))
        hits.sort(key=lambda x: -x[0])
        # 共有辞書の汚染を防ぐためコピーを返す
        return [{**d, "conditions": dict(d.get("conditions", {}))}
                for _, d in hits[:limit]]

    # --- t2ModelQuery ---
    def search_models(self, target_property: str, elements: list[str],
                      phase: str | None = None) -> list[ModelInfo]:
        self._maybe_fail("search_models")
        self.call_log.append({"tool": "search_models", "property": target_property})
        out = []
        for m in self.models:
            if m.target_property != target_property:
                continue
            if not set(elements) <= set(m.elements):
                continue
            if phase and m.phase and m.phase != phase:
                continue
            out.append(m)
        return out

    def get_model(self, model_id: str) -> ModelInfo | None:
        for m in self.models:
            if m.model_id == model_id:
                return m
        return None

    # --- t2PredictionJob ---
    def run_model(self, model_id: str, inputs: dict[str, Any]) -> dict[str, Any]:
        """モック推論。決定論的（入力ハッシュのシードで再現可能）。"""
        self._maybe_fail("run_model")
        self.call_log.append({"tool": "run_model", "model_id": model_id, "inputs": inputs})
        model = self.get_model(model_id)
        if model is None:
            raise ToolError(f"model_load_error: model {model_id} not found")
        if "composition" not in inputs:
            raise ToolError("missing input: composition")
        temp = inputs.get("temperature")
        unit = str(inputs.get("temperature_unit", "K"))
        if unit.strip().lower() != "k":
            raise ToolError(f"unit mismatch: temperature unit {unit} is not K")
        if temp is not None and model.temperature_range_K:
            lo, hi = model.temperature_range_K
            if not (lo <= float(temp) <= hi):
                raise ToolError(
                    f"out of domain: temperature {temp} K outside model range [{lo}, {hi}]"
                )
        seed = int(hashlib.md5(
            f"{model_id}:{sorted(inputs.items())!r}".encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        comp = inputs["composition"]
        x_al = float(comp.get("Al", 0.5))
        # Alアンチサイト仮説と整合するモック傾向: Al過剰で生成エンタルピー上昇（不安定化）
        base = -0.62 + 1.8 * max(0.0, x_al - 0.5)
        value = base + rng.gauss(0, 0.02)
        return {
            "model_id": model_id,
            "prediction": round(value, 4),
            "unit": "eV/atom",
            "uncertainty": round(abs(rng.gauss(0.03, 0.01)), 4),
            "in_domain": True,
            "independence_group": model.independence_group,
        }

    # --- t2Shell（サンドボックススクリプト実行。実行は必ず人間承認後） ---
    def run_script(
        self, script: str, timeout_s: int = 300, workdir: str | None = None
    ) -> dict[str, Any]:
        """承認済みシェルスクリプトを実行し、exit code / stdout / stderr を返す。

        ライブラリのインストール（pip 等）や計算スクリプトの実行に使う。
        呼び出し側（UI/API）が承認ゲートを通した後にのみ呼ぶこと。
        workdir を指定すると、その作業ディレクトリで実行し、生成された
        ファイル（図・CSV等）の一覧を generated_files として返す。
        """
        self._maybe_fail("run_script")
        self.call_log.append({"tool": "run_script", "script_head": script[:200]})
        if workdir:
            os.makedirs(workdir, exist_ok=True)
            before = set(os.listdir(workdir))
        fd, path = tempfile.mkstemp(suffix=".sh")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(script)
            proc = subprocess.run(
                ["bash", path], capture_output=True, text=True, timeout=timeout_s,
                cwd=workdir, check=False,
            )
            generated: list[str] = []
            if workdir:
                generated = sorted(set(os.listdir(workdir)) - before)
            return {
                "exit_code": proc.returncode,
                "stdout": proc.stdout[-8000:],
                "stderr": proc.stderr[-8000:],
                "workdir": workdir,
                "generated_files": generated,
            }
        except subprocess.TimeoutExpired:
            generated = sorted(set(os.listdir(workdir)) - before) if workdir else []
            return {"exit_code": -1, "stdout": "",
                    "stderr": f"timeout: 実行が {timeout_s} 秒を超過",
                    "workdir": workdir, "generated_files": generated}
        finally:
            os.unlink(path)

    # --- t2Python（統計評価） ---
    def analyze(self, values: list[float]) -> dict[str, float]:
        self._maybe_fail("analyze")
        if not values:
            return {"n": 0, "mean": 0.0, "std": 0.0}
        n = len(values)
        mean = sum(values) / n
        var = sum((v - mean) ** 2 for v in values) / max(1, n - 1)
        return {"n": n, "mean": mean, "std": var ** 0.5}
