"""OQMD（Open Quantum Materials Database）実データ接続。

oqmdapi（https://oqmd.org/oqmdapi）から生成エネルギー・安定性を取得し、
KnowledgeProvider として ToolGateway に登録すると文献検索と併せて
実データが証拠収集へ流れる。プロキシは httpx が環境変数
（http_proxy / https_proxy）から自動適用する。
"""

from __future__ import annotations

import re
from typing import Any

import httpx

from .tools import KnowledgeProvider, ToolError

_BASE_URL = "https://oqmd.org/oqmdapi"
_COMPOSITION_RE = re.compile(r"^(?:[A-Z][a-z]?\d*)+$")


class OQMDProvider(KnowledgeProvider):
    """OQMD の生成エネルギーデータを検索結果（文書スキーマ）として返す。"""

    def __init__(self, base_url: str = _BASE_URL, timeout_s: float = 30.0):
        super().__init__("oqmd")
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def get_formation_energies(self, composition: str,
                               limit: int = 10) -> list[dict[str, Any]]:
        """組成（例: 'Al2O3'、'Al-Mn'）の生成エネルギーエントリを取得する。"""
        if not _COMPOSITION_RE.match(composition.replace("-", "")):
            raise ToolError(f"不正な組成指定: {composition!r}")
        try:
            resp = httpx.get(
                f"{self.base_url}/formationenergy",
                params={
                    "composition": composition,
                    "limit": limit,
                    "fields": "name,entry_id,spacegroup,delta_e,stability,"
                              "band_gap,prototype",
                },
                timeout=self.timeout_s,
            )
            resp.raise_for_status()
            payload = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ToolError(f"OQMD API 呼出し失敗: {exc}") from exc
        return payload.get("data", [])

    def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """query から組成らしき語を抽出して OQMD を照会する。"""
        compositions = [
            tok for tok in re.split(r"[\s,、。]+", query)
            if len(tok) >= 2 and _COMPOSITION_RE.match(tok.replace("-", ""))
            and any(c.islower() or c.isdigit() or c == "-" for c in tok)
        ]
        docs: list[dict[str, Any]] = []
        for comp in compositions[:2]:
            for entry in self.get_formation_energies(comp, limit=limit):
                delta_e = entry.get("delta_e")
                stability = entry.get("stability")
                docs.append({
                    "title": f"OQMD entry {entry.get('entry_id')}: {entry.get('name')}",
                    "source_type": "database",
                    "claim": (
                        f"{entry.get('name')} の DFT 生成エネルギーは "
                        f"{delta_e if delta_e is not None else '不明'} eV/atom"
                        + (f"、convex hull からの距離は {stability} eV/atom"
                           if stability is not None else "")
                    ),
                    "evidence_type": "computation",
                    "conditions": {
                        "entry_id": entry.get("entry_id"),
                        "spacegroup": entry.get("spacegroup"),
                        "prototype": entry.get("prototype"),
                        "band_gap": entry.get("band_gap"),
                        "database": "OQMD",
                    },
                    "keywords": [str(entry.get("name")), "OQMD", "formation energy"],
                    "limitations": [
                        "OQMD の DFT (PBE) 計算値。実験値・他汎関数との差に注意",
                    ],
                })
        return docs[:limit]
