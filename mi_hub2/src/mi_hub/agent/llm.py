"""LLM 補助層（指示書 §16）。

LLM は自然言語の構造化・仮説候補生成・説明生成のみに使用する。
数値計算・権限判定・承認状態判定・仮説の最終確定には使用しない。
OPENAI_API_KEY が未設定の場合は決定論的フォールバックで動作する。
"""

from __future__ import annotations

import json
import os
from typing import Any


def llm_available() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def _chat_json(system: str, user: str) -> dict[str, Any] | None:
    """OpenAI へ JSON 応答を要求。失敗時は None（フォールバックへ）。"""
    if not llm_available():
        return None
    try:
        from openai import OpenAI

        client = OpenAI()
        resp = client.chat.completions.create(
            model=os.environ.get("MI_HUB_LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        return json.loads(resp.choices[0].message.content or "{}")
    except Exception:
        return None


def structure_goal(statement: str) -> dict[str, Any]:
    """研究目標文を構造化する。LLM 不可時はルールベース。"""
    out = _chat_json(
        "あなたは材料科学の研究目標を構造化するアシスタントです。"
        "JSON で target_material, target_phase, target_property, success_criteria を返してください。",
        statement,
    )
    if out:
        return out
    # フォールバック: 簡易キーワード抽出
    phase = None
    for p in ("B2", "L12", "L1_2", "FCC", "BCC", "HCP"):
        if p in statement:
            phase = p
            break
    material = None
    for m in ("Ni-Al", "NiAl", "Co-Ni-Ta", "Fe-V"):
        if m in statement:
            material = "Ni-Al" if m == "NiAl" else m
            break
    prop = "formation_enthalpy"
    if "格子定数" in statement:
        prop = "lattice_constant"
    elif "拡散" in statement:
        prop = "diffusivity"
    return {
        "target_material": material,
        "target_phase": phase,
        "target_property": prop,
        "success_criteria": [
            "主要仮説と対立仮説が形式化されている",
            "利用可能なモデル群による数値検証が完了している",
            "支持・反証・保留・限定支持の判定材料が提示されている",
        ],
    }


def generate_hypotheses(goal_statement: str, evidence_claims: list[str]) -> list[dict[str, Any]]:
    """主仮説・対立仮説の候補を生成する（最終採用は人間）。"""
    out = _chat_json(
        "材料科学の仮説候補を生成してください。JSON で "
        '{"hypotheses": [{"statement", "is_counter", "supporting_predictions", '
        '"falsification_conditions"}]} を返してください。',
        f"研究目標: {goal_statement}\n証拠: {json.dumps(evidence_claims, ensure_ascii=False)}",
    )
    if out and isinstance(out.get("hypotheses"), list):
        return out["hypotheses"]
    return [
        {
            "statement": f"主仮説: {goal_statement} に対する主要因が成立する",
            "is_counter": False,
            "supporting_predictions": ["独立モデル群の予測が同方向の傾向を示す"],
            "falsification_conditions": ["独立モデル群の過半が逆方向の傾向を示す"],
        },
        {
            "statement": "対立仮説: 別の欠陥・機構が主要因である",
            "is_counter": True,
            "supporting_predictions": ["対象因子を除外しても傾向が維持される"],
            "falsification_conditions": ["対象因子除外時に傾向が消失する"],
        },
    ]


def summarize_for_human(payload: dict[str, Any]) -> str:
    """人間向け説明の生成。LLM 不可時は構造化テキスト。"""
    out = _chat_json(
        "研究エージェントの状態を日本語で簡潔に要約してください。JSON {\"summary\": str}",
        json.dumps(payload, ensure_ascii=False, default=str),
    )
    if out and out.get("summary"):
        return str(out["summary"])
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)
