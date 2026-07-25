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


# モデルレジストリの物性語彙への正規化（LLM が日本語・自由語彙で返す場合の吸収）
_PROPERTY_ALIASES = {
    "相安定性": "phase_stability",
    "phase stability": "phase_stability",
    "生成エンタルピー": "formation_enthalpy",
    "formation enthalpy": "formation_enthalpy",
    "欠陥形成エネルギー": "defect_formation_energy",
    "defect formation": "defect_formation_energy",
    "格子定数": "lattice_constant",
    "lattice constant": "lattice_constant",
    "拡散": "diffusivity",
    "diffusivity": "diffusivity",
}


def _normalize_property(value: str | None) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    for alias, canonical in _PROPERTY_ALIASES.items():
        if alias.lower() in s.lower():
            return canonical
    return s.replace(" ", "_").lower()


def _normalize_goal(out: dict[str, Any]) -> dict[str, Any]:
    """LLM 出力の型・語彙のゆらぎを吸収する（success_criteria が文字列で返る等）。"""
    norm: dict[str, Any] = {}
    for key in ("target_material", "target_phase", "target_property"):
        v = out.get(key)
        if isinstance(v, list):
            v = v[0] if v else None
        norm[key] = str(v) if v is not None else None
    norm["target_property"] = _normalize_property(norm["target_property"])
    sc = out.get("success_criteria")
    if isinstance(sc, str):
        sc = [s.strip() for s in sc.replace("\n", "。").split("。") if s.strip()]
    elif isinstance(sc, list):
        sc = [str(s) for s in sc if s]
    else:
        sc = None
    norm["success_criteria"] = sc or None
    return norm


def structure_goal(statement: str) -> dict[str, Any]:
    """研究目標文を構造化する。LLM 不可時はルールベース。"""
    out = _chat_json(
        "あなたは材料科学の研究目標を構造化するアシスタントです。"
        "JSON で target_material (str), target_phase (str), target_property (str), "
        "success_criteria (list[str]) を返してください。"
        "target_property は英語スネークケース（例: phase_stability, formation_enthalpy, "
        "defect_formation_energy, lattice_constant, diffusivity）で返してください。",
        statement,
    )
    if out:
        return _normalize_goal(out)
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


_INTENTS = {"run", "pause", "resume", "complete", "approve", "reject", "script", "question"}


def classify_intent(message: str, has_pending_approval: bool) -> dict[str, Any]:
    """ユーザ発話の意図を分類する。実行系操作の最終判定は決定論的コード側で行う。

    返り値: {"intent": str, "script": str|None, "reason": str|None}
    intent: run（タスク実行の継続指示）/ pause / resume / complete /
            approve / reject（承認待ち操作への回答）/
            script（シェルスクリプト実行の依頼: ライブラリのインストールや計算実行）/
            question（質問・相談・その他）
    """
    out = _chat_json(
        "あなたは研究エージェントUIの意図分類器です。ユーザの発話を次のいずれかに分類し、"
        'JSON {"intent": str, "script": str|null, "reason": str} を返してください。\n'
        "- run: エージェントの計画タスクの実行を進める明確な指示（例: 実行を続けて、次に進めて）\n"
        "- pause / resume / complete: セッションの一時停止・再開・終了の指示\n"
        f"- approve / reject: 承認待ち操作への諾否（現在の承認待ち: {'あり' if has_pending_approval else 'なし'}）\n"
        "- script: シェル/Pythonでの計算実行やライブラリのインストールの依頼。"
        "この場合 script フィールドに bash として実行可能なスクリプトを生成すること。"
        "ライブラリは `pip install -q <pkg>`（`!pip` は不可）、Python コードは "
        "`python3 - <<'PY'` ... `PY` のヒアドキュメントで埋め込むこと。"
        "改行を保持した完全なスクリプトとし、物理定数・式は正確に書くこと。"
        "図・CSV等の成果物はカレントディレクトリに保存し、1回の実行で"
        "結果を全て出力・保存すること（同じ計算の再実行を避ける）。"
        "matplotlib の図はフォントサイズを既定の約2倍にし"
        "（plt.rcParams['font.size']=20 程度）、化学式・記号の添字/上付きは "
        "LaTeX 数式表記（例: L1$_2$, R$^2$）を使うこと\n"
        "- question: 上記以外（質問・相談・要約依頼など）\n"
        "迷った場合は question を選ぶこと。実行してよいかの最終判断は人間が行う。",
        message,
    )
    if out and out.get("intent") in _INTENTS:
        return {"intent": out["intent"],
                "script": out.get("script") if isinstance(out.get("script"), str) else None,
                "reason": out.get("reason")}
    # フォールバック: 明示的コマンドのみ反応し、それ以外は question
    stripped = message.strip()
    if "一時停止" in stripped:
        return {"intent": "pause", "script": None, "reason": None}
    if "再開" in stripped:
        return {"intent": "resume", "script": None, "reason": None}
    if "終了" in stripped:
        return {"intent": "complete", "script": None, "reason": None}
    if has_pending_approval and stripped in ("承認", "承認します", "OK", "ok", "はい"):
        return {"intent": "approve", "script": None, "reason": None}
    if has_pending_approval and stripped in ("却下", "却下します", "いいえ"):
        return {"intent": "reject", "script": None, "reason": None}
    if any(k in stripped for k in ("実行を続けて", "続けて", "進めて", "自動で実行")):
        return {"intent": "run", "script": None, "reason": None}
    return {"intent": "question", "script": None, "reason": None}


def chat_reply(context: dict[str, Any], history: list[dict[str, str]], message: str) -> str | None:
    """セッション状態を文脈にした自由対話の応答（説明・要約のみ）。LLM 不可時は None。

    科学的結論の確定・承認判定・数値計算は行わない（§16）。
    """
    if not llm_available():
        return None
    try:
        from openai import OpenAI

        client = OpenAI()
        system = (
            "あなたは材料研究エージェント MI-HUB2 の対話アシスタントです。"
            "以下のセッション状態を踏まえ、研究者と議論を前進させる相棒として日本語で答えてください。"
            "できること: 状態・計画・仮説・証拠・エラーの説明、研究方針の議論"
            "（特徴量設計、SQS・MLIP・CALPHAD等の手法比較、Materials Project/OQMD等の"
            "データソース活用案など）、次の一手の具体的な提案、次の操作の案内"
            "（実行は「実行を続けて」、一時停止/再開/終了、承認は承認タブまたはチャットで「承認」）。"
            "ライブラリのインストールや計算実行を求められたら、承認後にスクリプトとして"
            "実行できることを案内してください。"
            "してはいけないこと: 仮説の最終判定、承認の代行、数値の捏造。"
            "最終的な科学判断は研究者に委ねる旨を必要に応じて添えてください。\n\n"
            f"セッション状態:\n{json.dumps(context, ensure_ascii=False, default=str)}"
        )
        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        for msg in history[-10:]:
            if msg.get("role") in ("user", "assistant"):
                messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": message})
        resp = client.chat.completions.create(
            model=os.environ.get("MI_HUB_LLM_MODEL", "gpt-4o-mini"),
            messages=messages,
            temperature=0.3,
        )
        return resp.choices[0].message.content
    except Exception:
        return None


def summarize_for_human(payload: dict[str, Any]) -> str:
    """人間向け説明の生成。LLM 不可時は構造化テキスト。"""
    out = _chat_json(
        "研究エージェントの状態を日本語で簡潔に要約してください。JSON {\"summary\": str}",
        json.dumps(payload, ensure_ascii=False, default=str),
    )
    if out and out.get("summary"):
        return str(out["summary"])
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)
