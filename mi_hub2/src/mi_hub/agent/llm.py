"""LLM 補助層（指示書 §16）。

LLM は自然言語の構造化・仮説候補生成・説明生成のみに使用する。
数値計算・権限判定・承認状態判定・仮説の最終確定には使用しない。

複数の LLM プロバイダを OpenAI 互換 API 経由で切替できる：
- openai: OPENAI_API_KEY
- anthropic (Claude): ANTHROPIC_API_KEY（OpenAI 互換エンドポイント）
- gemini: GEMINI_API_KEY または GOOGLE_API_KEY（OpenAI 互換エンドポイント）
- local: Ollama 等の OpenAI 互換サーバ（MI_HUB_LLM_BASE_URL、既定
  http://localhost:11434/v1）
選択は MI_HUB_LLM_PROVIDER（未設定時はキーのある最初のプロバイダ）、
モデルは MI_HUB_LLM_MODEL で上書きできる。
キーが無い場合は決定論的フォールバックで動作する。
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

# provider -> (base_url, api_key_env_vars, default_model)
_PROVIDERS: dict[str, tuple[str | None, tuple[str, ...], str]] = {
    "openai": (None, ("OPENAI_API_KEY",), "gpt-4o-mini"),
    "anthropic": (
        "https://api.anthropic.com/v1/",
        ("ANTHROPIC_API_KEY",),
        "claude-3-5-haiku-latest",
    ),
    "gemini": (
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "gemini-2.0-flash",
    ),
    "local": (None, (), "llama3.1"),
}


def _provider_key(provider: str) -> str | None:
    for env in _PROVIDERS[provider][1]:
        if os.environ.get(env):
            return os.environ[env]
    return None


def available_providers() -> list[str]:
    """利用可能なプロバイダ一覧（local は base_url 設定時のみ）。"""
    out = [p for p in ("openai", "anthropic", "gemini") if _provider_key(p)]
    if os.environ.get("MI_HUB_LLM_BASE_URL"):
        out.append("local")
    return out


def current_provider() -> str | None:
    p = os.environ.get("MI_HUB_LLM_PROVIDER")
    if p in _PROVIDERS and (p == "local" or _provider_key(p)):
        return p
    avail = available_providers()
    return avail[0] if avail else None


def llm_available() -> bool:
    return current_provider() is not None


def _client_and_model():
    from openai import OpenAI

    provider = current_provider()
    if provider is None:
        return None, None
    base_url, _, default_model = _PROVIDERS[provider]
    if provider == "local":
        base_url = os.environ.get("MI_HUB_LLM_BASE_URL", "http://localhost:11434/v1")
        api_key = os.environ.get("MI_HUB_LLM_API_KEY", "local")
    else:
        api_key = _provider_key(provider)
    model = os.environ.get("MI_HUB_LLM_MODEL", default_model)
    return OpenAI(base_url=base_url, api_key=api_key), model


def _parse_json_text(text: str) -> dict[str, Any] | None:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        out = json.loads(text)
        return out if isinstance(out, dict) else None
    except json.JSONDecodeError:
        return None


def _chat_json(system: str, user: str) -> dict[str, Any] | None:
    """選択中のプロバイダへ JSON 応答を要求。失敗時は None（フォールバックへ）。"""
    if not llm_available():
        return None
    try:
        client, model = _client_and_model()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.2,
            )
        except Exception:
            # response_format 非対応のプロバイダ（一部ローカルLLM等）はテキストで再試行
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
            )
        return _parse_json_text(resp.choices[0].message.content or "{}")
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


def structure_calc_requirements(hypothesis: str, goal: str = "",
                                catalog: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """仮説文から計算要件（codes.CalcRequirements のフィールド）を構造化する。

    LLM は要件の抽出のみに使い、コードの採否は codes.recommend_codes の
    決定論的ルールで行う。LLM 不可時はキーワードベースのフォールバック。
    """
    out = _chat_json(
        "あなたは材料計算の計画を支援するアシスタントです。検証したい仮説から、"
        "必要な計算の要件を JSON で返してください: "
        '{"properties": list[str]（英語スネークケース、例: formation_enthalpy, '
        "phase_stability, defect_formation_energy, lattice_constant, diffusivity, "
        'phase_diagram, elastic_constants）, "elements": list[str]（元素記号）, '
        '"n_atoms": int（必要な系サイズの目安）, "temperature_dependent": bool, '
        '"dynamics": bool（MD・拡散等の時間発展が必要か）, '
        '"phase_diagram": bool, "accuracy": "screening"|"standard"|"benchmark", '
        '"magnetic": bool, "notes": str}。'
        "利用可能な計算コードの参考情報: "
        + json.dumps(catalog or [], ensure_ascii=False),
        f"研究目標: {goal}\n仮説: {hypothesis}",
    )
    if out:
        props_raw = out.get("properties")
        if isinstance(props_raw, str):
            props_raw = [props_raw]
        if isinstance(props_raw, list):
            normed = [_normalize_property(p) for p in props_raw if p]
            out["properties"] = list(dict.fromkeys(p for p in normed if p))
        return out
    # フォールバック: キーワード抽出
    text = f"{goal} {hypothesis}"
    props: list[str] = []
    for alias, canonical in _PROPERTY_ALIASES.items():
        if alias.lower() in text.lower() and canonical not in props:
            props.append(canonical)
    if "相図" in text or "phase diagram" in text.lower():
        props.append("phase_diagram")
    if "弾性" in text:
        props.append("elastic_constants")
    elements = re.findall(r"\b([A-Z][a-z]?)(?=[-\s、,/]|$)", hypothesis)
    return {
        "properties": props or ["formation_enthalpy"],
        "elements": list(dict.fromkeys(elements))[:6],
        "n_atoms": 100,
        "temperature_dependent": ("温度" in text or "Kで" in text),
        "dynamics": ("拡散" in text or "MD" in text),
        "phase_diagram": ("相図" in text),
        "accuracy": "standard",
        "magnetic": ("磁" in text),
        "notes": "ルールベース抽出（LLM不可時フォールバック）",
    }


_SCRIPT_RULES = (
    "bash として実行可能な完全なスクリプトを生成すること。"
    "ライブラリは `pip install -q <pkg>`（`!pip` は不可）、Python コードは "
    "`python3 - <<'PY'` ... `PY` のヒアドキュメントで埋め込むこと。"
    "図・CSV等の成果物はカレントディレクトリに保存し、1回の実行で"
    "結果を全て出力・保存すること（同じ計算の再実行を避ける）。"
    "matplotlib は `matplotlib.use('Agg')` を先頭で指定し、フォントサイズを"
    "既定の約2倍（plt.rcParams['font.size']=20 程度）、化学式・記号の"
    "添字/上付きは LaTeX 数式表記（例: L1$_2$, R$^2$）を使うこと。"
)


def generate_analysis_script(purpose: str, data_files: list[str],
                             context: str = "") -> str | None:
    """計算データの解析スクリプトを生成する。LLM 不可時は None。"""
    out = _chat_json(
        "あなたは材料計算データの解析スクリプトを書く専門家です。"
        'JSON {"script": str, "summary": str} を返してください。' + _SCRIPT_RULES +
        "データファイルは作業ディレクトリに既に存在するものだけを読み、"
        "存在しないファイルを仮定しないこと。解析結果（統計量・図・表）を"
        "標準出力へ日本語で分かりやすく出力すること。",
        json.dumps({"purpose": purpose, "data_files": data_files,
                    "context": context}, ensure_ascii=False),
    )
    if out and isinstance(out.get("script"), str) and out["script"].strip():
        return out["script"]
    return None


def fix_analysis_script(script: str, stdout: str, stderr: str) -> str | None:
    """実行に失敗した解析スクリプトをエラー内容から修正する。LLM 不可時は None。"""
    out = _chat_json(
        "あなたは解析スクリプトのデバッグ専門家です。実行に失敗したスクリプトと"
        "エラー出力から、原因を特定して修正済みスクリプト全文を返してください。"
        'JSON {"script": str, "diagnosis": str} を返すこと。' + _SCRIPT_RULES +
        "エラーの根本原因（ライブラリ不足・ファイル名誤り・型不一致等）を"
        "diagnosis に日本語で書くこと。修正はスクリプト側で行い、"
        "同じエラーを繰り返さないこと。",
        json.dumps({"script": script, "stdout": stdout[-4000:],
                    "stderr": stderr[-4000:]}, ensure_ascii=False),
    )
    if out and isinstance(out.get("script"), str) and out["script"].strip():
        return out["script"]
    return None


def summarize_analysis_result(purpose: str, stdout: str,
                              generated_files: list[str]) -> str | None:
    """解析結果の人間向け要約。LLM 不可時は None。"""
    out = _chat_json(
        "計算データ解析の実行結果を研究者向けに日本語で要約してください。"
        'JSON {"summary": str} を返すこと。summary は Markdown 箇条書きで、'
        "主要な数値結果の解釈 / 生成された成果物の説明 / 留意点・次の一手を含める。"
        "数値は出力にあるものだけを使い、捏造しないこと。",
        json.dumps({"purpose": purpose, "stdout": stdout[-6000:],
                    "generated_files": generated_files}, ensure_ascii=False),
    )
    if out and out.get("summary"):
        return str(out["summary"])
    return None


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
        "LaTeX 数式表記（例: L1$_2$, R$^2$）を使うこと。"
        "機械学習タスクでは scikit-learn / PyCaret（AutoML）/ XGBoost / LightGBM "
        "が利用可能で、交差検証と評価指標の出力を含めること。"
        "原子・分子スケールのエネルギー・構造計算は大学院レベルの手法を開始点とすること: "
        "原則として MLIP（CHGNet + ASE、インストール済み）または確立された "
        "EAM ポテンシャルを用い、可能なら構造緩和を含めること。"
        "Lennard-Jones 等の玩具的ペアポテンシャルは、ユーザが明示的に"
        "簡易計算を要求した場合を除き使用しないこと。"
        "手法の限界（MLIPはDFTの代替近似である等）を出力に明記すること\n"
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
        client, model = _client_and_model()
        context = dict(context)
        short_term = context.pop("short_term_memory", None)
        long_term = context.pop("long_term_memory", None)
        system = (
            "あなたは材料研究エージェント MI-HUB2 の対話アシスタントです。"
            "以下のセッション状態を踏まえ、研究者と議論を前進させる相棒として日本語で答えてください。"
            "どんなコメントに対しても必ず科学的な返答を行うこと。すなわち応答には常に"
            "(1) 科学的解釈（根拠となる物理・材料科学の知見や証拠IDを明示）"
            "(2) 妥当性・限界（近似、データ・手法の限界）"
            "(3) 推奨する次の一手、を含めること。雑談的なコメントであっても、"
            "研究目標・仮説・証拠と結びつけて科学的観点から応答すること。\n"
            "文脈は二層の記憶で与えられます。文脈維持のため両方を必ず参照すること。\n"
            "【短期記憶】現在セッションの直近状態（直近の評価・タスク・会話・証拠・"
            "未解決エラー）: 直前のやり取りとの一貫性維持に使うこと。\n"
            "【長期記憶】セッション全体の蓄積（研究目標・仮説・全証拠・評価履歴）: "
            "研究文脈全体との整合の維持に使うこと。\n"
            "できること: 状態・計画・仮説・証拠・エラーの説明、研究方針の議論"
            "（特徴量設計、SQS・MLIP・CALPHAD等の手法比較、Materials Project/OQMD等の"
            "データソース活用案など）、次の一手の具体的な提案、次の操作の案内"
            "（実行は「実行を続けて」、一時停止/再開/終了、承認は承認タブまたはチャットで「承認」）。"
            "ライブラリのインストールや計算実行を求められたら、承認後にスクリプトとして"
            "実行できることを案内してください。"
            "仮説の検証にどの計算コードが適するか聞かれたら、利用可能なコード"
            "（VASP=DFT高精度・小規模・HPC必要、MLIP=CHGNet/MACEで中精度・低コスト、"
            "LAMMPS=大規模MD・拡散・有限温度、pycalphad=相図・熱力学平衡）の"
            "適用範囲・精度・コストを比較して根拠付きで推薦し、入力スクリプトの"
            "自動生成と承認付きジョブ投入ができることを案内してください。"
            "承認待ちの操作（pending_approvals）がある場合は、応答の末尾で必ず"
            "「『（操作内容）』という承認事項があります。実行しますか？（承認/却下）」の形で"
            "具体的に問いかけてください。何か実行可能な提案をする場合も"
            "「〜という提案がありますが、実行しますか？」と明示的に確認してください。"
            "してはいけないこと: 仮説の最終判定、承認の代行、数値の捏造。"
            "最終的な科学判断は研究者に委ねる旨を必要に応じて添えてください。\n\n"
            f"【短期記憶】\n{json.dumps(short_term, ensure_ascii=False, default=str)}\n\n"
            f"【長期記憶】\n{json.dumps(long_term, ensure_ascii=False, default=str)}\n\n"
            f"【現在の状態・計画・承認待ち】\n"
            f"{json.dumps(context, ensure_ascii=False, default=str)}"
        )
        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        for msg in history[-10:]:
            if msg.get("role") in ("user", "assistant"):
                messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": message})
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
        )
        return resp.choices[0].message.content
    except Exception:
        return None


def science_comment(goal: str, kind: str, payload: dict[str, Any]) -> str | None:
    """仮説・計算結果への専門的コメント（解釈・妥当性・次の一手）。LLM 不可時は None。

    科学的結論の確定は行わず、判断材料の提示に徹する（最終判断は研究者）。
    """
    out = _chat_json(
        "あなたは材料科学の専門家として研究者に伴走するエージェントです。"
        f"研究目標「{goal}」の文脈で、与えられた{kind}に対する専門的コメントを"
        'JSON {"comment": str} で返してください。comment は日本語の Markdown 箇条書きで、'
        "次を含めること: 科学的解釈（数値・仮説の意味づけ、既知の物理・文献知見との整合/不整合）/ "
        "妥当性の留意点（近似・データの限界）/ 推奨する次の一手（検証手段を具体的に）。"
        "末尾は「（次の一手の内容）という提案がありますが、実行しますか？」の形で"
        "研究者へ明示的に問いかけること。"
        "数値は与えられたものだけを使い、捏造しないこと。"
        "結論の確定はせず、最終判断は研究者に委ねる姿勢を保つこと。",
        json.dumps(payload, ensure_ascii=False, default=str)[:6000],
    )
    if out and out.get("comment"):
        return str(out["comment"])
    return None


def _fallback_proposal_summary(script: str) -> str:
    """LLM 不可時のスクリプト提案要約（決定論的）。"""
    installs = re.findall(r"pip install\s+(?:-q\s+)?([^\n]+)", script)
    outputs = re.findall(r"savefig\(\s*[\"']([^\"']+)|to_csv\(\s*[\"']([^\"']+)"
                         r"|savetxt\(\s*[\"']([^\"']+)", script)
    files = sorted({x for tup in outputs for x in tup if x})
    lines = ["**この提案で行うこと**"]
    if installs:
        lines.append("- ライブラリの導入: " + ", ".join(i.strip() for i in installs))
    lines.append("- Pythonスクリプトをサンドボックス内で1回実行します")
    if files:
        lines.append("- 生成される成果物: " + ", ".join(files))
    lines.append("- 実行結果（出力・図・終了コード）は証拠タブに記録されます")
    lines.append("- 承認するまで実行されません。却下しても状態は変わりません")
    return "\n".join(lines)


def summarize_proposal(description: str, script: str) -> str:
    """スクリプト提案の人間向け要約（目的・内容・成果物・コスト・リスク）。"""
    out = _chat_json(
        "研究エージェントが承認を求めるスクリプト提案を、研究者が判断しやすいよう"
        "日本語で要約してください。JSON {\"summary\": str} を返すこと。summary は "
        "Markdown の箇条書きで、次の項目を含める: "
        "目的（何のための計算か）/ 何をするか（手法・モデル・主要パラメータ）/ "
        "生成される成果物（ファイル名）/ おおよその実行時間・計算コスト / "
        "リスク・限界（近似の程度、環境変更の有無）。"
        "スクリプトに書かれていないことは推測と明記し、数値を捏造しないこと。",
        json.dumps({"description": description, "script": script},
                   ensure_ascii=False),
    )
    if out and out.get("summary"):
        return str(out["summary"])
    return _fallback_proposal_summary(script)


def summarize_for_human(payload: dict[str, Any]) -> str:
    """人間向け説明の生成。LLM 不可時は構造化テキスト。"""
    out = _chat_json(
        "研究エージェントの状態を日本語で簡潔に要約してください。JSON {\"summary\": str}",
        json.dumps(payload, ensure_ascii=False, default=str),
    )
    if out and out.get("summary"):
        return str(out["summary"])
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)
