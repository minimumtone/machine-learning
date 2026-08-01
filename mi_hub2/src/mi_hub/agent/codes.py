"""計算コードカタログと仮説駆動のコード選択（§検証計画の拡張）。

検証したい仮説から導かれる計算要件（物性・系サイズ・温度・精度・予算）に対して、
利用可能な計算コード（VASP / MLIP / LAMMPS / pycalphad）から最適なものを
決定論的ルールで順位付けする。LLM は要件の構造化と説明文の生成にのみ使い、
コードの採否そのものはこのカタログのルールで決める（§16 と同じ方針）。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CalcRequirements(BaseModel):
    """仮説の検証に必要な計算の要件。"""

    properties: list[str] = Field(default_factory=list)  # 例: formation_enthalpy
    elements: list[str] = Field(default_factory=list)
    n_atoms: int = 100  # 想定系サイズ（原子数）
    temperature_dependent: bool = False  # 有限温度・温度依存性が必要か
    dynamics: bool = False  # 拡散・MD 等の時間発展が必要か
    phase_diagram: bool = False  # 相図・相分率・平衡計算が必要か
    accuracy: str = "standard"  # screening / standard / benchmark
    magnetic: bool = False
    notes: str = ""


class CodeSpec(BaseModel):
    """計算コードの適用範囲・精度・コストの記述。"""

    code: str
    method: str
    properties: list[str] = Field(default_factory=list)
    max_atoms: int = 10**9
    supports_dynamics: bool = False
    supports_phase_diagram: bool = False
    finite_temperature: bool = False
    accuracy_rank: int = 2  # 3=第一原理級, 2=MLIP/良質ポテンシャル, 1=経験則・DB内挿
    cost_rank: int = 2  # 3=高コスト(HPC必須), 2=中, 1=低（ローカル可）
    resource: str = "local"  # local / hpc
    limitations: list[str] = Field(default_factory=list)


#: 物性語彙は llm._PROPERTY_ALIASES と同じスネークケースを使う
CODE_CATALOG: list[CodeSpec] = [
    CodeSpec(
        code="vasp",
        method="DFT（平面波・PAW）",
        properties=["formation_enthalpy", "defect_formation_energy",
                    "lattice_constant", "phase_stability", "elastic_constants",
                    "electronic_structure", "magnetic_moment"],
        max_atoms=300,
        accuracy_rank=3, cost_rank=3, resource="hpc",
        limitations=["数百原子まで（SQSは~128原子が実用上限）",
                     "0 K 静的計算が基本（有限温度はフォノン等の追加計算が必要）",
                     "POTCAR ライセンスが必要（HPC側の VASP_PP_PATH）"],
    ),
    CodeSpec(
        code="mlip",
        method="機械学習ポテンシャル（CHGNet/MACE + ASE）",
        properties=["formation_enthalpy", "defect_formation_energy",
                    "lattice_constant", "phase_stability", "elastic_constants",
                    "diffusivity", "thermal_expansion"],
        max_atoms=20000, supports_dynamics=True, finite_temperature=True,
        accuracy_rank=2, cost_rank=1, resource="local",
        limitations=["DFT の代替近似（学習データ外の組成・構造では精度低下）",
                     "電子構造・磁性の直接予測は不可"],
    ),
    CodeSpec(
        code="lammps",
        method="古典/機械学習ポテンシャル MD（EAM/MEAM/MLIP）",
        properties=["diffusivity", "thermal_expansion", "melting_point",
                    "elastic_constants", "defect_formation_energy",
                    "mechanical_response"],
        max_atoms=10**7, supports_dynamics=True, finite_temperature=True,
        accuracy_rank=2, cost_rank=2, resource="hpc",
        limitations=["対象元素系の良質なポテンシャル（EAM/MLIP）の有無に依存",
                     "電子状態は扱えない"],
    ),
    CodeSpec(
        code="pycalphad",
        method="CALPHAD（熱力学データベース計算）",
        properties=["phase_stability", "phase_diagram", "phase_fraction",
                    "formation_enthalpy", "heat_capacity", "driving_force"],
        max_atoms=10**9, supports_phase_diagram=True, finite_temperature=True,
        accuracy_rank=1, cost_rank=1, resource="local",
        limitations=["評価済み TDB データベースが必要（未評価系は外挿）",
                     "原子スケールの構造・欠陥は扱えない"],
    ),
]


def catalog_summary() -> list[dict[str, Any]]:
    """LLM プロンプトや UI 表示用のカタログ要約。"""
    return [
        {"code": c.code, "method": c.method, "properties": c.properties,
         "max_atoms": c.max_atoms, "resource": c.resource,
         "limitations": c.limitations}
        for c in CODE_CATALOG
    ]


class CodeRecommendation(BaseModel):
    code: str
    method: str
    score: float
    reasons: list[str] = Field(default_factory=list)
    cautions: list[str] = Field(default_factory=list)
    resource: str = "local"


def recommend_codes(req: CalcRequirements) -> list[CodeRecommendation]:
    """要件に対する計算コードの順位付け（決定論的）。

    スコア: 物性の適合を主とし、系サイズ・温度・動力学・相図の要件で加減点。
    精度要求 benchmark は第一原理級を優先、screening は低コストを優先する。
    """
    out: list[CodeRecommendation] = []
    props = [p.strip().lower() for p in req.properties if p]
    for spec in CODE_CATALOG:
        reasons: list[str] = []
        cautions: list[str] = list(spec.limitations)
        matched = [p for p in props if p in spec.properties]
        if props and not matched:
            continue  # 対象物性を計算できないコードは候補にしない
        score = 2.0 * len(matched) if props else 1.0
        if matched:
            reasons.append(f"対象物性を直接計算可能: {', '.join(matched)}")
        if req.n_atoms > spec.max_atoms:
            score -= 5.0
            cautions.append(
                f"系サイズ {req.n_atoms} 原子は上限 {spec.max_atoms} 原子を超過")
        else:
            reasons.append(f"系サイズ {req.n_atoms} 原子は適用範囲内")
            if req.n_atoms > 1000 and spec.max_atoms >= 10**6:
                score += 1.0
                reasons.append("大規模系に十分な余裕がある")
        if req.dynamics:
            if spec.supports_dynamics:
                score += 2.0
                reasons.append("時間発展（MD）に対応")
            else:
                score -= 3.0
                cautions.append("MD・時間発展は非対応")
        if req.phase_diagram:
            if spec.supports_phase_diagram:
                score += 3.0
                reasons.append("相図・平衡計算に対応")
            else:
                score -= 2.0
                cautions.append("相図の直接計算は非対応")
        if req.temperature_dependent:
            if spec.finite_temperature:
                score += 1.0
                reasons.append("有限温度の効果を扱える")
            else:
                score -= 1.0
                cautions.append("0 K 計算が基本（有限温度は追加計算が必要）")
        if req.accuracy == "benchmark":
            score += 1.5 * spec.accuracy_rank
            if spec.accuracy_rank >= 3:
                reasons.append("ベンチマーク精度（第一原理級）")
        elif req.accuracy == "screening":
            score += 1.5 * (4 - spec.cost_rank)
            if spec.cost_rank <= 1:
                reasons.append("低コストでスクリーニングに適する")
        else:  # standard: 精度とコストのバランス
            score += 0.5 * spec.accuracy_rank + 0.5 * (4 - spec.cost_rank)
        if req.magnetic and spec.code != "vasp":
            cautions.append("磁性の直接的な取り扱いは限定的")
        out.append(CodeRecommendation(
            code=spec.code, method=spec.method, score=round(score, 2),
            reasons=reasons, cautions=cautions, resource=spec.resource,
        ))
    out.sort(key=lambda r: r.score, reverse=True)
    if not out and props:
        # 物性名が語彙と一致せず全コードが除外された場合のフォールバック:
        # 物性条件を外して全候補を提示し、その旨を留意点に明記する
        out = recommend_codes(req.model_copy(update={"properties": []}))
        for r in out:
            r.cautions.insert(
                0, f"指定物性（{', '.join(props)}）はカタログ語彙と一致せず、"
                   "物性条件を外して順位付けした")
    return out


def format_recommendation(req: CalcRequirements,
                          recs: list[CodeRecommendation]) -> str:
    """推薦結果の人間向け整形（チャット表示用）。"""
    lines = ["**計算コードの推薦（仮説検証向け）**", "",
             f"- 対象物性: {', '.join(req.properties) or '未指定'}",
             f"- 系サイズ: 約 {req.n_atoms} 原子 / 精度要求: {req.accuracy}"]
    flags = []
    if req.temperature_dependent:
        flags.append("温度依存性")
    if req.dynamics:
        flags.append("時間発展（MD）")
    if req.phase_diagram:
        flags.append("相図・平衡")
    if flags:
        lines.append(f"- 追加要件: {', '.join(flags)}")
    lines.append("")
    for i, r in enumerate(recs[:3], 1):
        lines.append(f"{i}. **{r.code}**（{r.method}、score={r.score}）")
        lines += [f"   - 根拠: {x}" for x in r.reasons]
        lines += [f"   - 留意点: {x}" for x in r.cautions[:2]]
    return "\n".join(lines)
