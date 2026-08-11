"""仮説の3値判定（Supported / Refuted / Inconclusive）。

数値結果に対する決定論的ルール評価で判定案を作る。
LLM の文章は補助情報であり、判定の主根拠にしない。
最終確定は研究者（Human-in-the-loop）が行う。
"""

from __future__ import annotations

from .models import Hypothesis, HypothesisJudgement, JudgementCriterion

# 効果方向の既定値: 主仮説は「独立変数の増加で目的量が増加する」を正方向とする
_DIRECTIONS = {"positive", "negative"}


def expected_direction(h: Hypothesis) -> str:
    """仮説の期待する効果方向（applicability.expected_direction、既定 positive）。"""
    d = str(h.applicability.get("expected_direction", "positive")).lower()
    return d if d in _DIRECTIONS else "positive"


def judge_hypothesis(
    h: Hypothesis,
    *,
    slope: float,
    mean_uncertainty: float,
    n_points: int,
    n_independent_groups: int,
) -> HypothesisJudgement:
    """効果の有意性・方向・再現性・データ量のルール評価から3値判定案を返す。

    - Refuted: 効果が有意で、かつ方向が仮説の予測と逆（反証条件に該当）
    - Supported: 効果が有意・方向一致・独立系列で再現・十分なデータ点
    - Inconclusive: 上記いずれも満たさない（不足している根拠を明示）
    """
    direction = expected_direction(h)
    threshold = 3.0 * (mean_uncertainty + 1e-9)
    significant = abs(slope) > threshold
    observed = "positive" if slope > 0 else "negative"
    direction_match = significant and observed == direction
    reproduced = n_independent_groups >= 2
    enough_points = n_points >= 3

    criteria = [
        JudgementCriterion(
            name="効果の有意性",
            passed=significant,
            detail=f"|slope|={abs(slope):.4g} vs 3σ閾値={threshold:.4g}",
        ),
        JudgementCriterion(
            name="効果方向の一致",
            passed=direction_match,
            detail=f"予測方向={direction} / 観測方向={observed if significant else '有意でない'}",
        ),
        JudgementCriterion(
            name="独立系列での再現",
            passed=reproduced,
            detail=f"独立モデル系列数={n_independent_groups}（2以上で再現とみなす）",
        ),
        JudgementCriterion(
            name="データ点数",
            passed=enough_points,
            detail=f"条件点数={n_points}（3以上を要求）",
        ),
    ]

    if significant and not direction_match:
        verdict = "refuted"
        rationale = (
            "効果は不確実性に対して有意だが、方向が仮説の予測と逆であり、"
            "反証条件（逆方向の有意な傾向）に該当する。"
        )
    elif significant and direction_match and reproduced and enough_points:
        verdict = "supported"
        rationale = (
            "効果が有意で方向が予測と一致し、独立なモデル系列でも再現され、"
            "十分な条件点数で確認された。"
        )
    else:
        verdict = "inconclusive"
        missing = [c.name for c in criteria if not c.passed]
        rationale = "判定に必要な基準が不足: " + "、".join(missing)

    return HypothesisJudgement(
        verdict=verdict,
        criteria=criteria,
        metrics={
            "slope": slope,
            "mean_uncertainty": mean_uncertainty,
            "n_points": n_points,
            "n_independent_groups": n_independent_groups,
            "expected_direction": direction,
        },
        rationale=rationale + "（最終確定は研究者の承認による）",
        decided_by="rule",
    )
