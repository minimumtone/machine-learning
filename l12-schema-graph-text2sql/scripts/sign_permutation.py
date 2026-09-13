"""Exact sign-permutation test of the Wilcoxon signed-rank statistic.

Both paired comparisons in the paper (ablation conditions vs. ``full`` and
EN vs. JA) use the same test: the two-sided p-value of the Wilcoxon
signed-rank statistic under the exact sign-permutation null, i.e. all
``2**k`` sign assignments of the midranks of the ``k`` non-zero paired
differences are enumerated and the assignments whose ``min(W+, W-)`` is at
least as extreme as the observed one are counted.

Tied |diff| values are handled with midranks, which is what makes this
differ from SciPy's ``wilcoxon(method="exact")``: that routine uses the
untied signed-rank distribution table and is therefore not exact when the
per-query means are heavily tied (they only take the values 0, 0.2, ...,
1.0 for five runs).  The enumeration is carried out as an exact integer
dynamic programme over the doubled midranks, so the result is deterministic,
independent of SciPy's version, and feasible for any ``k`` that occurs here
(``k <= 100``).

The paired differences are means of 0/1 (or per-query recall) values, so
mathematically equal differences can arrive with different binary
representations (``0.2`` vs ``0.19999999999999996``).  |diff| is therefore
rounded to ``TIE_DECIMALS`` decimals before ranking so that such values share
a midrank; genuinely distinct differences in these data are separated by far
more than ``10**-TIE_DECIMALS``.
"""
from __future__ import annotations

from collections.abc import Sequence

from scipy.stats import rankdata

TIE_DECIMALS = 9
TEST_LABEL = "sign-permutation-exact-midranks"
TEST_DESCRIPTION = (
    "exact sign-permutation test of the Wilcoxon signed-rank statistic "
    "(midranks for tied |diff|, |diff| rounded to "
    f"{TIE_DECIMALS} decimals before ranking) on per-query mean accuracy, "
    "full - condition (scripts/sign_permutation.py)")


def stats_meta(n_runs: int) -> dict:
    """``_meta`` block identifying the test behind stored ablation p-values."""
    return {
        "n_runs": n_runs,
        "test": TEST_DESCRIPTION,
        "test_label": TEST_LABEL,
        "tie_decimals": TIE_DECIMALS,
        "correction": "Holm-Bonferroni across the ablated conditions",
    }


def sign_permutation_pvalue(nonzero: Sequence[float]) -> float:
    """Two-sided exact sign-permutation p-value for non-zero paired differences.

    Returns 1.0 for an empty input.
    """
    diffs = [float(d) for d in nonzero]
    if not diffs:
        return 1.0
    # Midranks of |diff|; doubling makes them integers (ties give .5 ranks).
    magnitudes = [round(abs(d), TIE_DECIMALS) for d in diffs]
    ranks2 = [int(round(2 * r)) for r in rankdata(magnitudes)]
    total2 = sum(ranks2)
    w_plus2 = sum(r for r, d in zip(ranks2, diffs) if d > 0)
    observed2 = min(w_plus2, total2 - w_plus2)

    # count[s] = number of sign assignments whose positive-rank sum (doubled) is s
    count = [0] * (total2 + 1)
    count[0] = 1
    for r in ranks2:
        for s in range(total2, r - 1, -1):
            count[s] += count[s - r]

    n_extreme = sum(c for s, c in enumerate(count)
                    if min(s, total2 - s) <= observed2)
    return n_extreme / 2 ** len(ranks2)
