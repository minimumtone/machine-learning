"""Stricter, configurable re-implementation of the execution-result metrics.

This module does NOT replace ``evaluation/metrics.py``.  It exists so that the
leniencies baked into the historical metric can be measured, reported, and (if
desired) tightened, without silently changing any number that has already been
published.

Leniencies in the historical ``execution_accuracy_full()`` that this module
makes explicit and controllable:

1. **Empty gold result sets score 1.0 unconditionally.**  If the gold query
   returns zero rows and the generated query also returns zero rows, the
   historical metric awards recall = precision = 1.0.  Any wrong query that
   happens to return nothing therefore scores full marks.  Controlled by
   ``empty_gold``.

2. **Column-subset projection.**  The historical metric intersects the gold
   column names with the result column names and scores only on the
   intersection.  A generated query that omits gold columns is scored on the
   reduced projection, i.e. dropping columns can only help.  Controlled by
   ``column_policy``.

3. **Positional fallback.**  When no column names overlap, the historical
   metric compares the first ``min(len(rc), len(ec))`` columns by position.
   There is no guarantee those columns are semantically aligned.  Controlled
   by ``allow_positional``.

4. **Set semantics.**  Rows are compared as a Python ``set``, so row
   multiplicity is ignored, and row order is ignored even when the gold SQL
   has an ORDER BY.  Controlled by ``multiset`` and ``ordered``.

Usage::

    from evaluation.metrics_strict import score, ScoringPolicy

    policy = ScoringPolicy(column_policy="require_all", empty_gold="exclude")
    result = score(rows, expected_rows, cols, expected_cols, policy=policy)
    result["recall"], result["status"]

``ScoringPolicy.historical()`` reproduces ``execution_accuracy_full()`` exactly
and is verified against it by ``scripts/audit_scoring.py``.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal

__all__ = ["ScoringPolicy", "exact_result_set_match", "score"]


def _normalize_value(v: Any) -> str:
    """Normalize a cell value to a string for comparison.

    Kept byte-compatible with ``evaluation.metrics._normalize_value`` so that
    the historical policy reproduces the historical numbers exactly.
    """
    if v is None:
        return "__NULL__"
    s = str(v).strip()
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
        return f"{f:.6g}"
    except (ValueError, OverflowError):
        return s.lower()


@dataclass(frozen=True)
class ScoringPolicy:
    """How strictly to score an execution result against the gold result.

    Attributes:
        column_policy:
            ``"intersect"`` — score on gold columns that the result also has
            (historical behaviour, lenient).
            ``"require_all"`` — every gold column must be present in the
            result, otherwise the query scores 0.
        allow_positional:
            When no column names overlap, fall back to positional comparison
            (historical behaviour).  When False, such a query scores 0.
        empty_gold:
            ``"credit"`` — empty gold + empty result scores 1.0 (historical).
            ``"exclude"`` — such queries are reported separately and left out
            of the aggregate denominator, because returning nothing is not
            positive evidence of correctness.
        multiset:
            Compare rows as multisets (row multiplicity matters) instead of
            sets.
        ordered:
            Compare rows as ordered sequences.  Only meaningful when the gold
            query has an ORDER BY; the caller decides when to enable it.
            Implies multiset semantics.
    """

    column_policy: Literal["intersect", "require_all"] = "intersect"
    allow_positional: bool = True
    empty_gold: Literal["credit", "exclude"] = "credit"
    multiset: bool = False
    ordered: bool = False

    @classmethod
    def historical(cls) -> "ScoringPolicy":
        """The policy implemented by ``evaluation.metrics``."""
        return cls()

    @classmethod
    def strict(cls) -> "ScoringPolicy":
        """All leniencies removed except row ordering."""
        return cls(
            column_policy="require_all",
            allow_positional=False,
            empty_gold="exclude",
            multiset=True,
            ordered=False,
        )


_ZERO = {"recall": 0.0, "precision": 0.0, "f1": 0.0, "exact_match": 0.0}


def _result(status: str, **kw: float) -> dict[str, Any]:
    out: dict[str, Any] = dict(_ZERO)
    out.update(kw)
    out["status"] = status
    return out


def _project(rows: list[list[Any]], indices: list[int] | None) -> list[tuple[str, ...]]:
    projected = []
    for r in rows:
        if indices is None:
            projected.append(tuple(_normalize_value(v) for v in r))
        else:
            projected.append(tuple(_normalize_value(r[i]) for i in indices if i < len(r)))
    return projected


def _overlap_counts(
    result: list[tuple[str, ...]],
    expected: list[tuple[str, ...]],
    policy: ScoringPolicy,
) -> tuple[int, int, int]:
    """Return (matched, n_result, n_expected) under the configured semantics."""
    if policy.ordered:
        n = min(len(result), len(expected))
        matched = sum(1 for i in range(n) if result[i] == expected[i])
        return matched, len(result), len(expected)
    if policy.multiset:
        rc, ec = Counter(result), Counter(expected)
        matched = sum((rc & ec).values())
        return matched, sum(rc.values()), sum(ec.values())
    rs, es = set(result), set(expected)
    return len(rs & es), len(rs), len(es)


def score(
    result_rows: list[list[Any]],
    expected_rows: list[list[Any]],
    result_columns: list[str] | None = None,
    expected_columns: list[str] | None = None,
    *,
    policy: ScoringPolicy | None = None,
) -> dict[str, Any]:
    """Score one execution result against the gold result.

    Returns a dict with ``recall``, ``precision``, ``f1``, ``exact_match`` and
    a ``status`` string.  ``status == "excluded_empty_gold"`` means the query
    carries no information under this policy and should be dropped from the
    aggregate rather than counted as right or wrong.
    """
    policy = policy or ScoringPolicy.historical()

    if not expected_rows:
        if policy.empty_gold == "exclude":
            return _result("excluded_empty_gold")
        if not result_rows:
            return _result("empty_both", recall=1.0, precision=1.0, f1=1.0, exact_match=1.0)
        return _result("empty_gold_nonempty_result")

    if not result_rows:
        return _result("empty_result")

    r_idx: list[int] | None
    e_idx: list[int] | None
    if result_columns and expected_columns:
        rc = [c.lower() for c in result_columns]
        ec = [c.lower() for c in expected_columns]
        common = [c for c in ec if c in rc]
        missing = [c for c in ec if c not in rc]

        if policy.column_policy == "require_all" and missing:
            return _result("missing_gold_columns")

        if common:
            r_idx = [rc.index(c) for c in common]
            e_idx = [ec.index(c) for c in common]
        else:
            if not policy.allow_positional:
                return _result("no_column_overlap")
            n = min(len(rc), len(ec))
            if n == 0:
                return _result("no_columns")
            r_idx = list(range(n))
            e_idx = list(range(n))
    else:
        r_idx = e_idx = None

    result = _project(result_rows, r_idx)
    expected = _project(expected_rows, e_idx)

    matched, n_result, n_expected = _overlap_counts(result, expected, policy)
    if n_expected == 0:
        return _result("empty_gold_after_projection")

    recall = matched / n_expected
    precision = matched / n_result if n_result else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    exact = 1.0 if (recall == 1.0 and precision == 1.0) else 0.0
    return _result("scored", recall=recall, precision=precision, f1=f1, exact_match=exact)


def exact_result_set_match(
    result_rows: list[list[Any]],
    expected_rows: list[list[Any]],
    result_columns: list[str] | None,
    expected_columns: list[str] | None,
    *,
    ordered: bool = False,
) -> bool:
    """Canonical exact result-set match.

    True only when
    - the result column names equal the gold column names exactly
      (same names, same order, case-insensitive), and
    - the rows agree as multisets of normalized value tuples, and
    - when ``ordered`` is True (the gold query has a total ORDER BY),
      the rows agree as an ordered sequence.

    This is the single "exact" definition of the package.  The lenient
    common-column variant reported historically is exposed separately
    as ``common_column_exact_overlap`` by ``scripts/audit_scoring.py``.
    """
    if expected_columns is not None:
        rc = [c.lower() for c in (result_columns or [])]
        ec = [c.lower() for c in expected_columns]
        if rc != ec:
            return False
    got = [tuple(_normalize_value(v) for v in r) for r in result_rows]
    exp = [tuple(_normalize_value(v) for v in r) for r in expected_rows]
    if ordered:
        return got == exp
    return Counter(got) == Counter(exp)
