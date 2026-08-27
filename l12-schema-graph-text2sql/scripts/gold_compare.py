#!/usr/bin/env python3
"""Comparison policy for gold-SQL results versus expected_results JSON.

Shared by scripts/check_expected_results.py and
scripts/run_gold_verification.py so both verifiers apply exactly the same
semantics:

- Cell normalization is type-respecting: Decimal -> float,
  date/datetime/time -> ISO-8601 string, str stays str (numeric-looking
  TEXT such as "001" is never coerced to a number), int/bool/None pass
  through unchanged.
- Numeric cells (int/float, excluding bool) compare with
  math.isclose(rel_tol=1e-9, abs_tol=1e-8); all other types compare by
  strict equality.
- Column names from cursor.description must equal the stored "columns"
  list exactly (aliases are part of the gold contract).
- Every expected JSON must carry an explicit "ordered" boolean (it is a
  required key, never inferred at verification time, so deleting an
  ORDER BY from a gold SQL cannot silently downgrade the contract to an
  unordered comparison).
- Queries whose outermost statement has an ORDER BY clause carry
  "ordered": true in the expected JSON and compare as sequences (row
  order is significant).
- All other queries compare as multisets (bag equality): duplicate rows
  are preserved, order is not.  Never set equality — duplicate semantics
  matter for Text-to-SQL evaluation.  Pairing uses a sorted fast path
  and falls back to O(n^2) tolerance-aware matching so near-tolerance
  numeric rows cannot be mispaired by the sort key.
"""
from __future__ import annotations

import datetime
import math
import re
from decimal import Decimal

import sqlglot
import sqlglot.errors

REL_TOL = 1e-9
ABS_TOL = 1e-8

_COMMENT_RE = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)
_STRING_RE = re.compile(r"'(?:[^']|'')*'")
_ORDER_BY_RE = re.compile(r"\bORDER\s+BY\b", re.IGNORECASE)


def normalize_cell(v: object) -> object:
    """Normalize one cell to a JSON-native value without type coercion."""
    if isinstance(v, bool) or v is None or isinstance(v, (int, float, str)):
        if isinstance(v, float) and not math.isfinite(v):
            return str(v)
        return v
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, (datetime.date, datetime.datetime, datetime.time)):
        return v.isoformat()
    return str(v)


def normalize_rows(rows: list) -> list[list]:
    """Normalize all cells of all rows (order preserved)."""
    return [[normalize_cell(c) for c in row] for row in rows]


def _sql_is_ordered_lexer(sql: str) -> bool:
    """Lexer-based detection: top-level ORDER BY outside strings/comments.

    Handles line/block comments, single-quoted strings and parenthesis
    depth only. Dollar-quoted strings and quoted identifiers containing
    "ORDER BY" or parentheses are not modeled; the sqlglot cross-check in
    sql_is_ordered guards against such cases for parseable statements.
    """
    stripped = _STRING_RE.sub("''", _COMMENT_RE.sub(" ", sql))
    depth = 0
    for m in re.finditer(r"[()]|\bORDER\s+BY\b", stripped, re.IGNORECASE):
        tok = m.group(0)
        if tok == "(":
            depth += 1
        elif tok == ")":
            depth -= 1
        elif depth == 0:
            return True
    return False


def sql_is_ordered(sql: str) -> bool:
    """True when the outermost statement has a top-level ORDER BY clause.

    Primary detection is a real SQL parse (sqlglot, postgres dialect); the
    simple lexer serves as a cross-check and as fallback for statements
    sqlglot cannot parse. A disagreement between the two is raised rather
    than silently resolved.
    """
    lexer_result = _sql_is_ordered_lexer(sql)
    try:
        expr = sqlglot.parse_one(sql, dialect="postgres")
    except sqlglot.errors.ParseError:
        return lexer_result
    ast_result = expr.args.get("order") is not None
    if ast_result != lexer_result:
        raise ValueError(
            "ORDER BY detection disagreement between sqlglot AST "
            f"({ast_result}) and lexer ({lexer_result}) for SQL: "
            f"{sql[:200]!r}"
        )
    return ast_result


def _cells_equal(a: object, b: object) -> bool:
    """Strict equality, with tolerance only for numeric (non-bool) pairs."""
    a_num = isinstance(a, (int, float)) and not isinstance(a, bool)
    b_num = isinstance(b, (int, float)) and not isinstance(b, bool)
    if a_num and b_num:
        return math.isclose(float(a), float(b),
                            rel_tol=REL_TOL, abs_tol=ABS_TOL)
    return type(a) is type(b) and a == b


def _rows_equal(a: list, b: list) -> bool:
    return (len(a) == len(b)
            and all(_cells_equal(x, y) for x, y in zip(a, b)))


def _sort_key(row: list) -> str:
    """Canonical pairing key for multiset comparison.

    Floats are keyed at 6 decimals only to pair rows between the two
    multisets; the paired rows are then compared with the full tolerance
    policy, so this key does not loosen the comparison itself.
    """
    return repr([round(c, 6) if isinstance(c, float) else c for c in row])


def _multiset_match(actual: list, expected: list) -> bool:
    """Tolerance-aware multiset matching (O(n^2) greedy pairing)."""
    matched = [False] * len(expected)
    for a in actual:
        for i, e in enumerate(expected):
            if not matched[i] and _rows_equal(a, e):
                matched[i] = True
                break
        else:
            return False
    return True


def rows_match(actual: list, expected: list, ordered: bool) -> bool:
    """Compare normalized row lists under the sequence/multiset policy."""
    if len(actual) != len(expected):
        return False
    if ordered:
        return all(_rows_equal(a, e) for a, e in zip(actual, expected))
    a_sorted = sorted(actual, key=_sort_key)
    e_sorted = sorted(expected, key=_sort_key)
    if all(_rows_equal(a, e) for a, e in zip(a_sorted, e_sorted)):
        return True
    return _multiset_match(actual, expected)


def validate_expected_schema(expected: object) -> str | None:
    """Return an error string when the expected JSON is malformed."""
    if not isinstance(expected, dict):
        return "expected JSON is not an object"
    cols = expected.get("columns")
    if not isinstance(cols, list) or not all(isinstance(c, str) for c in cols):
        return "'columns' missing or not a list of strings"
    rows = expected.get("rows")
    if not isinstance(rows, list) or not all(isinstance(r, list) for r in rows):
        return "'rows' missing or not a list of lists"
    ncols = len(cols)
    if any(len(row) != ncols for row in rows):
        return "'rows' contains a row whose length does not match 'columns'"
    if "ordered" not in expected:
        return "'ordered' key is missing"
    if not isinstance(expected["ordered"], bool):
        return "'ordered' is not a boolean"
    return None
