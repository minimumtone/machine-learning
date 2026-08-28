#!/usr/bin/env python3
"""Audit gold SQL ORDER BY clauses for total ordering.

For every gold SQL (main, transfer, AND the obfuscated transfer suite)
with a top-level ORDER BY, this maps each ORDER BY expression to a
SELECT-output column (by alias or bare column name), executes the query
WITH ITS OUTER LIMIT/OFFSET/FETCH REMOVED, and reports queries whose
candidate rows contain duplicate ORDER BY key tuples — i.e. whose ORDER BY
is not a total order over the full candidate set, so the returned row
sequence (and the LIMIT boundary itself) is not deterministic across
PostgreSQL plans. Removing the limit is essential: a tie at or beyond the
LIMIT boundary is invisible in the limited result but still makes the
returned rows plan-dependent.

ORDER BY expressions that cannot be mapped to an output column are
audited by re-executing the query with the ORDER BY expressions injected
at the head of the SELECT list. Injection is refused (and the query is
reported for MANUAL review, which fails the audit) whenever the rewrite
could change semantics: SELECT DISTINCT, window functions, set-returning
functions, or volatile functions anywhere in the statement. "Injection
executed" is therefore never silently equated with "safely audited".

Usage:
    L12_DSN=... TRANSFER_DSN=... OBF_TRANSFER_DSN=... \
    python scripts/audit_order_totality.py

Exit status: 0 when no ties, nothing unmapped, nothing needing manual
review, and nothing skipped; 1 otherwise.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import psycopg

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

SUITES = [
    # (name, gold dir, DSN env var, qid filter)
    ("main", "gold_sql", "L12_DSN",
     lambda qid: not qid.startswith("q_transfer")),
    ("transfer", "gold_sql", "TRANSFER_DSN",
     lambda qid: qid.startswith("q_transfer")),
    ("obfuscated", "gold_sql_obfuscated", "OBF_TRANSFER_DSN",
     lambda qid: True),
]

_COMMENT_RE = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)
_STRING_RE = re.compile(r"'(?:[^']|'')*'")
# Constructs that make SELECT-list injection semantically unsafe: window
# functions, set-returning functions (extra SELECT columns can change row
# multiplication), and volatile functions (re-execution is not comparable).
_UNSAFE_INJECT_RE = re.compile(
    r"\bOVER\s*\(|\b(generate_series|unnest|regexp_split_to_table|"
    r"string_to_table|random|clock_timestamp|timeofday)\s*\(",
    re.IGNORECASE)


def _blank(sql: str) -> str:
    """Blank comments and string literals, preserving length/positions."""
    def repl(m: re.Match) -> str:
        return " " * len(m.group(0))
    return _STRING_RE.sub(repl, _COMMENT_RE.sub(repl, sql))


def top_level_order_by(sql: str) -> str | None:
    """Return the text of the outermost trailing ORDER BY list, if any."""
    blanked = _blank(sql)
    depth = 0
    start = None
    for m in re.finditer(r"[()]|\bORDER\s+BY\b", blanked, re.IGNORECASE):
        tok = m.group(0)
        if tok == "(":
            depth += 1
        elif tok == ")":
            depth -= 1
        elif depth == 0:
            start = m.end()
    if start is None:
        return None
    tail = blanked[start:]
    stop = re.search(r"\b(LIMIT|OFFSET|FETCH|FOR)\b|;", tail, re.IGNORECASE)
    end = start + (stop.start() if stop else len(tail))
    return sql[start:end].strip()


def strip_outer_limit(sql: str) -> str:
    """Remove the outermost LIMIT/OFFSET/FETCH clause (keep ORDER BY).

    The audit must inspect the full candidate set: a tie hidden at or
    beyond the LIMIT boundary still makes the limited result
    plan-dependent even though the returned rows show no duplicate keys.
    """
    blanked = _blank(sql)
    depth = 0
    cut = None
    for m in re.finditer(r"[()]|\b(LIMIT|OFFSET|FETCH)\b", blanked,
                         re.IGNORECASE):
        tok = m.group(0)
        if tok == "(":
            depth += 1
        elif tok == ")":
            depth -= 1
        elif depth == 0:
            cut = m.start()
            break
    if cut is None:
        return sql
    return sql[:cut].rstrip().rstrip(";") + ";"


def split_exprs(clause: str) -> list[str]:
    """Split an ORDER BY list on top-level commas."""
    parts, depth, cur = [], 0, []
    for ch in clause:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur).strip())
    return parts


def key_name(expr: str) -> str:
    """Reduce one ORDER BY expression to its bare key name."""
    e = strip_direction(expr)
    e = e.split(".")[-1].strip()
    return e


def strip_direction(expr: str) -> str:
    """Remove ASC/DESC/NULLS FIRST|LAST from one ORDER BY expression."""
    return re.sub(r"\b(ASC|DESC|NULLS\s+(FIRST|LAST))\b", "", expr,
                  flags=re.IGNORECASE).strip()


def inject_keys_sql(sql: str, exprs: list[str]) -> str | None:
    """Prepend the ORDER BY expressions to the outermost SELECT list.

    Returns None when the outermost SELECT cannot be safely rewritten:
    SELECT DISTINCT (extra columns change the row set), or the statement
    contains window/set-returning/volatile functions (extra SELECT
    columns or re-execution can change semantics). Refusals are surfaced
    as MANUAL review items by the caller, never silently passed.
    """
    blanked = _blank(sql)
    if _UNSAFE_INJECT_RE.search(blanked):
        return None
    depth = 0
    pos = None
    for m in re.finditer(r"[()]|\bSELECT\b", blanked, re.IGNORECASE):
        tok = m.group(0)
        if tok == "(":
            depth += 1
        elif tok == ")":
            depth -= 1
        elif depth == 0:
            pos = m.end()
            break
    if pos is None:
        return None
    if re.match(r"\s*DISTINCT\b", blanked[pos:], re.IGNORECASE):
        return None
    keys = ", ".join(strip_direction(e) for e in exprs)
    return sql[:pos] + " " + keys + ", " + sql[pos:]


def main() -> int:
    if not os.environ.get("L12_DSN"):
        print("ERROR: set L12_DSN", file=sys.stderr)
        return 1
    conns: dict[str, psycopg.Connection | None] = {}
    for _, _, env, _ in SUITES:
        if env in conns:
            continue
        dsn = os.environ.get(env)
        conns[env] = psycopg.connect(dsn) if dsn else None
    # READ ONLY + REPEATABLE READ: one snapshot per connection covers the
    # whole audit run, so every audited query sees the same DB state.
    for c in conns.values():
        if c is not None:
            c.read_only = True
            c.isolation_level = psycopg.IsolationLevel.REPEATABLE_READ
            with c.cursor() as cur:
                cur.execute("SET statement_timeout = '30s'")
            c.commit()

    ties: list[str] = []
    unmapped: list[str] = []
    manual: list[str] = []
    skipped: list[str] = []
    n_ordered = 0
    for suite, gold_dirname, env, qid_filter in SUITES:
        gold_dir = PROJECT / "evaluation" / gold_dirname
        conn = conns[env]
        for sql_path in sorted(gold_dir.glob("*.sql")):
            qid = sql_path.stem
            if not qid_filter(qid):
                continue
            sql = sql_path.read_text()
            clause = top_level_order_by(sql)
            if clause is None:
                continue
            n_ordered += 1
            label = f"{suite}:{qid}"
            if conn is None:
                skipped.append(label)
                continue
            # Audit the full candidate set: strip the outer LIMIT/OFFSET/
            # FETCH so ties beyond the limit boundary are also detected.
            audit_base = strip_outer_limit(sql)
            with conn.cursor() as cur:
                cur.execute(audit_base)  # type: ignore[arg-type]
                cols = ([d.name for d in cur.description]
                        if cur.description else [])
                rows = cur.fetchall()
            exprs = split_exprs(clause)
            # Map each ORDER BY expression to an output column index;
            # collect the rest for SELECT-list injection.
            mapped: dict[int, int] = {}
            to_inject: list[tuple[int, str]] = []
            for i, expr in enumerate(exprs):
                name = key_name(expr)
                if re.fullmatch(r"\d+", name):
                    mapped[i] = int(name) - 1
                elif name in cols:
                    mapped[i] = cols.index(name)
                else:
                    to_inject.append((i, expr))
            if to_inject:
                audit_sql = inject_keys_sql(
                    audit_base, [e for _, e in to_inject])
                if audit_sql is None:
                    manual.append(
                        f"{label}: {', '.join(e for _, e in to_inject)}")
                    continue
                keys = None
                try:
                    # SAVEPOINT: a failing injected query must roll back
                    # only itself, not the outer REPEATABLE READ
                    # transaction, so the rest of the audit still sees
                    # the same snapshot.
                    with conn.cursor() as cur:
                        cur.execute("SAVEPOINT audit_query")
                        try:
                            cur.execute(audit_sql)  # type: ignore[arg-type]
                            n_inj = len(to_inject)
                            keys = []
                            for row in cur.fetchall():
                                key = []
                                inj_pos = {i: p for p, (i, _)
                                           in enumerate(to_inject)}
                                for i in range(len(exprs)):
                                    if i in mapped:
                                        key.append(row[n_inj + mapped[i]])
                                    else:
                                        key.append(row[inj_pos[i]])
                                keys.append(tuple(key))
                            cur.execute("RELEASE SAVEPOINT audit_query")
                        except psycopg.Error:
                            cur.execute("ROLLBACK TO SAVEPOINT audit_query")
                            cur.execute("RELEASE SAVEPOINT audit_query")
                            raise
                except psycopg.Error:
                    pass
                if keys is None:
                    unmapped.append(
                        f"{label}: {', '.join(e for _, e in to_inject)}")
                    continue
                if len(keys) != len(set(keys)):
                    dup = sorted({k for k in keys if keys.count(k) > 1})[:2]
                    ties.append(
                        f"{label}: duplicate ORDER BY keys e.g. {dup}")
                continue
            keys = [tuple(row[mapped[i]] for i in range(len(exprs)))
                    for row in rows]
            if len(keys) != len(set(keys)):
                dup = sorted({k for k in keys if keys.count(k) > 1})[:2]
                ties.append(f"{label}: duplicate ORDER BY keys e.g. {dup}")
    for name, items in (("NON-TOTAL ORDER", ties), ("UNMAPPED", unmapped),
                        ("MANUAL REVIEW", manual),
                        ("SKIPPED (no DSN)", skipped)):
        for item in items:
            print(f"{name}: {item}")
    print(f"\nordered_queries={n_ordered} non_total_order={len(ties)} "
          f"unmapped={len(unmapped)} manual={len(manual)} "
          f"skipped={len(skipped)}")
    return 1 if (ties or unmapped or manual or skipped) else 0


if __name__ == "__main__":
    sys.exit(main())
