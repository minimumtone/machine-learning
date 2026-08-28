#!/usr/bin/env python3
"""Audit gold SQL ORDER BY clauses for total ordering.

For every gold SQL with a top-level ORDER BY, this maps each ORDER BY
expression to a SELECT-output column (by alias or bare column name),
executes the query, and reports queries whose returned rows contain
duplicate ORDER BY key tuples — i.e. whose ORDER BY is not a total order,
so the row sequence is not deterministic across PostgreSQL plans.

ORDER BY expressions that cannot be mapped to an output column are
audited by re-executing the query with the ORDER BY expressions injected
at the head of the SELECT list; expressions that cannot be injected either
(e.g. SELECT-list aliases) are reported as UNMAPPED for manual review.

Usage:
    L12_DSN=... TRANSFER_DSN=... python scripts/audit_order_totality.py

Exit status: 0 when no ties and nothing unmapped; 1 otherwise.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import psycopg

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

GOLD_DIR = PROJECT / "evaluation" / "gold_sql"

_COMMENT_RE = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)
_STRING_RE = re.compile(r"'(?:[^']|'')*'")


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

    Returns None when the outermost SELECT cannot be safely rewritten
    (e.g. SELECT DISTINCT, where extra columns change the row set).
    """
    blanked = _blank(sql)
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
    dsn_main = os.environ.get("L12_DSN")
    dsn_transfer = os.environ.get("TRANSFER_DSN")
    if not dsn_main:
        print("ERROR: set L12_DSN", file=sys.stderr)
        return 1
    conns: dict[bool, psycopg.Connection | None] = {
        False: psycopg.connect(dsn_main),
        True: psycopg.connect(dsn_transfer) if dsn_transfer else None,
    }
    for c in conns.values():
        if c is not None:
            with c.cursor() as cur:
                cur.execute(
                    "SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY")
                cur.execute("SET statement_timeout = '30s'")
            c.commit()

    ties: list[str] = []
    unmapped: list[str] = []
    skipped: list[str] = []
    n_ordered = 0
    for sql_path in sorted(GOLD_DIR.glob("*.sql")):
        qid = sql_path.stem
        sql = sql_path.read_text()
        clause = top_level_order_by(sql)
        if clause is None:
            continue
        n_ordered += 1
        conn = conns[qid.startswith("q_transfer")]
        if conn is None:
            skipped.append(qid)
            continue
        with conn.cursor() as cur:
            cur.execute(sql)  # type: ignore[arg-type]
            cols = [d.name for d in cur.description] if cur.description else []
            rows = cur.fetchall()
        exprs = split_exprs(clause)
        # Map each ORDER BY expression to an output column index; collect
        # the rest for SELECT-list injection.
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
            audit_sql = inject_keys_sql(sql, [e for _, e in to_inject])
            keys = None
            if audit_sql is not None:
                try:
                    with conn.cursor() as cur:
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
                except psycopg.Error:
                    conn.rollback()
            if keys is None:
                unmapped.append(
                    f"{qid}: {', '.join(e for _, e in to_inject)}")
                continue
            if len(keys) != len(set(keys)):
                dup = sorted({k for k in keys if keys.count(k) > 1})[:2]
                ties.append(f"{qid}: duplicate ORDER BY keys e.g. {dup}")
            continue
        keys = [tuple(row[mapped[i]] for i in range(len(exprs)))
                for row in rows]
        if len(keys) != len(set(keys)):
            dup = sorted({k for k in keys if keys.count(k) > 1})[:2]
            ties.append(f"{qid}: duplicate ORDER BY keys e.g. {dup}")
    for name, items in (("NON-TOTAL ORDER", ties), ("UNMAPPED", unmapped),
                        ("SKIPPED (no DSN)", skipped)):
        for item in items:
            print(f"{name}: {item}")
    print(f"\nordered_queries={n_ordered} non_total_order={len(ties)} "
          f"unmapped={len(unmapped)} skipped={len(skipped)}")
    return 1 if (ties or unmapped or skipped) else 0


if __name__ == "__main__":
    sys.exit(main())
