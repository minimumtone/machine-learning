#!/usr/bin/env python3
"""Vocabulary audit: gold-SQL string literals vs actual database values.

Execution-based verification cannot notice that a gold query filters on
a literal spelled differently from the stored controlled vocabulary
(e.g. gold 'transition_metal' vs stored 'transition metal'): the query
still runs and simply matches nothing. This audit closes that gap:

1. Parse every gold SQL file (sqlglot, postgres dialect) and collect
   equality-style comparisons of a column against string literals
   (=, <>, !=, IN, NOT IN, IS [NOT] DISTINCT FROM).
2. Map each column name to every text-typed column of the same name in
   the suite's database and collect the actual DISTINCT values.
3. Report any literal that does not appear among the actual values of
   any same-named column, printing the column, its actual distinct
   values, and the gold literals.

Columns compared with LIKE/ILIKE or inequality ranges are out of scope
(pattern/range semantics, not vocabulary membership). Literals listed in
INTENTIONAL_ZERO_MATCH are negative controls that deliberately query
values absent from the fixture.

Exit 0 when every audited literal is backed by a stored value, 1
otherwise.

Usage:
    L12_DSN=... TRANSFER_DSN=... OBF_TRANSFER_DSN=... \
        python scripts/audit_vocabulary.py
    (main suite falls back to the local CONNINFO defaults; transfer
    suites are skipped with a warning when their DSN is unset unless
    --require-all is given)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import psycopg
import sqlglot
import sqlglot.errors
from sqlglot import expressions as exp

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from scripts.db_conninfo import CONNINFO  # noqa: E402

SUITES = [
    # (name, gold dir, DSN env var, qid filter) — mirrors
    # run_gold_verification.SUITES.
    ("main", "gold_sql", "L12_DSN",
     lambda qid: not qid.startswith("q_transfer")),
    ("transfer", "gold_sql", "TRANSFER_DSN",
     lambda qid: qid.startswith("q_transfer")),
    ("obfuscated", "gold_sql_obfuscated", "OBF_TRANSFER_DSN",
     lambda qid: True),
]

# (suite, column, literal) tuples that intentionally have no stored match.
# chemical_system stores alphabetized systems ('Al-Ni'); q_expert_038's
# gold defensively also accepts the user-order spelling 'Ni-Al'.
INTENTIONAL_ZERO_MATCH: set[tuple[str, str, str]] = {
    ("main", "chemical_system", "Ni-Al"),
}

TEXT_TYPES = {"text", "character varying", "character"}


def _column_of(node: exp.Expression) -> str | None:
    if isinstance(node, exp.Column):
        return node.name.lower()
    if isinstance(node, exp.Cast) and isinstance(node.this, exp.Column):
        return node.this.name.lower()
    return None


def _literals_of(node: exp.Expression) -> list[str] | None:
    """String literal(s) on one side of a comparison; None when not pure."""
    if isinstance(node, exp.Literal) and node.is_string:
        return [node.this]
    return None


def extract_literal_filters(sql: str) -> list[tuple[str, str]]:
    """(column, literal) pairs from equality-style comparisons in SQL."""
    pairs: list[tuple[str, str]] = []
    try:
        statements = sqlglot.parse(sql, dialect="postgres")
    except sqlglot.errors.ParseError as e:
        raise ValueError(f"unparseable gold SQL: {e}") from e
    for stmt in statements:
        if stmt is None:
            continue
        for node in stmt.walk():
            if isinstance(node, (exp.EQ, exp.NEQ,
                                 exp.NullSafeEQ, exp.NullSafeNEQ)):
                for a, b in ((node.this, node.expression),
                             (node.expression, node.this)):
                    col = _column_of(a)
                    lits = _literals_of(b)
                    if col and lits:
                        pairs.extend((col, lit) for lit in lits)
            elif isinstance(node, exp.In):
                col = _column_of(node.this)
                if col:
                    for item in node.expressions:
                        lits = _literals_of(item)
                        if lits:
                            pairs.extend((col, lit) for lit in lits)
    return pairs


def db_text_column_values(dsn: str) -> dict[str, set[str]]:
    """column name -> union of DISTINCT values over all same-named
    text-typed columns of ordinary tables and views in schema public."""
    values: dict[str, set[str]] = {}
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY")
            cur.execute("SET statement_timeout = '30s'")
            cur.execute("""
                SELECT c.table_name, c.column_name
                FROM information_schema.columns c
                JOIN information_schema.tables t
                  ON t.table_schema = c.table_schema
                 AND t.table_name = c.table_name
                WHERE c.table_schema = 'public'
                  AND c.data_type = ANY(%s)
                  AND t.table_type IN ('BASE TABLE', 'VIEW')
            """, (list(TEXT_TYPES),))
            targets = cur.fetchall()
        for table, column in targets:
            with conn.cursor() as cur:
                cur.execute(
                    f'SELECT DISTINCT "{column}" FROM "{table}" '
                    f'WHERE "{column}" IS NOT NULL')
                col_values = {row[0] for row in cur.fetchall()}
            values.setdefault(column.lower(), set()).update(col_values)
    return values


def audit_suite(name: str, gold_dir: Path, dsn: str,
                qid_filter) -> list[str]:
    failures: list[str] = []
    db_values = db_text_column_values(dsn)
    for path in sorted(gold_dir.glob("*.sql")):
        if not qid_filter(path.stem):
            continue
        for col, lit in extract_literal_filters(path.read_text()):
            if col not in db_values:
                # Aliased output columns / obfuscated names resolve at
                # runtime; only stored text columns carry vocabulary.
                continue
            if (name, col, lit) in INTENTIONAL_ZERO_MATCH:
                continue
            if lit not in db_values[col]:
                actual = sorted(db_values[col])
                shown = actual if len(actual) <= 15 else actual[:15] + ["..."]
                failures.append(
                    f"[{name}] {path.name}: column {col!r} literal {lit!r} "
                    f"not among actual values {shown}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-all", action="store_true",
        help="fail (instead of warn) when a transfer suite DSN is unset")
    args = parser.parse_args()

    failures: list[str] = []
    n_audited = 0
    for name, subdir, env_var, qid_filter in SUITES:
        gold_dir = PROJECT / "evaluation" / subdir
        dsn = os.environ.get(env_var)
        if not dsn and name == "main":
            dsn = CONNINFO
        if not dsn:
            msg = f"suite {name}: DSN {env_var} not set"
            if args.require_all:
                failures.append(msg)
            else:
                print(f"WARNING: {msg}; suite skipped")
            continue
        failures.extend(audit_suite(name, gold_dir, dsn, qid_filter))
        n_audited += 1

    for f in failures:
        print(f"VOCABULARY MISMATCH: {f}")
    print(f"\nsuites_audited={n_audited} "
          f"vocabulary_mismatch={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
