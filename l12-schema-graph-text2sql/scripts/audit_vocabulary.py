#!/usr/bin/env python3
"""Vocabulary audit: gold-SQL string literals vs actual database values.

Execution-based verification cannot notice that a gold query filters on
a literal spelled differently from the stored controlled vocabulary
(e.g. gold 'transition_metal' vs stored 'transition metal'): the query
still runs and simply matches nothing. This audit closes that gap:

1. Parse every gold SQL file (sqlglot, postgres dialect) and collect,
   with scope/alias resolution to the real (table, column):
   - equality-style comparisons of a column against string literals
     (=, <>, !=, IN, NOT IN, IS [NOT] DISTINCT FROM);
   - LIKE / ILIKE patterns against a column.
2. For equality literals, compare against the DISTINCT values of that
   specific table's column (NOT a union over same-named columns of
   other tables, which would hide a wrong literal that happens to exist
   elsewhere). Columns projected through CTEs/subqueries are followed
   through the projection lineage down to their base table (so a
   literal compared OUTSIDE a CTE against a CTE-projected column is
   still audited against the underlying table). A reference that cannot
   be traced to any base table (e.g. a computed projection compared to
   a literal) is a FAILURE by default, counted as unresolved — it must
   never be silently assumed to be "audited elsewhere". A reference
   that resolves to a base table whose column is non-text is counted as
   skipped (equality against non-text columns is type-checked by
   execution itself).
3. For LIKE/ILIKE, require at least one stored row to match the
   pattern (EXISTS probe), so a typo'd pattern ('%theraml%') fails the
   audit instead of silently returning zero rows.
4. Report NULL statistics (non_null/null counts) for every equality-
   audited stored column and fail when such a controlled-vocabulary
   column contains NULLs, unless declared in NULLS_ALLOWED.

Literals/patterns listed in INTENTIONAL_ZERO_MATCH /
INTENTIONAL_ZERO_MATCH_LIKE are negative controls that deliberately
query values absent from the fixture.

Before auditing, each suite's DB is guarded: the main suite requires
the version='007' marker + unchanged schema fingerprint + a fresh pass
of validate_fixture_integrity(); the transfer and obfuscated suites
require the transfer_initialization_status marker + a fresh pass of
validate_transfer_integrity(); the MP suite requires
assert_valid_mp_transfer() (pinned snapshot digest + schema
fingerprint).

Exit 0 when every audited literal is backed by a stored value and
every reference resolved (unresolved=0), 1 otherwise.

Usage:
    L12_DSN=... TRANSFER_DSN=... OBF_TRANSFER_DSN=... MP_DSN=... \
        python scripts/audit_vocabulary.py
    (main and MP suites fall back to the local conninfo defaults; a
    missing transfer-suite DSN is a failure unless --allow-skip is
    given)
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
from sqlglot.optimizer.scope import Scope, build_scope

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from scripts.db_conninfo import CONNINFO, mp_conninfo  # noqa: E402
from scripts.fixture_guard import assert_valid_fixture  # noqa: E402
from scripts.mp_guard import assert_valid_mp_transfer  # noqa: E402
from scripts.transfer_guard import assert_valid_transfer  # noqa: E402

SUITES = [
    # (name, gold dir, DSN env var, qid filter, guard) — mirrors
    # run_gold_verification.SUITES.
    ("main", "gold_sql", "L12_DSN",
     lambda qid: not qid.startswith("q_transfer"), assert_valid_fixture),
    ("transfer", "gold_sql", "TRANSFER_DSN",
     lambda qid: qid.startswith("q_transfer"), assert_valid_transfer),
    ("obfuscated", "gold_sql_obfuscated", "OBF_TRANSFER_DSN",
     lambda qid: True, assert_valid_transfer),
    ("mp", "gold_sql_mp", "MP_DSN",
     lambda qid: True, assert_valid_mp_transfer),
]

# (suite, table, column, literal) equality literals that intentionally
# have no stored match (negative controls).
INTENTIONAL_ZERO_MATCH: set[tuple[str, str, str, str]] = set()

# (suite, table, column, pattern) LIKE/ILIKE patterns that intentionally
# match no stored row.
INTENTIONAL_ZERO_MATCH_LIKE: set[tuple[str, str, str, str]] = set()

# (suite, table, column) stored text columns used in equality predicates
# where NULLs are part of the fixture contract (with justification).
NULLS_ALLOWED: set[tuple[str, str, str]] = {
    # Pure-element ground-state structures have no Strukturbericht label
    # in this fixture (89 pure-element rows).
    ("main", "structure", "strukturbericht"),
    ("main", "prototype_definition", "strukturbericht"),
    # Pure-element compositions carry no site label (single-site,
    # site-resolved rows exist only for multi-element prototypes).
    ("main", "composition", "site_label"),
    ("transfer", "oqmd_element_ratios", "wyckoff_site"),
    # Same column as oqmd_element_ratios.wyckoff_site under the
    # obfuscated schema mapping.
    ("obfuscated", "tbl_juliet", "col_zulu"),
}

TEXT_TYPES = {"text", "character varying", "character"}


def _column_of(node: exp.Expression) -> exp.Column | None:
    if isinstance(node, exp.Column):
        return node
    if isinstance(node, exp.Cast) and isinstance(node.this, exp.Column):
        return node.this
    return None


def _literals_of(node: exp.Expression) -> list[str] | None:
    """String literal(s) on one side of a comparison; None when not pure."""
    if isinstance(node, exp.Literal) and node.is_string:
        return [node.this]
    return None


def _resolve_source(src: object, name: str,
                    table_columns: dict[str, set[str]],
                    unqualified: bool) -> list[str]:
    """Base tables a column named `name` in source `src` traces back to."""
    if isinstance(src, exp.Table):
        t = src.name.lower()
        if unqualified:
            return [t] if name in table_columns.get(t, set()) else []
        return [t]
    if isinstance(src, Scope):
        inner = src.expression
        if not isinstance(inner, exp.Select):
            # UNION/VALUES-shaped sources are not traced; caller reports
            # the reference as unresolved.
            return []
        for sel in inner.selects:
            if isinstance(sel, exp.Star):
                # SELECT *: the column passes through unchanged; resolve
                # it as an unqualified reference inside the inner scope.
                out: list[str] = []
                for isrc in src.sources.values():
                    out.extend(_resolve_source(
                        isrc, name, table_columns, unqualified=True))
                return out
            if sel.alias_or_name.lower() == name:
                target = sel.this if isinstance(sel, exp.Alias) else sel
                col = _column_of(target)
                if col is None:
                    # Computed projection (expression/function): its
                    # vocabulary cannot be attributed to a base table.
                    return []
                return _resolve_tables(src, col, table_columns)
        return []
    return []


def _resolve_tables(scope: Scope, column: exp.Column,
                    table_columns: dict[str, set[str]]) -> list[str]:
    """Base table names a column reference in a scope traces back to.

    Qualified references resolve through the scope's alias map;
    references to CTE/subquery sources are followed through the
    projection lineage (plain column projections and SELECT *) down to
    the base tables. Unqualified references resolve to every source in
    scope that can supply the column name. An empty result means the
    reference is UNRESOLVED and must be reported by the caller.
    """
    name = column.name.lower()
    if column.table:
        src = scope.sources.get(column.table)
        return _resolve_source(src, name, table_columns, unqualified=False)
    candidates: list[str] = []
    for src in scope.sources.values():
        candidates.extend(
            _resolve_source(src, name, table_columns, unqualified=True))
    return candidates


def extract_filters(
    sql: str, table_columns: dict[str, set[str]],
) -> tuple[list[tuple[list[str], str, str]],
           list[tuple[list[str], str, str, bool]]]:
    """Extract (candidate tables, column, literal) equality filters and
    (candidate tables, column, pattern, is_ilike) LIKE filters."""
    eq_filters: list[tuple[list[str], str, str]] = []
    like_filters: list[tuple[list[str], str, str, bool]] = []
    try:
        statements = sqlglot.parse(sql, dialect="postgres")
    except sqlglot.errors.ParseError as e:
        raise ValueError(f"unparseable gold SQL: {e}") from e
    for stmt in statements:
        if stmt is None:
            continue
        root = build_scope(stmt)
        if root is None:
            continue
        for scope in root.traverse():
            for node in scope.walk():
                if isinstance(node, (exp.EQ, exp.NEQ,
                                     exp.NullSafeEQ, exp.NullSafeNEQ)):
                    for a, b in ((node.this, node.expression),
                                 (node.expression, node.this)):
                        col = _column_of(a)
                        lits = _literals_of(b)
                        if col is not None and lits:
                            tables = _resolve_tables(
                                scope, col, table_columns)
                            eq_filters.extend(
                                (tables, col.name.lower(), lit)
                                for lit in lits)
                elif isinstance(node, exp.In):
                    col = _column_of(node.this)
                    if col is not None:
                        tables = _resolve_tables(scope, col, table_columns)
                        for item in node.expressions:
                            lits = _literals_of(item)
                            if lits:
                                eq_filters.extend(
                                    (tables, col.name.lower(), lit)
                                    for lit in lits)
                elif isinstance(node, (exp.Like, exp.ILike)):
                    col = _column_of(node.this)
                    lits = _literals_of(node.expression)
                    if col is not None and lits:
                        tables = _resolve_tables(scope, col, table_columns)
                        like_filters.extend(
                            (tables, col.name.lower(), lit,
                             isinstance(node, exp.ILike))
                            for lit in lits)
    return eq_filters, like_filters


def db_catalog(
    conn: psycopg.Connection,
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """(all columns, text-typed columns) per table/view name.

    Resolution must see EVERY column (so an unqualified reference to a
    numeric column resolves to its table and is counted as skipped
    rather than unresolved); vocabulary auditing itself only applies to
    the text-typed subset.
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT c.table_name, c.column_name, c.data_type
            FROM information_schema.columns c
            JOIN information_schema.tables t
              ON t.table_schema = c.table_schema
             AND t.table_name = c.table_name
            WHERE c.table_schema = 'public'
              AND t.table_type IN ('BASE TABLE', 'VIEW')
        """)
        all_cols: dict[str, set[str]] = {}
        text_cols: dict[str, set[str]] = {}
        for table, column, data_type in cur.fetchall():
            all_cols.setdefault(table.lower(), set()).add(column.lower())
            if data_type in TEXT_TYPES:
                text_cols.setdefault(
                    table.lower(), set()).add(column.lower())
    return all_cols, text_cols


class ColumnStats:
    """Lazily cached DISTINCT values and NULL counts per (table, column)."""

    def __init__(self, conn: psycopg.Connection):
        self.conn = conn
        self._values: dict[tuple[str, str], set[str]] = {}
        self._nulls: dict[tuple[str, str], tuple[int, int]] = {}

    def values(self, table: str, column: str) -> set[str]:
        key = (table, column)
        if key not in self._values:
            with self.conn.cursor() as cur:
                cur.execute(
                    f'SELECT DISTINCT "{column}" FROM "{table}" '
                    f'WHERE "{column}" IS NOT NULL')
                self._values[key] = {row[0] for row in cur.fetchall()}
        return self._values[key]

    def null_counts(self, table: str, column: str) -> tuple[int, int]:
        """(non_null_count, null_count) for the column."""
        key = (table, column)
        if key not in self._nulls:
            with self.conn.cursor() as cur:
                cur.execute(
                    f'SELECT COUNT("{column}"), '
                    f'COUNT(*) - COUNT("{column}") FROM "{table}"')
                row = cur.fetchone()
                assert row is not None
                self._nulls[key] = (int(row[0]), int(row[1]))
        return self._nulls[key]

    def like_matches(self, table: str, column: str, pattern: str,
                     ilike: bool) -> bool:
        op = "ILIKE" if ilike else "LIKE"
        with self.conn.cursor() as cur:
            cur.execute(
                f'SELECT EXISTS (SELECT 1 FROM "{table}" '
                f'WHERE "{column}" {op} %s)', (pattern,))
            row = cur.fetchone()
            assert row is not None
            return bool(row[0])


def audit_suite(name: str, gold_dir: Path, dsn: str, qid_filter,
                guard) -> tuple[list[str], list[str], int]:
    failures: list[str] = []
    notes: list[str] = []
    n_skipped_nontext = 0
    audited_columns: set[tuple[str, str]] = set()
    with psycopg.connect(dsn) as conn:
        # One REPEATABLE READ READ ONLY snapshot per suite so every
        # audited literal is compared against the same DB state.
        conn.read_only = True
        conn.isolation_level = psycopg.IsolationLevel.REPEATABLE_READ
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout = '30s'")
        guard(conn)
        all_cols, text_cols = db_catalog(conn)
        stats = ColumnStats(conn)
        for path in sorted(gold_dir.glob("*.sql")):
            if not qid_filter(path.stem):
                continue
            eq_filters, like_filters = extract_filters(
                path.read_text(), all_cols)
            for tables, col, lit in eq_filters:
                if not tables:
                    failures.append(
                        f"[{name}] {path.name}: UNRESOLVED column "
                        f"reference {col!r} compared to {lit!r} — could "
                        "not be traced to a base table")
                    continue
                stored = [t for t in tables
                          if col in text_cols.get(t, set())]
                if not stored:
                    # Resolved to base table(s) but the column is not
                    # text-typed there; equality against non-text
                    # columns is outside vocabulary auditing.
                    n_skipped_nontext += 1
                    continue
                if any((name, t, col, lit) in INTENTIONAL_ZERO_MATCH
                       for t in stored):
                    continue
                if not any(lit in stats.values(t, col) for t in stored):
                    t = stored[0]
                    actual = sorted(stats.values(t, col))
                    shown = (actual if len(actual) <= 15
                             else actual[:15] + ["..."])
                    failures.append(
                        f"[{name}] {path.name}: {t}.{col} literal "
                        f"{lit!r} not among actual values {shown}")
                audited_columns.update((t, col) for t in stored)
            for tables, col, pattern, ilike in like_filters:
                if not tables:
                    failures.append(
                        f"[{name}] {path.name}: UNRESOLVED column "
                        f"reference {col!r} matched against "
                        f"{pattern!r} — could not be traced to a base "
                        "table")
                    continue
                stored = [t for t in tables
                          if col in text_cols.get(t, set())]
                if not stored:
                    n_skipped_nontext += 1
                    continue
                if any((name, t, col, pattern)
                       in INTENTIONAL_ZERO_MATCH_LIKE for t in stored):
                    continue
                op = "ILIKE" if ilike else "LIKE"
                if not any(stats.like_matches(t, col, pattern, ilike)
                           for t in stored):
                    failures.append(
                        f"[{name}] {path.name}: {stored[0]}.{col} {op} "
                        f"{pattern!r} matches no stored row")
        for table, col in sorted(audited_columns):
            non_null, null = stats.null_counts(table, col)
            notes.append(
                f"[{name}] {table}.{col}: non_null={non_null} null={null}")
            if null and (name, table, col) not in NULLS_ALLOWED:
                failures.append(
                    f"[{name}] {table}.{col} used as controlled "
                    f"vocabulary in gold SQL but has {null} NULLs "
                    f"(declare in NULLS_ALLOWED if intentional)")
    return failures, notes, n_skipped_nontext


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-skip", action="store_true",
        help="warn (instead of fail) when a transfer suite DSN is unset")
    parser.add_argument(
        "--show-null-stats", action="store_true",
        help="print non_null/null counts for every audited column")
    args = parser.parse_args()

    failures: list[str] = []
    missing_suites: list[str] = []
    n_audited = 0
    n_skipped = 0
    for name, subdir, env_var, qid_filter, guard in SUITES:
        gold_dir = PROJECT / "evaluation" / subdir
        dsn = os.environ.get(env_var)
        if not dsn and name == "main":
            dsn = CONNINFO
        if not dsn and name == "mp":
            dsn = mp_conninfo()
        if not dsn:
            msg = f"suite {name}: DSN {env_var} not set"
            if args.allow_skip:
                print(f"WARNING: {msg}; suite skipped")
            else:
                missing_suites.append(msg)
            continue
        suite_failures, notes, suite_skipped = audit_suite(
            name, gold_dir, dsn, qid_filter, guard)
        failures.extend(suite_failures)
        n_skipped += suite_skipped
        if args.show_null_stats:
            for note in notes:
                print(f"NULL STATS: {note}")
        n_audited += 1

    n_unresolved = sum(1 for f in failures if "UNRESOLVED" in f)
    for f in failures:
        print(f"VOCABULARY MISMATCH: {f}")
    for m in missing_suites:
        print(f"SUITE MISSING: {m}")
    print(f"\nsuites_audited={n_audited} "
          f"vocabulary_mismatch={len(failures) - n_unresolved} "
          f"unresolved={n_unresolved} skipped_nontext={n_skipped} "
          f"suites_missing={len(missing_suites)}")
    return 1 if failures or missing_suites else 0


if __name__ == "__main__":
    sys.exit(main())
