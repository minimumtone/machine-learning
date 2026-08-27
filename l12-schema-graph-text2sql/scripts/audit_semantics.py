#!/usr/bin/env python3
"""Semantic audit: formula vs composition vs prototype stoichiometry.

The schema stores overlapping chemistry facts in separate columns:
material_entry.formula / reduced_formula, composition.element /
atomic_fraction, and structure.prototype -> prototype_definition
.formula_type. FK/CHECK constraints cannot compare a parsed formula
against fraction rows, so this Python audit closes that gap:

1. FORMULA vs COMPOSITION: parse material_entry.formula into element
   counts, normalize to atomic fractions, and compare against the summed
   composition.atomic_fraction per element (site-resolved rows are summed)
   with tolerance 1e-8.
2. REDUCED FORMULA: the gcd-reduced counts of formula must equal the
   parsed counts of reduced_formula (element-by-element).
3. PROTOTYPE STOICHIOMETRY: the entry's sorted fraction multiset must
   match the stoichiometry declared by its prototype's formula_type
   (e.g. A3B -> {0.75, 0.25}, AB -> {0.5, 0.5}, A -> {1.0}).

Read-only; exits 0 when every entry passes all three audits, 1 otherwise.

Usage:
    L12_DSN=postgresql://... python scripts/audit_semantics.py
    (falls back to the local CONNINFO defaults when L12_DSN is unset)
"""
from __future__ import annotations

import math
import os
import re
import sys
from fractions import Fraction
from pathlib import Path

import psycopg

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from scripts.db_conninfo import CONNINFO  # noqa: E402

TOL = 1e-8

_TOKEN_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def parse_formula(formula: str) -> dict[str, int] | None:
    """Parse 'Ni3Al' -> {'Ni': 3, 'Al': 1}; None when not fully parseable.

    Only the flat element-count grammar used by this fixture is supported
    (no parentheses/hydrates); anything else is reported as a failure
    rather than silently skipped.
    """
    counts: dict[str, int] = {}
    pos = 0
    for m in _TOKEN_RE.finditer(formula):
        if m.start() != pos:
            return None
        counts[m.group(1)] = counts.get(m.group(1), 0) \
            + (int(m.group(2)) if m.group(2) else 1)
        pos = m.end()
    if pos != len(formula) or not counts:
        return None
    return counts


def reduce_counts(counts: dict[str, int]) -> dict[str, int]:
    """Divide all counts by their gcd."""
    g = math.gcd(*counts.values())
    return {el: n // g for el, n in counts.items()}


def formula_type_fractions(formula_type: str) -> list[Fraction] | None:
    """'A3B' -> [3/4, 1/4] (descending); None when not parseable."""
    parts = re.findall(r"([A-Z])(\d*)", formula_type)
    if not parts or "".join(p[0] + p[1] for p in parts) != formula_type:
        return None
    counts = [int(n) if n else 1 for _, n in parts]
    total = sum(counts)
    return sorted((Fraction(c, total) for c in counts), reverse=True)


def main() -> int:
    dsn = os.environ.get("L12_DSN") or CONNINFO
    conn = psycopg.connect(dsn)
    with conn.cursor() as cur:
        cur.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY")
        cur.execute("SET statement_timeout = '30s'")
    conn.commit()

    failures: list[str] = []
    with conn.cursor() as cur:
        cur.execute("""
            SELECT m.entry_id, m.formula, m.reduced_formula, s.prototype,
                   pd.formula_type
            FROM material_entry m
            JOIN structure s ON s.entry_id = m.entry_id
            JOIN prototype_definition pd ON pd.prototype_id = s.prototype
            ORDER BY m.entry_id
        """)
        entries = cur.fetchall()
        cur.execute("""
            SELECT entry_id, element, SUM(atomic_fraction)
            FROM composition
            GROUP BY entry_id, element
        """)
        comp: dict[str, dict[str, float]] = {}
        for entry_id, element, frac in cur.fetchall():
            comp.setdefault(entry_id, {})[element] = float(frac)
    conn.close()

    n_checked = 0
    for entry_id, formula, reduced_formula, prototype, ftype in entries:
        n_checked += 1
        counts = parse_formula(formula)
        if counts is None:
            failures.append(f"{entry_id}: unparseable formula {formula!r}")
            continue
        total = sum(counts.values())
        fractions = {el: n / total for el, n in counts.items()}
        stored = comp.get(entry_id, {})
        # 1. formula vs composition (element sets and fractions)
        if set(fractions) != set(stored):
            failures.append(
                f"{entry_id}: formula elements {sorted(fractions)} != "
                f"composition elements {sorted(stored)}")
        else:
            for el, f in fractions.items():
                if abs(f - stored[el]) > TOL:
                    failures.append(
                        f"{entry_id}: {el} formula fraction {f} != "
                        f"composition fraction {stored[el]}")
        # 2. reduced_formula agrees with gcd-reduced formula counts
        if reduced_formula is not None:
            red = parse_formula(reduced_formula)
            if red is None:
                failures.append(f"{entry_id}: unparseable reduced_formula "
                                f"{reduced_formula!r}")
            elif red != reduce_counts(counts):
                failures.append(
                    f"{entry_id}: reduced_formula {reduced_formula!r} != "
                    f"reduced counts of formula {formula!r}")
        # 3. prototype stoichiometry
        expected = formula_type_fractions(ftype)
        if expected is None:
            failures.append(
                f"{entry_id}: prototype {prototype} has unparseable "
                f"formula_type {ftype!r}")
            continue
        actual = sorted(stored.values(), reverse=True)
        if len(actual) != len(expected) or any(
                abs(a - float(e)) > TOL for a, e in zip(actual, expected)):
            failures.append(
                f"{entry_id}: composition fractions {actual} do not match "
                f"prototype {prototype} formula_type {ftype}")

    for f in failures:
        print(f"SEMANTIC MISMATCH: {f}")
    print(f"\nentries_checked={n_checked} "
          f"semantic_mismatch={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
