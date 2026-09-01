"""Database-level integrity tests (require a loaded l12_materials DB).

Skipped automatically when the database is unreachable, so the pure-Python
test suite still runs without PostgreSQL — run pytest with -ra so skips
are visible. Set FULL_DB_TEST=1 to turn an unreachable database into a
hard failure instead of a skip. Covers:

- the reference-set divergence fixture: SQL that joins
  pure_element_reference WITHOUT a reference_set condition must return a
  different result than the correct same-set join;
- NaN / +-Infinity rejection on representative DOUBLE PRECISION columns
  (positive-only and sign-free) and NaN rejection on a NUMERIC column.
"""
from __future__ import annotations

import os

import pytest

if os.environ.get("FULL_DB_TEST"):
    import psycopg
else:
    psycopg = pytest.importorskip("psycopg")

from scripts.eval_ablation import CONNINFO  # noqa: E402

FIXTURE_SET = "L12-FIXTURE-PBE-v1"
DIVERGENCE_SET = "L12-FIXTURE-DIVERGENCE-TEST-v1"


@pytest.fixture(scope="module")
def conn():
    try:
        connection = psycopg.connect(CONNINFO)
    except psycopg.OperationalError:
        if os.environ.get("FULL_DB_TEST"):
            pytest.fail("FULL_DB_TEST=1: l12_materials database is not "
                        "reachable — DB tests may not be skipped")
        pytest.skip("l12_materials database is not reachable")
    yield connection
    connection.close()


def _one(conn, sql: str, params: tuple = ()):  # noqa: ANN001
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchone()


class TestReferenceSetDivergence:
    def test_both_reference_sets_loaded(self, conn):
        row = _one(
            conn,
            "SELECT COUNT(DISTINCT reference_set) FROM pure_element_reference "
            "WHERE reference_set IN (%s, %s)",
            (FIXTURE_SET, DIVERGENCE_SET),
        )
        assert row is not None and row[0] == 2

    def test_divergence_set_delta_e_differs(self, conn):
        row = _one(
            conn,
            "SELECT COUNT(*) FROM pure_element_reference a "
            "JOIN pure_element_reference b "
            "  ON b.element_symbol = a.element_symbol "
            " AND b.reference_set = %s "
            "WHERE a.reference_set = %s AND a.delta_e = b.delta_e",
            (DIVERGENCE_SET, FIXTURE_SET),
        )
        assert row is not None and row[0] == 0

    def test_divergence_set_not_loadable_for_materials(self, conn):
        row = _one(
            conn,
            "SELECT COUNT(*) FROM fixture_source_reference_set "
            "WHERE reference_set = %s",
            (DIVERGENCE_SET,),
        )
        assert row is not None and row[0] == 0
        row = _one(
            conn,
            "SELECT COUNT(*) FROM phase_stability WHERE reference_set = %s",
            (DIVERGENCE_SET,),
        )
        assert row is not None and row[0] == 0

    def test_missing_reference_set_join_changes_result(self, conn):
        """A join lacking the reference_set condition must diverge.

        With two elemental conventions loaded, omitting
        `per.reference_set = ps.reference_set` double-joins every element
        and shifts the weighted reference sum, so the re-referenced energy
        is wrong. This is exactly the bug class the divergence fixture
        exists to expose.
        """
        correct = _one(
            conn,
            "SELECT ROUND(SUM(w)::numeric, 6) FROM ("
            "  SELECT SUM(c.atomic_fraction * per.delta_e) AS w"
            "  FROM phase_stability ps"
            "  JOIN composition c ON c.entry_id = ps.entry_id"
            "  JOIN pure_element_reference per"
            "    ON per.element_symbol = c.element"
            "   AND per.reference_set = ps.reference_set"
            "  GROUP BY ps.entry_id"
            ") t",
        )
        missing = _one(
            conn,
            "SELECT ROUND(SUM(w)::numeric, 6) FROM ("
            "  SELECT SUM(c.atomic_fraction * per.delta_e) AS w"
            "  FROM phase_stability ps"
            "  JOIN composition c ON c.entry_id = ps.entry_id"
            "  JOIN pure_element_reference per"
            "    ON per.element_symbol = c.element"
            "  GROUP BY ps.entry_id"
            ") t",
        )
        assert correct is not None and missing is not None
        assert correct[0] != missing[0]


class TestNonFiniteRejection:
    @pytest.mark.parametrize("literal", ["NaN", "Infinity"])
    def test_positive_double_rejects_nonfinite(self, conn, literal):
        with pytest.raises(psycopg.errors.CheckViolation):
            with conn.transaction():
                conn.execute(
                    "UPDATE elastic_tensor "
                    "SET bulk_modulus_vrh = %s::double precision "
                    "WHERE elastic_id = ("
                    "  SELECT elastic_id FROM elastic_tensor LIMIT 1)",
                    (literal,),
                )

    @pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
    def test_signfree_double_rejects_nonfinite(self, conn, literal):
        with pytest.raises(psycopg.errors.CheckViolation):
            with conn.transaction():
                conn.execute(
                    "UPDATE pure_element_reference "
                    "SET delta_e = %s::double precision "
                    "WHERE element_symbol = 'Ni' AND reference_set = %s",
                    (literal, FIXTURE_SET),
                )

    def test_numeric_rejects_nan(self, conn):
        with pytest.raises(psycopg.errors.CheckViolation):
            with conn.transaction():
                conn.execute(
                    "UPDATE element SET atomic_mass = 'NaN'::numeric "
                    "WHERE symbol = 'Ni'",
                )
