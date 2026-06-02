"""Evaluation metrics for Text-to-SQL comparison."""
from __future__ import annotations

import re
from typing import Any

try:
    import sqlglot
    HAS_SQLGLOT = True
except ImportError:
    HAS_SQLGLOT = False


def syntax_validity(sql: str) -> bool:
    """Check whether SQL parses without error."""
    if HAS_SQLGLOT:
        try:
            sqlglot.parse(sql, dialect="postgres")
            return True
        except Exception:
            return False
    return sql.strip().upper().startswith("SELECT")


def execution_validity(result: dict[str, Any]) -> bool:
    """Check whether execution succeeded."""
    return result.get("success", False)


def execution_accuracy(
    result_rows: list[list[Any]],
    expected_rows: list[list[Any]],
    result_columns: list[str] | None = None,
    expected_columns: list[str] | None = None,
) -> float:
    """Compute row-set overlap between result and expected.

    When column metadata is provided, matching is done on the
    intersection of column names so that extra SELECT columns in the
    generated SQL do not penalise accuracy.
    """
    if not expected_rows:
        return 1.0 if not result_rows else 0.0

    if result_columns and expected_columns:
        rc = [c.lower() for c in result_columns]
        ec = [c.lower() for c in expected_columns]
        common = [c for c in ec if c in rc]
        if common:
            ri = [rc.index(c) for c in common]
            ei = [ec.index(c) for c in common]
            result_set = {tuple(r[i] for i in ri) for r in result_rows}
            expected_set = {tuple(r[i] for i in ei) for r in expected_rows}
            if not expected_set:
                return 0.0
            return len(result_set & expected_set) / len(expected_set)

    result_set = {tuple(r) for r in result_rows}
    expected_set = {tuple(r) for r in expected_rows}
    if not expected_set:
        return 0.0
    return len(result_set & expected_set) / len(expected_set)


def hallucinated_table_rate(
    generated_tables: list[str],
    allowed_tables: list[str],
) -> float:
    """Fraction of generated tables not in allowed list."""
    if not generated_tables:
        return 0.0
    allowed_lower = {t.lower() for t in allowed_tables}
    bad = [t for t in generated_tables if t.lower() not in allowed_lower]
    return len(bad) / len(generated_tables)


def hallucinated_column_rate(
    generated_columns: list[str],
    allowed_columns: list[str],
) -> float:
    """Fraction of generated columns not in allowed list."""
    if not generated_columns:
        return 0.0
    allowed_lower = {c.lower() for c in allowed_columns}
    bad = [c for c in generated_columns if c.lower() not in allowed_lower]
    return len(bad) / len(generated_columns)


def hallucinated_join_rate(
    generated_joins: list[str],
    allowed_joins: list[str],
) -> float:
    """Fraction of joins that are not in the allowed list."""
    if not generated_joins:
        return 0.0

    def normalize(j: str) -> str:
        return re.sub(r"\s+", " ", j.strip().lower())

    allowed_set = {normalize(j) for j in allowed_joins}
    bad = [j for j in generated_joins if normalize(j) not in allowed_set]
    return len(bad) / len(generated_joins)


def multi_hop_success(hop_count: int, is_correct: bool) -> dict[str, Any]:
    """Tag multi-hop success."""
    return {
        "hop_count": hop_count,
        "is_multi_hop": hop_count >= 3,
        "correct": is_correct,
    }


def compute_metrics(
    sql: str,
    exec_result: dict[str, Any],
    expected_rows: list[list[Any]],
    allowed_tables: list[str],
    allowed_columns: list[str],
    allowed_joins: list[str],
    hop_count: int,
    tokens: int = 0,
    latency_ms: int = 0,
) -> dict[str, Any]:
    """Compute all metrics for a single query."""
    from safety.sql_validator import extract_tables_from_sql, extract_columns_from_sql

    gen_tables = extract_tables_from_sql(sql)
    gen_columns = extract_columns_from_sql(sql)

    gen_joins: list[str] = []
    for m in re.finditer(
        r"JOIN\s+\w+\s+\w+\s+ON\s+([\w.]+\s*=\s*[\w.]+)", sql, re.IGNORECASE,
    ):
        gen_joins.append(m.group(1))

    is_correct = execution_accuracy(
        exec_result.get("rows", []), expected_rows,
    ) >= 0.8

    return {
        "syntax_valid": syntax_validity(sql),
        "execution_valid": execution_validity(exec_result),
        "execution_accuracy": execution_accuracy(
            exec_result.get("rows", []), expected_rows,
        ),
        "hallucinated_table_rate": hallucinated_table_rate(gen_tables, allowed_tables),
        "hallucinated_column_rate": hallucinated_column_rate(gen_columns, allowed_columns),
        "hallucinated_join_rate": hallucinated_join_rate(gen_joins, allowed_joins),
        "multi_hop": multi_hop_success(hop_count, is_correct),
        "token_usage": tokens,
        "latency_ms": latency_ms,
    }
