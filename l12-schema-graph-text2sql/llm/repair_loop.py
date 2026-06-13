"""SQL repair loop: retry SQL generation when validation or execution fails.

Supports four failure modes:
1. Validation failure (SQLGuard reject) — repair with validation error message
2. Execution failure (PostgreSQL error) — repair with DB error message
3. Empty result (0 rows returned) — repair with coverage diagnostic hints
4. Superset result (too many rows / missing WHERE) — repair with condition feedback
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any, Callable

from llm.sql_generator import _fix_known_literals, _normalize_column_aliases


def _load_repair_template() -> str:
    path = Path(__file__).parent / "prompt_templates" / "sql_repair_prompt.md"
    return path.read_text(encoding="utf-8")


def _call_llm(prompt: str, system_msg: str, model: str, api_key: str) -> tuple[str, int]:
    """Call the LLM and return (sql, token_count)."""
    import openai
    from llm.sql_generator import extract_sql_from_response

    client = openai.OpenAI(api_key=api_key)
    create_kwargs: dict[str, Any] = dict(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ],
    )
    _is_new_model = model and any(t in model for t in ("gpt-5", "o1", "o3", "o4"))
    if _is_new_model:
        create_kwargs["max_completion_tokens"] = 4096
    else:
        create_kwargs["temperature"] = 0.0
        create_kwargs["max_tokens"] = 4096  # Fix B7: unified budget
    try:
        resp = client.chat.completions.create(**create_kwargs)
    except Exception:
        return "", 0
    raw = resp.choices[0].message.content or ""
    sql = extract_sql_from_response(raw)
    usage = resp.usage
    tokens = (usage.prompt_tokens + usage.completion_tokens) if usage else 0
    return sql, tokens


def attempt_repair(
    original_sql: str,
    error_message: str,
    allowed_tables: list[str],
    allowed_columns: list[str],
    allowed_joins: list[str],
    model: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Attempt to repair a failed SQL query via LLM."""
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY", "")
    if model is None:
        model = os.getenv("LLM_MODEL", "gpt-5.5")

    template = _load_repair_template()
    prompt = template.format(
        original_sql=original_sql,
        error_message=error_message,
        allowed_tables=", ".join(allowed_tables),
        allowed_columns=", ".join(allowed_columns),
        allowed_joins=", ".join(allowed_joins),
    )

    if not api_key or api_key == "your_api_key_here":
        return {
            "repaired_sql": original_sql,
            "repair_prompt": prompt,
            "success": False,
            "tokens": 0,
            "reason": "No API key available for repair",
        }

    sql, tokens = _call_llm(
        prompt,
        "You are a PostgreSQL expert for a materials science database. Fix the SQL.",
        model, api_key,
    )
    if sql:
        sql = _fix_known_literals(sql)
        sql = _normalize_column_aliases(sql)
    return {
        "repaired_sql": sql if sql else original_sql,
        "repair_prompt": prompt,
        "success": bool(sql),
        "tokens": tokens,
    }


def repair_loop(
    original_sql: str,
    validate_fn: Any,
    allowed_tables: list[str],
    allowed_columns: list[str],
    allowed_joins: list[str],
    max_retries: int = 3,
) -> dict[str, Any]:
    """Try to repair SQL up to max_retries times (validation-only)."""
    sql = original_sql
    attempts: list[dict[str, Any]] = []

    for i in range(max_retries):
        result = validate_fn(sql)
        if result.get("valid", False):
            return {"sql": sql, "valid": True, "attempts": attempts}

        errors = result.get("errors", [])
        error_msg = "; ".join(errors) if errors else "Unknown validation error"
        repair_result = attempt_repair(
            sql, error_msg, allowed_tables, allowed_columns, allowed_joins,
        )
        attempts.append({"attempt": i + 1, "error": error_msg, "repair": repair_result})

        if not repair_result.get("success"):
            break
        sql = repair_result["repaired_sql"]

    return {"sql": sql, "valid": False, "attempts": attempts}


# ── Superset / condition-insufficiency detection ──


def count_sql_conditions(sql: str) -> int:
    """Count the number of WHERE/AND/OR filtering conditions in SQL.

    Strips string literals first to avoid false positives.
    """
    cleaned = re.sub(r"'[^']*'", "'X'", sql)
    cleaned = re.sub(r"/\*.*?\*/", " ", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"--[^\n]*", " ", cleaned)
    upper = cleaned.upper()
    count = 0
    # Count WHERE clause presence
    if "WHERE" in upper:
        count += 1
    # Count AND conditions (each adds a restriction)
    count += len(re.findall(r"\bAND\b", upper))
    # Fix B13: OR *expands* the result set, not restricts it — don't count
    return count


def count_expected_conditions(conditions: dict[str, Any]) -> int:
    """Estimate the expected number of SQL WHERE conditions from extracted entities.

    Each recognized entity type maps to at least one WHERE condition.
    """
    count = 0
    if conditions.get("prototype"):
        count += 1
    elements = conditions.get("contains_elements", [])
    if elements:
        count += len(elements)  # Each element → subquery or AND
    if conditions.get("stability"):
        count += 1
    props = conditions.get("properties", [])
    if props:
        count += len(props)
    numeric = conditions.get("numeric_conditions", [])
    if numeric:
        count += len(numeric)
    if conditions.get("formula"):
        count += 1
    return count


def detect_superset(
    sql: str,
    row_count: int,
    conditions: dict[str, Any],
    total_db_rows: int | None = None,
) -> dict[str, Any]:
    """Detect if a query result is likely a superset (missing WHERE conditions).

    Returns a dict with:
      - is_superset: bool
      - reason: str (diagnostic message)
      - sql_conditions: int
      - expected_conditions: int
      - row_ratio: float
    """
    # Fix B13: Don't hardcode total_db_rows; query count from DB or use safe default
    if total_db_rows is None:
        total_db_rows = 1500  # Conservative estimate; callers should pass actual count
    sql_conds = count_sql_conditions(sql)
    expected_conds = count_expected_conditions(conditions)
    row_ratio = row_count / total_db_rows if total_db_rows > 0 else 0.0

    is_superset = False
    reasons: list[str] = []

    # Heuristic 1: SQL has significantly fewer conditions than expected
    if expected_conds > 0 and sql_conds < expected_conds:
        missing = expected_conds - sql_conds
        if missing >= 2 or (expected_conds >= 2 and sql_conds == 0):
            is_superset = True
            reasons.append(
                f"SQL has {sql_conds} conditions but query mentions "
                f"{expected_conds} constraints (missing ~{missing})"
            )

    # Heuristic 2: Row count is suspiciously high relative to DB size
    if row_ratio > 0.5 and expected_conds >= 2:
        is_superset = True
        reasons.append(
            f"Query returned {row_count} rows ({row_ratio:.0%} of DB), "
            f"suggesting insufficient filtering"
        )

    # Heuristic 3: Row count > 100 with multiple expected conditions but few SQL conditions
    if row_count > 100 and expected_conds >= 2 and sql_conds <= 1:
        is_superset = True
        reasons.append(
            f"High row count ({row_count}) with only {sql_conds} SQL condition(s) "
            f"despite {expected_conds} expected constraints"
        )

    return {
        "is_superset": is_superset,
        "reason": "; ".join(reasons) if reasons else "No superset detected",
        "sql_conditions": sql_conds,
        "expected_conditions": expected_conds,
        "row_ratio": round(row_ratio, 3),
    }


def _build_superset_msg(
    sql: str,
    question: str,
    row_count: int,
    conditions: dict[str, Any],
    superset_info: dict[str, Any],
) -> str:
    """Build diagnostic message for superset results."""
    parts = [
        f"The query returned {row_count} rows, which appears to be a SUPERSET "
        f"of the expected results (too many rows due to missing WHERE conditions).",
        f"Original question: {question}",
        f"Diagnosis: {superset_info['reason']}",
        "",
        "Missing conditions detected from the query:",
    ]

    # List specific missing conditions
    if conditions.get("contains_elements"):
        parts.append(f"- Element filter: {conditions['contains_elements']}")
    if conditions.get("prototype"):
        parts.append(f"- Prototype/structure filter: {conditions['prototype']}")
    if conditions.get("stability"):
        parts.append(f"- Stability condition: {conditions['stability']}")
    if conditions.get("properties"):
        parts.append(f"- Property filters: {conditions['properties']}")
    if conditions.get("numeric_conditions"):
        for nc in conditions["numeric_conditions"]:
            parts.append(f"- Numeric condition: {nc}")
    if conditions.get("formula"):
        parts.append(f"- Formula condition: {conditions['formula']}")

    parts.append("")
    parts.append(
        "Please add the missing WHERE conditions to narrow the result set. "
        "Each constraint from the original question should map to a WHERE clause."
    )
    return "\n".join(parts)


# ── Enhanced execution-aware repair loop ──


def _build_execution_error_msg(
    sql: str,
    db_error: str,
    coverage: dict[str, Any] | None = None,
) -> str:
    """Build a diagnostic error message for execution failures."""
    parts = [f"PostgreSQL execution error: {db_error}"]
    if coverage:
        unrecognized = coverage.get("unrecognized_terms", [])
        if unrecognized:
            parts.append(
                f"Possibly unrecognized query terms: {', '.join(unrecognized)}"
            )
    return "\n".join(parts)


def _build_empty_result_msg(
    sql: str,
    question: str,
    coverage: dict[str, Any] | None = None,
) -> str:
    """Build a diagnostic message for 0-row results."""
    parts = [
        "The query executed successfully but returned 0 rows.",
        f"Original question: {question}",
        "Possible causes:",
        "- WHERE conditions too restrictive (e.g. exact match vs LIKE/ILIKE)",
        "- Missing JOIN causing empty cartesian product",
        "- Column value mismatch (e.g. 'L12' vs 'L1_2' vs 'Cu3Au')",
        "Please relax conditions or fix value matching to return results.",
    ]
    if coverage:
        unrecognized = coverage.get("unrecognized_terms", [])
        if unrecognized:
            parts.append(
                f"Note: these terms were not recognized by the entity extractor "
                f"and may be misspelled or unmapped: {', '.join(unrecognized)}"
            )
    return "\n".join(parts)


def execution_repair_loop(
    original_sql: str,
    question: str,
    execute_fn: Callable[[str], dict[str, Any]],
    allowed_tables: list[str],
    allowed_columns: list[str],
    allowed_joins: list[str],
    coverage: dict[str, Any] | None = None,
    conditions: dict[str, Any] | None = None,
    max_retries: int = 3,
    model: str | None = None,
    api_key: str | None = None,
    allow_empty_result: bool = False,
) -> dict[str, Any]:
    """Retry SQL when execution fails, returns 0 rows, or returns a superset.

    This is called **after** the initial SQL has passed SQLGuard validation.
    It handles three failure modes:
      1. DB execution error (syntax ok but runtime failure)
      2. Empty result set (query runs but returns nothing)
      3. Superset result (too many rows / missing WHERE conditions)

    If allow_empty_result is True, 0-row results are treated as success
    (the correct answer may legitimately be an empty set).

    Returns dict with: sql, exec_result, attempts, repair_tokens
    """
    sql = original_sql
    attempts: list[dict[str, Any]] = []
    total_repair_tokens = 0
    total_repair_latency_ms = 0  # Fix B14: track repair latency

    exec_result: dict[str, Any] = {}
    for i in range(max_retries + 1):  # +1 for the original attempt
        t0 = time.time()
        exec_result = execute_fn(sql)
        exec_latency = int((time.time() - t0) * 1000)
        total_repair_latency_ms += exec_latency

        row_count = exec_result.get("row_count", 0)
        if exec_result.get("success") and (row_count > 0 or allow_empty_result):
            # Check for superset (only if conditions were extracted)
            if conditions and i < max_retries:
                row_count = exec_result.get("row_count", 0)
                superset_info = detect_superset(sql, row_count, conditions)
                if superset_info["is_superset"]:
                    error_msg = _build_superset_msg(
                        sql, question, row_count, conditions, superset_info,
                    )
                    t_repair = time.time()
                    repair_result = attempt_repair(
                        sql, error_msg, allowed_tables, allowed_columns,
                        allowed_joins, model=model, api_key=api_key,
                    )
                    repair_lat = int((time.time() - t_repair) * 1000)
                    repair_tokens = repair_result.get("tokens", 0)
                    total_repair_tokens += repair_tokens
                    total_repair_latency_ms += repair_lat
                    attempts.append({
                        "attempt": i + 1,
                        "reason": "superset",
                        "error": error_msg,
                        "repair": repair_result,
                        "tokens": repair_tokens,
                        "latency_ms": repair_lat,
                        "superset_info": superset_info,
                    })
                    if repair_result.get("success"):
                        new_sql = repair_result["repaired_sql"]
                        if new_sql != sql:
                            sql = new_sql
                            continue  # Re-execute with repaired SQL
            return {
                "sql": sql,
                "exec_result": exec_result,
                "attempts": attempts,
                "repair_tokens": total_repair_tokens,
                "repair_latency_ms": total_repair_latency_ms,
                "repaired": i > 0,
            }

        if i == max_retries:
            break

        # Determine error type and build diagnostic message
        if not exec_result.get("success"):
            db_errors = exec_result.get("errors", [exec_result.get("error", "Unknown error")])
            error_msg = _build_execution_error_msg(
                sql, "; ".join(str(e) for e in db_errors), coverage,
            )
            repair_reason = "execution_error"
        else:
            error_msg = _build_empty_result_msg(sql, question, coverage)
            repair_reason = "empty_result"

        t_repair = time.time()
        repair_result = attempt_repair(
            sql, error_msg, allowed_tables, allowed_columns, allowed_joins,
            model=model, api_key=api_key,
        )
        repair_latency_llm = int((time.time() - t_repair) * 1000)
        repair_tokens = repair_result.get("tokens", 0)
        total_repair_tokens += repair_tokens
        total_repair_latency_ms += repair_latency_llm

        attempts.append({
            "attempt": i + 1,
            "reason": repair_reason,
            "error": error_msg,
            "repair": repair_result,
            "tokens": repair_tokens,
            "latency_ms": repair_latency_llm,
        })

        if not repair_result.get("success"):
            break

        new_sql = repair_result["repaired_sql"]
        if new_sql == sql:
            break  # LLM returned same SQL, no point retrying
        sql = new_sql

    return {
        "sql": sql,
        "exec_result": exec_result,
        "attempts": attempts,
        "repair_tokens": total_repair_tokens,
        "repair_latency_ms": total_repair_latency_ms,
        "repaired": len(attempts) > 0 and exec_result.get("success", False),
    }
