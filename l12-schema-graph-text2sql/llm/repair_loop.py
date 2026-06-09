"""SQL repair loop: retry SQL generation when validation or execution fails.

Supports three failure modes:
1. Validation failure (SQLGuard reject) — repair with validation error message
2. Execution failure (PostgreSQL error) — repair with DB error message
3. Empty result (0 rows returned) — repair with coverage diagnostic hints
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable


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
        create_kwargs["max_tokens"] = 512
    resp = client.chat.completions.create(**create_kwargs)
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
        model = os.getenv("LLM_MODEL", "gpt-4.1-mini")

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
    return {
        "repaired_sql": sql,
        "repair_prompt": prompt,
        "success": True,
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
    max_retries: int = 2,
    model: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Retry SQL when execution fails or returns 0 rows.

    This is called **after** the initial SQL has passed SQLGuard validation.
    It handles two failure modes:
      1. DB execution error (syntax ok but runtime failure)
      2. Empty result set (query runs but returns nothing)

    Returns dict with: sql, exec_result, attempts, repair_tokens
    """
    sql = original_sql
    attempts: list[dict[str, Any]] = []
    total_repair_tokens = 0

    for i in range(max_retries + 1):  # +1 for the original attempt
        exec_result = execute_fn(sql)

        if exec_result.get("success") and exec_result.get("row_count", 0) > 0:
            return {
                "sql": sql,
                "exec_result": exec_result,
                "attempts": attempts,
                "repair_tokens": total_repair_tokens,
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

        repair_result = attempt_repair(
            sql, error_msg, allowed_tables, allowed_columns, allowed_joins,
            model=model, api_key=api_key,
        )
        repair_tokens = repair_result.get("tokens", 0)
        total_repair_tokens += repair_tokens

        attempts.append({
            "attempt": i + 1,
            "reason": repair_reason,
            "error": error_msg,
            "repair": repair_result,
            "tokens": repair_tokens,
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
        "repaired": len(attempts) > 0 and exec_result.get("success", False),
    }
