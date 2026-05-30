"""SQL repair loop: retry SQL generation when validation or execution fails."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _load_repair_template() -> str:
    path = Path(__file__).parent / "prompt_templates" / "sql_repair_prompt.md"
    return path.read_text(encoding="utf-8")


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
            "reason": "No API key available for repair",
        }

    import openai
    from llm.sql_generator import extract_sql_from_response

    client = openai.OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a PostgreSQL expert. Fix the SQL."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=512,
    )
    raw = resp.choices[0].message.content or ""
    sql = extract_sql_from_response(raw)
    return {
        "repaired_sql": sql,
        "repair_prompt": prompt,
        "success": True,
    }


def repair_loop(
    original_sql: str,
    validate_fn: Any,
    allowed_tables: list[str],
    allowed_columns: list[str],
    allowed_joins: list[str],
    max_retries: int = 3,
) -> dict[str, Any]:
    """Try to repair SQL up to max_retries times."""
    sql = original_sql
    attempts: list[dict[str, Any]] = []

    for i in range(max_retries):
        result = validate_fn(sql)
        if result.get("valid", False):
            return {"sql": sql, "valid": True, "attempts": attempts}

        error_msg = result.get("error", "Unknown validation error")
        repair_result = attempt_repair(
            sql, error_msg, allowed_tables, allowed_columns, allowed_joins,
        )
        attempts.append({"attempt": i + 1, "error": error_msg, "repair": repair_result})

        if not repair_result.get("success"):
            break
        sql = repair_result["repaired_sql"]

    return {"sql": sql, "valid": False, "attempts": attempts}
