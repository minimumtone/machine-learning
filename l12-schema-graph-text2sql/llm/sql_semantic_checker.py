"""SQL semantic checker: verify generated SQL matches the user's intent.

Reverse-translates the generated SQL to natural language and compares
it with the original query. If the semantics diverge (e.g., user asked
for "格子定数を一覧" but SQL does COUNT(*)), the checker flags the
mismatch so the caller can trigger re-generation.

This acts as a safety net for intent classification errors.
"""
from __future__ import annotations

import logging
import os
import re
import time
from typing import Any

logger = logging.getLogger(__name__)

_REVERSE_PROMPT = """あなたはSQL専門家です。以下のPostgreSQLクエリが「何を返すか」を日本語1文で要約してください。
専門用語はそのまま使い、SELECT句のカラムと集約関数に注目してください。

SQL:
{sql}

このSQLが返すもの（1文で）:"""

_COMPARE_PROMPT = """以下の2つの文が意味的に整合しているか判定してください。

ユーザの質問: {user_query}
SQLが返すもの: {sql_summary}

判定基準:
- ユーザが求めているデータの種類（一覧/件数/平均/分布等）とSQLの出力が一致するか
- ユーザが言及したカラム（格子定数、形成エネルギー等）がSQLの出力に含まれるか
- 完全一致は不要。ユーザの意図をSQLがおおむね満たしていればOK

回答は以下のJSON形式のみ:
{{"consistent": true/false, "reason": "理由（1文）"}}"""


def _call_llm_simple(prompt: str, system_msg: str, model: str, api_key: str) -> str:
    """Call the LLM and return raw text response."""
    import openai

    client = openai.OpenAI(api_key=api_key)
    create_kwargs: dict[str, Any] = dict(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ],
    )
    _is_new = model and any(t in model for t in ("gpt-5", "o1", "o3", "o4"))
    if _is_new:
        create_kwargs["max_completion_tokens"] = 512
    else:
        create_kwargs["temperature"] = 0.0
        create_kwargs["max_tokens"] = 512
    try:
        resp = client.chat.completions.create(**create_kwargs)
        return resp.choices[0].message.content or ""
    except Exception as e:
        logger.warning("Semantic checker LLM call failed: %s", e)
        return ""


def reverse_translate_sql(sql: str, model: str | None = None,
                          api_key: str | None = None) -> str:
    """Convert SQL to a natural language summary of what it returns."""
    if not sql.strip():
        return ""
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY", "")
    if model is None:
        model = os.getenv("LLM_MODEL", "gpt-5.5")
    if not api_key:
        return ""

    prompt = _REVERSE_PROMPT.format(sql=sql)
    return _call_llm_simple(
        prompt, "あなたはSQLを日本語に要約する専門家です。", model, api_key
    ).strip()


def check_semantic_consistency(
    user_query: str,
    generated_sql: str,
    model: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Check if the generated SQL semantically matches the user's query.

    Returns:
      - consistent: bool
      - sql_summary: str (reverse-translated SQL)
      - reason: str (explanation if inconsistent)
      - tokens: int (approximate token usage)
    """
    if not generated_sql.strip():
        return {"consistent": False, "sql_summary": "", "reason": "Empty SQL",
                "tokens": 0}

    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY", "")
    if model is None:
        model = os.getenv("LLM_MODEL", "gpt-5.5")
    if not api_key:
        return {"consistent": True, "sql_summary": "", "reason": "No API key",
                "tokens": 0}

    # Step 1: Reverse translate SQL
    sql_summary = reverse_translate_sql(generated_sql, model, api_key)
    if not sql_summary:
        return {"consistent": True, "sql_summary": "", "reason": "Reverse translation failed",
                "tokens": 0}

    # Step 2: Compare with user query
    compare_prompt = _COMPARE_PROMPT.format(
        user_query=user_query, sql_summary=sql_summary
    )
    response = _call_llm_simple(
        compare_prompt, "あなたは意味整合性を判定する専門家です。JSONのみ回答してください。",
        model, api_key,
    )

    # Parse JSON response
    import json
    consistent = True
    reason = ""
    try:
        # Extract JSON from response
        json_match = re.search(r'\{[^}]+\}', response)
        if json_match:
            result = json.loads(json_match.group())
            consistent = bool(result.get("consistent", True))
            reason = result.get("reason", "")
    except (json.JSONDecodeError, KeyError):
        logger.warning("Failed to parse semantic check response: %s", response[:200])

    return {
        "consistent": consistent,
        "sql_summary": sql_summary,
        "reason": reason,
        "tokens": 0,  # approximate; caller can track via API usage
    }


def check_and_flag(
    user_query: str,
    generated_sql: str,
    model: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Convenience wrapper: check consistency and return actionable result.

    Returns dict with:
      - needs_regeneration: bool
      - sql_summary: str
      - reason: str
    """
    result = check_semantic_consistency(user_query, generated_sql, model, api_key)
    return {
        "needs_regeneration": not result["consistent"],
        "sql_summary": result["sql_summary"],
        "reason": result["reason"],
    }
