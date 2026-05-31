"""Generate constrained SQL from natural language using LLM with schema graph constraints."""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from llm.entity_extractor import extract_conditions
from llm.few_shot_store import add_example, format_few_shot_block, retrieve_similar
from llm.intent_classifier import classify_intent
from llm.schema_linker import link_schema


def _load_prompt_template() -> str:
    path = Path(__file__).parent / "prompt_templates" / "sql_generation_prompt.md"
    return path.read_text(encoding="utf-8")


def build_constrained_prompt(
    user_query: str,
    allowed_tables: list[str],
    allowed_columns: list[str],
    allowed_joins: list[str],
    few_shot_examples: list[dict[str, Any]] | None = None,
) -> str:
    """Build the LLM prompt with schema constraints and optional few-shot examples."""
    template = _load_prompt_template()
    prompt = template.format(
        user_query=user_query,
        allowed_tables="\n".join(f"- {t}" for t in allowed_tables),
        allowed_columns="\n".join(f"- {c}" for c in allowed_columns),
        allowed_joins="\n".join(f"- {j}" for j in allowed_joins),
    )
    if few_shot_examples:
        few_shot_block = format_few_shot_block(few_shot_examples)
        prompt = prompt.replace("\nUser query:", few_shot_block + "\nUser query:")
    return prompt


def extract_sql_from_response(response: str) -> str:
    """Extract SQL from LLM response, stripping markdown fences."""
    sql = response.strip()
    match = re.search(r"```(?:sql)?\s*\n?(.*?)```", sql, re.DOTALL)
    if match:
        sql = match.group(1).strip()
    lines = [ln for ln in sql.split("\n") if not ln.strip().startswith("--")]
    return "\n".join(lines).strip()


def generate_sql_via_llm(
    user_query: str,
    allowed_tables: list[str],
    allowed_columns: list[str],
    allowed_joins: list[str],
    model: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Call the OpenAI API to generate SQL from a constrained prompt.

    Returns a dict with keys: sql, prompt, model, tokens, latency_ms.
    If the API key is not set, falls back to rule-based generation.
    """
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY", "")
    if model is None:
        model = os.getenv("LLM_MODEL", "gpt-5.5")

    prompt = build_constrained_prompt(
        user_query, allowed_tables, allowed_columns, allowed_joins,
    )

    # Retrieve similar few-shot examples
    few_shot = retrieve_similar(user_query, top_k=3)

    # Check coverage score for fallback decision
    conditions = extract_conditions(user_query)
    coverage_info = conditions.get("_coverage", {})
    coverage_action = coverage_info.get("action", "execute_rule_based")

    if not api_key or api_key == "your_api_key_here":
        sql = _rule_based_fallback(user_query, allowed_tables, allowed_columns, allowed_joins)
        return {
            "sql": sql,
            "prompt": prompt,
            "model": "rule_based_fallback",
            "tokens": 0,
            "latency_ms": 0,
            "few_shot_count": len(few_shot),
            "few_shot_queries": [e["nl_query"] for e in few_shot],
            "coverage": coverage_info,
        }

    # If coverage is low and we have an API key, force LLM mode
    # If coverage action is clarification_required, still try LLM but flag it
    use_llm = True  # We have API key, so always use LLM when available

    # Rebuild prompt with few-shot examples for LLM mode
    prompt = build_constrained_prompt(
        user_query, allowed_tables, allowed_columns, allowed_joins,
        few_shot_examples=few_shot,
    )

    import openai
    client = openai.OpenAI(api_key=api_key)
    t0 = time.time()
    create_kwargs: dict[str, Any] = dict(
        model=model,
        messages=[
            {"role": "system", "content": "You are a PostgreSQL expert for materials databases."},
            {"role": "user", "content": prompt},
        ],
    )
    # GPT-5 / o-series: no temperature, use max_completion_tokens
    _is_new_model = model and any(t in model for t in ("gpt-5", "o1", "o3", "o4"))
    if _is_new_model:
        create_kwargs["max_completion_tokens"] = 4096
    else:
        create_kwargs["temperature"] = 0.0
        create_kwargs["max_tokens"] = 512
    resp = client.chat.completions.create(**create_kwargs)
    latency_ms = int((time.time() - t0) * 1000)
    raw = resp.choices[0].message.content or ""
    sql = extract_sql_from_response(raw)
    usage = resp.usage
    tokens = (usage.prompt_tokens + usage.completion_tokens) if usage else 0
    return {
        "sql": sql,
        "prompt": prompt,
        "model": model,
        "tokens": tokens,
        "latency_ms": latency_ms,
        "few_shot_count": len(few_shot),
        "few_shot_queries": [e["nl_query"] for e in few_shot],
        "coverage": coverage_info,
    }


def _rule_based_fallback(
    user_query: str,
    allowed_tables: list[str],
    allowed_columns: list[str],
    allowed_joins: list[str],
) -> str:
    """Generate SQL deterministically when no LLM API key is available."""
    conditions = extract_conditions(user_query)
    linked = link_schema(conditions)

    select_cols = ["m.entry_id", "m.formula"]
    where_clauses: list[str] = []
    order_by = ""
    joins: list[str] = []

    tables_needed = set(linked["required_tables"])
    tables_needed.discard("material_entry")

    alias_map = {
        "composition": "c",
        "structure": "s",
        "phase_stability": "ps",
        "calculation": "calc",
        "calculated_property": "cp",
    }

    indirect_join_map = {
        "calculated_property": (
            "JOIN calculation calc ON calc.entry_id = m.entry_id\n"
            "    JOIN calculated_property cp ON cp.calculation_id = calc.calculation_id"
        ),
    }

    sorted_tables = sorted(tables_needed)
    for t in sorted_tables:
        if t not in tables_needed:
            continue
        if t in indirect_join_map:
            joins.append(indirect_join_map[t])
            tables_needed.discard("calculation")
        else:
            a = alias_map.get(t, t[:2])
            joins.append(f"JOIN {t} {a} ON {a}.entry_id = m.entry_id")

    has_exists_elements = False
    for frag in linked["sql_fragments"]:
        sql_f = frag["sql_fragment"]
        if frag["type"] == "sort":
            order_by = sql_f
        elif frag["type"] == "element_exists":
            has_exists_elements = True
            where_clauses.append(sql_f)
        else:
            where_clauses.append(sql_f)

    if has_exists_elements:
        joins = [j for j in joins if not j.startswith("JOIN composition")]

    if "structure" in tables_needed:
        select_cols.append("s.prototype")
        select_cols.append("s.lattice_a")
        select_cols.append("s.space_group")
    if "phase_stability" in tables_needed:
        select_cols.append("ps.formation_energy_per_atom")
        select_cols.append("ps.energy_above_hull")
        select_cols.append("ps.band_gap")

    sql_parts = [f"SELECT DISTINCT\n    {', '.join(select_cols)}"]
    sql_parts.append("FROM material_entry m")
    for j in joins:
        sql_parts.append(f"    {j}")
    if where_clauses:
        sql_parts.append("WHERE\n    " + "\n    AND ".join(where_clauses))
    if order_by:
        sql_parts.append(order_by)
    sql_parts.append("LIMIT 100;")

    return "\n".join(sql_parts)


def pipeline(
    user_query: str,
    join_list: list[str] | None = None,
    store_on_success: bool = False,
    skip_intent_check: bool = False,
) -> dict[str, Any]:
    """Full pipeline: intent classify -> extract -> link -> generate SQL.

    When *store_on_success* is True the result is persisted in the few-shot
    store after successful DB execution so that future queries can benefit
    from it as a few-shot example.

    When *skip_intent_check* is True, bypass intent classification
    (useful for benchmarking or when intent is pre-verified).
    """
    # Intent classification gate
    if not skip_intent_check:
        intent = classify_intent(user_query)
        if intent["intent"] in ("out_of_scope", "greeting"):
            return {
                "sql": "",
                "mode": "rejected",
                "intent": intent,
                "reason": intent["reason"],
            }
        if intent["intent"] == "unsafe":
            return {
                "sql": "",
                "mode": "rejected",
                "intent": intent,
                "reason": f"Unsafe input detected: {intent['reason']}",
            }

    conditions = extract_conditions(user_query)
    linked = link_schema(conditions)

    if join_list is None:
        join_list = [
            "composition.entry_id = material_entry.entry_id",
            "structure.entry_id = material_entry.entry_id",
            "phase_stability.entry_id = material_entry.entry_id",
            "calculation.entry_id = material_entry.entry_id",
            "calculated_property.calculation_id = calculation.calculation_id",
        ]

    all_columns = [
        "material_entry.entry_id", "material_entry.formula",
        "material_entry.reduced_formula", "material_entry.chemical_system",
        "composition.element", "composition.atomic_fraction", "composition.site_label",
        "structure.prototype", "structure.strukturbericht", "structure.lattice_a",
        "structure.lattice_b", "structure.lattice_c", "structure.volume_per_atom",
        "structure.formula_type", "structure.space_group_number",
        "structure.space_group",
        "phase_stability.formation_energy_per_atom",
        "phase_stability.energy_above_hull", "phase_stability.is_stable",
        "phase_stability.band_gap",
        "calculation.method", "calculation.functional",
        "calculated_property.property_name", "calculated_property.value",
        "calculated_property.unit",
    ]
    result = generate_sql_via_llm(
        user_query=user_query,
        allowed_tables=linked["required_tables"],
        allowed_columns=[
            c for c in all_columns
            if c.split(".")[0] in linked["required_tables"]
        ],
        allowed_joins=[
            j for j in join_list
            if any(t in j for t in linked["required_tables"])
        ],
    )
    result["conditions"] = conditions
    result["linked_schema"] = linked

    if store_on_success:
        add_example(
            nl_query=user_query,
            sql=result["sql"],
            conditions=conditions,
            row_count=-1,
            source="pipeline",
        )

    return result
