"""Generate constrained SQL from natural language using LLM with schema graph constraints."""
from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from llm.entity_extractor import extract_conditions
from llm.few_shot_store import add_example, format_few_shot_block, retrieve_similar
from llm.intent_classifier import classify_intent, classify_query_type
from llm.output_schema_specifier import specify_output_schema
from llm.schema_linker import link_schema


def _fix_known_literals(sql: str) -> str:
    """Post-process generated SQL to correct known DB literal values."""
    # site_label: 'A' -> 'A-site', 'B' -> 'B-site'
    sql = re.sub(r"site_label\s*=\s*'A'", "site_label = 'A-site'", sql)
    sql = re.sub(r"site_label\s*=\s*'B'", "site_label = 'B-site'", sql)
    sql = re.sub(r"site_label\s+IN\s*\('A',\s*'B'\)", "site_label IN ('A-site', 'B-site')", sql)
    # functional: 'PBE' -> 'GGA-PBE'
    sql = re.sub(r"functional\s*=\s*'PBE'", "functional = 'GGA-PBE'", sql, flags=re.IGNORECASE)
    sql = re.sub(r"functional\s*=\s*'GGA'", "functional = 'GGA-PBE'", sql, flags=re.IGNORECASE)
    return sql


def _normalize_column_aliases(sql: str) -> str:
    """Normalize common LLM column alias variants to standard forms.

    The LLM sometimes generates verbose aliases like 'a_site_element'
    instead of the gold SQL's 'a_site'. This normalizes to the shorter
    standard forms used in gold SQL.
    """
    # Normalize common verbose aliases
    alias_map = [
        (r'\bAS\s+a_site_element\b', 'AS a_site'),
        (r'\bAS\s+b_site_element\b', 'AS b_site'),
        (r'\bAS\s+avg_formation_energy_per_atom\b', 'AS avg_eform'),
        (r'\bAS\s+avg_formation_energy\b', 'AS avg_eform'),
        (r'\bAS\s+avg_eform_per_atom\b', 'AS avg_eform'),
        (r'\bAS\s+lattice_a_difference_from_ni3al\b', 'AS lattice_diff'),
        (r'\bAS\s+lattice_a_difference\b', 'AS lattice_diff'),
        (r'\bAS\s+lattice_a_mismatch\b', 'AS lattice_diff'),
        (r'\bAS\s+lattice_mismatch_to_ni3al\b', 'AS lattice_diff'),
        (r'\bAS\s+lattice_mismatch\b', 'AS lattice_diff'),
        (r'\bAS\s+stability_category\b', 'AS stability'),
        (r'\bAS\s+stability_class\b', 'AS stability'),
        (r'\bAS\s+stable_count\b', 'AS count'),
        (r'\bAS\s+l12_count\b', 'AS count'),
        (r'\bAS\s+compound_count\b', 'AS count'),
    ]
    for pattern, replacement in alias_map:
        sql = re.sub(pattern, replacement, sql, flags=re.IGNORECASE)
    return sql


def _load_prompt_template() -> str:
    path = Path(__file__).parent / "prompt_templates" / "sql_generation_prompt.md"
    return path.read_text(encoding="utf-8")


def _format_columns_by_table(columns: list[str]) -> str:
    """Group columns by table for clearer prompt presentation."""
    by_table: dict[str, list[str]] = {}
    for col in columns:
        parts = col.split(".", 1)
        if len(parts) == 2:
            table, colname = parts
            by_table.setdefault(table, []).append(colname)
        else:
            by_table.setdefault("_other", []).append(col)
    lines: list[str] = []
    for table in sorted(by_table):
        cols = sorted(by_table[table])
        lines.append(f"  {table}: {', '.join(cols)}")
    return "\n".join(lines)


def build_constrained_prompt(
    user_query: str,
    allowed_tables: list[str],
    allowed_columns: list[str],
    allowed_joins: list[str],
    few_shot_examples: list[dict[str, Any]] | None = None,
    query_type_instruction: str = "",
    column_hint: str = "",
) -> str:
    """Build the LLM prompt with schema constraints and optional few-shot examples."""
    template = _load_prompt_template()
    prompt = template.format(
        user_query=user_query,
        allowed_tables="\n".join(f"- {t}" for t in allowed_tables),
        allowed_columns=_format_columns_by_table(allowed_columns),
        allowed_joins="\n".join(f"- {j}" for j in allowed_joins),
        query_type_instruction=query_type_instruction or "Follow the question's intent.",
        column_hint=column_hint or "Return only columns relevant to the question.",
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

    # Check API key early so fallback doesn't need the prompt template
    conditions = extract_conditions(user_query)
    coverage_info = conditions.get("_coverage", {})
    few_shot = retrieve_similar(user_query, top_k=3)

    # Classify query type and determine output schema
    query_type_info = classify_query_type(user_query)
    column_hint = specify_output_schema(user_query, conditions, allowed_columns)

    if not api_key or api_key == "your_api_key_here":
        sql = _rule_based_fallback(user_query, allowed_tables, allowed_columns, allowed_joins)
        return {
            "sql": sql,
            "prompt": "",
            "model": "rule_based_fallback",
            "tokens": 0,
            "latency_ms": 0,
            "few_shot_count": len(few_shot),
            "few_shot_queries": [e["nl_query"] for e in few_shot],
            "coverage": coverage_info,
        }

    prompt = build_constrained_prompt(
        user_query, allowed_tables, allowed_columns, allowed_joins,
        few_shot_examples=few_shot,
        query_type_instruction=query_type_info["instruction"],
        column_hint=column_hint,
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
        create_kwargs["max_tokens"] = 4096  # Fix B7: unified budget
    resp = client.chat.completions.create(**create_kwargs)
    latency_ms = int((time.time() - t0) * 1000)
    raw = resp.choices[0].message.content or ""
    sql = extract_sql_from_response(raw)
    sql = _fix_known_literals(sql)
    sql = _normalize_column_aliases(sql)
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
    conditions = extract_conditions(user_query)  # cached at caller if possible
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
        "elastic_tensor": "et",
        "thermal_property": "tp",
        "magnetic_property": "mp",
    }

    indirect_join_map = {
        "calculated_property": (
            "JOIN calculation calc ON calc.entry_id = m.entry_id\n"
            "    JOIN calculated_property cp ON cp.calculation_id = calc.calculation_id"
        ),
        "literature_reference": (
            "JOIN material_reference mr ON mr.entry_id = m.entry_id\n"
            "    JOIN literature_reference lr ON lr.reference_id = mr.reference_id"
        ),
    }

    sorted_tables = sorted(tables_needed)
    for t in sorted_tables:
        if t not in tables_needed:
            continue
        if t in indirect_join_map:
            joins.append(indirect_join_map[t])
            tables_needed.discard("calculation")
            tables_needed.discard("material_reference")
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
    if "calculated_property" in tables_needed:
        select_cols.append("cp.property_name")
        select_cols.append("cp.value")
        select_cols.append("cp.unit")
    if "elastic_tensor" in tables_needed:
        et_alias = alias_map.get("elastic_tensor", "et")
        select_cols.append(f"{et_alias}.bulk_modulus_vrh")
        select_cols.append(f"{et_alias}.shear_modulus_vrh")
    if "thermal_property" in tables_needed:
        tp_alias = alias_map.get("thermal_property", "tp")
        select_cols.append(f"{tp_alias}.debye_temperature_k")

    sql_parts = [f"SELECT DISTINCT\n    {', '.join(select_cols)}"]
    sql_parts.append("FROM material_entry m")
    for j in joins:
        sql_parts.append(f"    {j}")
    if where_clauses:
        sql_parts.append("WHERE\n    " + "\n    AND ".join(where_clauses))
    if order_by:
        sql_parts.append(order_by)
    row_limit = int(os.getenv("SQL_ROW_LIMIT", "100"))
    sql_parts.append(f"LIMIT {row_limit};")

    return "\n".join(sql_parts)


def pipeline(
    user_query: str,
    join_list: list[str] | None = None,
    store_on_success: bool = False,
    skip_intent_check: bool = False,
) -> dict[str, Any]:
    """Full pipeline: intent classify -> extract -> link -> generate SQL.

    Parameters
    ----------
    join_list : list[str] | None
        Pre-computed join conditions from ``get_allowed_join_list()``.
        When *None* (default) the pipeline falls back to a hard-coded
        5-table core join set.  For full 30-table schema graph traversal,
        callers should build the join list explicitly via a live DB
        connection and pass it here.

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

    all_columns: list[str] | None = None
    if join_list is None:
        # Schema graph auto-construction requires a live DB connection.
        # When no join_list is provided and no connection is available,
        # fall back to the core 5-table join set with an explicit warning.
        logger.warning(
            "join_list not provided; using hard-coded 5-table fallback. "
            "Pass join_list explicitly via get_allowed_join_list() for "
            "full 30-table schema graph traversal."
        )
        join_list = [
            "composition.entry_id = material_entry.entry_id",
            "structure.entry_id = material_entry.entry_id",
            "phase_stability.entry_id = material_entry.entry_id",
            "calculation.entry_id = material_entry.entry_id",
            "calculated_property.calculation_id = calculation.calculation_id",
        ]
        all_columns = None

    if all_columns is None:
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

    # NOTE: store_on_success is deferred — caller must invoke add_example()
    # after DB execution confirms the SQL is valid and returns rows.
    result["_store_on_success"] = store_on_success

    return result
