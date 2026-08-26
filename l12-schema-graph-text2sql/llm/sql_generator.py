"""Generate constrained SQL from natural language using LLM with schema graph constraints."""
from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from graph.schema_parser import get_foreign_keys
from llm.entity_extractor import extract_conditions
from llm.few_shot_store import format_few_shot_block, retrieve_similar
from llm.intent_classifier import classify_intent, classify_query_type
from llm.output_schema_specifier import specify_output_schema
from llm.schema_linker import link_schema

if TYPE_CHECKING:
    from graph.join_path_generator import generate_joins_for_tables, get_allowed_join_list

logger = logging.getLogger(__name__)

# Scoring weights for SQL candidate ranking (total = 100)
_SCORE_SQLGUARD = 30
_SCORE_EXEC_SUCCESS = 20
_SCORE_HAS_ROWS = 10
_SCORE_CONDITIONS = 20
_SCORE_ROW_RANGE_FULL = 10
_SCORE_ROW_RANGE_PARTIAL = 5
_SCORE_DISTINCT = 5
_SCORE_LIMIT = 5

# Row count thresholds for scoring
_ROW_COUNT_MAX_IDEAL = 500

_HAS_GRAPH = importlib.util.find_spec("networkx") is not None
if not TYPE_CHECKING and _HAS_GRAPH:
    from graph.join_path_generator import generate_joins_for_tables, get_allowed_join_list


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
    """Load the SQL generation prompt template.

    The template path can be overridden via the ``SQL_PROMPT_TEMPLATE``
    environment variable (used e.g. for schema-transfer experiments).
    """
    override = os.getenv("SQL_PROMPT_TEMPLATE")
    if override:
        return Path(override).read_text(encoding="utf-8")
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
    temperature: float | None = None,
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
        if temperature is not None and temperature > 0:
            create_kwargs["temperature"] = temperature
    else:
        create_kwargs["temperature"] = temperature if temperature is not None else 0.0
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
    conditions: dict[str, Any] | None = None,
    linked: dict[str, Any] | None = None,
) -> str:
    """Generate SQL deterministically when no LLM API key is available."""
    if conditions is None:
        conditions = extract_conditions(user_query)
    if linked is None:
        linked = link_schema(conditions)

    # Detect COUNT-type queries.
    # Exclude false positives: "定数を" (constant), "係数を" (coefficient),
    # "指数を" (exponent), "変数を" (variable), "パラメータ数を" (parameter count
    # — ambiguous, but usually means "count of parameters" so kept as count).
    _q = user_query.lower()
    _count_keywords = ["総数", "何件", "いくつ", "何種類"]
    is_count_query = any(kw in _q for kw in _count_keywords)
    if not is_count_query:
        _search_start = 0
        while not is_count_query:
            idx = _q.find("数を", _search_start)
            if idx == -1:
                break
            preceding = _q[max(0, idx - 1):idx]
            if preceding not in ("定", "係", "指", "変", "常"):
                is_count_query = True
            _search_start = idx + 1
    if not is_count_query and "count" in _q:
        is_count_query = True

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
        "application_domain": (
            "JOIN material_application ma ON ma.entry_id = m.entry_id\n"
            "    JOIN application_domain ad ON ad.domain_id = ma.domain_id"
        ),
    }

    # Map each indirect join to the prerequisite tables it includes
    _indirect_discards: dict[str, list[str]] = {
        "calculated_property": ["calculation"],
        "literature_reference": ["material_reference"],
        "application_domain": ["material_application"],
    }

    sorted_tables = sorted(tables_needed)
    for t in sorted_tables:
        if t not in tables_needed:
            continue
        if t in indirect_join_map:
            joins.append(indirect_join_map[t])
            for dep in _indirect_discards.get(t, []):
                tables_needed.discard(dep)
        else:
            a = alias_map.get(t, t[:2])
            joins.append(f"JOIN {t} {a} ON {a}.entry_id = m.entry_id")

    has_exists_elements = False
    for frag in linked["sql_fragments"]:
        sql_f = frag["sql_fragment"]
        if frag["type"] == "sort":
            order_by = sql_f
        elif frag["type"] in ("element_exists", "element_or"):
            has_exists_elements = True
            where_clauses.append(sql_f)
        else:
            where_clauses.append(sql_f)

    if has_exists_elements:
        has_site_label = any(
            "site_label" in frag["sql_fragment"]
            for frag in linked["sql_fragments"]
            if frag["type"] not in ("sort", "element_exists", "element_or")
        )
        if has_site_label:
            # Convert site_label conditions to EXISTS subqueries
            new_where: list[str] = []
            for wc in where_clauses:
                if "c.site_label" in wc:
                    new_where.append(
                        f"EXISTS (SELECT 1 FROM composition c_sl "
                        f"WHERE c_sl.entry_id = m.entry_id AND c_sl.{wc.split('c.', 1)[1]})"
                    )
                else:
                    new_where.append(wc)
            where_clauses = new_where
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
        # Detect which calculated property is queried and add filter
        _prop_keywords = {
            "bulk_modulus": ["バルクモジュラス", "体積弾性率", "bulk modulus", "弾性係数"],
            "shear_modulus": ["せん断弾性率", "シアモジュラス", "shear modulus"],
            "youngs_modulus": ["ヤング率", "youngs modulus", "young's modulus"],
        }
        _matched_props: list[str] = []
        for pname, kws in _prop_keywords.items():
            if any(kw in _q for kw in kws):
                _matched_props.append(pname)
        if len(_matched_props) == 1:
            where_clauses.append(f"cp.property_name = '{_matched_props[0]}'")
        elif _matched_props:
            _in = ", ".join(f"'{p}'" for p in _matched_props)
            where_clauses.append(f"cp.property_name IN ({_in})")
    if "elastic_tensor" in tables_needed:
        et_alias = alias_map.get("elastic_tensor", "et")
        select_cols.append(f"{et_alias}.bulk_modulus_vrh")
        select_cols.append(f"{et_alias}.shear_modulus_vrh")
    if "thermal_property" in tables_needed:
        tp_alias = alias_map.get("thermal_property", "tp")
        select_cols.append(f"{tp_alias}.debye_temperature_k")
    if "literature_reference" in tables_needed:
        select_cols.append("lr.doi")
        select_cols.append("lr.title")
        select_cols.append("lr.year")

    if is_count_query:
        sql_parts = ["SELECT COUNT(*) AS total_count"]
    else:
        sql_parts = [f"SELECT DISTINCT\n    {', '.join(select_cols)}"]
    sql_parts.append("FROM material_entry m")
    for j in joins:
        sql_parts.append(f"    {j}")
    if where_clauses:
        sql_parts.append("WHERE\n    " + "\n    AND ".join(where_clauses))
    if not is_count_query:
        if order_by:
            sql_parts.append(order_by)
        sql_parts.append(";")
    else:
        sql_parts.append(";")

    return "\n".join(sql_parts)


_CORE_5_TABLE_JOINS = [
    "composition.entry_id = material_entry.entry_id",
    "structure.entry_id = material_entry.entry_id",
    "phase_stability.entry_id = material_entry.entry_id",
    "calculation.entry_id = material_entry.entry_id",
    "calculated_property.calculation_id = calculation.calculation_id",
]

_CORE_5_TABLE_COLUMNS = [
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


def build_schema_context_from_db(
    conn: Any,
) -> dict[str, Any]:
    """Build full 30-table schema context from a live DB connection.

    Returns a dict with ``join_list`` and ``all_columns`` suitable for
    passing to :func:`pipeline`.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT table_name, column_name "
            "FROM information_schema.columns "
            "WHERE table_schema = 'public' "
            "ORDER BY table_name, ordinal_position"
        )
        col_rows = cur.fetchall()
    all_columns = [f"{t}.{c}" for t, c in col_rows]
    all_tables = sorted({t for t, _ in col_rows})

    # pg_constraint keeps composite-FK column pairs positionally aligned
    # (information_schema's constraint_column_usage cross-products them).
    fks = get_foreign_keys(conn)
    join_list = [
        f"{fk.source_table}.{fk.source_column}={fk.target_table}.{fk.target_column}"
        for fk in fks
    ]

    return {
        "join_list": join_list,
        "all_columns": all_columns,
        "all_tables": all_tables,
        "n_tables": len(all_tables),
        "n_columns": len(all_columns),
        "n_joins": len(join_list),
    }


def _score_sql_candidate(
    sql: str,
    conditions: dict[str, Any],
    execute_fn: Any | None = None,
) -> dict[str, Any]:
    """Score a SQL candidate by validation, execution, and domain heuristics.

    Returns dict with score (0-100), breakdown, and execution result.
    """
    from safety.sql_validator import validate_sql

    score = 0
    breakdown: dict[str, int] = {}

    # 1. SQLGuard validation
    validation = validate_sql(sql)
    if validation.get("valid"):
        score += _SCORE_SQLGUARD
        breakdown["sqlguard"] = _SCORE_SQLGUARD
    else:
        breakdown["sqlguard"] = 0
        return {"score": score, "breakdown": breakdown, "exec_result": None,
                "validation": validation}

    # 2. Execution success
    exec_result = None
    if execute_fn is not None:
        try:
            exec_result = execute_fn(sql)
            if exec_result and exec_result.get("success"):
                score += _SCORE_EXEC_SUCCESS
                breakdown["exec_success"] = _SCORE_EXEC_SUCCESS
                row_count = exec_result.get("row_count", 0)
                if row_count > 0:
                    score += _SCORE_HAS_ROWS
                    breakdown["has_rows"] = _SCORE_HAS_ROWS
                else:
                    breakdown["has_rows"] = 0
            else:
                breakdown["exec_success"] = 0
                breakdown["has_rows"] = 0
        except Exception:
            breakdown["exec_success"] = 0
            breakdown["has_rows"] = 0
    else:
        breakdown["exec_success"] = 0
        breakdown["has_rows"] = 0

    # 3. Domain heuristics
    sql_upper = sql.upper()

    # Expected conditions coverage
    expected = 0
    found = 0
    if conditions.get("prototype"):
        expected += 1
        if "PROTOTYPE" in sql_upper or "STRUKTURBERICHT" in sql_upper:
            found += 1
    if conditions.get("contains_elements"):
        expected += 1
        if "ELEMENT" in sql_upper or "EXISTS" in sql_upper:
            found += 1
    if conditions.get("stability"):
        expected += 1
        if "ENERGY_ABOVE_HULL" in sql_upper or "IS_STABLE" in sql_upper:
            found += 1
    if conditions.get("sort_by"):
        expected += 1
        if "ORDER BY" in sql_upper:
            found += 1
    cond_score = int(_SCORE_CONDITIONS * (found / expected)) if expected > 0 else _SCORE_CONDITIONS
    score += cond_score
    breakdown["conditions"] = cond_score

    # Appropriate row count
    if exec_result and exec_result.get("success"):
        row_count = exec_result.get("row_count", 0)
        if 1 <= row_count <= _ROW_COUNT_MAX_IDEAL:
            score += _SCORE_ROW_RANGE_FULL
            breakdown["row_range"] = _SCORE_ROW_RANGE_FULL
        elif row_count > _ROW_COUNT_MAX_IDEAL:
            score += _SCORE_ROW_RANGE_PARTIAL
            breakdown["row_range"] = _SCORE_ROW_RANGE_PARTIAL
        else:
            breakdown["row_range"] = 0
    else:
        breakdown["row_range"] = 0

    # DISTINCT usage
    if "DISTINCT" in sql_upper:
        score += _SCORE_DISTINCT
        breakdown["distinct"] = _SCORE_DISTINCT
    else:
        breakdown["distinct"] = 0

    # LIMIT presence
    if "LIMIT" in sql_upper:
        score += _SCORE_LIMIT
        breakdown["limit"] = _SCORE_LIMIT
    else:
        breakdown["limit"] = 0

    # 4. Domain validation penalty (up to -30)
    if exec_result and exec_result.get("success"):
        from llm.domain_validator import validate_result_rows
        domain_result = validate_result_rows(
            rows=exec_result.get("rows", []),
            columns=exec_result.get("columns", []),
            conditions=conditions,
        )
        penalty = domain_result.get("score_penalty", 0)
        if penalty > 0:
            score = max(0, score - penalty)
            breakdown["domain_penalty"] = -penalty

    return {
        "score": score,
        "breakdown": breakdown,
        "exec_result": exec_result,
        "validation": validation,
    }


def pipeline(
    user_query: str,
    join_list: list[str] | None = None,
    all_columns: list[str] | None = None,
    store_on_success: bool = False,
    skip_intent_check: bool = False,
    n_best: int = 1,
    execute_fn: Any | None = None,
    table_graph: Any | None = None,
) -> dict[str, Any]:
    """Full pipeline: intent classify -> extract -> link -> generate SQL.

    Parameters
    ----------
    join_list : list[str] | None
        Pre-computed join conditions from ``get_allowed_join_list()`` or
        ``build_schema_context_from_db()``.  When *None* the pipeline
        falls back to the core 5-table join set.
    all_columns : list[str] | None
        Full ``table.column`` list from ``build_schema_context_from_db()``.
        When *None* the pipeline uses the core 5-table column set.
    store_on_success : bool
        Persist result in the few-shot store after successful DB execution.
    skip_intent_check : bool
        Bypass intent classification (useful for benchmarking).
    n_best : int
        Number of SQL candidates to generate (default 1). When > 1,
        generates multiple candidates and selects the highest-scored one.
    execute_fn : callable | None
        SQL execution function for n-best scoring. Signature:
        ``execute_fn(sql) -> {"success": bool, "row_count": int, ...}``.
        Required when n_best > 1 for execution-based scoring.
    table_graph : nx.Graph | None
        Schema graph for JOIN path generation. When provided, pipeline
        uses ``generate_joins_for_tables()`` to compute JOIN clauses
        from schema graph traversal instead of filtering pre-computed
        join_list.
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
        logger.warning(
            "join_list not provided; using hard-coded 5-table fallback. "
            "Pass join_list via build_schema_context_from_db() for "
            "full 30-table schema graph traversal."
        )
        join_list = _CORE_5_TABLE_JOINS
    if all_columns is None:
        all_columns = _CORE_5_TABLE_COLUMNS

    # Restrict linked tables to those actually present in the target schema;
    # if none match (e.g. an unseen schema), expose the full schema instead.
    schema_tables = {c.split(".")[0] for c in all_columns}
    required_tables = [
        t for t in linked["required_tables"] if t in schema_tables
    ]
    if not required_tables:
        logger.debug(
            "No linked tables exist in the target schema; "
            "falling back to the full schema"
        )
        required_tables = sorted(schema_tables)
    linked = {**linked, "required_tables": required_tables}

    filtered_columns = [
        c for c in all_columns
        if c.split(".")[0] in linked["required_tables"]
    ]

    # Schema Graph JOIN path generation: when table_graph is provided,
    # compute JOIN clauses via BFS traversal instead of filtering.
    if table_graph is not None and _HAS_GRAPH:
        join_clause = generate_joins_for_tables(
            table_graph, linked["required_tables"]
        )
        if join_clause:
            req = set(linked["required_tables"])
            filtered_joins = [
                j for j in get_allowed_join_list(table_graph)
                if any(t in j for t in req)
            ]
        else:
            filtered_joins = [
                j for j in join_list
                if any(t in j for t in linked["required_tables"])
            ]
    else:
        filtered_joins = [
            j for j in join_list
            if any(t in j for t in linked["required_tables"])
        ]

    if n_best <= 1:
        result = generate_sql_via_llm(
            user_query=user_query,
            allowed_tables=linked["required_tables"],
            allowed_columns=filtered_columns,
            allowed_joins=filtered_joins,
        )
        result["conditions"] = conditions
        result["linked_schema"] = linked
        result["_store_on_success"] = store_on_success
        return result

    # n-best: generate multiple candidates and score
    candidates: list[dict[str, Any]] = []
    total_tokens = 0
    total_latency = 0

    current_model = os.getenv("LLM_MODEL", "gpt-5.5")
    _is_new = current_model and any(
        t in current_model for t in ("gpt-5", "o1", "o3", "o4")
    )

    for i in range(n_best):
        # For GPT-5/o-series: temperature may not be supported; use None
        # and rely on seed variation for candidate diversity.
        # For other models: vary temperature (0.0, 0.3, 0.6, ...).
        if _is_new:
            temp_override = None
        else:
            temp_override = min(round(i * 0.3, 1), 2.0) if i > 0 else 0.0
        gen_result = generate_sql_via_llm(
            user_query=user_query,
            allowed_tables=linked["required_tables"],
            allowed_columns=filtered_columns,
            allowed_joins=filtered_joins,
            temperature=temp_override,
        )
        sql = gen_result.get("sql", "")
        total_tokens += gen_result.get("tokens", 0)
        total_latency += gen_result.get("latency_ms", 0)

        if not sql:
            candidates.append({"sql": "", "score": 0, "gen_result": gen_result})
            continue

        scored = _score_sql_candidate(sql, conditions, execute_fn)
        candidates.append({
            "sql": sql,
            "score": scored["score"],
            "breakdown": scored["breakdown"],
            "exec_result": scored.get("exec_result"),
            "gen_result": gen_result,
        })

    # Also include rule-based as a candidate (pass pre-computed data to avoid
    # redundant extract_conditions/link_schema/reranker calls)
    rb_sql = _rule_based_fallback(
        user_query, linked["required_tables"],
        filtered_columns, filtered_joins,
        conditions=conditions, linked=linked,
    )
    if rb_sql:
        rb_scored = _score_sql_candidate(rb_sql, conditions, execute_fn)
        candidates.append({
            "sql": rb_sql,
            "score": rb_scored["score"],
            "breakdown": rb_scored["breakdown"],
            "exec_result": rb_scored.get("exec_result"),
            "gen_result": {
                "sql": rb_sql, "model": "rule_based", "tokens": 0,
                "latency_ms": 0,
            },
        })

    # Rerank candidates using LLM semantic scoring
    from llm.reranker import rerank_sql_candidates
    candidates = rerank_sql_candidates(user_query, candidates)

    # Select best candidate
    candidates.sort(key=lambda c: c["score"], reverse=True)
    best = candidates[0]

    result = best.get("gen_result", {})
    result["sql"] = best["sql"]
    result["conditions"] = conditions
    result["linked_schema"] = linked
    result["_store_on_success"] = store_on_success
    result["n_best_info"] = {
        "n_candidates": len(candidates),
        "best_score": best["score"],
        "best_breakdown": best.get("breakdown", {}),
        "all_scores": [c["score"] for c in candidates],
        "reranked": any(c.get("reranker_score") is not None for c in candidates),
        "total_tokens": total_tokens,
        "total_latency_ms": total_latency,
    }

    return result
