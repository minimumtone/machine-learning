"""Run baseline Text-to-SQL methods for comparison.

DEPRECATED: This is the legacy standalone baseline runner.
Use scripts/run_full_evaluation.py which runs ALL methods (baselines + proposed)
in a single pass with unified metrics, token budgets, and repair loops.
This file is kept for reference only and should NOT be used for evaluation.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


def _load_schema_lists() -> tuple[list[str], list[str]]:
    """Load allowed tables/columns from allowed_schema.yaml (30 tables)."""
    import yaml
    schema_path = Path(__file__).parent.parent / "safety" / "allowed_schema.yaml"
    with schema_path.open(encoding="utf-8") as f:
        schema = yaml.safe_load(f)
    return schema.get("allowed_tables", []), schema.get("allowed_columns", [])


ALL_TABLES, ALL_COLUMNS = _load_schema_lists()

ALL_JOINS = [
    "composition.entry_id = material_entry.entry_id",
    "structure.entry_id = material_entry.entry_id",
    "calculation.entry_id = material_entry.entry_id",
    "calculated_property.calculation_id = calculation.calculation_id",
    "phase_stability.entry_id = material_entry.entry_id",
]

FK_LIST_TEXT = "\n".join(f"- {j}" for j in ALL_JOINS)


def baseline1_prompt(question: str) -> str:
    """Baseline 1: LLM only, no schema info."""
    return (
        "Generate a PostgreSQL SELECT query for a materials database.\n"
        f"Question: {question}\n"
        "SQL:"
    )


def baseline2_prompt(question: str) -> str:
    """Baseline 2: LLM + full schema dump."""
    schema_text = "Tables and columns:\n"
    for t in ALL_TABLES:
        cols = [c.split(".")[1] for c in ALL_COLUMNS if c.startswith(f"{t}.")]
        schema_text += f"  {t}: {', '.join(cols)}\n"
    return (
        "Generate a PostgreSQL SELECT query.\n"
        f"{schema_text}\n"
        f"Question: {question}\n"
        "SQL:"
    )


def baseline3_rule_based(question: str) -> str:
    """Baseline 3: Rule-based SQL generation (no LLM). Dictionary pattern matching."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from llm.entity_extractor import extract_conditions
    from llm.condition_mapper import map_prototype_condition, map_stability_condition

    conditions = extract_conditions(question)
    where_parts: list[str] = []
    tables_needed: set[str] = {"material_entry"}

    proto = conditions.get("prototype")
    if proto:
        result = map_prototype_condition(proto)
        if result:
            if isinstance(result, list):
                for r in result:
                    where_parts.append(r["sql_fragment"])
                    tables_needed.update(r.get("tables", []))
            else:
                where_parts.append(result["sql_fragment"])
                tables_needed.update(result.get("tables", []))

    stability = conditions.get("stability")
    if stability:
        result = map_stability_condition(stability)
        if result:
            where_parts.append(result["sql_fragment"])
            tables_needed.update(result.get("tables", []))

    elements = conditions.get("contains_elements", [])
    if elements:
        tables_needed.add("composition")
        elem_conds = " OR ".join(f"c.element = '{e}'" for e in elements)
        where_parts.append(f"({elem_conds})")

    # Build SQL
    joins = []
    if "structure" in tables_needed:
        joins.append("JOIN structure s ON s.entry_id = m.entry_id")
    if "phase_stability" in tables_needed:
        joins.append("JOIN phase_stability ps ON ps.entry_id = m.entry_id")
    if "composition" in tables_needed:
        joins.append("JOIN composition c ON c.entry_id = m.entry_id")

    sql = "SELECT m.entry_id, m.formula\nFROM material_entry m\n"
    sql += "\n".join(joins)
    if where_parts:
        sql += "\nWHERE " + " AND ".join(where_parts)
    row_limit = int(os.getenv("SQL_ROW_LIMIT", "100"))
    sql += f"\nLIMIT {row_limit};"
    return sql


def baseline4_prompt(question: str) -> str:
    """Baseline 4: LLM + FK list."""
    return (
        "Generate a PostgreSQL SELECT query.\n"
        f"Foreign keys:\n{FK_LIST_TEXT}\n\n"
        f"Question: {question}\n"
        "SQL:"
    )


def run_baseline(
    baseline_id: str,
    prompt_fn: Any,
    dataset_path: Path,
    output_path: Path,
    model: str | None = None,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """Run a single baseline over the evaluation dataset."""
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY", "")
    if model is None:
        model = os.getenv("LLM_MODEL", "gpt-5.5")

    results: list[dict[str, Any]] = []
    with dataset_path.open() as f:
        queries = [json.loads(line) for line in f if line.strip()]

    for q in queries:
        prompt = prompt_fn(q["question"])
        t0 = time.time()

        if api_key and api_key != "your_api_key_here":
            import openai
            from llm.sql_generator import extract_sql_from_response
            client = openai.OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=512,
            )
            raw = resp.choices[0].message.content or ""
            sql = extract_sql_from_response(raw)
            tokens = (resp.usage.prompt_tokens + resp.usage.completion_tokens) if resp.usage else 0
        else:
            sql = "SELECT 'baseline_placeholder' AS result LIMIT 1;"
            tokens = 0

        latency_ms = int((time.time() - t0) * 1000)
        results.append({
            "id": q["id"],
            "baseline": baseline_id,
            "question": q["question"],
            "difficulty": q["difficulty"],
            "sql": sql,
            "tokens": tokens,
            "latency_ms": latency_ms,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results


def run_baseline3(
    dataset_path: Path,
    output_path: Path,
) -> list[dict[str, Any]]:
    """Run Baseline 3 (rule-based, no LLM)."""
    results: list[dict[str, Any]] = []
    with dataset_path.open() as f:
        queries = [json.loads(line) for line in f if line.strip()]

    for q in queries:
        t0 = time.time()
        sql = baseline3_rule_based(q["question"])
        latency_ms = int((time.time() - t0) * 1000)
        results.append({
            "id": q["id"],
            "baseline": "baseline3",
            "question": q["question"],
            "difficulty": q["difficulty"],
            "sql": sql,
            "tokens": 0,
            "latency_ms": latency_ms,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return results


def run_all_baselines(dataset_path: Path | None = None) -> None:
    """Run all four baselines."""
    if dataset_path is None:
        dataset_path = Path(__file__).parent / "evaluation_dataset.jsonl"
    out_dir = dataset_path.parent

    # Baselines 1, 2, 4 use LLM
    llm_baselines = [
        ("baseline1", baseline1_prompt),
        ("baseline2", baseline2_prompt),
        ("baseline4", baseline4_prompt),
    ]
    for bid, fn in llm_baselines:
        print(f"Running {bid} ...")
        run_baseline(bid, fn, dataset_path, out_dir / f"{bid}_result.json")

    # Baseline 3 is rule-based (no LLM)
    print("Running baseline3 (rule-based) ...")
    run_baseline3(dataset_path, out_dir / "baseline3_result.json")

    print("All baselines complete.")


if __name__ == "__main__":
    run_all_baselines()
