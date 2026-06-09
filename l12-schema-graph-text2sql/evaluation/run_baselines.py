"""Run baseline Text-to-SQL methods for comparison."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


ALL_TABLES = [
    "material_entry", "composition", "structure",
    "calculation", "calculated_property", "phase_stability",
    "prototype_definition",
]

ALL_COLUMNS = [
    "material_entry.entry_id", "material_entry.formula",
    "material_entry.reduced_formula", "material_entry.chemical_system",
    "material_entry.number_of_elements",
    "composition.composition_id", "composition.entry_id",
    "composition.element", "composition.atomic_fraction", "composition.site_label",
    "structure.structure_id", "structure.entry_id", "structure.prototype",
    "structure.strukturbericht", "structure.formula_type",
    "structure.space_group_number", "structure.crystal_system",
    "structure.lattice_a", "structure.lattice_b", "structure.lattice_c",
    "structure.volume_per_atom",
    "calculation.calculation_id", "calculation.entry_id",
    "calculation.method", "calculation.functional", "calculation.calculation_type",
    "calculated_property.property_id", "calculated_property.calculation_id",
    "calculated_property.property_name", "calculated_property.value",
    "calculated_property.unit",
    "phase_stability.stability_id", "phase_stability.entry_id",
    "phase_stability.formation_energy_per_atom",
    "phase_stability.energy_above_hull", "phase_stability.is_stable",
    "prototype_definition.prototype_id", "prototype_definition.prototype_name",
    "prototype_definition.strukturbericht", "prototype_definition.formula_type",
    "prototype_definition.description",
]

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


def baseline3_prompt(question: str) -> str:
    """Baseline 3: LLM + embedding-based schema retrieval (simulated)."""
    return baseline2_prompt(question)


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


def run_all_baselines(dataset_path: Path | None = None) -> None:
    """Run all four baselines."""
    if dataset_path is None:
        dataset_path = Path(__file__).parent / "evaluation_dataset.jsonl"
    out_dir = dataset_path.parent

    baselines = [
        ("baseline1", baseline1_prompt),
        ("baseline2", baseline2_prompt),
        ("baseline3", baseline3_prompt),
        ("baseline4", baseline4_prompt),
    ]
    for bid, fn in baselines:
        print(f"Running {bid} ...")
        run_baseline(bid, fn, dataset_path, out_dir / f"{bid}_result.json")
    print("All baselines complete.")


if __name__ == "__main__":
    run_all_baselines()
