#!/usr/bin/env python3
"""Prepare queries, gold SQL, expected results, and prompt assets for MP transfer test."""
from __future__ import annotations

import json
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

import psycopg

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from scripts.build_mp_transfer_db import mp_conninfo  # noqa: E402

EVAL_DIR = PROJECT / "evaluation"
EXPECTED_DIR = EVAL_DIR / "expected_results"
PROMPT_DIR = PROJECT / "llm" / "prompt_templates"
FEW_SHOT_DIR = PROJECT / "llm"

QUERIES: list[dict[str, Any]] = [
    {
        "id": "q_mp_001",
        "difficulty": "easy",
        "question": "材料エントリー（mp_entries）は全体で何件ありますか？",
        "gold_sql": "SELECT COUNT(*) FROM mp_entries;",
    },
    {
        "id": "q_mp_002",
        "difficulty": "easy",
        "question": "格子定数aが5未満の材料の式を3つ教えてください。",
        "gold_sql": "SELECT formula FROM mp_entries WHERE lattice_a < 5 ORDER BY formula, entry_id LIMIT 3;",
    },
    {
        "id": "q_mp_003",
        "difficulty": "easy",
        "question": "Niを含む材料の式を5つ列挙してください。",
        "gold_sql": (
            "SELECT DISTINCT e.formula FROM mp_entries e "
            "JOIN mp_element_ratios r ON e.entry_id = r.entry_id "
            "WHERE r.element = 'Ni' ORDER BY e.formula LIMIT 5;"
        ),
    },
    {
        "id": "q_mp_004",
        "difficulty": "easy",
        "question": "熱力学的に安定な（energy_above_hull=0）エントリーは何件ですか？",
        "gold_sql": "SELECT COUNT(*) FROM mp_entries WHERE is_stable = TRUE;",
    },
    {
        "id": "q_mp_005",
        "difficulty": "easy",
        "question": "mp_elementsに含まれる元素の種類数を教えてください。",
        "gold_sql": "SELECT COUNT(*) FROM mp_elements;",
    },
    {
        "id": "q_mp_006",
        "difficulty": "medium",
        "question": "結晶系ごとの材料数を教えてください。",
        "gold_sql": (
            "SELECT crystal_system, COUNT(*) FROM mp_entries "
            "GROUP BY crystal_system ORDER BY COUNT(*) DESC;"
        ),
    },
    {
        "id": "q_mp_007",
        "difficulty": "medium",
        "question": "Al-Ni系の材料は何件ありますか？",
        "gold_sql": "SELECT COUNT(*) FROM mp_entries WHERE chemsys = 'Al-Ni';",
    },
    {
        "id": "q_mp_008",
        "difficulty": "medium",
        "question": "バンドギャップを持つ材料（band_gap > 0）の割合を教えてください。",
        "gold_sql": (
            "SELECT COUNT(*) FILTER(WHERE band_gap > 0) * 100.0 / COUNT(*) "
            "FROM mp_entries;"
        ),
    },
    {
        "id": "q_mp_009",
        "difficulty": "medium",
        "question": "体積が最も大きい材料の式を教えてください。",
        "gold_sql": "SELECT formula FROM mp_entries ORDER BY volume DESC NULLS LAST, formula LIMIT 1;",
    },
    {
        "id": "q_mp_010",
        "difficulty": "hard",
        "question": "Coを含む材料の中で、energy_above_hullが最小の材料の式と値を教えてください。",
        "gold_sql": (
            "SELECT e.formula, e.energy_above_hull FROM mp_entries e "
            "JOIN mp_element_ratios r ON e.entry_id = r.entry_id "
            "WHERE r.element = 'Co' ORDER BY e.energy_above_hull ASC, e.formula LIMIT 1;"
        ),
    },
    {
        "id": "q_mp_011",
        "difficulty": "hard",
        "question": "各元素系（chemsys）ごとに最も安定な材料の式を教えてください。",
        "gold_sql": (
            "SELECT DISTINCT ON (chemsys) chemsys, formula, energy_above_hull "
            "FROM mp_entries ORDER BY chemsys, energy_above_hull ASC, formula;"
        ),
    },
    {
        "id": "q_mp_012",
        "difficulty": "hard",
        "question": "立方晶（Cubic）でバンドギャップが1 eVを超える材料は何件ありますか？",
        "gold_sql": (
            "SELECT COUNT(*) FROM mp_entries "
            "WHERE crystal_system = 'Cubic' AND band_gap > 1;"
        ),
    },
    {
        "id": "q_mp_013",
        "difficulty": "very_hard",
        "question": "Co-Ti系の材料の中で、バンドギャップが最大の材料の式を教えてください。",
        "gold_sql": (
            "SELECT formula FROM mp_entries WHERE chemsys = 'Co-Ti' "
            "ORDER BY band_gap DESC NULLS LAST, formula LIMIT 1;"
        ),
    },
    {
        "id": "q_mp_014",
        "difficulty": "very_hard",
        "question": "CoとTiの両方を含む材料の平均energy_above_hullを教えてください。",
        "gold_sql": (
            "SELECT AVG(e.energy_above_hull) FROM mp_entries e "
            "JOIN mp_element_ratios r1 ON e.entry_id = r1.entry_id AND r1.element = 'Co' "
            "JOIN mp_element_ratios r2 ON e.entry_id = r2.entry_id AND r2.element = 'Ti';"
        ),
    },
    {
        "id": "q_mp_015",
        "difficulty": "very_hard",
        "question": "結晶系ごとの平均バンドギャップを大きい順に教えてください。",
        "gold_sql": (
            "SELECT crystal_system, AVG(band_gap) FROM mp_entries "
            "GROUP BY crystal_system ORDER BY AVG(band_gap) DESC NULLS LAST;"
        ),
    },
]

FEW_SHOT: list[dict[str, str]] = [
    {
        "question": "mp_entriesの総数を教えて。",
        "sql": "SELECT COUNT(*) AS entry_count FROM mp_entries;",
    },
    {
        "question": "Niを含む材料の式を5つ。",
        "sql": (
            "SELECT DISTINCT e.formula FROM mp_entries e "
            "JOIN mp_element_ratios r ON e.entry_id = r.entry_id "
            "WHERE r.element = 'Ni' ORDER BY e.formula LIMIT 5;"
        ),
    },
    {
        "question": "立方晶で最も体積の大きい材料。",
        "sql": (
            "SELECT formula FROM mp_entries WHERE crystal_system = 'Cubic' "
            "ORDER BY volume DESC LIMIT 1;"
        ),
    },
    {
        "question": "Coを含み、energy_above_hullが最小の材料。",
        "sql": (
            "SELECT e.formula, e.energy_above_hull FROM mp_entries e "
            "JOIN mp_element_ratios r ON e.entry_id = r.entry_id "
            "WHERE r.element = 'Co' ORDER BY e.energy_above_hull ASC LIMIT 1;"
        ),
    },
    {
        "question": "Al-Ni系の材料数。",
        "sql": "SELECT COUNT(*) FROM mp_entries WHERE chemsys = 'Al-Ni';",
    },
    {
        "question": "結晶系ごとに材料を数えよ。",
        "sql": (
            "SELECT crystal_system, COUNT(*) FROM mp_entries "
            "GROUP BY crystal_system ORDER BY COUNT(*) DESC;"
        ),
    },
    {
        "question": "バンドギャップがある材料の割合。",
        "sql": (
            "SELECT COUNT(*) FILTER(WHERE band_gap > 0) * 100.0 / COUNT(*) "
            "FROM mp_entries;"
        ),
    },
    {
        "question": "CoとTiの両方を含む材料の平均energy_above_hull。",
        "sql": (
            "SELECT AVG(e.energy_above_hull) FROM mp_entries e "
            "JOIN mp_element_ratios r1 ON e.entry_id = r1.entry_id AND r1.element = 'Co' "
            "JOIN mp_element_ratios r2 ON e.entry_id = r2.entry_id AND r2.element = 'Ti';"
        ),
    },
]

PROMPT_TEMPLATE = """You are a Text-to-SQL generator for a Materials Project-flavored materials database (PostgreSQL).
Generate only one PostgreSQL SELECT query.

Rules:
- Use ONLY the provided tables and columns. Do NOT invent column names.
- Use ONLY the provided JOIN clauses.
- Do not use INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE.
- Return SQL only.
- Always include a LIMIT clause (default LIMIT 10000).
- For "最小/最大/最も" questions, use ORDER BY + LIMIT 1.
- For queries that ask for `N` examples or a limited list (e.g. "3つ", "5つ"), add ORDER BY on a stable column (e.g. `formula`, `entry_id`) before LIMIT so the result is deterministic.
- For "何件/数を教えて" questions, use COUNT(*) with appropriate WHERE and a descriptive alias.
- For "割合" (ratio/percentage) questions, use COUNT(*) FILTER(WHERE condition) * 100.0 / COUNT(*).
- IMPORTANT: Follow the "Output structure instruction" below. If it says to return individual rows, do NOT use GROUP BY or aggregate functions.
- Few-shot examples (if any) may reference a DIFFERENT schema. Reuse only their SQL patterns (JOIN structure, aggregation) — table and column names MUST come from the allowed lists below.

Allowed tables:
{allowed_tables}

Allowed columns (ONLY use these exact column names — do NOT invent or guess column names):
{allowed_columns}

Allowed JOINs:
{allowed_joins}

Output structure instruction:
{query_type_instruction}

Column selection guidance:
{column_hint}

User query:
{user_query}

SQL:
"""


def _convert(v: Any) -> Any:
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, (list, tuple)):
        return [_convert(x) for x in v]
    return v


def execute_gold(conn, sql: str) -> dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute("SET statement_timeout = '30s'")
        cur.execute(sql)
        columns = [d[0] for d in cur.description] if cur.description else []
        rows = [_convert(list(r)) for r in cur.fetchall()]
    return {"columns": columns, "rows": rows}


def main() -> None:
    conn = psycopg.connect(mp_conninfo())

    EXPECTED_DIR.mkdir(parents=True, exist_ok=True)

    with open(EVAL_DIR / "mp_transfer_evaluation_dataset.jsonl", "w", encoding="utf-8") as f:
        for q in QUERIES:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    for q in QUERIES:
        try:
            result = execute_gold(conn, q["gold_sql"])
        except Exception as exc:
            print(f"ERROR {q['id']}: {exc}")
            result = {"columns": [], "rows": [], "error": str(exc)}
        with open(EXPECTED_DIR / f"{q['id']}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"{q['id']}: rows={len(result.get('rows', []))}")

    with open(PROMPT_DIR / "sql_generation_prompt_mp.md", "w", encoding="utf-8") as f:
        f.write(PROMPT_TEMPLATE)

    mp_few_shot = [{"nl_query": ex["question"], "sql": ex["sql"]} for ex in FEW_SHOT]
    with open(FEW_SHOT_DIR / "few_shot_examples_mp.json", "w", encoding="utf-8") as f:
        json.dump(mp_few_shot, f, ensure_ascii=False, indent=2)

    print("Prepared MP transfer assets.")
    conn.close()


if __name__ == "__main__":
    main()
