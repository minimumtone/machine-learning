#!/usr/bin/env python3
"""Create the obfuscated-transfer evaluation dataset from the original transfer dataset.

Translates the gold SQL with the mapping produced by build_obfuscated_transfer_db.py,
executes it against the obfuscated DB, and writes expected results + a new JSONL dataset.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import psycopg
from psycopg.conninfo import make_conninfo

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from graph.schema_parser import get_foreign_keys  # noqa: E402
from scripts.gold_compare import sql_is_ordered  # noqa: E402
from scripts.transfer_guard import assert_valid_transfer  # noqa: E402

SRC_DATASET = PROJECT / "evaluation" / "transfer_evaluation_dataset.jsonl"
MAP_PATH = PROJECT / "db" / "obfuscated_transfer_mapping.json"
GOLD_DIR = PROJECT / "evaluation" / "gold_sql_obfuscated"
EXPECTED_DIR = PROJECT / "evaluation" / "expected_results_obfuscated"
OUT_DATASET = PROJECT / "evaluation" / "transfer_obfuscated_evaluation_dataset.jsonl"
PROMPT_TEMPLATE = PROJECT / "llm" / "prompt_templates" / "sql_generation_prompt_transfer_obfuscated.md"

COLUMN_DESCRIPTIONS: dict[str, str] = {
    "ratio_key": "element-ratio identifier",
    "entry_key": "unique entry identifier",
    "symbol": "chemical element symbol (e.g. Ni)",
    "atomic_ratio": "stoichiometric ratio of the element in the compound",
    "wyckoff_site": "Wyckoff site letter",
    "element_name": "element name (e.g. nickel)",
    "atomic_number": "atomic number Z",
    "atomic_mass": "atomic mass (amu)",
    "composition_formula": "chemical formula of the compound (e.g. Ni3Al)",
    "prototype_label": "prototype structure label (e.g. L12, B2)",
    "spacegroup_number": "space group number",
    "crystal_system": "crystal system (cubic / hexagonal / tetragonal / ...)",
    "lattice_param_a": "lattice parameter a (angstrom)",
    "cell_volume_pa": "unit-cell volume per atom",
    "fe_key": "formation-energy identifier",
    "delta_e": "formation energy ΔE relative to reference states (eV per atom)",
    "hull_distance": "distance from the thermodynamic convex hull (eV per atom)",
    "on_hull": "whether the structure is on the convex hull (boolean)",
    "gap_ev": "electronic band gap (eV)",
    "ref_key": "reference-state identifier",
    "gs_spacegroup": "ground-state Hermann\u2013Mauguin space-group symbol (e.g. Fm-3m)",
    "reference_delta_e": "elemental reference formation energy per atom (same convention as delta_e; subtract the ratio-weighted sum from delta_e to re-reference to elemental ground states)",
    "volume_pa": "ground-state atomic volume",
    "polymorph_count": "number of OQMD single-element structure entries (polymorph candidates) for this element",
}


def _obf_conninfo(db: str) -> str:
    password = os.environ.get("POSTGRES_PASSWORD")
    if not password:
        raise RuntimeError("POSTGRES_PASSWORD environment variable is required")
    return make_conninfo(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=db,
        user=os.getenv("POSTGRES_USER", "l12_user"),
        password=password,
    )


def load_translation(mapping_path: Path) -> dict[str, str]:
    mapping = json.loads(mapping_path.read_text())
    trans: dict[str, str] = {}
    for old_t, new_t in mapping["tables"].items():
        trans[old_t] = new_t
    for tdata in mapping["columns"].values():
        for old_c, new_c in tdata["columns"].items():
            if old_c not in trans:
                # global column mapping: keep the first new name encountered
                trans[old_c] = new_c
    return trans


def translate_sql(sql: str, translation: dict[str, str]) -> str:
    """Replace identifiers with their obfuscated counterparts."""
    # Sort by length descending to avoid partial substitutions.
    for old in sorted(translation, key=len, reverse=True):
        new = translation[old]
        sql = re.sub(rf"\b{re.escape(old)}\b", new, sql)
    return sql


def execute_and_save(sql: str, conn: psycopg.Connection, out_path: Path,
                     ordered: bool) -> dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute("SAVEPOINT obf_gold")
        try:
            cur.execute(sql)  # type: ignore[arg-type]
            rows = cur.fetchall()
            cols = [desc[0] for desc in cur.description] if cur.description else []
            cur.execute("RELEASE SAVEPOINT obf_gold")
        except psycopg.Error:
            cur.execute("ROLLBACK TO SAVEPOINT obf_gold")
            cur.execute("RELEASE SAVEPOINT obf_gold")
            raise
    data: list[list[Any]] = []
    for row in rows:
        record: list[Any] = []
        for val in row:
            if isinstance(val, (dict, list)):
                record.append(val)
            elif hasattr(val, "__float__") and not isinstance(val, (bool, int, str)):
                record.append(float(val))
            else:
                record.append(val)
        data.append(record)
    payload = {"columns": cols, "ordered": ordered, "rows": data}
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def build_prompt_template(conn: psycopg.Connection, mapping: dict[str, Any]) -> str:
    """Create a prompt template that describes obfuscated columns in plain English."""
    fks = get_foreign_keys(conn)
    join_lines: list[str] = []
    # De-duplicate FK join clauses
    seen = set()
    for fk in fks:
        line = f"{fk.source_table}.{fk.source_column} = {fk.target_table}.{fk.target_column}"
        if line not in seen:
            seen.add(line)
            join_lines.append(line)

    table_names = {tdata["new_table"] for tdata in mapping["columns"].values()}
    table_descriptions: dict[str, str] = {
        "oqmd_element_ratios": "stoichiometric element ratios for each compound",
        "oqmd_elements": "periodic-table element properties",
        "oqmd_entries": "material entries (formula, prototype, lattice)",
        "oqmd_formation_energies": "thermodynamic formation energies",
        "oqmd_reference_states": "pure-element ground-state reference data",
    }

    # Per-table columns for prompt
    by_table: dict[str, list[tuple[str, str]]] = {}
    for old_t, tdata in mapping["columns"].items():
        new_t = tdata["new_table"]
        cols: list[tuple[str, str]] = []
        for old_c, new_c in tdata["columns"].items():
            desc = str(COLUMN_DESCRIPTIONS.get(old_c, old_c.replace("_", " ")))
            cols.append((str(new_c), desc))
        by_table[new_t] = cols

    allowed_tables_block = "\n".join(
        f"  {t}: {table_descriptions.get(next((ot for ot, td in mapping['columns'].items() if td['new_table']==t), t), t)}"
        for t in sorted(table_names)
    )
    allowed_columns_block = "\n".join(
        f"  {t}:\n" + "\n".join(f"    - {c}: {desc}" for c, desc in sorted(cols))
        for t, cols in sorted(by_table.items())
    )
    allowed_joins_block = "\n".join(f"  - {j}" for j in join_lines)

    return f"""You are a Text-to-SQL generator for a materials database (PostgreSQL).
Generate only one PostgreSQL SELECT query.

Rules:
- Use ONLY the provided tables and columns. Do NOT invent column names.
- Use ONLY the provided JOIN clauses.
- Do not use INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE.
- Return SQL only.
- Always include a LIMIT clause (default LIMIT 10000).
- For "最小/最大/最も" questions, use ORDER BY + LIMIT 1.
- For "何件/数を教えて" questions, use COUNT(*) with appropriate WHERE and a descriptive alias.
- IMPORTANT: Follow the "Output structure instruction" below. If it says to return individual rows, do NOT use GROUP BY or aggregate functions.
- The table and column names are anonymized. Use the plain-English descriptions in parentheses to choose the right columns.
- Few-shot examples (if any) may reference a DIFFERENT schema. Reuse only their SQL patterns (JOIN structure, aggregation, CTE style) — table and column names MUST come from the allowed lists below.

Aggregation patterns (use ONLY when the Output structure instruction says to aggregate):
- "割合" (ratio/percentage): SELECT COUNT(*) FILTER(WHERE condition) * 100.0 / COUNT(*) or use CASE+SUM
- "最も多い/少ない" (most/least): GROUP BY + ORDER BY COUNT(*) DESC/ASC LIMIT 1
- Default: return individual rows with ORDER BY. Do NOT use GROUP BY unless explicitly instructed.

Allowed tables:
{allowed_tables_block}

Allowed columns (ONLY use these exact column names — use the descriptions in parentheses):
{allowed_columns_block}

Allowed JOINs:
{allowed_joins_block}

Output structure instruction:
{{query_type_instruction}}

Column selection guidance:
Return only the columns directly relevant to answering the question. Do NOT add entry identifiers or other auxiliary columns unless explicitly requested.

User query:
{{user_query}}

SQL:
""".replace("{{query_type_instruction}}", "{query_type_instruction}").replace("{{user_query}}", "{user_query}")


def main() -> int:
    obf_db = os.getenv("TRANSFER_DB", "oqmd_transfer") + "_obfuscated"
    translation = load_translation(MAP_PATH)
    mapping = json.loads(MAP_PATH.read_text())
    conn = psycopg.connect(_obf_conninfo(obf_db))
    # Same contract as the other expected-result generators: one
    # REPEATABLE READ READ ONLY snapshot for the whole run, and the
    # destination must be a valid (marker + fingerprint) transfer DB.
    conn.read_only = True
    conn.isolation_level = psycopg.IsolationLevel.REPEATABLE_READ
    with conn.cursor() as cur:
        cur.execute("SET statement_timeout = '30s'")
    assert_valid_transfer(conn)
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    EXPECTED_DIR.mkdir(parents=True, exist_ok=True)

    prompt = build_prompt_template(conn, mapping)
    PROMPT_TEMPLATE.write_text(prompt)

    out_lines: list[dict[str, Any]] = []
    n_failed = 0
    for line in SRC_DATASET.read_text().strip().splitlines():
        item = json.loads(line)
        old_id = item["id"]
        new_id = old_id.replace("q_transfer_", "q_transfer_obf_")
        old_gold_path = PROJECT / "evaluation" / item["gold_sql_path"]
        gold = old_gold_path.read_text()
        gold_obf = translate_sql(gold, translation)
        gold_obf_path = GOLD_DIR / f"{new_id}.sql"
        gold_obf_path.write_text(gold_obf)

        # A generation failure must not leave a valid-looking expected
        # JSON behind: remove any stale file, roll back the aborted
        # transaction, and fail the whole run at the end.
        expected_path = EXPECTED_DIR / f"{new_id}.json"
        try:
            execute_and_save(gold_obf, conn, expected_path,
                             ordered=sql_is_ordered(gold_obf))
        except Exception as exc:
            print(f"FAILED {new_id}: {exc}")
            n_failed += 1
            expected_path.unlink(missing_ok=True)
            continue

        out_lines.append({
            "id": new_id,
            "question": item["question"],
            "difficulty": item["difficulty"],
            "gold_sql_path": f"gold_sql_obfuscated/{new_id}.sql",
            "expected_result_path": f"expected_results_obfuscated/{new_id}.json",
        })

    OUT_DATASET.write_text("\n".join(json.dumps(line, ensure_ascii=False) for line in out_lines))
    conn.close()
    print(f"Wrote {OUT_DATASET}")
    if n_failed:
        print(f"generation_failed={n_failed}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
