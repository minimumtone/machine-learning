#!/usr/bin/env python3
"""LLM-only baseline: raw single-shot SQL generation with no pipeline aids.

The prompt contains only the raw schema (table.column listing plus FK
pairs) and the natural-language question.  No term dictionary, no
few-shot examples, no schema linking, no graph-constrained JOIN paths,
no SQLGuard, no repair loop, no n-best generation, no reranker, and no
literal/alias post-processing are applied.  The only post-processing is
extraction of the SQL text from the model response (code-fence
stripping), which is response parsing rather than a pipeline aid.

Metrics per query: recall/precision/F1 against expected results, syntax
validity, execution validity, and hallucinated table/column/JOIN rates.

Output: evaluation/llm_only_results.json
        evaluation/generated_sql/llm_only/<qid>.sql
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import psycopg  # noqa: E402

from evaluation.metrics import (  # noqa: E402
    execution_accuracy_full,
    hallucinated_column_rate,
    hallucinated_join_rate,
    hallucinated_table_rate,
    syntax_validity,
)
from llm.sql_generator import extract_sql_from_response  # noqa: E402

EVAL_DIR = PROJECT / "evaluation"
RESULTS_DIR = EVAL_DIR / "expected_results"
GOLD_SQL_DIR = EVAL_DIR / "gold_sql"
SQL_OUT_DIR = EVAL_DIR / "generated_sql" / "llm_only"

CONNINFO = (
    f"host={os.getenv('POSTGRES_HOST', 'localhost')} "
    f"port={os.getenv('POSTGRES_PORT', '5432')} "
    f"dbname={os.getenv('POSTGRES_DB', 'l12_materials')} "
    f"user={os.getenv('POSTGRES_USER', 'l12_user')} "
    f"password={os.getenv('POSTGRES_PASSWORD', 'l12_password')}"
)


def load_queries() -> list[dict[str, Any]]:
    queries = []
    with open(EVAL_DIR / "evaluation_dataset.jsonl") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    return queries


def load_expected(qid: str) -> dict[str, Any]:
    path = RESULTS_DIR / f"{qid}.json"
    data: dict[str, Any] = {}
    if path.exists():
        with open(path) as f:
            data = json.load(f)
    gold_path = GOLD_SQL_DIR / f"{qid}.sql"
    if gold_path.exists():
        data["gold_sql"] = gold_path.read_text(encoding="utf-8").strip()
    return data


def execute_sql(conn: Any, sql: str) -> dict[str, Any]:
    try:
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout = '10s'")
            cur.execute(sql)
            columns = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchall()
        return {
            "success": True, "columns": columns,
            "rows": [list(r) for r in rows], "row_count": len(rows),
        }
    except Exception as e:
        conn.rollback()
        return {
            "success": False, "error": str(e),
            "rows": [], "row_count": 0, "columns": [],
        }


def build_raw_schema_text(conn: Any) -> tuple[str, list[str], list[str], list[str]]:
    """Serialize the live schema as a plain table/column/FK listing."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT table_name, column_name, data_type "
            "FROM information_schema.columns "
            "WHERE table_schema = 'public' "
            "ORDER BY table_name, ordinal_position"
        )
        col_rows = cur.fetchall()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT tc.table_name, kcu.column_name, "
            "       ccu.table_name, ccu.column_name "
            "FROM information_schema.table_constraints tc "
            "JOIN information_schema.key_column_usage kcu "
            "  ON tc.constraint_name = kcu.constraint_name "
            " AND tc.table_schema = kcu.table_schema "
            "JOIN information_schema.constraint_column_usage ccu "
            "  ON tc.constraint_name = ccu.constraint_name "
            " AND ccu.table_schema = tc.table_schema "
            "WHERE tc.constraint_type = 'FOREIGN KEY' "
            " AND tc.table_schema = 'public'"
        )
        fk_rows = cur.fetchall()

    by_table: dict[str, list[str]] = {}
    for t, c, dt in col_rows:
        by_table.setdefault(t, []).append(f"{c} {dt}")
    lines = []
    for t in sorted(by_table):
        lines.append(f"TABLE {t} ({', '.join(by_table[t])})")
    lines.append("")
    lines.append("FOREIGN KEYS:")
    for a, ac, b, bc in sorted(fk_rows):
        lines.append(f"  {a}.{ac} -> {b}.{bc}")

    allowed_tables = sorted(by_table)
    allowed_columns = [f"{t}.{c.split(' ')[0]}" for t in by_table for c in by_table[t]]
    allowed_joins = [f"{a}.{ac}={b}.{bc}" for a, ac, b, bc in fk_rows]
    return "\n".join(lines), allowed_tables, allowed_columns, allowed_joins


def generate_sql_raw(question: str, schema_text: str, model: str) -> tuple[str, int]:
    import openai
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    prompt = (
        "Given the following PostgreSQL database schema, write a single SQL "
        "query that answers the question. Output only the SQL query.\n\n"
        f"{schema_text}\n\n"
        f"Question: {question}\n"
    )
    create_kwargs: dict[str, Any] = dict(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    if any(t in model for t in ("gpt-5", "o1", "o3", "o4")):
        create_kwargs["max_completion_tokens"] = 16384
    else:
        create_kwargs["temperature"] = 0.0
        create_kwargs["max_tokens"] = 4096
    resp = client.chat.completions.create(**create_kwargs)
    raw = resp.choices[0].message.content or ""
    usage = resp.usage
    tokens = (usage.prompt_tokens + usage.completion_tokens) if usage else 0
    return extract_sql_from_response(raw), tokens


def _extract_from_join_tables(sql: str) -> list[str]:
    """Table names referenced in FROM/JOIN, excluding CTE names."""
    if not sql:
        return []
    cte_names: set[str] = set()
    for m in re.finditer(r"\bWITH\b\s+(\w+)\s+AS\b", sql, re.IGNORECASE):
        cte_names.add(m.group(1).lower())
    for m in re.finditer(r",\s*(\w+)\s+AS\s*\(", sql, re.IGNORECASE):
        cte_names.add(m.group(1).lower())
    tables = []
    for m in re.finditer(r"(?:FROM|JOIN)\s+(\w+)", sql, re.IGNORECASE):
        t = m.group(1).lower()
        if t not in cte_names and t not in tables:
            tables.append(t)
    return tables


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="",
                        help="comma-separated qids to (re)run; merge into "
                             "existing results")
    args = parser.parse_args()
    only_qids = {q for q in args.only.split(",") if q}

    out_path = EVAL_DIR / "llm_only_results.json"
    SQL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = os.getenv("LLM_MODEL", "gpt-5.5")

    print("Connecting to PostgreSQL...")
    conn = psycopg.connect(CONNINFO)

    print("Serializing raw schema...")
    schema_text, allowed_tables, allowed_columns, allowed_joins = (
        build_raw_schema_text(conn)
    )
    print(f"{len(allowed_tables)} tables, {len(allowed_columns)} columns, "
          f"{len(allowed_joins)} FKs")

    all_queries = load_queries()
    print(f"Total queries: {len(all_queries)}")

    prior: dict[str, dict[str, Any]] = {}
    if only_qids:
        with open(out_path) as f:
            prior = {r["qid"]: r for r in json.load(f)["results"]}
        all_queries = [q for q in all_queries if q["id"] in only_qids]
        print(f"Rerunning {len(all_queries)} queries: "
              f"{sorted(only_qids)}")

    results = []
    if not only_qids and out_path.exists():
        with open(out_path) as f:
            saved = json.load(f).get("results", [])
        done = {r["qid"] for r in saved}
        results = [r for r in saved if r["qid"] in done]
        all_queries = [q for q in all_queries if q["id"] not in done]
        print(f"Resuming: {len(done)} done, {len(all_queries)} remaining")
    for i, q in enumerate(all_queries):
        qid = q["id"]
        question = q["question"]
        difficulty = q["difficulty"]
        print(f"[{i+1}/{len(all_queries)}] {qid} ({difficulty})...",
              end=" ", flush=True)

        expected = load_expected(qid)
        expected_rows = expected.get("rows", [])
        expected_columns = expected.get("columns", [])

        t0 = time.time()
        try:
            gen_sql, tokens = generate_sql_raw(question, schema_text, model)
        except Exception as e:
            elapsed = time.time() - t0
            print(f"API ERROR: {type(e).__name__}: {e!s:.80s}")
            results.append({
                "qid": qid, "difficulty": difficulty, "gen_sql": "",
                "recall": 0.0, "precision": 0.0, "f1": 0.0,
                "syntax_valid": False, "execution_valid": False,
                "hallucinated_table_rate": 0.0,
                "hallucinated_column_rate": 0.0,
                "hallucinated_join_rate": 0.0,
                "has_table_hallucination": False,
                "has_column_hallucination": False,
                "has_join_hallucination": False,
                "exec_error": f"api_error: {e}",
                "tokens": 0, "latency_s": round(elapsed, 1),
            })
            continue
        elapsed = time.time() - t0

        (SQL_OUT_DIR / f"{qid}.sql").write_text(gen_sql + "\n", encoding="utf-8")

        gen_result = execute_sql(conn, gen_sql) if gen_sql else {
            "success": False, "rows": [], "columns": [], "error": "empty_sql",
        }
        acc = execution_accuracy_full(
            gen_result.get("rows", []), expected_rows,
            gen_result.get("columns", []), expected_columns,
        )
        syn_valid = syntax_validity(gen_sql)
        exec_valid = gen_result.get("success", False)

        gen_tables = _extract_from_join_tables(gen_sql)
        gen_cols_list = [
            f"{t}.{c}" for t, c in re.findall(r"(\w+)\.(\w+)", gen_sql)
        ] if gen_sql else []
        hall_table = hallucinated_table_rate(gen_tables, allowed_tables)
        hall_col = hallucinated_column_rate(gen_cols_list, allowed_columns, gen_sql)
        hall_join = hallucinated_join_rate(gen_sql, allowed_joins)

        print(f"recall={acc['recall']:.1%} exec={'Y' if exec_valid else 'N'} "
              f"hT={hall_table:.0%} hC={hall_col:.0%} hJ={hall_join:.0%} "
              f"{elapsed:.1f}s")

        results.append({
            "qid": qid, "difficulty": difficulty, "gen_sql": gen_sql,
            "recall": acc["recall"], "precision": acc["precision"],
            "f1": acc["f1"],
            "syntax_valid": syn_valid, "execution_valid": exec_valid,
            "hallucinated_table_rate": hall_table,
            "hallucinated_column_rate": hall_col,
            "hallucinated_join_rate": hall_join,
            "has_table_hallucination": hall_table > 0,
            "has_column_hallucination": hall_col > 0,
            "has_join_hallucination": hall_join > 0,
            "exec_error": gen_result.get("error", ""),
            "tokens": tokens, "latency_s": round(elapsed, 1),
        })

        if not only_qids:
            with open(out_path, "w") as f:
                json.dump({"model": model, "results": results}, f,
                          ensure_ascii=False, indent=2)

    if only_qids:
        for r in results:
            prior[r["qid"]] = r
        results = [prior[q["id"]] for q in load_queries()]

    n = len(results)
    agg: dict[str, Any] = {
        "n_queries": n,
        "recall_mean": sum(r["recall"] for r in results) / n,
        "precision_mean": sum(r["precision"] for r in results) / n,
        "f1_mean": sum(r["f1"] for r in results) / n,
        "syntax_validity_rate": sum(1 for r in results if r["syntax_valid"]) / n,
        "execution_validity_rate": sum(1 for r in results if r["execution_valid"]) / n,
        "hallucinated_table_rate_mean": sum(r["hallucinated_table_rate"] for r in results) / n,
        "hallucinated_column_rate_mean": sum(r["hallucinated_column_rate"] for r in results) / n,
        "hallucinated_join_rate_mean": sum(r["hallucinated_join_rate"] for r in results) / n,
        "queries_with_table_hallucination": sum(1 for r in results if r["has_table_hallucination"]),
        "queries_with_column_hallucination": sum(1 for r in results if r["has_column_hallucination"]),
        "queries_with_join_hallucination": sum(1 for r in results if r["has_join_hallucination"]),
        "latency_mean_s": sum(r["latency_s"] for r in results) / n,
    }
    by_diff: dict[str, dict[str, Any]] = {}
    for diff in ["easy", "medium", "hard", "very_hard"]:
        dr = [r for r in results if r["difficulty"] == diff]
        if not dr:
            continue
        nd = len(dr)
        by_diff[diff] = {
            "n": nd,
            "recall": sum(r["recall"] for r in dr) / nd,
            "precision": sum(r["precision"] for r in dr) / nd,
            "f1": sum(r["f1"] for r in dr) / nd,
            "execution_validity": sum(1 for r in dr if r["execution_valid"]) / nd,
            "queries_with_column_hallucination": sum(
                1 for r in dr if r["has_column_hallucination"]),
            "queries_with_join_hallucination": sum(
                1 for r in dr if r["has_join_hallucination"]),
        }

    output = {
        "model": model,
        "condition": "llm_only_raw_schema_single_shot",
        "aggregate": agg,
        "by_difficulty": by_diff,
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}")
    print("LLM-ONLY BASELINE SUMMARY")
    for k, v in agg.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
