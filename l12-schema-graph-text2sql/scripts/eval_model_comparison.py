#!/usr/bin/env python3
"""LLM model comparison: evaluate the full pipeline with different LLM backends.

Models tested:
  - gpt-5.5 (baseline, from 5-run ablation)
  - gpt-4o

Additional models can be appended to MODELS (e.g. Anthropic models require
ANTHROPIC_API_KEY); the manuscript comparison table covers gpt-5.5 vs gpt-4o.

Output: evaluation/model_comparison_results.json
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))


from evaluation.metrics import execution_accuracy_full, normalize_limit  # noqa: E402
from scripts.provenance import (  # noqa: E402
    assert_resumable,
    build_provenance,
)
from graph.graph_builder import build_table_graph  # noqa: E402
from graph.join_path_generator import get_allowed_join_list  # noqa: E402
from graph.schema_parser import get_foreign_keys, get_tables, get_columns  # noqa: E402
from scripts.eval_db import open_eval_connection, run_model_sql  # noqa: E402

EVAL_DIR = PROJECT / "evaluation"
RESULTS_DIR = EVAL_DIR / "expected_results"

CONNINFO = (
    f"host={os.getenv('POSTGRES_HOST', 'localhost')} "
    f"port={os.getenv('POSTGRES_PORT', '5432')} "
    f"dbname={os.getenv('POSTGRES_DB', 'l12_materials')} "
    f"user={os.getenv('POSTGRES_USER', 'l12_user')} "
    f"password={os.getenv('POSTGRES_PASSWORD', 'l12_password')}"
)

# Models to evaluate
MODELS = [
    {"name": "gpt-4o", "provider": "openai", "model_id": "gpt-4o"},
]


def models_config_sha256(models: list[dict[str, str]]) -> str:
    """Stable hash of the compared model conditions.

    Covers every condition's name / provider / model_id so a stored
    result produced under a different model configuration is rejected
    as stale on resume instead of being silently skipped.
    """
    canon = sorted(
        (m["name"], m["provider"], m["model_id"]) for m in models
    )
    return hashlib.sha256(json.dumps(canon).encode()).hexdigest()


def _current_provenance() -> dict[str, str]:
    prov = build_provenance(EVAL_DIR / "evaluation_dataset.jsonl")
    prov["models_config_sha256"] = models_config_sha256(MODELS)
    return prov


def load_queries():
    queries = []
    with open(EVAL_DIR / "evaluation_dataset.jsonl") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    return queries


def load_expected(qid):
    path = RESULTS_DIR / f"{qid}.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        return data.get("rows", []), data.get("columns", [])
    return [], []


def execute_sql(conn, sql):
    return run_model_sql(conn, sql)



def compute_accuracy(conn, sql, qid):
    expected_rows, expected_columns = load_expected(qid)
    if not sql:
        return 0.0
    exec_result = execute_sql(conn, sql)
    if not exec_result.get("success"):
        return 0.0
    metrics = execution_accuracy_full(
        exec_result["rows"], expected_rows,
        exec_result["columns"], expected_columns,
    )
    return metrics.get("recall", 0.0)


def _call_anthropic(prompt: str, model_id: str) -> str:
    """Call Anthropic Claude API and return the SQL response."""
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model=model_id,
        max_tokens=4096,
        system="You are a PostgreSQL expert for materials databases.",
        messages=[{"role": "user", "content": prompt}],
    )
    if not resp.content:
        return ""
    block = resp.content[0]
    if isinstance(block, anthropic.types.TextBlock):
        return block.text
    return ""


def run_model_condition(
    conn, queries, model_info,
    allowed_joins, allowed_columns, table_graph, exec_fn,
):
    """Run the full pipeline with a specific LLM model."""
    from llm.sql_generator import (
        build_constrained_prompt,
        classify_query_type,
        extract_conditions,
        extract_sql_from_response,
        specify_output_schema,
        _fix_known_literals,
        _normalize_column_aliases,
    )
    from llm.few_shot_store import retrieve_similar

    provider = model_info["provider"]
    model_id = model_info["model_id"]

    results = []
    for i, q in enumerate(queries):
        qid = q["id"]
        question = q["question"]
        difficulty = q["difficulty"]

        print(
            f"  [{i+1}/{len(queries)}] {qid} ({difficulty})...",
            end=" ", flush=True,
        )

        t0 = time.time()
        try:
            # Build the same prompt as the pipeline
            conditions = extract_conditions(question)
            conditions.get("_coverage", {})
            few_shot = retrieve_similar(question, top_k=3)
            query_type_info = classify_query_type(question)
            column_hint = specify_output_schema(
                question, conditions, allowed_columns,
            )

            prompt = build_constrained_prompt(
                question,
                [t for t in allowed_joins],
                allowed_columns,
                allowed_joins,
                few_shot_examples=few_shot,
                query_type_instruction=query_type_info["instruction"],
                column_hint=column_hint,
            )

            if provider == "openai":
                import openai
                client = openai.OpenAI(
                    api_key=os.environ.get("OPENAI_API_KEY", ""),
                )
                create_kwargs: dict[str, Any] = dict(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": "You are a PostgreSQL expert for materials databases."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=4096,
                )
                resp = client.chat.completions.create(**create_kwargs)
                raw = resp.choices[0].message.content or ""
            elif provider == "anthropic":
                raw = _call_anthropic(prompt, model_id)
            else:
                raw = ""

            sql = extract_sql_from_response(raw)
            sql = _fix_known_literals(sql)
            sql = _normalize_column_aliases(sql)

            if sql:
                sql = normalize_limit(sql)

            # Execute and score
            acc = compute_accuracy(conn, sql, qid)
            elapsed = time.time() - t0
            print(f"acc={acc:.1%}  {elapsed:.1f}s")

            results.append({
                "qid": qid,
                "difficulty": difficulty,
                "accuracy": acc,
                "latency_s": round(elapsed, 1),
                "sql": sql,
            })
        except Exception as e:
            elapsed = time.time() - t0
            print(f"ERROR: {type(e).__name__}: {e!s:.80s}  {elapsed:.1f}s")
            results.append({
                "qid": qid,
                "difficulty": difficulty,
                "accuracy": 0.0,
                "latency_s": round(elapsed, 1),
                "sql": "",
            })

    return results


def main():
    out_path = PROJECT / "evaluation" / "model_comparison_results.json"

    # Allow specifying which model to start from
    start_from = os.getenv("MODEL_START", "")

    models = MODELS
    if start_from:
        idx = next((i for i, m in enumerate(MODELS) if m["name"] == start_from), 0)
        models = MODELS[idx:]

    print(f"Models: {[m['name'] for m in models]}")
    print("Connecting to PostgreSQL...")
    conn = open_eval_connection(CONNINFO, suite="main")

    print("Loading schema...")
    tables = get_tables(conn)
    columns = {}
    for t in tables:
        columns[t] = get_columns(conn, t)
    fks = get_foreign_keys(conn)
    table_graph = build_table_graph(fks)
    if not table_graph.has_edge("composition", "element"):
        table_graph.add_edge(
            "composition", "element",
            source_column="element", target_column="symbol",
        )
    allowed_columns = [
        f"{t}.{c.column_name}" for t, cols in columns.items() for c in cols
    ]
    allowed_joins = get_allowed_join_list(table_graph)

    print("Loading queries...")
    all_queries = load_queries()
    print(f"Total queries: {len(all_queries)}")

    def exec_fn(sql):
        return execute_sql(conn, sql)

    # Load existing results if resuming
    all_results: dict[str, Any] = {}
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
        assert_resumable(
            existing.get("provenance", {}),
            _current_provenance(),
            force="--force-stale-resume" in sys.argv,
            what=out_path.name,
            extra_keys=("models_config_sha256",))
        all_results = existing.get("models", {})
        print(f"Loaded existing results: {list(all_results.keys())}")

    for model_info in models:
        model_name = model_info["name"]
        if model_name in all_results:
            print(f"\nSkipping {model_name} (already exists)")
            continue

        print(f"\n{'='*70}")
        print(f"MODEL: {model_name} ({model_info['provider']})")
        print(f"{'='*70}")

        results = run_model_condition(
            conn, all_queries, model_info,
            allowed_joins, allowed_columns, table_graph, exec_fn,
        )

        # Compute summary
        total_acc = sum(r["accuracy"] for r in results) / len(results)
        by_diff: dict[str, list[float]] = {}
        for r in results:
            by_diff.setdefault(r["difficulty"], []).append(r["accuracy"])
        diff_summary = {
            d: sum(accs) / len(accs) for d, accs in by_diff.items()
        }
        avg_latency = sum(r["latency_s"] for r in results) / len(results)

        print(f"\n  Overall: {total_acc:.1%}")
        for d in ["easy", "medium", "hard", "very_hard"]:
            if d in diff_summary:
                print(f"  {d:12s}: {diff_summary[d]:.1%}")
        print(f"  Avg latency: {avg_latency:.1f}s")

        all_results[model_name] = {
            "provider": model_info["provider"],
            "model_id": model_info["model_id"],
            "overall": total_acc,
            "by_difficulty": diff_summary,
            "avg_latency": avg_latency,
            "results": results,
        }

        # Save after each model (incremental)
        with open(out_path, "w") as f:
            json.dump({
                "provenance": _current_provenance(),
                "n_queries": len(all_queries),
                "models": all_results,
            }, f, ensure_ascii=False, indent=2)
        print(f"  Saved to {out_path}")

    # Final summary
    print(f"\n{'='*70}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*70}")

    # Include gpt-5.5 baseline from ablation
    baseline_path = PROJECT / "evaluation" / "ablation_multirun_stats.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            stats = json.load(f)
        full_cond = stats.get("conditions", {}).get("full", {})
        if full_cond:
            print(
                f"{'gpt-5.5':>20s} {full_cond['overall_mean']:.1%} "
                f"(5-run mean, baseline)"
            )

    for model_info in MODELS:
        name = model_info["name"]
        if name not in all_results:
            continue
        r = all_results[name]
        diff = r["by_difficulty"]
        print(
            f"{name:>20s} {r['overall']:7.1%} "
            f"E={diff.get('easy',0):.0%} M={diff.get('medium',0):.0%} "
            f"H={diff.get('hard',0):.0%} VH={diff.get('very_hard',0):.0%} "
            f"{r['avg_latency']:.1f}s"
        )

    conn.close()
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
