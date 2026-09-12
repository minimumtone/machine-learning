#!/usr/bin/env python3
"""Derive all main-corpus artifacts from the single canonical inference run.

The canonical main run is ``evaluation/multiaxis_results.json`` (one
pipeline inference over all 245 main-corpus questions, with the
generated SQL stored per query).  Everything downstream is DERIVED from
that stored SQL, deterministically and without calling any LLM:

  evaluation/generated_sql/main/<qid>.sql       (245 files, re-exported)
  evaluation/generated_sql/main/manifest.json   (+ provenance, source hash)
  evaluation/main_eval_with_sql.json            (re-scored from stored SQL)
  evaluation/generated_sql/llm_only/<qid>.sql   (245 files, re-exported)
  evaluation/generated_sql/llm_only/manifest.json

Per-query scores are recomputed by executing the stored SQL against the
fixture (READ ONLY + REPEATABLE READ + fixture guard + SAVEPOINT) and
are self-checked against the recall stored in the canonical run; any
mismatch is a hard failure.  The canonical exact-result-set metric
(exact column list + row multiset + row order only when the expected
result's semantic_ordered flag is set, i.e. the question itself asks for
an ordered answer) is computed here as well.

The exact_match field of the canonical run itself is also normalized to
the canonical exact_result_set_match definition (recomputed from the
stored SQL, no LLM call), so a single "exact" definition holds across
the whole package.

Run ``scripts/build_failure_analysis.py`` afterwards; it reads the same
canonical run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from evaluation.metrics import execution_accuracy_full  # noqa: E402
from evaluation.metrics_strict import exact_result_set_match  # noqa: E402
from scripts.db_conninfo import main_conninfo  # noqa: E402
from scripts.eval_db import open_eval_connection, run_model_sql  # noqa: E402
from scripts.provenance import sha256_file  # noqa: E402

EVAL = PROJECT / "evaluation"
CANONICAL = EVAL / "multiaxis_results.json"
LLM_ONLY = EVAL / "llm_only_results.json"


def load_dataset() -> dict[str, dict[str, Any]]:
    out = {}
    for line in open(EVAL / "main_evaluation_dataset.jsonl"):
        if line.strip():
            r = json.loads(line)
            out[r["id"]] = r
    return out


def export_sql(subdir: str, results: list[dict[str, Any]],
               sql_key: str) -> dict[str, Path]:
    target = EVAL / "generated_sql" / subdir
    target.mkdir(parents=True, exist_ok=True)
    for stale in target.glob("*.sql"):
        stale.unlink()
    paths = {}
    for r in results:
        p = target / f"{r['qid']}.sql"
        p.write_text((r.get(sql_key) or "").rstrip() + "\n", encoding="utf-8")
        paths[r["qid"]] = p
    return paths


def write_manifest(subdir: str, source_file: Path, source_data: dict[str, Any],
                   entries: list[dict[str, Any]], eval_file: str) -> None:
    target = EVAL / "generated_sql" / subdir
    missing = [e["sql_path"] for e in entries
               if not (PROJECT / e["sql_path"]).exists()]
    if missing:
        raise RuntimeError(f"{subdir}: {len(missing)} manifest sql_path "
                           f"missing: {missing[:5]}")
    manifest = {
        "source": subdir,
        "model": source_data.get("model"),
        "provenance": source_data.get("provenance"),
        "eval_file": eval_file,
        "source_result_file": source_file.name,
        "source_result_sha256": sha256_file(source_file),
        "n_queries": len(entries),
        "queries": entries,
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{subdir}] {len(entries)} SQL files + manifest")


def main() -> int:
    dataset = load_dataset()
    canonical = json.load(open(CANONICAL))
    results = canonical["results"]
    if len(results) != len(dataset):
        raise RuntimeError(
            f"canonical run has {len(results)} results, dataset has "
            f"{len(dataset)} questions")

    sql_paths = export_sql("main", results, "gen_sql")

    conn = open_eval_connection(main_conninfo(), suite="main",
                                statement_timeout="30s")
    mismatches: list[tuple[str, float, float]] = []
    derived: list[dict[str, Any]] = []
    manifest_entries: list[dict[str, Any]] = []
    for r in results:
        qid = r["qid"]
        meta = dataset[qid]
        exp = json.load(open(EVAL / meta["expected_result_path"]))
        sql = r.get("gen_sql") or ""
        if sql.strip():
            got = run_model_sql(conn, sql)
        else:
            got = {"success": False, "rows": [], "columns": []}
        acc = execution_accuracy_full(
            got.get("rows", []), exp.get("rows", []),
            got.get("columns", []), exp.get("columns", []))
        if abs(acc["recall"] - r["recall"]) > 1e-6:
            mismatches.append((qid, acc["recall"], r["recall"]))
        exact = exact_result_set_match(
            got.get("rows", []), exp.get("rows", []),
            got.get("columns", []), exp.get("columns", []),
            ordered=bool(exp.get("semantic_ordered")))
        derived.append({
            "qid": qid,
            "difficulty": r["difficulty"],
            "question": meta["question"],
            "execution_recall": acc["recall"],
            "exact_result_set_match": exact,
            "latency_s": r["latency_s"],
            "sql": sql,
        })
        manifest_entries.append({
            "qid": qid,
            "difficulty": r["difficulty"],
            "execution_recall": acc["recall"],
            "latency_s": r["latency_s"],
            "sql_path": str(sql_paths[qid].relative_to(PROJECT)),
        })
    conn.close()

    if mismatches:
        for qid, new, old in mismatches[:10]:
            print(f"SELF-CHECK MISMATCH {qid}: recomputed={new} stored={old}")
        return 1

    # Normalize the canonical run's exact metric to the canonical
    # definition (deterministic re-score of the stored SQL).
    exact_by_qid = {d["qid"]: d["exact_result_set_match"] for d in derived}
    changed = False
    for r in results:
        if r.get("exact_match") != exact_by_qid[r["qid"]]:
            r["exact_match"] = exact_by_qid[r["qid"]]
            changed = True
    note = ("exact_match = evaluation.metrics_strict.exact_result_set_match: "
            "exact gold column list + row multiset match + row order only "
            "when the expected result's semantic_ordered flag is set (the "
            "question itself asks for an ordered answer; normalized "
            "deterministically from the stored SQL by "
            "scripts/derive_main_artifacts.py)")
    if canonical.get("exact_metric") != note:
        canonical["exact_metric"] = note
        changed = True
    agg = canonical.get("aggregate", {})
    new_rate = sum(1 for r in results if r["exact_match"]) / len(results)
    if agg.get("exact_match_rate") != new_rate:
        agg["exact_match_rate"] = new_rate
        changed = True
    for diff, block in canonical.get("by_difficulty", {}).items():
        dr = [r for r in results if r["difficulty"] == diff]
        rate = sum(1 for r in dr if r["exact_match"]) / len(dr)
        if block.get("exact_match") != rate:
            block["exact_match"] = rate
            changed = True
    if changed:
        CANONICAL.write_text(
            json.dumps(canonical, ensure_ascii=False, indent=2),
            encoding="utf-8")
        print("[canonical] multiaxis_results.json exact_match normalized "
              "to exact_result_set_match")

    n = len(derived)
    total = sum(d["execution_recall"] for d in derived) / n
    exact_rate = sum(1 for d in derived if d["exact_result_set_match"]) / n
    by_diff: dict[str, list[float]] = {}
    for d in derived:
        by_diff.setdefault(d["difficulty"], []).append(d["execution_recall"])
    out = {
        "model": canonical.get("model"),
        "provenance": canonical.get("provenance"),
        "derived_from": CANONICAL.name,
        "source_result_sha256": sha256_file(CANONICAL),
        "derivation": (
            "re-scored deterministically from the generated SQL stored in "
            "the canonical run by scripts/derive_main_artifacts.py; no LLM "
            "call is involved"),
        "metric": "historical execution recall (row-set recall over "
                  "common columns; see scripts/audit_scoring.py for the "
                  "strict and exact-match co-metrics)",
        "exact_metric": "exact_result_set_match = exact gold column list "
                        "+ row multiset match + row order only when the "
                        "expected result's semantic_ordered flag is set "
                        "(evaluation.metrics_strict.exact_result_set_match)",
        "n_queries": n,
        "overall_execution_recall": total,
        "exact_result_set_match_rate": exact_rate,
        "by_difficulty": {d: sum(v) / len(v) for d, v in by_diff.items()},
        "exact_by_difficulty": {
            d: sum(1 for x in derived
                   if x["difficulty"] == d and x["exact_result_set_match"])
               / len(v)
            for d, v in by_diff.items()},
        "avg_latency": sum(d["latency_s"] for d in derived) / n,
        "results": derived,
    }
    (EVAL / "main_eval_with_sql.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[main] historical execution recall {total:.1%}, "
          f"exact_result_set_match {exact_rate:.1%} (n={n})")

    write_manifest("main", CANONICAL, canonical, manifest_entries,
                   "main_eval_with_sql.json")

    llm = json.load(open(LLM_ONLY))
    llm_results = llm["results"]
    if len(llm_results) != len(dataset):
        raise RuntimeError(
            f"llm_only run has {len(llm_results)} results, dataset has "
            f"{len(dataset)} questions")
    llm_paths = export_sql("llm_only", llm_results, "gen_sql")
    write_manifest("llm_only", LLM_ONLY, llm, [
        {
            "qid": r["qid"],
            "difficulty": r["difficulty"],
            "recall": r["recall"],
            "latency_s": r["latency_s"],
            "sql_path": str(llm_paths[r["qid"]].relative_to(PROJECT)),
        }
        for r in llm_results
    ], "llm_only_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
