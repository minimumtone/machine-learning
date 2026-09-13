#!/usr/bin/env python3
"""Export per-query ablation results as machine-readable CSV tables.

These files replace the former supplementary Table S3 (per-query results
of the 100-query ablation subset, first run of the *full* condition) and
Table S4 (per-query recall by ablation condition).  They are DERIVED
from saved artifacts only -- no LLM call, no database, no randomness:

  evaluation/query_catalog.csv             (question text, difficulty)
  evaluation/ablation_run_1.json ..5.json  (per-query recall / exact / latency)

Outputs (byte-identical across repeated runs):

  evaluation/per_query_results.csv
      qid, eval_set, difficulty, question, execution_recall, exact_match,
      latency_s, passed                       -- run 1, full condition
  evaluation/per_query_by_condition.csv
      qid, difficulty, full, no_fewshot, no_dict, no_reranker, no_guard,
      no_nbest, no_graph                      -- run 1, execution recall
  evaluation/per_query_by_condition_mean5.csv
      same columns, mean execution recall over runs 1..5
  evaluation/per_query_tables_provenance.json
      SHA-256 of every input file, row counts, consistency checks and
      SHA-256 of the three CSV files

`passed` is 1 when execution recall >= PASS_THRESHOLD (0.8), the criterion
used by the former Table S3.  `eval_set` is copied from query_catalog.csv;
all 100 rows are the ablation subset of the 245-query main evaluation.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from statistics import mean

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from scripts.provenance import sha256_file  # noqa: E402

EVAL = PROJECT / "evaluation"
CATALOG = EVAL / "query_catalog.csv"
RUN_FILES = [EVAL / f"ablation_run_{i}.json" for i in range(1, 6)]
CONDITIONS = ["full", "no_fewshot", "no_dict", "no_reranker",
              "no_guard", "no_nbest", "no_graph"]
PASS_THRESHOLD = 0.8
N_QUERIES = 100
DIFFICULTY_ORDER = {"easy": 0, "medium": 1, "hard": 2, "very_hard": 3}
EXPECTED_DIFFICULTY_COUNTS = {"easy": 20, "medium": 30, "hard": 30, "very_hard": 20}

OUT_RESULTS = EVAL / "per_query_results.csv"
OUT_BY_COND = EVAL / "per_query_by_condition.csv"
OUT_BY_COND_MEAN = EVAL / "per_query_by_condition_mean5.csv"
OUT_PROV = EVAL / "per_query_tables_provenance.json"


def fmt(x: float) -> str:
    return f"{x:.4f}"


def load_catalog() -> dict[str, dict[str, str]]:
    with CATALOG.open(encoding="utf-8", newline="") as fh:
        rows = {r["qid"]: r for r in csv.DictReader(fh)}
    if len(rows) == 0:
        raise RuntimeError(f"{CATALOG.name}: empty catalog")
    return rows


def load_runs() -> list[dict]:
    runs = []
    for p in RUN_FILES:
        if not p.is_file():
            raise FileNotFoundError(f"missing saved run: {p.name}")
        run = json.loads(p.read_text(encoding="utf-8"))
        if run.get("n_queries") != N_QUERIES:
            raise RuntimeError(f"{p.name}: n_queries={run.get('n_queries')} != {N_QUERIES}")
        conds = run["conditions"]
        if set(conds) != set(CONDITIONS):
            raise RuntimeError(f"{p.name}: conditions {sorted(conds)} != {sorted(CONDITIONS)}")
        for cond in CONDITIONS:
            qids = [r["qid"] for r in conds[cond]["results"]]
            if len(qids) != N_QUERIES or len(set(qids)) != N_QUERIES:
                raise RuntimeError(f"{p.name}/{cond}: expected {N_QUERIES} unique qids, "
                                   f"got {len(qids)} ({len(set(qids))} unique)")
        runs.append(run)
    ids = [r["run_id"] for r in runs]
    if ids != [1, 2, 3, 4, 5]:
        raise RuntimeError(f"run_id sequence {ids} != [1..5]")
    return runs


def by_qid(run: dict, cond: str) -> dict[str, dict]:
    return {r["qid"]: r for r in run["conditions"][cond]["results"]}


def sort_key(qid: str, difficulty: str) -> tuple[int, str]:
    return DIFFICULTY_ORDER[difficulty], qid


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def main() -> int:
    catalog = load_catalog()
    runs = load_runs()
    run1 = runs[0]
    full1 = by_qid(run1, "full")

    qid_set = set(full1)
    for run in runs:
        for cond in CONDITIONS:
            if set(by_qid(run, cond)) != qid_set:
                raise RuntimeError(f"run {run['run_id']}/{cond}: qid set differs from run1/full")
    missing = sorted(q for q in qid_set if q not in catalog)
    if missing:
        raise RuntimeError(f"qids absent from {CATALOG.name}: {missing[:5]}")

    difficulty = {q: full1[q]["difficulty"] for q in qid_set}
    for q in qid_set:
        if catalog[q]["difficulty_original"] != difficulty[q]:
            raise RuntimeError(f"{q}: catalog difficulty_original="
                               f"{catalog[q]['difficulty_original']} != run {difficulty[q]}")
        for run in runs:
            for cond in CONDITIONS:
                d = by_qid(run, cond)[q]["difficulty"]
                if d != difficulty[q]:
                    raise RuntimeError(f"{q}: difficulty differs in run {run['run_id']}/{cond}")
    counts = {lvl: sum(1 for q in qid_set if difficulty[q] == lvl) for lvl in DIFFICULTY_ORDER}
    if counts != EXPECTED_DIFFICULTY_COUNTS:
        raise RuntimeError(f"difficulty counts {counts} != {EXPECTED_DIFFICULTY_COUNTS}")

    order = sorted(qid_set, key=lambda q: sort_key(q, difficulty[q]))

    # --- per_query_results.csv (run 1, full condition) -----------------
    results_rows = []
    for q in order:
        r = full1[q]
        results_rows.append([
            q,
            catalog[q]["eval_set"],
            difficulty[q],
            catalog[q]["question"],
            fmt(r["recall"]),
            str(int(round(r["exact_match"]))),
            f"{r['latency_s']:.1f}",
            "1" if r["recall"] >= PASS_THRESHOLD else "0",
        ])
    write_csv(OUT_RESULTS,
              ["qid", "eval_set", "difficulty", "question", "execution_recall",
               "exact_match", "latency_s", "passed"],
              results_rows)

    # --- per_query_by_condition.csv (run 1) ----------------------------
    cond1 = {c: by_qid(run1, c) for c in CONDITIONS}
    cond_rows = [[q, difficulty[q]] + [fmt(cond1[c][q]["recall"]) for c in CONDITIONS]
                 for q in order]
    write_csv(OUT_BY_COND, ["qid", "difficulty"] + CONDITIONS, cond_rows)

    # --- per_query_by_condition_mean5.csv (mean over runs 1..5) --------
    per_run = {c: [by_qid(run, c) for run in runs] for c in CONDITIONS}
    mean_rows = [[q, difficulty[q]]
                 + [fmt(mean(m[q]["recall"] for m in per_run[c])) for c in CONDITIONS]
                 for q in order]
    write_csv(OUT_BY_COND_MEAN, ["qid", "difficulty"] + CONDITIONS, mean_rows)

    # --- consistency checks against the saved run summaries -------------
    checks = {}
    for run in runs:
        for c in CONDITIONS:
            recomputed = mean(r["recall"] for r in run["conditions"][c]["results"])
            stored = run["conditions"][c]["overall"]
            if abs(recomputed - stored) > 1e-9:
                raise RuntimeError(f"run {run['run_id']}/{c}: mean recall {recomputed} "
                                   f"!= stored overall {stored}")
        checks[f"run_{run['run_id']}"] = {
            c: round(run["conditions"][c]["overall"] * 100, 1) for c in CONDITIONS}
    n_passed = sum(1 for row in results_rows if row[-1] == "1")
    n_diff_run1 = sum(
        1 for q in order
        if len({fmt(cond1[c][q]["recall"]) for c in CONDITIONS}) > 1)

    prov = {
        "_note": ("Derived deterministically from saved artifacts only (no LLM, no DB). "
                  "Replaces former supplementary Table S3 (per_query_results.csv) and "
                  "Table S4 (per_query_by_condition.csv). All 100 rows are the ablation "
                  "subset of the 245-query main evaluation."),
        "generator": "scripts/build_per_query_tables.py",
        "inputs": {p.name: sha256_file(p) for p in [CATALOG, *RUN_FILES]},
        "dataset_file": run1["provenance"].get("dataset_file"),
        "dataset_sha256": run1["provenance"].get("dataset_sha256"),
        "model": run1.get("model"),
        "pass_threshold": PASS_THRESHOLD,
        "n_queries": len(order),
        "difficulty_counts": counts,
        "run1_full_overall_pct": round(run1["conditions"]["full"]["overall"] * 100, 1),
        "run1_n_passed": n_passed,
        "run1_n_queries_differing_across_conditions": n_diff_run1,
        "overall_pct_by_run_and_condition": checks,
        "outputs": {p.name: sha256_file(p) for p in [OUT_RESULTS, OUT_BY_COND, OUT_BY_COND_MEAN]},
    }
    OUT_PROV.write_text(json.dumps(prov, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")

    print(f"{OUT_RESULTS.name}: {len(results_rows)} rows "
          f"(run1 full {prov['run1_full_overall_pct']}%, passed {n_passed}/{len(order)})")
    print(f"{OUT_BY_COND.name}: {len(cond_rows)} rows "
          f"({n_diff_run1} queries differ across conditions in run 1)")
    print(f"{OUT_BY_COND_MEAN.name}: {len(mean_rows)} rows")
    print(f"{OUT_PROV.name} written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
