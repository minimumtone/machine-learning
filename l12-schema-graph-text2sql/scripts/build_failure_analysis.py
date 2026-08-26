#!/usr/bin/env python3
"""Build evaluation/failure_analysis.json from the stored multiaxis run.

Selects every query whose stored recall is below 0.8 from
evaluation/multiaxis_results.json and pairs it with its gold SQL and the
per-query SELECT-column precision / JOIN match rate recorded in the same
run, so the error-distribution figure and the failure-mode table are
derived from a single stored run.
"""
from __future__ import annotations

import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
EVAL = PROJECT / "evaluation"


def main() -> None:
    data = json.load(open(EVAL / "multiaxis_results.json"))
    dataset = {}
    for line in open(EVAL / "evaluation_dataset.jsonl"):
        r = json.loads(line)
        dataset[r["id"]] = r

    failures = []
    for rec in data["results"]:
        if rec["recall"] >= 0.8:
            continue
        meta = dataset[rec["qid"]]
        gold_sql = (EVAL / meta["gold_sql_path"]).read_text()
        failures.append({
            "qid": rec["qid"],
            "difficulty": rec["difficulty"],
            "question": meta["question"],
            "recall": rec["recall"],
            "select_col_prec": rec["select_column_precision"],
            "join_match_rate": rec["join_match_rate"],
            "gen_sql": rec["gen_sql"],
            "gold_sql": gold_sql,
        })

    out = {"failures": failures, "n_total": len(data["results"])}
    (EVAL / "failure_analysis.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2))
    print(f"{len(failures)} failures (recall<0.8) out of {len(data['results'])}")
    for f in failures:
        print(f"  {f['qid']} {f['difficulty']} recall={f['recall']:.2f}")


if __name__ == "__main__":
    main()
