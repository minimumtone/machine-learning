#!/usr/bin/env python3
"""Summarize the language-dependence evaluations (paired ja/en + independent EN).

Reads evaluation/language_paired_{ja,en}_run{1..3}.json and
evaluation/independent_en_run{1..3}.json, and writes
evaluation/language_eval_summary.json with per-language mean +- SD overall
recall and per-difficulty means.

Difficulty labels are taken from the current dataset files (id -> difficulty),
not from the labels stored inside the saved run records, so that label
corrections (e.g. re-scoring the independent EN set with the unified
complexity score) propagate to the breakdown without re-running evaluations.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT / "evaluation"


def load_difficulty_map(dataset: Path) -> dict[str, str]:
    return {
        rec["id"]: rec["difficulty"]
        for rec in (json.loads(line) for line in dataset.read_text().splitlines())
    }


def summarize(paths: list[Path], diff_map: dict[str, str]) -> dict:
    runs = [json.loads(p.read_text()) for p in paths]
    overall = [r["overall_recall"] * 100 for r in runs]
    diffs: dict[str, list[float]] = {}
    for r in runs:
        per_level: dict[str, list[float]] = {}
        for res in r["results"]:
            per_level.setdefault(diff_map[res["qid"]], []).append(res["recall"])
        for d, vals in per_level.items():
            diffs.setdefault(d, []).append(statistics.mean(vals) * 100)
    counts: dict[str, int] = {}
    for res in runs[0]["results"]:
        d = diff_map[res["qid"]]
        counts[d] = counts.get(d, 0) + 1
    return {
        "n_runs": len(runs),
        "n_queries": len(runs[0]["results"]),
        "overall_recall_pct_mean": round(statistics.mean(overall), 1),
        "overall_recall_pct_sd": round(statistics.stdev(overall), 1) if len(overall) > 1 else None,
        "overall_recall_pct_runs": [round(v, 1) for v in overall],
        "difficulty_counts": dict(sorted(counts.items())),
        "by_difficulty_mean_pct": {d: round(statistics.mean(v), 1) for d, v in sorted(diffs.items())},
    }


def main() -> None:
    out = {}
    paired_map = load_difficulty_map(EVAL_DIR / "evaluation_dataset_en.jsonl")
    for lang in ("ja", "en"):
        paths = sorted(EVAL_DIR.glob(f"language_paired_{lang}_run*.json"))
        if paths:
            out[f"paired_{lang}"] = summarize(paths, paired_map)
    indep = sorted(EVAL_DIR.glob("independent_en_run*.json"))
    if indep:
        indep_map = load_difficulty_map(EVAL_DIR / "independent_en_dataset.jsonl")
        out["independent_en"] = summarize(indep, indep_map)
    dst = EVAL_DIR / "language_eval_summary.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"Saved: {dst}")


if __name__ == "__main__":
    main()
