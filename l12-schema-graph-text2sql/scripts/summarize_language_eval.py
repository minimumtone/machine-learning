#!/usr/bin/env python3
"""Summarize the language-dependence evaluations (paired ja/en + independent EN).

Reads evaluation/language_paired_{ja,en}_run{1..3}.json and
evaluation/independent_en_run{1..3}.json, and writes
evaluation/language_eval_summary.json with per-language mean +- SD overall
recall and per-difficulty means.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT / "evaluation"


def summarize(paths: list[Path]) -> dict:
    runs = [json.loads(p.read_text()) for p in paths]
    overall = [r["overall_recall"] * 100 for r in runs]
    diffs: dict[str, list[float]] = {}
    for r in runs:
        for d, v in r["by_difficulty"].items():
            diffs.setdefault(d, []).append(v * 100)
    return {
        "n_runs": len(runs),
        "n_queries": len(runs[0]["results"]),
        "overall_recall_pct_mean": round(statistics.mean(overall), 1),
        "overall_recall_pct_sd": round(statistics.stdev(overall), 1) if len(overall) > 1 else None,
        "overall_recall_pct_runs": [round(v, 1) for v in overall],
        "by_difficulty_mean_pct": {d: round(statistics.mean(v), 1) for d, v in sorted(diffs.items())},
    }


def main() -> None:
    out = {}
    for lang in ("ja", "en"):
        paths = sorted(EVAL_DIR.glob(f"language_paired_{lang}_run*.json"))
        if paths:
            out[f"paired_{lang}"] = summarize(paths)
    indep = sorted(EVAL_DIR.glob("independent_en_run*.json"))
    if indep:
        out["independent_en"] = summarize(indep)
    dst = EVAL_DIR / "language_eval_summary.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"Saved: {dst}")


if __name__ == "__main__":
    main()
