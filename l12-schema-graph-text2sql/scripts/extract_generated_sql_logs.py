#!/usr/bin/env python3
"""Extract per-query generated SQL from existing evaluation JSON files.

The main ablation JSON (`ablation_run_*.json`, `ablation_results.json`) does not
contain generated SQL.  This script only extracts SQL from evaluation result
files that already store it (independent, transfer, prototype, CTE, MP transfer).
For the main ablation logs, see `scripts/capture_main_eval_sql.py` or re-run
`scripts/eval_ablation.py` with SQL logging enabled.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT / "evaluation"
OUT_DIR = EVAL_DIR / "generated_sql"

# Files known to contain per-query SQL
SOURCES = {
    "independent": "independent_eval_results.json",
    "transfer": "transfer_eval_results.json",
    "transfer_obfuscated": "transfer_obfuscated_eval_results.json",
    "prototype": "prototype_eval_results.json",
    "cte": "cte_eval_results.json",
    "mp_transfer": "mp_transfer_eval_results.json",
}


def safe_qid(qid: str) -> str:
    """Return a filesystem-safe qid string."""
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", qid)


def extract_from_file(path: Path, key: str):
    if not path.exists():
        print(f"[skip] {path.name} not found")
        return 0
    with open(path) as f:
        data = json.load(f)

    results = data.get("results", [])
    if not results:
        # Transfer results may be stored under a different key
        for maybe in data.values():
            if isinstance(maybe, list) and maybe and "qid" in maybe[0]:
                results = maybe
                break

    target_dir = OUT_DIR / key
    target_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    manifest: list[dict] = []
    for r in results:
        qid = r.get("qid")
        sql = r.get("sql")
        if not qid or not sql:
            continue
        filename = f"{safe_qid(qid)}.sql"
        sql_path = target_dir / filename
        sql_path.write_text(sql.rstrip() + "\n", encoding="utf-8")
        written += 1
        manifest.append({
            "qid": qid,
            "difficulty": r.get("difficulty"),
            "accuracy": r.get("accuracy"),
            "latency_s": r.get("latency_s"),
            "source_file": path.name,
            "sql_path": str(sql_path.relative_to(PROJECT)),
        })

    if manifest:
        manifest_path = target_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps({
                "source": key,
                "n_queries": len(manifest),
                "eval_file": path.name,
                "queries": manifest,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"[{key}] wrote {written} SQL files to {target_dir}")
    return written


def main():
    total = 0
    for key, filename in SOURCES.items():
        total += extract_from_file(EVAL_DIR / filename, key)
    print(f"Total extracted SQL logs: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
