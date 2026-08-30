#!/usr/bin/env python3
"""Extract per-query generated SQL from existing evaluation JSON files.

Covers the suites whose result files store per-query SQL (independent,
transfer, obfuscated transfer, prototype, CTE, MP transfer).  The main
corpus and the LLM-only baseline are exported from the canonical stored
runs by `scripts/derive_main_artifacts.py` instead.

Every result in a source file MUST yield one .sql file; a result without
SQL is a hard error, and each manifest copies the source file's
provenance plus its SHA-256 so the exported logs are mechanically tied
to the run they came from.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT / "evaluation"
OUT_DIR = EVAL_DIR / "generated_sql"
sys.path.insert(0, str(PROJECT))

from scripts.provenance import build_provenance, sha256_file  # noqa: E402

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
    for stale in target_dir.glob("*.sql"):
        stale.unlink()
    written = 0
    manifest: list[dict] = []
    for r in results:
        qid = r.get("qid")
        sql = r.get("sql") or r.get("gen_sql")
        if not qid or sql is None:
            raise RuntimeError(
                f"[{key}] result without qid/sql in {path.name}: "
                f"{ {k: r.get(k) for k in ('qid', 'difficulty')} }")
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

    if written != len(results):
        raise RuntimeError(
            f"[{key}] exported {written} SQL files but {path.name} has "
            f"{len(results)} results")
    missing = [m["sql_path"] for m in manifest
               if not (PROJECT / m["sql_path"]).exists()]
    if missing:
        raise RuntimeError(f"[{key}] manifest sql_path missing: {missing[:5]}")

    provenance = data.get("provenance")
    if provenance is None and data.get("dataset"):
        # Older result files predate provenance embedding; annotate the
        # manifest post hoc from the dataset the run recorded, and say so.
        ds = EVAL_DIR / data["dataset"]
        if ds.exists():
            provenance = build_provenance(ds)
            provenance["annotated_post_hoc"] = (
                "the source result file has no embedded provenance; these "
                "hashes were computed from the current packaged inputs at "
                "manifest-generation time, not at inference time")

    manifest_path = target_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps({
            "source": key,
            "model": data.get("model"),
            "provenance": provenance,
            "source_result_file": path.name,
            "source_result_sha256": sha256_file(path),
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
