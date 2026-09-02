"""Unified complexity score for gold SQL (paper section 2.1.2).

score = 3 * (distinct base tables referenced)
      + 1 * (atomic WHERE/HAVING predicates)
      + 2 * (GROUP BY clauses)
      + 3 * (EXISTS subqueries)
      + 3 * (derived subqueries: CTEs, derived FROM tables, scalar/IN subqueries)

Easy < 8, Medium 8--11, Hard 12--16, Very Hard >= 17.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import sqlglot
from sqlglot import expressions as exp

ROOT = Path(__file__).resolve().parent.parent


def count_predicates(node: exp.Expression | None) -> int:
    if node is None:
        return 0
    cond = node.this
    n = 1
    for _ in cond.find_all(exp.And, exp.Or):
        n += 1
    return n


def score_sql(sql: str) -> dict:
    tree = sqlglot.parse_one(sql, read="postgres")
    cte_names = {c.alias_or_name for c in tree.find_all(exp.CTE)}
    tables = {
        t.name
        for t in tree.find_all(exp.Table)
        if t.name not in cte_names
    }
    n_pred = 0
    n_group = 0
    n_exists = 0
    n_derived = 0
    for w in tree.find_all(exp.Where):
        n_pred += count_predicates(w)
    for h in tree.find_all(exp.Having):
        n_pred += count_predicates(h)
    n_group = len(list(tree.find_all(exp.Group)))
    n_exists = len(list(tree.find_all(exp.Exists)))
    n_derived = len(cte_names)
    for sq in tree.find_all(exp.Subquery):
        if not sq.find_ancestor(exp.Exists):
            n_derived += 1
    score = 3 * len(tables) + n_pred + 2 * n_group + 3 * n_exists + 3 * n_derived
    return {
        "n_tables": len(tables),
        "n_predicates": n_pred,
        "n_group_by": n_group,
        "n_exists": n_exists,
        "n_derived": n_derived,
        "score": score,
        "difficulty": classify(score),
    }


def classify(score: int) -> str:
    if score < 8:
        return "easy"
    if score <= 11:
        return "medium"
    if score <= 16:
        return "hard"
    return "very_hard"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="evaluation/independent_en_dataset.jsonl")
    ap.add_argument("--apply", action="store_true",
                    help="rewrite the dataset difficulty labels in place")
    args = ap.parse_args()
    path = ROOT / args.dataset
    rows = [json.loads(line) for line in path.open()]
    n_match = 0
    for row in rows:
        gp = ROOT / row["gold_sql_path"]
        if not gp.exists():
            gp = ROOT / "evaluation" / row["gold_sql_path"]
        sql = gp.read_text()
        res = score_sql(sql)
        cur = row["difficulty"].lower().replace(" ", "_")
        mark = "==" if cur == res["difficulty"] else "!="
        if cur == res["difficulty"]:
            n_match += 1
        print(
            f"{row['id']}: {cur:9s} {mark} {res['difficulty']:9s} "
            f"score={res['score']:3d} (T={res['n_tables']} P={res['n_predicates']} "
            f"G={res['n_group_by']} E={res['n_exists']} D={res['n_derived']})"
        )
    print(f"match {n_match}/{len(rows)}")
    if args.apply:
        for row in rows:
            gp = ROOT / row["gold_sql_path"]
            if not gp.exists():
                gp = ROOT / "evaluation" / row["gold_sql_path"]
            row["difficulty"] = score_sql(gp.read_text())["difficulty"]
        with path.open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Rewrote difficulty labels in {path}")
        refresh_run_provenance(path)


def refresh_run_provenance(dataset: Path) -> None:
    """Synchronize saved run files with the relabeled dataset: re-pin
    dataset_sha256, relabel results[].difficulty, and re-aggregate
    by_difficulty (same convention as rescore_stored_results.py).
    Per-query model outputs, recalls and latencies are left untouched."""
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    diff_map = {
        rec["id"]: rec["difficulty"]
        for rec in (json.loads(line) for line in dataset.read_text().splitlines())
    }
    for run in sorted(dataset.parent.glob("independent_en_run*.json")):
        data = json.loads(run.read_text())
        prov = data.get("provenance")
        if not isinstance(prov, dict) or prov.get("dataset_file") != dataset.name:
            continue
        changed = prov.get("dataset_sha256") != digest
        for res in data["results"]:
            new_label = diff_map[res["qid"]]
            if res["difficulty"] != new_label:
                res["difficulty"] = new_label
                changed = True
        per_level: dict[str, list[float]] = {}
        for res in data["results"]:
            per_level.setdefault(res["difficulty"], []).append(res["recall"])
        by_diff = {
            d: sum(v) / len(v)
            for d in ("easy", "medium", "hard", "very_hard")
            if (v := per_level.get(d))
        }
        if data.get("by_difficulty") != by_diff:
            data["by_difficulty"] = by_diff
            changed = True
        if not changed:
            continue
        prov["dataset_sha256"] = digest
        prov["dataset_relabel_note"] = (
            "difficulty labels updated post hoc from the unified "
            "structural-complexity score (compute_unified_difficulty.py) "
            "and by_difficulty re-aggregated; model outputs, per-query "
            "recalls and latencies unchanged"
        )
        run.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        print(f"Synchronized difficulty labels in {run.name}")


if __name__ == "__main__":
    main()
