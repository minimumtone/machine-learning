#!/usr/bin/env python3
"""Re-score every stored evaluation result against the current gold/expected.

No LLM is called.  For each evaluation JSON that stores per-query
generated SQL, the stored SQL is re-executed against the corresponding
database (READ ONLY + REPEATABLE READ + suite guard + SAVEPOINT, see
scripts/eval_db.py) and the execution metrics (recall / precision / f1 /
accuracy / exact_result_set_match) are recomputed against the current
expected_results.  Aggregates and per-difficulty summaries are then
recomputed from the per-query values, and the provenance block is
refreshed so its dataset/gold/prompt/expected hashes describe the exact
inputs the scores were computed from.

Run this after any gold-SQL / expected-result correction (e.g. the R22A
semantic fix), then:

    python scripts/derive_main_artifacts.py
    python scripts/extract_generated_sql_logs.py
    python scripts/build_failure_analysis.py
    python scripts/compute_ablation_stats_v2.py
    python scripts/recompute_significance.py --apply
    python scripts/audit_scoring.py

Usage:
    python scripts/rescore_stored_results.py
"""
from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from evaluation.metrics import execution_accuracy_full  # noqa: E402
from evaluation.metrics_strict import exact_result_set_match  # noqa: E402
from scripts.build_obfuscated_transfer_db import obfuscated_conninfo  # noqa: E402
from scripts.build_transfer_db import transfer_conninfo  # noqa: E402
from scripts.db_conninfo import main_conninfo, mp_conninfo  # noqa: E402
from scripts.eval_db import open_eval_connection, run_model_sql  # noqa: E402
from scripts.provenance import (  # noqa: E402
    _git_commit,
    _sha256_file,
    _sha256_gold_dir,
    _sha256_json_dir,
)

EVAL = PROJECT / "evaluation"


def _find_prompt_path(name: str) -> Path | None:
    for p in (PROJECT / "llm" / "prompt_templates" / name,
              PROJECT / name, PROJECT / "llm" / name):
        if p.is_file():
            return p
    return None

RESCORE_NOTE = (
    "re-scored deterministically from the stored generated SQL against "
    "the current gold/expected corpus by scripts/rescore_stored_results.py; "
    "no LLM call is involved (the generated SQL itself is unchanged)")

# suite -> (conninfo factory, expected dir)
SUITES = {
    "main": (main_conninfo, "expected_results", "main"),
    "transfer": (transfer_conninfo, "expected_results", "transfer"),
    "obfuscated": (obfuscated_conninfo, "expected_results_obfuscated",
                   "transfer"),
    "mp": (mp_conninfo, "expected_results_mp_transfer", "mp_transfer"),
}


class Rescorer:
    def __init__(self) -> None:
        self._conns: dict[str, Any] = {}
        self.n_queries = 0
        self.n_changed = 0
        self.changed_qids: set[str] = set()

    def conn(self, suite: str):
        if suite not in self._conns:
            factory, _, guard = SUITES[suite]
            self._conns[suite] = open_eval_connection(
                factory(), suite=guard, statement_timeout="30s")
        return self._conns[suite]

    def close(self) -> None:
        for c in self._conns.values():
            c.close()

    def score(self, suite: str, qid: str, sql: str) -> dict[str, Any]:
        """Execute stored SQL and score it against the current expected."""
        expected_dir = EVAL / SUITES[suite][1]
        exp_path = expected_dir / f"{qid}.json"
        exp = json.loads(exp_path.read_text(encoding="utf-8"))
        if sql and sql.strip():
            got = run_model_sql(self.conn(suite), sql)
        else:
            got = {"success": False, "rows": [], "columns": []}
        m = execution_accuracy_full(
            got.get("rows", []), exp.get("rows", []),
            got.get("columns", []), exp.get("columns", []))
        exact = exact_result_set_match(
            got.get("rows", []), exp.get("rows", []),
            got.get("columns", []), exp.get("columns", []),
            ordered=bool(exp.get("semantic_ordered")))
        self.n_queries += 1
        return {"recall": m["recall"], "precision": m["precision"],
                "f1": m["f1"], "exact": exact,
                "success": bool(got.get("success"))}

    def update_row(self, suite: str, row: dict[str, Any],
                   sql_key: str = "sql",
                   exact_as_bool: bool = False,
                   prefix: str = "") -> None:
        """Update the execution-metric fields present in one result row."""
        qid = row["qid"]
        s = self.score(suite, qid, row.get(sql_key) or "")
        exact_val: Any = s["exact"] if exact_as_bool else (
            1.0 if s["exact"] else 0.0)
        updates = {
            prefix + "accuracy": s["recall"],
            prefix + "recall": s["recall"],
            prefix + "precision": s["precision"],
            prefix + "f1": s["f1"],
            prefix + "exact_match": exact_val,
            prefix + "execution_valid": s["success"],
        }
        changed = False
        for k, v in updates.items():
            if k in row and row[k] != v:
                row[k] = v
                changed = True
        if changed:
            self.n_changed += 1
            self.changed_qids.add(f"{suite}:{qid}")


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals)


def refresh_provenance(data: dict[str, Any], expected_dir: str) -> None:
    prov = data.get("provenance")
    if not isinstance(prov, dict):
        return
    if isinstance(prov.get("dataset_file"), str):
        ds = EVAL / prov["dataset_file"]
        prov["dataset_sha256"] = _sha256_file(ds)
    if isinstance(prov.get("gold_dir"), str):
        prov["gold_sha256"] = _sha256_gold_dir(EVAL / prov["gold_dir"])
    for file_key, hash_key in (("prompt_template_file",
                                "prompt_template_sha256"),
                               ("prompt_file", "prompt_sha256")):
        name = prov.get(file_key)
        if isinstance(name, str):
            p = _find_prompt_path(name)
            if p is not None:
                prov[hash_key] = _sha256_file(p)
    # Pin the expected corpus the scores were computed against.
    prov["expected_dir"] = expected_dir
    prov["expected_sha256"] = _sha256_json_dir(EVAL / expected_dir)
    prov["git_commit"] = _git_commit()
    prov["rescored_at"] = datetime.datetime.now(
        datetime.timezone.utc).isoformat(timespec="seconds")
    prov["rescore_note"] = RESCORE_NOTE


def _by_difficulty(rows: list[dict[str, Any]], field: str) -> dict[str, float]:
    groups: dict[str, list[float]] = {}
    for r in rows:
        groups.setdefault(r["difficulty"], []).append(float(r[field]))
    return {d: _mean(v) for d, v in groups.items()}


def rescore_multiaxis(rs: Rescorer) -> None:
    path = EVAL / "multiaxis_results.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["results"]
    for r in rows:
        rs.update_row("main", r, sql_key="gen_sql", exact_as_bool=True)
    n = len(rows)
    agg = data["aggregate"]
    agg["recall_mean"] = _mean([r["recall"] for r in rows])
    agg["precision_mean"] = _mean([r["precision"] for r in rows])
    agg["f1_mean"] = _mean([r["f1"] for r in rows])
    agg["exact_match_rate"] = sum(1 for r in rows if r["exact_match"]) / n
    if "execution_validity_rate" in agg:
        agg["execution_validity_rate"] = (
            sum(1 for r in rows if r.get("execution_valid")) / n)
    for diff, block in data.get("by_difficulty", {}).items():
        dr = [r for r in rows if r["difficulty"] == diff]
        block["recall"] = _mean([r["recall"] for r in dr])
        block["precision"] = _mean([r["precision"] for r in dr])
        block["f1"] = _mean([r["f1"] for r in dr])
        block["exact_match"] = sum(1 for r in dr if r["exact_match"]) / len(dr)
    refresh_provenance(data, "expected_results")
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(f"[multiaxis] recall_mean={agg['recall_mean']:.4f} "
          f"exact={agg['exact_match_rate']:.4f}")


def rescore_llm_only(rs: Rescorer) -> None:
    path = EVAL / "llm_only_results.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["results"]
    for r in rows:
        rs.update_row("main", r, sql_key="gen_sql")
    n = len(rows)
    agg = data["aggregate"]
    agg["recall_mean"] = _mean([r["recall"] for r in rows])
    agg["precision_mean"] = _mean([r["precision"] for r in rows])
    agg["f1_mean"] = _mean([r["f1"] for r in rows])
    if "execution_validity_rate" in agg:
        agg["execution_validity_rate"] = (
            sum(1 for r in rows if r.get("execution_valid")) / n)
    for diff, block in data.get("by_difficulty", {}).items():
        dr = [r for r in rows if r["difficulty"] == diff]
        block["recall"] = _mean([r["recall"] for r in dr])
        block["precision"] = _mean([r["precision"] for r in dr])
        block["f1"] = _mean([r["f1"] for r in dr])
        if "execution_validity" in block:
            block["execution_validity"] = (
                sum(1 for r in dr if r.get("execution_valid")) / len(dr))
    refresh_provenance(data, "expected_results")
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(f"[llm_only] recall_mean={agg['recall_mean']:.4f}")


def _refresh_summary(data: dict[str, Any]) -> None:
    rows = data["results"]
    s = data["summary"]
    s["overall"] = _mean([r["accuracy"] for r in rows])
    s["overall_recall"] = _mean([r["recall"] for r in rows])
    s["overall_precision"] = _mean([r["precision"] for r in rows])
    s["overall_f1"] = _mean([r["f1"] for r in rows])
    s["overall_exact_match"] = _mean([r["exact_match"] for r in rows])
    s["by_difficulty"] = _by_difficulty(rows, "accuracy")


def rescore_summary_file(rs: Rescorer, filename: str, suite: str,
                         expected_dir: str) -> None:
    path = EVAL / filename
    data = json.loads(path.read_text(encoding="utf-8"))
    for r in data["results"]:
        rs.update_row(suite, r)
    _refresh_summary(data)
    refresh_provenance(data, expected_dir)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(f"[{filename}] overall={data['summary']['overall']:.4f}")


def _refresh_condition(rs: Rescorer, cond: dict[str, Any]) -> None:
    rows = cond["results"]
    for r in rows:
        rs.update_row("main", r)
    cond["overall"] = _mean([r["accuracy"] for r in rows])
    cond["by_difficulty"] = _by_difficulty(rows, "accuracy")


def rescore_condition_file(rs: Rescorer, filename: str,
                           groups_key: str = "conditions") -> None:
    path = EVAL / filename
    data = json.loads(path.read_text(encoding="utf-8"))
    for name, cond in data[groups_key].items():
        if isinstance(cond, dict) and isinstance(cond.get("results"), list):
            _refresh_condition(rs, cond)
    refresh_provenance(data, "expected_results")
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    overall = {n: round(c.get("overall", 0.0), 4)
               for n, c in data[groups_key].items()
               if isinstance(c, dict) and "overall" in c}
    print(f"[{filename}] {overall}")


def rescore_reranker_ab(rs: Rescorer) -> None:
    path = EVAL / "reranker_eval_results.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["results"]
    for r in rows:
        s = rs.score("main", r["qid"], r.get("sql") or "")
        b = rs.score("main", r["qid"], r.get("sql_baseline") or "")
        if (r["accuracy_reranker"] != s["recall"]
                or r["accuracy_baseline"] != b["recall"]):
            rs.n_changed += 1
            rs.changed_qids.add(f"main:{r['qid']}")
        r["accuracy_reranker"] = s["recall"]
        r["accuracy_baseline"] = b["recall"]
    data["overall_reranker"] = _mean([r["accuracy_reranker"] for r in rows])
    data["overall_baseline"] = _mean([r["accuracy_baseline"] for r in rows])
    data["delta"] = data["overall_reranker"] - data["overall_baseline"]
    refresh_provenance(data, "expected_results")
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(f"[reranker_ab] reranker={data['overall_reranker']:.4f} "
          f"baseline={data['overall_baseline']:.4f}")


def main() -> int:
    rs = Rescorer()
    try:
        rescore_multiaxis(rs)
        rescore_llm_only(rs)
        rescore_summary_file(rs, "cte_eval_results.json", "main",
                             "expected_results")
        rescore_summary_file(rs, "prototype_eval_results.json", "main",
                             "expected_results")
        rescore_summary_file(rs, "independent_eval_results.json", "main",
                             "expected_results")
        rescore_summary_file(rs, "transfer_eval_results.json", "transfer",
                             "expected_results")
        rescore_summary_file(rs, "transfer_obfuscated_eval_results.json",
                             "obfuscated", "expected_results_obfuscated")
        rescore_summary_file(rs, "mp_transfer_eval_results.json", "mp",
                             "expected_results_mp_transfer")
        for i in (1, 2, 3, 4, 5):
            rescore_condition_file(rs, f"ablation_run_{i}.json")
        rescore_condition_file(rs, "ablation_results.json")
        rescore_condition_file(rs, "fewshot_sensitivity_results.json")
        rescore_condition_file(rs, "dict_sensitivity_results.json")
        rescore_condition_file(rs, "model_comparison_results.json",
                               groups_key="models")
        rescore_reranker_ab(rs)
    finally:
        rs.close()
    print(f"\nre-scored {rs.n_queries} stored queries; "
          f"{rs.n_changed} row(s) changed "
          f"({len(rs.changed_qids)} distinct suite:qid)")
    for q in sorted(rs.changed_qids):
        print(f"  changed: {q}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
