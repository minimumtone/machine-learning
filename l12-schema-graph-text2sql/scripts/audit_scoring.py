#!/usr/bin/env python3
"""Audit how much of the reported score comes from metric leniency.

Naming note: the shipped headline number is the mean of the
"historical" policy below.  It is a row-set RECALL over the common
columns (superset answers still score 1.0), NOT a strict execution
accuracy, and must be reported as "historical execution recall"
alongside the strict score and the exact-match rate.

Re-executes the packaged generated-SQL logs against the verification database
and re-scores them under several policies.  No LLM and no API key are needed:
everything required is already in the reproduction package.

The script first reproduces the shipped numbers exactly (self-check), then
reports what the same runs score once each leniency is removed:

  historical        historical execution recall — the metric used for
                    the reported numbers
  exact_result_set_match   the canonical exact metric: exact gold column
                    list + row multiset match + row order when the gold
                    query is ordered
  common_column_exact_overlap  the LEGACY lenient "exact": recall=1 and
                    precision=1 over the common columns only (a result
                    missing gold columns can still count)
  require_all_cols  every gold column must be present in the result
  no_positional     no positional fallback when column names do not overlap
  drop_empty_gold   queries whose gold result is empty are excluded
  multiset          row multiplicity is respected
  ordered           row order is respected (gold queries with ORDER BY only)
  strict            all of the above except ordering

Usage:
    python scripts/audit_scoring.py
    python scripts/audit_scoring.py --datasets main independent
    python scripts/audit_scoring.py --out evaluation/scoring_audit.json

Environment: POSTGRES_HOST / POSTGRES_PORT / POSTGRES_USER / POSTGRES_PASSWORD
(same variables as the rest of the package).  Datasets whose database has not
been built (e.g. oqmd_transfer) are skipped with a note.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import psycopg  # noqa: E402

from evaluation.metrics import execution_accuracy_full  # noqa: E402
from evaluation.metrics_strict import (  # noqa: E402
    ScoringPolicy,
    exact_result_set_match,
    score,
)
from scripts.eval_db import open_eval_connection, run_model_sql  # noqa: E402
from scripts.provenance import build_provenance, sha256_file  # noqa: E402

EVAL = PROJECT / "evaluation"

# name -> (dataset file, generated-SQL subdir, database, shipped result file)
# "main" is audited against the derivation of the canonical inference run
# (multiaxis_results.json -> main_eval_with_sql.json; see
# scripts/derive_main_artifacts.py).
DATASETS: dict[str, tuple[str, str, str, str | None]] = {
    "main": ("main_evaluation_dataset.jsonl", "main", "l12_materials",
             "main_eval_with_sql.json"),
    "independent": ("expert_evaluation_dataset.jsonl", "independent", "l12_materials",
                    "independent_eval_results.json"),
    "transfer": ("transfer_evaluation_dataset.jsonl", "transfer", "oqmd_transfer",
                 "transfer_eval_results.json"),
    "transfer_obfuscated": ("transfer_obfuscated_evaluation_dataset.jsonl",
                            "transfer_obfuscated", "oqmd_transfer_obfuscated",
                            "transfer_obfuscated_eval_results.json"),
}

POLICIES: dict[str, ScoringPolicy] = {
    "historical": ScoringPolicy.historical(),
    "require_all_cols": ScoringPolicy(column_policy="require_all"),
    "no_positional": ScoringPolicy(allow_positional=False),
    "drop_empty_gold": ScoringPolicy(empty_gold="exclude"),
    "multiset": ScoringPolicy(multiset=True),
    "strict": ScoringPolicy.strict(),
}

SUITE_GUARD = {
    "main": "main",
    "independent": "main",
    "transfer": "transfer",
    "transfer_obfuscated": "transfer",
}

ORDER_BY = re.compile(r"\border\s+by\b", re.IGNORECASE)


def conninfo(db: str) -> str:
    return (
        f"host={os.getenv('POSTGRES_HOST', 'localhost')} "
        f"port={os.getenv('POSTGRES_PORT', '5432')} "
        f"dbname={db} "
        f"user={os.getenv('POSTGRES_USER', 'l12_user')} "
        f"password={os.getenv('POSTGRES_PASSWORD', 'l12_password')}"
    )


def is_placeholder(sql: str) -> bool:
    """True for the audit placeholders that record a failed generation."""
    body = [ln for ln in sql.splitlines() if ln.strip() and not ln.strip().startswith("--")]
    return not body


def shipped_per_query(name: str) -> dict[str, float]:
    _, _, _, resf = DATASETS[name]
    if not resf or not (EVAL / resf).exists():
        return {}
    data = json.load(open(EVAL / resf))
    for value in data.values():
        if isinstance(value, list) and value and isinstance(value[0], dict) and "qid" in value[0]:
            return {
                r["qid"]: float(
                    r.get("execution_recall", r.get("accuracy", 0.0)))
                for r in value}
    return {}


def audit_dataset(name: str) -> dict[str, Any] | None:
    dsf, subdir, db, _ = DATASETS[name]
    rows = [json.loads(line) for line in open(EVAL / dsf) if line.strip()]
    sql_dir = EVAL / "generated_sql" / subdir
    if not sql_dir.exists():
        print(f"[{name}] skipped: {sql_dir} not found", file=sys.stderr)
        return None
    try:
        conn = open_eval_connection(conninfo(db), suite=SUITE_GUARD[name],
                                    statement_timeout="30s")
    except psycopg.OperationalError as exc:
        print(f"[{name}] skipped: cannot connect to database '{db}' "
              f"({str(exc).splitlines()[0]})", file=sys.stderr)
        if db.startswith("oqmd_"):
            print(f"[{name}] hint: run `python scripts/build_transfer_db.py` first",
                  file=sys.stderr)
        return None

    shipped = shipped_per_query(name)
    totals = {p: [0.0, 0] for p in POLICIES}          # policy -> [sum, n_counted]
    common_col_exact_sum = 0.0
    canonical_exact_sum = 0.0
    ordered_sum, ordered_n = 0.0, 0
    excluded: list[str] = []
    missing_cols: list[str] = []
    no_overlap: list[str] = []
    empty_gold_credited: list[str] = []
    single_col_credit: list[dict[str, Any]] = []
    minority_col_credit: list[dict[str, Any]] = []
    selfcheck_mismatch: list[tuple[str, float, float]] = []
    per_query: dict[str, dict[str, float]] = {}

    for r in rows:
        qid = r["id"]
        gold_sql = (EVAL / r["gold_sql_path"]).read_text()
        exp = json.load(open(EVAL / r["expected_result_path"]))
        exp_rows = exp["rows"] if isinstance(exp, dict) and "rows" in exp else exp
        exp_cols = exp.get("columns") if isinstance(exp, dict) else None

        sql_file = sql_dir / f"{qid}.sql"
        if not sql_file.exists():
            raise FileNotFoundError(
                f"[{name}] generated SQL log missing: {sql_file} — the "
                "package must ship one .sql file per query; re-run "
                "scripts/derive_main_artifacts.py / "
                "scripts/extract_generated_sql_logs.py")
        sql = sql_file.read_text()
        if sql.strip() and not is_placeholder(sql):
            res = run_model_sql(conn, sql)
            got, cols = res["rows"], res["columns"]
        else:
            got, cols = [], []

        # Self-check against the shipped per-query accuracy.
        hist = execution_accuracy_full(got, exp_rows, cols, exp_cols)
        if qid in shipped and abs(hist["recall"] - shipped[qid]) > 1e-6:
            selfcheck_mismatch.append((qid, hist["recall"], shipped[qid]))

        common_col_exact_sum += 1.0 if (
            hist["recall"] == 1.0 and hist["precision"] == 1.0) else 0.0
        canonical_exact_sum += 1.0 if exact_result_set_match(
            got, exp_rows, cols, exp_cols,
            ordered=bool(exp.get("ordered")) if isinstance(exp, dict) else False,
        ) else 0.0
        q_scores: dict[str, float] = {}

        for pname, policy in POLICIES.items():
            s = score(got, exp_rows, cols, exp_cols, policy=policy)
            if s["status"] == "excluded_empty_gold":
                if pname == "drop_empty_gold":
                    excluded.append(qid)
                continue
            totals[pname][0] += s["recall"]
            totals[pname][1] += 1
            q_scores[pname] = s["recall"]
            if pname == "require_all_cols" and s["status"] == "missing_gold_columns":
                missing_cols.append(qid)
            if pname == "no_positional" and s["status"] == "no_column_overlap":
                no_overlap.append(qid)

        if not exp_rows and not got:
            empty_gold_credited.append(qid)

        # Queries that scored a perfect recall while only a small part of the
        # gold projection was actually compared.  These are the concrete
        # false-positive candidates a reviewer would want to see.
        if hist["recall"] == 1.0 and exp_cols and cols:
            rc = {c.lower() for c in cols}
            ecl = [c.lower() for c in exp_cols]
            common = [c for c in ecl if c in rc]
            record = {"qid": qid, "n_common": len(common), "n_gold_cols": len(ecl),
                      "common": common}
            if len(common) == 1 and len(ecl) > 1:
                single_col_credit.append(record)
            elif len(common) > 1 and common and len(common) * 2 < len(ecl):
                minority_col_credit.append(record)

        if ORDER_BY.search(gold_sql):
            s = score(got, exp_rows, cols, exp_cols,
                      policy=ScoringPolicy(ordered=True, multiset=True))
            if s["status"] != "excluded_empty_gold":
                ordered_sum += s["recall"]
                ordered_n += 1
                q_scores["ordered"] = s["recall"]

        per_query[qid] = q_scores

    conn.close()
    n = len(rows)
    out: dict[str, Any] = {
        "metric_naming": (
            "the 'historical' policy is a historical execution recall "
            "(row-set recall over common columns), not a strict "
            "execution accuracy; the canonical exact metric is "
            "'exact_result_set_match_pct' (exact gold column list + row "
            "multiset + order when the gold query is ordered); "
            "'common_column_exact_overlap_pct' is the LEGACY lenient "
            "exact over common columns only and is kept for comparison"),
        "n_queries": n,
        "policies": {
            p: {
                "mean_recall_pct": round(totals[p][0] / totals[p][1] * 100, 1) if totals[p][1] else None,
                "n_scored": totals[p][1],
            }
            for p in POLICIES
        },
        "exact_result_set_match_pct": round(canonical_exact_sum / n * 100, 1),
        "common_column_exact_overlap_pct": round(common_col_exact_sum / n * 100, 1),
        "ordered": {
            "mean_recall_pct": round(ordered_sum / ordered_n * 100, 1) if ordered_n else None,
            "n_gold_with_order_by": ordered_n,
        },
        "diagnostics": {
            "empty_gold_credited": empty_gold_credited,
            "n_excluded_empty_gold": len(excluded),
            "queries_missing_gold_columns": missing_cols,
            "queries_no_column_overlap": no_overlap,
            "credited_on_single_column": single_col_credit,
            "credited_on_minority_of_columns": minority_col_credit,
        },
        "selfcheck": {
            "compared_against": DATASETS[name][3],
            "n_mismatch": len(selfcheck_mismatch),
            "mismatches": selfcheck_mismatch[:10],
        },
        "per_query": per_query,
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS),
                    choices=list(DATASETS))
    ap.add_argument("--out", default=str(EVAL / "scoring_audit.json"))
    args = ap.parse_args()

    report: dict[str, Any] = {}
    for name in args.datasets:
        res = audit_dataset(name)
        if res is not None:
            report[name] = res

    if not report:
        print("No dataset could be audited.", file=sys.stderr)
        return 1

    cols = ["historical", "require_all_cols", "no_positional", "drop_empty_gold",
            "multiset", "strict"]
    header = f"{'dataset':22s}{'n':>4s}" + "".join(f"{c[:14]:>16s}" for c in cols) \
             + f"{'exact':>8s}{'cc-ex':>8s}{'ordered':>9s}"
    print(header)
    print("-" * len(header))
    for name, res in report.items():
        line = f"{name:22s}{res['n_queries']:4d}"
        for c in cols:
            v = res["policies"][c]["mean_recall_pct"]
            line += f"{(f'{v:.1f}%' if v is not None else '--'):>16s}"
        line += f"{res['exact_result_set_match_pct']:7.1f}%"
        line += f"{res['common_column_exact_overlap_pct']:7.1f}%"
        ov = res["ordered"]["mean_recall_pct"]
        line += f"{(f'{ov:.1f}%' if ov is not None else '--'):>9s}"
        print(line)

    print()
    bad = 0
    for name, res in report.items():
        sc = res["selfcheck"]
        d = res["diagnostics"]
        status = "OK" if sc["n_mismatch"] == 0 else f"{sc['n_mismatch']} MISMATCH"
        print(f"[{name}] self-check vs shipped per-query accuracy: {status}")
        bad += sc["n_mismatch"]
        if d["empty_gold_credited"]:
            print(f"  gold returns 0 rows and the generated SQL also returns 0 rows "
                  f"-> scored 1.0 for free: {len(d['empty_gold_credited'])} queries "
                  f"({', '.join(d['empty_gold_credited'][:6])}"
                  f"{' ...' if len(d['empty_gold_credited']) > 6 else ''})")
        if d["queries_missing_gold_columns"]:
            print(f"  scored on a subset of the gold columns: "
                  f"{len(d['queries_missing_gold_columns'])} queries")
        if d["queries_no_column_overlap"]:
            print(f"  no column-name overlap, scored positionally: "
                  f"{len(d['queries_no_column_overlap'])} queries")
        n_single = len(d["credited_on_single_column"])
        n_minor = len(d["credited_on_minority_of_columns"])
        if n_single or n_minor:
            print(f"  REVIEW: recall=1.0 while compared on 1 gold column: {n_single}; "
                  f"on a minority of gold columns: {n_minor}")
            for rec in (d["credited_on_single_column"] + d["credited_on_minority_of_columns"])[:8]:
                print(f"          {rec['qid']}: {rec['n_common']}/{rec['n_gold_cols']} cols "
                      f"({', '.join(rec['common'])})")

    report["provenance"] = build_provenance(EVAL / "main_evaluation_dataset.jsonl")
    report["sources"] = {
        name: {
            "source_result_file": DATASETS[name][3],
            "source_result_sha256": sha256_file(EVAL / DATASETS[name][3]),
        }
        for name in report
        if name in DATASETS and DATASETS[name][3]
        and (EVAL / DATASETS[name][3]).exists()
    }
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nWrote {args.out}")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
