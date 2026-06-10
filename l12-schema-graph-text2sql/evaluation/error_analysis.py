"""Error analysis for baseline vs proposed comparison."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from evaluation.metrics import syntax_validity
from safety.sql_validator import extract_tables_from_sql


def _load_allowed_tables() -> list[str]:
    """Load the 30-table allowed list from allowed_schema.yaml."""
    schema_path = Path(__file__).parent.parent / "safety" / "allowed_schema.yaml"
    if schema_path.exists():
        with schema_path.open(encoding="utf-8") as f:
            schema = yaml.safe_load(f)
        return schema.get("allowed_tables", [])
    return []


ALL_ALLOWED_TABLES = _load_allowed_tables()


def analyze_results(
    results: list[dict[str, Any]],
    method_name: str,
) -> dict[str, Any]:
    """Analyze a set of results and return summary statistics."""
    total = len(results)
    if total == 0:
        return {"method": method_name, "total": 0}

    syntax_ok = sum(1 for r in results if syntax_validity(r.get("sql", "")))

    hallucinated_tables: list[str] = []
    for r in results:
        tables = extract_tables_from_sql(r.get("sql", ""))
        allowed = {t.lower() for t in ALL_ALLOWED_TABLES}
        for t in tables:
            if t.lower() not in allowed:
                hallucinated_tables.append(t)

    by_difficulty = Counter(r["difficulty"] for r in results)

    return {
        "method": method_name,
        "total": total,
        "syntax_validity_rate": syntax_ok / total,
        "hallucinated_table_count": len(hallucinated_tables),
        "hallucinated_tables": list(set(hallucinated_tables)),
        "by_difficulty": dict(by_difficulty),
    }


def generate_error_report(
    baseline_paths: list[Path],
    proposed_path: Path,
    output_path: Path | None = None,
) -> str:
    """Generate a Markdown error analysis report.

    Output format matches run_full_evaluation.py's write_error_analysis().
    """
    if output_path is None:
        output_path = proposed_path.parent / "error_analysis_report.md"

    sections: list[str] = [
        "# Error Analysis Report\n",
        "> Paper ref: Table (tab:error_analysis) -- error breakdown by method\n",
    ]

    all_paths = list(baseline_paths) + [proposed_path]
    for bp in all_paths:
        data = _load_results_file(bp)
        if not data:
            continue
        method = bp.stem
        # Normalize method name from CSV filename
        if method.startswith("baseline_result_"):
            method = method.replace("baseline_result_", "")
        total = len(data)
        sections.append(f"\n## {method}\n")
        sections.append(f"- Total queries: {total}")

        # Execution failures
        exec_fail = sum(
            1 for r in data
            if str(r.get("execution_valid", "True")).lower() in ("false", "0", "")
        )
        sections.append(
            f"- Execution failures: {exec_fail} ({exec_fail / total * 100:.1f}%)"
        )

        # Syntax errors
        syntax_err = sum(
            1 for r in data
            if str(r.get("syntax_valid", "True")).lower() in ("false", "0")
        )
        sections.append(f"- Syntax errors: {syntax_err}")

        # Hallucinated tables / joins
        ht = sum(
            1 for r in data if float(r.get("hallucinated_table_rate", 0)) > 0
        )
        hj = sum(
            1 for r in data if float(r.get("hallucinated_join_rate", 0)) > 0
        )
        sections.append(f"- Hallucinated tables: {ht}")
        sections.append(f"- Hallucinated joins: {hj}")

        # Lowest accuracy queries
        sorted_by_acc = sorted(data, key=lambda r: float(r.get("execution_accuracy", 0)))
        sections.append(f"\n### Lowest accuracy queries:")
        for r in sorted_by_acc[:5]:
            sections.append(
                f"- {r['query_id']} ({r['difficulty']}): "
                f"acc={float(r.get('execution_accuracy', 0)):.2f}"
            )
        sections.append("")

    report = "\n".join(sections)
    output_path.write_text(report, encoding="utf-8")
    print(f"Error analysis report saved to {output_path}")
    return report


def _load_results_file(path: Path) -> list[dict[str, Any]]:
    """Load results from JSON or CSV file."""
    if not path.exists():
        return []
    if path.suffix == ".csv":
        with path.open(encoding="utf-8") as f:
            return list(csv.DictReader(f))
    else:
        with path.open(encoding="utf-8") as f:
            return json.load(f)


if __name__ == "__main__":
    eval_dir = Path(__file__).parent
    # CSV names match run_full_evaluation.py output
    _baseline_suffixes = {
        1: "llm_only",
        2: "full_schema",
        3: "rule_based",
        4: "fk_list",
    }
    baseline_paths = []
    for i in range(1, 5):
        csv_path = eval_dir / f"baseline_result_baseline{i}_{_baseline_suffixes[i]}.csv"
        json_path = eval_dir / f"baseline{i}_result.json"
        if csv_path.exists():
            baseline_paths.append(csv_path)
        else:
            baseline_paths.append(json_path)

    proposed_csv = eval_dir / "proposed_result.csv"
    proposed_json = eval_dir / "proposed_result.json"
    proposed_path = proposed_csv if proposed_csv.exists() else proposed_json

    generate_error_report(baseline_paths, proposed_path)
