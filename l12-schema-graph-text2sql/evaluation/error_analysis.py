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
    """Generate a Markdown error analysis report."""
    if output_path is None:
        output_path = proposed_path.parent / "error_analysis_report.md"

    sections: list[str] = ["# Error Analysis Report\n"]

    for bp in baseline_paths:
        data = _load_results_file(bp)
        if not data:
            continue
        analysis = analyze_results(data, bp.stem)
        sections.append(f"## {analysis['method']}\n")
        sections.append(f"- Total queries: {analysis['total']}")
        sections.append(
            f"- Syntax validity: {analysis['syntax_validity_rate']:.1%}"
        )
        sections.append(
            f"- Hallucinated tables: {analysis['hallucinated_table_count']}"
        )
        if analysis["hallucinated_tables"]:
            sections.append(
                f"  - Examples: {', '.join(analysis['hallucinated_tables'][:5])}"
            )
        sections.append("")

    data = _load_results_file(proposed_path)
    if data:
        analysis = analyze_results(data, "proposed")
        sections.append(f"## Proposed Method\n")
        sections.append(f"- Total queries: {analysis['total']}")
        sections.append(
            f"- Syntax validity: {analysis['syntax_validity_rate']:.1%}"
        )
        sections.append(
            f"- Hallucinated tables: {analysis['hallucinated_table_count']}"
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
    # Try CSV first (output of run_full_evaluation.py), then JSON
    baseline_paths = []
    for i in range(1, 5):
        csv_path = eval_dir / f"baseline_result_baseline{i}_llm_only.csv"
        json_path = eval_dir / f"baseline{i}_result.json"
        if csv_path.exists():
            baseline_paths.append(csv_path)
        else:
            baseline_paths.append(json_path)

    proposed_csv = eval_dir / "proposed_result.csv"
    proposed_json = eval_dir / "proposed_result.json"
    proposed_path = proposed_csv if proposed_csv.exists() else proposed_json

    generate_error_report(baseline_paths, proposed_path)
