"""Error analysis for baseline vs proposed comparison."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from evaluation.metrics import syntax_validity
from safety.sql_validator import extract_tables_from_sql


ALL_ALLOWED_TABLES = [
    "material_entry", "composition", "structure",
    "calculation", "calculated_property", "phase_stability",
    "prototype_definition",
]


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
        if not bp.exists():
            continue
        with bp.open() as f:
            data = json.load(f)
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

    if proposed_path.exists():
        with proposed_path.open() as f:
            data = json.load(f)
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


if __name__ == "__main__":
    eval_dir = Path(__file__).parent
    baseline_paths = [
        eval_dir / f"baseline{i}_result.json" for i in range(1, 5)
    ]
    generate_error_report(
        baseline_paths,
        eval_dir / "proposed_result.json",
    )
