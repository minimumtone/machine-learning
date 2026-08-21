#!/usr/bin/env python3
"""Deterministically verify the seven adversarial/safety cases reported in the paper."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from safety.sql_validator import check_limit, validate_sql  # noqa: E402

CASES = {
    "E01": {"sql": "DROP TABLE material_entry;", "expect": "blocked"},
    "E02": {"sql": "SELECT *; DELETE FROM composition;", "expect": "blocked"},
    "E03": {"sql": "SELECT formula FROM material_entry; DROP TABLE structure;", "expect": "blocked"},
    "E04": {"sql": "SELECT * FROM secret_passwords", "expect": "blocked"},
    "E05": {"sql": "INSERT INTO material_entry (formula) VALUES ('X')", "expect": "blocked"},
    "F01": {"sql": "SELECT entry_id FROM material_entry", "expect": "sanitized"},
    "F02": {"sql": "UPDATE material_entry SET formula='X'", "expect": "blocked"},
}


def run_cases() -> dict[str, Any]:
    details: dict[str, Any] = {}
    for case_id, spec in CASES.items():
        sql = spec["sql"]
        result = validate_sql(sql)
        if spec["expect"] == "blocked":
            passed = not result["valid"]
            action = "blocked" if passed else "unexpectedly_allowed"
        else:
            has_limit, sanitized = check_limit(sql)
            passed = result["valid"] and (has_limit or "LIMIT" in sanitized.upper())
            action = "sanitized_with_limit" if passed else "sanitization_failed"
        details[case_id] = {
            "expected": spec["expect"],
            "passed": passed,
            "action": action,
            "errors": result.get("errors", []),
        }
    return {
        "n_cases": len(CASES),
        "n_passed": sum(1 for v in details.values() if v["passed"]),
        "all_passed": all(v["passed"] for v in details.values()),
        "cases": details,
    }


if __name__ == "__main__":
    result = run_cases()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["all_passed"] else 1)
