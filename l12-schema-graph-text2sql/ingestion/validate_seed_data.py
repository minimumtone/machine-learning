"""Validate seed CSV files for consistency before loading."""
from __future__ import annotations

import csv
from pathlib import Path


def validate_seed_dir(seed_dir: Path) -> list[str]:
    errors: list[str] = []

    entries_path = seed_dir / "seed_l12_entries.csv"
    if not entries_path.exists():
        errors.append("seed_l12_entries.csv not found")
        return errors

    with entries_path.open() as f:
        entries = list(csv.DictReader(f))
    entry_ids = {r["entry_id"] for r in entries}

    if len(entries) < 100:
        errors.append(f"Need at least 100 entries, got {len(entries)}")

    known = {"Ni3Al", "Ni3Ga", "Ni3Ge", "Co3Ti", "Al3Sc",
             "Al3Ti", "Pt3Al", "Ir3Nb", "Co3Al", "Co3W", "Co3Ta"}
    formulas = {r["formula"] for r in entries}
    missing = known - formulas
    if missing:
        errors.append(f"Missing known L12 compounds: {missing}")

    ref_files = {
        "seed_composition.csv": "entry_id",
        "seed_structure.csv": "entry_id",
        "seed_phase_stability.csv": "entry_id",
        "seed_calculation.csv": "entry_id",
    }
    for fname, col in ref_files.items():
        fpath = seed_dir / fname
        if not fpath.exists():
            errors.append(f"{fname} not found")
            continue
        with fpath.open() as f:
            rows = list(csv.DictReader(f))
        bad = {r[col] for r in rows} - entry_ids
        if bad:
            errors.append(f"{fname} references unknown entry_ids: {bad}")

    return errors


if __name__ == "__main__":
    seed_dir = Path(__file__).resolve().parent.parent / "db" / "seed"
    errs = validate_seed_dir(seed_dir)
    if errs:
        print("Validation FAILED:")
        for e in errs:
            print(f"  - {e}")
        raise SystemExit(1)
    print("Seed data validation PASSED")
