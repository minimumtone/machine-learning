#!/usr/bin/env python3
"""入力体積と一致する未緩和のVASP計算行を抽出する。"""

from __future__ import annotations

import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
DATA = ROOT / "data" / "sqs_results.csv"
OUTPUT = PAPER / "unrelaxed_volume_rows.csv"

sys.path.insert(0, str(ROOT / "vasp_inputs"))
from reanalyze_all import VASP_ATOMIC_VOLUMES  # noqa: E402


COMPOSITION_PATTERN = re.compile(
    r"^([A-Z][a-z]?)(\d+)([A-Z][a-z]?)(\d+)$"
)
OUTPUT_FIELDS = [
    "dir",
    "structure_root",
    "natoms",
    "relax_converged",
    "volume_A3",
    "v_per_atom",
    "expected_input_v_per_atom",
    "note",
]


def flagged_row(row: dict[str, str]) -> dict[str, str] | None:
    if row["status"] != "OK" or not row["volume_A3"].strip():
        return None
    match = COMPOSITION_PATTERN.fullmatch(row["dir"])
    if match is None:
        return None
    element_a, count_a, element_b, count_b = match.groups()
    if (
        element_a not in VASP_ATOMIC_VOLUMES
        or element_b not in VASP_ATOMIC_VOLUMES
    ):
        return None
    natoms = int(row["natoms"])
    volume = float(row["volume_A3"])
    expected = (
        int(count_a) * VASP_ATOMIC_VOLUMES[element_a]
        + int(count_b) * VASP_ATOMIC_VOLUMES[element_b]
    ) / natoms
    v_per_atom = volume / natoms
    if abs(v_per_atom - expected) >= 1e-6:
        return None
    return {
        "dir": row["dir"],
        "structure_root": row["structure_root"],
        "natoms": row["natoms"],
        "relax_converged": row["relax_converged"],
        "volume_A3": row["volume_A3"],
        "v_per_atom": f"{v_per_atom:.12g}",
        "expected_input_v_per_atom": f"{expected:.12g}",
        "note": "UNRELAXED_VOLUME",
    }


def equal_composition_pairs(
    rows: list[dict[str, str]], structure_root: str, natoms: int, count: int
) -> list[str]:
    pairs = set()
    for row in rows:
        if row["structure_root"] != structure_root:
            continue
        if int(row["natoms"]) != natoms:
            continue
        match = COMPOSITION_PATTERN.fullmatch(row["dir"])
        if match is None:
            continue
        element_a, count_a, element_b, count_b = match.groups()
        if int(count_a) == count and int(count_b) == count:
            pairs.add("-".join(sorted((element_a, element_b))))
    return sorted(pairs)


def main() -> None:
    with DATA.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    flagged = [item for row in rows if (item := flagged_row(row)) is not None]
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=OUTPUT_FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(flagged)

    counts = Counter((row["structure_root"], int(row["natoms"])) for row in flagged)
    print("フラグ件数（structure_root, natoms）:")
    for key in sorted(counts):
        print(f"  {key}: {counts[key]}")

    print("等量組成に該当するペア名一覧:")
    categories = [
        ("BCC_SQS", 16, 8, "BCC 8:8 / 16原子"),
        ("FCC_SQS", 32, 16, "FCC 16:16 / 32原子"),
        ("BCC_SQS", 128, 64, "BCC 64:64 / 128原子"),
    ]
    for structure_root, natoms, count, label in categories:
        pairs = equal_composition_pairs(flagged, structure_root, natoms, count)
        print(f"  {label}: {', '.join(pairs) if pairs else '(なし)'}")
    print(f"出力行数: {len(flagged)}")


if __name__ == "__main__":
    main()
