#!/usr/bin/env python3
"""Verify that numeric values in the manuscript are present in paper_data.json.

This is an audit helper, not a strict proof: it compares the set of scalar
numbers contained in paper/paper_data.json with the numbers that appear in the
LaTeX source(s).  Numbers that appear in the manuscript but not in the JSON
are reported for manual review (page numbers, constants, version strings, etc.).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parent.parent

NUM_RE = re.compile(r"-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?")
LATEX_NUM_RE = re.compile(
    r"(?<![\d.,-])"
    r"(-?\d{1,3}(?:,\d{3})*(?:\.\d+)?)"
    r"(?![\d.,])"
    r"(?:\\?%)?",
    re.UNICODE,
)


def _normalize(value: Any) -> float | int:
    """Convert a scalar value to a comparable normalized number."""
    if isinstance(value, bool):
        raise ValueError("booleans are not numeric")
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        value = value.replace(",", "").replace("\\%", "").replace("%", "").strip()
        if not value:
            raise ValueError("empty string")
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError as exc:
            raise ValueError(f"not numeric: {value}") from exc
    raise ValueError(f"unsupported type {type(value)}")


def collect_numbers(obj: Any, path: str = "", seen: dict[float | int, list[str]] | None = None) -> dict[float | int, list[str]]:
    """Recursively collect all numeric leaf values from paper_data.json."""
    if seen is None:
        seen = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_path = f"{path}.{k}" if path else k
            collect_numbers(v, new_path, seen)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            collect_numbers(v, f"{path}[{i}]", seen)
    else:
        try:
            n = _normalize(obj)
        except ValueError:
            return seen
        seen.setdefault(n, []).append(path)
    return seen


def format_number(n: float | int) -> str:
    if isinstance(n, float) and n.is_integer():
        return f"{int(n)}"
    return f"{n}"


def search_in_tex(tex_text: str, n: float | int) -> list[str]:
    """Return contexts where *n* appears in the TeX source."""
    contexts: list[str] = []
    s = format_number(n)
    # Candidate patterns: raw, percent, comma-grouped, scientific-ish
    candidates = {s, f"{s}\\%", f"{s}%"}
    if "." not in s:
        candidates.add(f"{int(n):,}")
    for cand in candidates:
        for m in re.finditer(re.escape(cand), tex_text):
            start = max(0, m.start() - 15)
            end = min(len(tex_text), m.end() + 15)
            contexts.append(tex_text[start:end].replace("\n", " "))
            if len(contexts) >= 3:
                break
        if contexts:
            break
    return contexts


def extract_tex_numbers(tex_text: str) -> list[tuple[str, float | int, int, int]]:
    """Extract normalized numbers from TeX and return (matched_text, value, start, end)."""
    found: list[tuple[str, float | int, int, int]] = []
    for m in LATEX_NUM_RE.finditer(tex_text):
        raw = m.group(1)
        has_percent = bool(re.match(r".*(?:\\?%)$", m.group(0)))
        text = raw
        if has_percent and "." not in text:
            pass
        try:
            val = _normalize(text)
        except ValueError:
            continue
        found.append((m.group(0), val, m.start(), m.end()))
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paper-dir",
        type=Path,
        default=PROJECT / "paper",
        help="Directory containing paper_data.json and .tex files.",
    )
    parser.add_argument(
        "--tex",
        nargs="*",
        default=["stam-m.tex", "stam-m_ja.tex"],
        help="LaTeX files to audit (relative to --paper-dir).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional TSV report path.  If omitted, only stdout summary is printed.",
    )
    parser.add_argument(
        "--tex-only",
        action="store_true",
        help="Only report TeX numbers not found in JSON; do not report missing JSON numbers.",
    )
    args = parser.parse_args(argv)

    paper_dir = args.paper_dir
    data_path = paper_dir / "paper_data.json"
    if not data_path.exists():
        print(f"ERROR: {data_path} not found", file=sys.stderr)
        return 1

    with open(data_path) as f:
        paper_data = json.load(f)
    json_numbers = collect_numbers(paper_data)

    combined_tex = ""
    file_map: list[tuple[str, int, str]] = []  # (filename, offset, text)
    for tex_name in args.tex:
        tex_path = paper_dir / tex_name
        if not tex_path.exists():
            print(f"WARNING: {tex_path} not found, skipping", file=sys.stderr)
            continue
        text = tex_path.read_text(encoding="utf-8")
        file_map.append((tex_name, len(combined_tex), text))
        combined_tex += f"\n\n% FILE: {tex_name}\n\n" + text

    if not combined_tex:
        print("ERROR: no TeX files to audit", file=sys.stderr)
        return 1

    # Remove comments to avoid matching magic numbers in comments
    combined_tex_no_comment = re.sub(r"(?<!\\)%.*", "", combined_tex)

    # 1. JSON -> TeX
    missing_in_tex: list[tuple[float | int, list[str]]] = []
    for n, paths in json_numbers.items():
        if not search_in_tex(combined_tex_no_comment, n):
            missing_in_tex.append((n, paths))

    # 2. TeX -> JSON
    tex_numbers = extract_tex_numbers(combined_tex_no_comment)
    tex_not_in_json: list[tuple[str, float | int, str]] = []
    for raw, val, start, end in tex_numbers:
        if val not in json_numbers:
            ctx_start = max(0, start - 25)
            ctx_end = min(len(combined_tex_no_comment), end + 25)
            ctx = combined_tex_no_comment[ctx_start:ctx_end].replace("\n", " ")
            tex_not_in_json.append((raw, val, ctx))

    print(f"JSON numeric values: {len(json_numbers)}")
    print(f"TeX numeric tokens:  {len(tex_numbers)}")
    print(f"JSON numbers not found in TeX: {len(missing_in_tex)}")
    print(f"TeX numbers not found in JSON: {len(tex_not_in_json)}")

    if missing_in_tex:
        print("\n--- JSON numbers missing from TeX (review) ---")
        for n, paths in sorted(missing_in_tex, key=lambda x: x[0], reverse=True)[:100]:
            print(f"  {n:>12}  paths={', '.join(paths[:3])}")

    if tex_not_in_json:
        print("\n--- TeX numbers not in JSON (manual values / constants) ---")
        for raw, val, ctx in tex_not_in_json[:100]:
            print(f"  {raw:>12} -> {val:>12}  ...{ctx}...")

    if args.report:
        rows = []
        for n, paths in missing_in_tex:
            rows.append(("json_missing_in_tex", str(n), "", ";".join(paths[:5])))
        for raw, val, ctx in tex_not_in_json:
            rows.append(("tex_not_in_json", raw, str(val), ctx))
        with open(args.report, "w", encoding="utf-8") as f:
            f.write("type\traw\tvalue\tcontext\n")
            for row in rows:
                f.write("\t".join(row) + "\n")
        print(f"\nReport written to {args.report}")

    # Return non-zero only when a JSON number is completely absent from the manuscript
    return 1 if missing_in_tex and not args.tex_only else 0


if __name__ == "__main__":
    sys.exit(main())
