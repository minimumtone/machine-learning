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


def clean_tex_for_numbers(tex_text: str) -> str:
    r"""Normalize LaTeX punctuation that splits numeric tokens.

    Removes ``{,}`` thin-space commas, turns math-mode ``$-$`` into a plain
    minus, and strips TeX thin spaces so numbers like ``1{,}559`` or
    ``$-$7.4\,pp`` are treated as contiguous tokens.
    """
    text = re.sub(r"\\textminus\{\}", "-", tex_text)
    text = re.sub(r"\$-\$", "-", text)
    text = re.sub(r"\{,\}", "", text)
    text = re.sub(r"\\,", "", text)
    text = re.sub(r"\\%", "%", text)
    return text


STRIP_ENVIRONMENTS = ("tikzpicture", "lstlisting", "verbatim", "filecontents")

# JSON values computed for completeness but deliberately not cited in the
# manuscript body (auxiliary latencies / sub-breakdowns).  They are still
# reported for review but do not fail the audit.
UNCITED_JSON_PATHS = (
    "database.n_unique_formulas",
    "ablation.cte_query_results.no_dict_cte_accuracy_pct",
    "transfer_evaluation_variants.C_obfuscated.by_difficulty_pct",
    "transfer_evaluation_variants.D_materials_project.avg_latency_s",
    "jp_reranker_comparison.jp_xsmall_latency_s",
    "jp_reranker_comparison.ms_marco_latency_s",
    "cte_evaluation_15.avg_latency_s",
)


def strip_non_prose(tex_text: str) -> str:
    """Remove TeX regions whose numbers are not manuscript data values.

    Strips drawing/listing environments (TikZ coordinates, listing colour
    specs) and the front matter before \\maketitle (addresses, postal codes).
    """
    for env in STRIP_ENVIRONMENTS:
        tex_text = re.sub(
            rf"\\begin\{{{env}\*?\}}.*?\\end\{{{env}\*?\}}",
            "",
            tex_text,
            flags=re.DOTALL,
        )
    parts = tex_text.split("\\maketitle", 1)
    if len(parts) == 2:
        preamble, body = parts
        doc_start = preamble.find("\\begin{document}")
        if doc_start != -1:
            tex_text = preamble[:doc_start] + body
        else:
            tex_text = body
    return tex_text


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
        text = strip_non_prose(tex_path.read_text(encoding="utf-8"))
        file_map.append((tex_name, len(combined_tex), text))
        combined_tex += f"\n\n% FILE: {tex_name}\n\n" + text

    if not combined_tex:
        print("ERROR: no TeX files to audit", file=sys.stderr)
        return 1

    # Remove comments to avoid matching magic numbers in comments
    combined_tex_no_comment = re.sub(r"(?<!\\)%.*", "", combined_tex)
    combined_tex_clean = clean_tex_for_numbers(combined_tex_no_comment)

    # 1. JSON -> TeX (this side gates the exit code).  Exact p-values are
    # reported in the manuscript as inequalities (e.g. p<0.001), so p_value
    # paths are reported for review but do not fail the audit.
    missing_in_tex: list[tuple[float | int, list[str]]] = []
    for n, paths in json_numbers.items():
        if not search_in_tex(combined_tex_clean, n):
            missing_in_tex.append((n, paths))
    gating_missing = [
        (n, paths)
        for n, paths in missing_in_tex
        if not all(
            "p_value" in p or p.startswith(UNCITED_JSON_PATHS) for p in paths
        )
    ]

    # 2. TeX -> JSON
    tex_numbers = extract_tex_numbers(combined_tex_clean)
    tex_not_in_json: list[tuple[str, float | int, str]] = []
    for raw, val, start, end in tex_numbers:
        if val not in json_numbers:
            ctx_start = max(0, start - 25)
            ctx_end = min(len(combined_tex_no_comment), end + 25)
            ctx = combined_tex_no_comment[ctx_start:ctx_end].replace("\n", " ")
            tex_not_in_json.append((raw, val, ctx))

    print(f"JSON numeric values: {len(json_numbers)}")
    print(f"TeX numeric tokens:  {len(tex_numbers)}")
    print(f"JSON numbers not found in TeX: {len(missing_in_tex)} (gating: {len(gating_missing)})")
    print(f"TeX numbers not found in JSON: {len(tex_not_in_json)} (informational only)")

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

    # Return non-zero only when a non-p-value JSON number is completely
    # absent from the manuscript; TeX->JSON is informational only.
    return 1 if gating_missing and not args.tex_only else 0


if __name__ == "__main__":
    sys.exit(main())
