#!/usr/bin/env python3
"""Check whether the "single source of truth" claim actually holds.

``paper/paper_data.json`` declares itself the "Single source of truth for paper
numerical values".  This script tests that claim mechanically instead of taking
it on trust.  It performs five checks:

  1. FIGURE PROVENANCE -- does ``generate_figures.py`` read
     ``paper_data.json``, or does it read the raw evaluation JSONs directly?
     If it reads them directly, the figures do not go through the SSOT and can
     drift from the reported numbers.

  2. HARD-CODED LITERALS -- numeric values written directly into
     ``compute_all_figures.py`` and emitted into ``paper_data.json`` as if they
     had been derived.  These are the values that silently go stale.

  3. SQLGUARD CHECK COUNT -- ``n_sqlguard_checks`` against the number of
     ``check_*`` functions actually invoked by ``validate_sql()``, and against
     the length of the enumerated list in the manuscript.

  4. DERIVABLE INVARIANTS -- values in paper_data.json that can be recomputed
     from the packaged artefacts without a database or an LLM.

  5. PROVENANCE FIELDS -- whether ``_meta.git_commit`` identifies a real commit.

Exit code 0 if every check passes, 1 otherwise.  Deterministic, no LLM, no DB.

Usage:
    python paper_scripts/verify_ssot.py
    python paper_scripts/verify_ssot.py --json sql_package/evaluation/ssot_audit.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent


def first_existing(*paths: Path) -> Path:
    """Return the first path that exists, or the primary path for errors."""
    for path in paths:
        if path.exists():
            return path
    return paths[0]


PAPER_DATA = PROJECT / "paper" / "paper_data.json"
GEN_FIGURES = first_existing(
    PROJECT / "scripts" / "generate_figures.py",
    SCRIPT_DIR / "generate_figures.py",
)
COMPUTE = first_existing(
    PROJECT / "scripts" / "compute_all_figures.py",
    SCRIPT_DIR / "compute_all_figures.py",
)
VALIDATOR = first_existing(
    PROJECT / "safety" / "sql_validator.py",
    PROJECT / "sql_package" / "safety" / "sql_validator.py",
)
EVALUATION_DIR = first_existing(
    PROJECT / "evaluation",
    PROJECT / "sql_package" / "evaluation",
)
# Only the maintained Japanese manuscript is synchronized with the SSOT;
# the English manuscript is intentionally frozen at an earlier revision.
TEX_FILES = ["paper/stam-m_ja.tex"]


class Result:
    def __init__(self) -> None:
        self.findings: list[dict[str, Any]] = []
        self.failed = 0

    def add(self, check: str, ok: bool, detail: str, data: Any = None) -> None:
        self.findings.append({"check": check, "ok": ok, "detail": detail, "data": data})
        if not ok:
            self.failed += 1
        print(f"[{'PASS' if ok else 'FAIL'}] {check}")
        for line in detail.splitlines():
            print(f"       {line}")


def check_figure_provenance(r: Result) -> None:
    src = GEN_FIGURES.read_text(encoding="utf-8")
    reads_ssot = "paper_data" in src
    direct = sorted(set(re.findall(r'"evaluation"\s*/\s*"([a-z0-9_]+\.json)"', src)))
    ok = reads_ssot and not direct
    if reads_ssot and direct:
        detail = ("generate_figures.py reads paper_data.json AND "
                  f"{len(direct)} evaluation files directly: {', '.join(direct)}")
    elif reads_ssot:
        detail = "generate_figures.py reads paper_data.json only."
    else:
        detail = ("generate_figures.py does NOT read paper_data.json; it reads "
                  f"{len(direct)} evaluation files directly:\n  " + "\n  ".join(direct) +
                  "\nThe figures therefore bypass the SSOT.")
    r.add("figure provenance goes through the SSOT", ok, detail, direct)


HARDCODE_PATTERNS = [
    (r'"n_conditions":\s*(\d+)', "ablation.n_conditions"),
    (r'n_sqlguard_checks\s*=\s*(\d+)', "safety.n_sqlguard_checks"),
]


def check_hardcoded_literals(r: Result) -> None:
    src = COMPUTE.read_text(encoding="utf-8")
    hits: list[str] = []
    for pat, label in HARDCODE_PATTERNS:
        m = re.search(pat, src)
        if m:
            hits.append(f"{label} = {m.group(1)} (literal in compute_all_figures.py)")
    # numbers embedded in _note strings, which reach paper_data.json verbatim
    for note in re.findall(r'"_note":\s*"((?:[^"\\]|\\.)*)"', src):
        for num in re.findall(r"\b\d+\b", note):
            hits.append(f'_note literal "{num}" in: "{note[:60]}..."')
    # a hand-written list emitted as data
    if re.search(r"known_l12\s*=\s*\[", src):
        hits.append("materials_evaluation.known_l12_seed_list is a hand-written list")
    detail = ("No hard-coded numeric literals reach paper_data.json."
              if not hits else
              "Values presented as derived but written by hand:\n  " + "\n  ".join(hits))
    r.add("paper_data.json contains no hand-written numbers", not hits, detail, hits)


def check_sqlguard_count(r: Result) -> None:
    data = json.loads(PAPER_DATA.read_text(encoding="utf-8"))
    claimed = data.get("safety", {}).get("n_sqlguard_checks")
    src = VALIDATOR.read_text(encoding="utf-8")
    body = re.search(r"^def validate_sql\b.*?(?=\n^def |\Z)", src, re.M | re.S)
    invoked = sorted(set(re.findall(r"\b(check_[a-z_]+)\(", body.group(0)))) if body else []
    tex_enum_patterns = [
        r"comprises\s+(\d+)\s+checks\b.*?\\begin\{enumerate\}(.*?)\\end\{enumerate\}",
        r"(\d+)のチェック関数から成る.*?\\begin\{enumerate\}(.*?)\\end\{enumerate\}",
    ]
    tex_counts: dict[str, tuple[int, int]] = {}
    unmatched: list[str] = []
    for rel in TEX_FILES:
        p = PROJECT / rel
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8")
        for pat in tex_enum_patterns:
            m = re.search(pat, text, re.S)
            if m:
                tex_counts[rel] = (int(m.group(1)),
                                   len(re.findall(r"\\item", m.group(2))))
                break
        else:
            unmatched.append(rel)
    ok = (claimed == len(invoked)
          and not unmatched
          and all(stated == claimed and items == claimed
                  for stated, items in tex_counts.values()))
    lines = [f"paper_data.json safety.n_sqlguard_checks = {claimed}",
             f"check_* functions invoked by validate_sql() = {len(invoked)}"]
    lines += [f"{rel}: states {stated}, enumerates {items} items"
              for rel, (stated, items) in tex_counts.items()]
    lines += [f"{rel}: SQLGuard check enumeration not found" for rel in unmatched]
    if not ok:
        lines.append("The manuscript enumerates items that are not separate checks in "
                     "the implementation; verify each listed item actually fires.")
    r.add("SQLGuard check count is consistent", ok, "\n".join(lines),
          {"claimed": claimed, "invoked": invoked, "tex": tex_counts})


def check_derivable_invariants(r: Result) -> None:
    data = json.loads(PAPER_DATA.read_text(encoding="utf-8"))
    problems: list[str] = []
    stats_path = EVALUATION_DIR / "ablation_multirun_stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        n_real = len(stats.get("conditions", {}))
        n_claim = data.get("ablation", {}).get("n_conditions")
        if n_claim != n_real:
            problems.append(f"ablation.n_conditions = {n_claim} but "
                            f"ablation_multirun_stats.json has {n_real} conditions")
        n_runs_claim = data.get("ablation", {}).get("n_runs")
        n_runs_real = len(list(EVALUATION_DIR.glob("ablation_run_*.json")))
        if n_runs_claim != n_runs_real:
            problems.append(f"ablation.n_runs = {n_runs_claim} but "
                            f"{n_runs_real} ablation_run_*.json files are packaged")
        total = data.get("ablation", {}).get("total_evaluations")
        expect = (n_real * (data.get("ablation", {}).get("n_queries_per_condition") or 0)
                  * (n_runs_real or 0))
        if total is not None and expect and total != expect:
            problems.append(f"ablation.total_evaluations = {total} but "
                            f"conditions x queries x runs = {expect}")
    detail = ("All derivable invariants agree with the packaged artefacts."
              if not problems else "\n".join(problems))
    r.add("derivable invariants agree", not problems, detail, problems)


def check_provenance_fields(r: Result) -> None:
    data = json.loads(PAPER_DATA.read_text(encoding="utf-8"))
    meta = data.get("_meta", {})
    commit = meta.get("git_commit")
    has_git = (PROJECT / ".git").exists()
    ok = bool(commit) and commit != "unknown"
    detail = (f"_meta.git_commit = {commit!r}; .git present: {has_git}")
    if not ok:
        detail += ("\nRegenerating without a .git directory yields 'unknown', so the "
                   "packaged value cannot be reproduced by a reviewer.")
    r.add("provenance fields are reproducible", ok, detail, {"git_commit": commit})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", help="write the findings to this path")
    args = ap.parse_args()

    if not PAPER_DATA.exists():
        print(f"{PAPER_DATA} not found", file=sys.stderr)
        return 1

    r = Result()
    check_figure_provenance(r)
    check_hardcoded_literals(r)
    check_sqlguard_count(r)
    check_derivable_invariants(r)
    check_provenance_fields(r)

    print(f"\n{len(r.findings) - r.failed}/{len(r.findings)} checks passed")
    if r.failed:
        print("The 'Single source of truth' claim in paper_data.json._meta and in "
              "README_REPRO.md should be narrowed to what actually holds, or the "
              "pipeline changed so that it does hold.")
    if args.json:
        Path(args.json).write_text(
            json.dumps(r.findings, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Wrote {args.json}")
    return 1 if r.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
