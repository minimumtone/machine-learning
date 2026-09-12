#!/usr/bin/env python3
"""Build the distributable SQL reproduction package (ZIP).

Assembles the package from the project working tree with an explicit
top-level include list, excludes caches and paper-only tooling, writes
GIT_COMMIT from the current revision, and refuses to package a tree
whose static verification (scripts/verify_all.py --static-only) fails.

Usage:
    python scripts/build_sql_package.py [--output PATH] [--skip-verify]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent

# Top-level directories and files shipped in the package.
INCLUDE_DIRS = [
    "db", "docker", "evaluation", "graph", "ingestion", "llm",
    "safety", "scripts", "tests",
]
INCLUDE_FILES = [
    ".env.example", "README_SQL.md", "MANIFEST.md", "few_shot_examples.json",
    "pytest.ini", "requirements-repro.txt",
]

# Paper-pipeline tooling: requires paper/ (not shipped) and is not part
# of the SQL reproduction package.
EXCLUDE_SCRIPTS = {
    "verify_ssot.py",
    "verify_paper_numbers.py",
    "generate_figures.py",
    "compute_all_figures.py",
    # Writes paper/jp_reranker_vh_results.json (a paper-figure input that
    # predates the R22A gold fixes and stores no per-query SQL, so it cannot
    # be deterministically re-scored without new inference).
    "eval_jp_reranker_vh.py",
    # Paper-manuscript editing tools (frontmatter repair / EN->JA
    # translation); not part of the SQL reproduction workflow, and
    # repair_stam_ja.py would restore an obsolete manuscript revision.
    "repair_stam_ja.py",
    "translate_stam_ja_v2.py",
}

EXCLUDE_DIR_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo"}


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT,
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def iter_package_files() -> list[Path]:
    files: list[Path] = []
    for name in INCLUDE_FILES:
        p = PROJECT / name
        if not p.is_file():
            raise FileNotFoundError(f"required package file missing: {name}")
        files.append(p)
    for d in INCLUDE_DIRS:
        root = PROJECT / d
        if not root.is_dir():
            raise FileNotFoundError(f"required package dir missing: {d}")
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(PROJECT)
            if any(part in EXCLUDE_DIR_NAMES for part in rel.parts):
                continue
            if p.suffix in EXCLUDE_SUFFIXES:
                continue
            if rel.parts[0] == "scripts" and p.name in EXCLUDE_SCRIPTS:
                continue
            files.append(p)
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=None,
                        help="output ZIP path (default: "
                             "l12_sql_package_r22.zip in the project root)")
    parser.add_argument("--skip-verify", action="store_true",
                        help="skip the pre-package static verification")
    args = parser.parse_args()

    commit = _git_head()
    (PROJECT / "GIT_COMMIT").write_text(commit + "\n", encoding="utf-8")
    print(f"GIT_COMMIT = {commit}")

    if not args.skip_verify:
        print("running scripts/verify_all.py --static-only ...")
        r = subprocess.run(
            [sys.executable, str(PROJECT / "scripts" / "verify_all.py"),
             "--static-only"], cwd=PROJECT)
        if r.returncode != 0:
            print("static verification FAILED; refusing to package",
                  file=sys.stderr)
            return 1

    out = Path(args.output) if args.output else (
        PROJECT / "l12_sql_package_r22.zip")
    files = iter_package_files() + [PROJECT / "GIT_COMMIT"]
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(files):
            zf.write(p, p.relative_to(PROJECT).as_posix())
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, "
          f"{len(files)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
