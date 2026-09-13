#!/usr/bin/env python3
"""Build the three-part submission package (ZIP).

Layout of the ZIP (see package/README.md for the reviewer-facing description):

    README.md, FINAL_CHECKLIST.md, SHA256SUMS.txt
    01_manuscript/          manuscript sources + built PDF (+ main-text extract)
    02_supplementary/       supplementary-material PDF extract + machine-readable tables
    03_mdr_scripts_data/    self-contained reproduction package for the data repository
        sql_package/        (same file set as scripts/build_sql_package.py)
        paper/              frozen manuscript sources, SSOT, freeze files, figures
        paper_scripts/      SSOT / figure / number-audit scripts
        verification_logs/

Preconditions (the script refuses to package otherwise):
  * paper/stam-m_ja.pdf exists (build it with LuaLaTeX + BibTeX first);
  * paper/SHA256SUMS_ja.txt matches the working tree (the ja-final freeze set);
  * scripts/verify_all.py --static-only passes (unless --skip-verify);
  * the ``pypdf`` package is importable (used to split the PDF into main-text
    and supplement parts without duplicating embedded fonts).

Usage:
    python scripts/build_submission_package.py [--output PATH] [--skip-verify]
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from pypdf import PdfReader, PdfWriter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_sql_package import PROJECT, _git_head, iter_package_files  # noqa: E402

PAPER = PROJECT / "paper"
PACKAGE_DOCS = PROJECT / "package"

TEX = "stam-m_ja.tex"
PDF = "stam-m_ja.pdf"
SUPPLEMENT_TITLE = "補足資料"

MANUSCRIPT_SOURCES = [TEX, "references.bib", "interact.cls", "tfnlm.bst"]
FREEZE_FILES = ["paper_data.json", "SHA256SUMS_ja.txt", "ja_numbers.tsv",
                "glossary.tsv", "jp_reranker_vh_results.json"]
FIGURES = sorted(p.name for p in (PAPER / "figures").glob("*.png"))
PAPER_SCRIPTS = ["compute_all_figures.py", "generate_figures.py",
                 "verify_ssot.py", "verify_paper_numbers.py"]
SUPPLEMENT_TABLES = ["per_query_results.csv", "per_query_by_condition.csv",
                     "per_query_by_condition_mean5.csv",
                     "per_query_tables_provenance.json"]
COVER_LETTER = "cover_letter_ja.md"

D1, D2, D3 = "01_manuscript", "02_supplementary", "03_mdr_scripts_data"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_freeze() -> None:
    bad = []
    for line in (PAPER / "SHA256SUMS_ja.txt").read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        if sha256(PAPER / name.strip()) != digest:
            bad.append(name.strip())
    if bad:
        raise SystemExit("paper/SHA256SUMS_ja.txt does not match the working tree "
                         f"(regenerate the freeze list first): {bad}")


def supplement_start_page(reader: PdfReader) -> int:
    """1-based number of the first page whose first non-blank text line is the supplement title."""
    for i, page in enumerate(reader.pages, start=1):
        lines = [ln.strip() for ln in page.extract_text().splitlines() if ln.strip()]
        if lines and lines[0] == SUPPLEMENT_TITLE:
            return i
    raise SystemExit(f"supplement title page ({SUPPLEMENT_TITLE!r}) not found")


def write_pages(reader: PdfReader, pages: range, out: Path) -> None:
    """Write the 0-based ``pages`` of ``reader`` to ``out`` (shared fonts kept once)."""
    writer = PdfWriter()
    for i in pages:
        writer.add_page(reader.pages[i])
    writer.compress_identical_objects(remove_duplicates=True, remove_unreferenced=True)
    with out.open("wb") as f:
        writer.write(f)


def split_pdf(pdf: Path, workdir: Path) -> tuple[int, int]:
    """Split ``pdf`` into main-text and supplement PDFs; return (first_sup_page, n_pages)."""
    reader = PdfReader(pdf)
    n_pages = len(reader.pages)
    first_sup = supplement_start_page(reader)
    write_pages(reader, range(0, first_sup - 1), workdir / "stam-m_ja_main.pdf")
    write_pages(reader, range(first_sup - 1, n_pages), workdir / "stam-m_ja_supplementary.pdf")
    return first_sup, n_pages


def build_entries(workdir: Path) -> dict[str, Path]:
    """Map ZIP member name -> source path."""
    entries: dict[str, Path] = {}

    def add(zip_name: str, src: Path) -> None:
        if not src.is_file():
            raise FileNotFoundError(f"required file missing: {src}")
        if zip_name in entries:
            raise ValueError(f"duplicate ZIP member: {zip_name}")
        entries[zip_name] = src

    # root docs
    add("README.md", PACKAGE_DOCS / "README.md")
    add("FINAL_CHECKLIST.md", PACKAGE_DOCS / "FINAL_CHECKLIST.md")

    # 01_manuscript
    add(f"{D1}/README.md", PACKAGE_DOCS / "README_01_manuscript.md")
    for name in MANUSCRIPT_SOURCES:
        add(f"{D1}/{name}", PAPER / name)
    for fig in FIGURES:
        add(f"{D1}/figures/{fig}", PAPER / "figures" / fig)
    add(f"{D1}/{PDF}", PAPER / PDF)
    add(f"{D1}/{COVER_LETTER}", PAPER / COVER_LETTER)

    # 02_supplementary
    add(f"{D2}/README.md", PACKAGE_DOCS / "README_02_supplementary.md")
    for name in SUPPLEMENT_TABLES:
        add(f"{D2}/tables/{name}", PROJECT / "evaluation" / name)

    # 03_mdr_scripts_data
    add(f"{D3}/README.md", PACKAGE_DOCS / "README_03_mdr_scripts_data.md")
    for p in iter_package_files() + [PROJECT / "GIT_COMMIT"]:
        add(f"{D3}/sql_package/{p.relative_to(PROJECT).as_posix()}", p)
    for name in MANUSCRIPT_SOURCES + FREEZE_FILES:
        add(f"{D3}/paper/{name}", PAPER / name)
    for fig in FIGURES:
        add(f"{D3}/paper/figures/{fig}", PAPER / "figures" / fig)
    for name in PAPER_SCRIPTS:
        add(f"{D3}/paper_scripts/{name}", PROJECT / "scripts" / name)
    add(f"{D3}/paper_scripts/requirements-paper.txt",
        PACKAGE_DOCS / "requirements-paper.txt")
    logs = PACKAGE_DOCS / "verification_logs"
    for p in sorted(logs.rglob("*")):
        if p.is_file():
            add(f"{D3}/verification_logs/{p.relative_to(logs).as_posix()}", p)

    # derived PDFs (built into workdir by split_pdf)
    add(f"{D1}/stam-m_ja_main.pdf", workdir / "stam-m_ja_main.pdf")
    add(f"{D2}/stam-m_ja_supplementary.pdf", workdir / "stam-m_ja_supplementary.pdf")
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default=None,
                        help="output ZIP path (default: l12_text2sql_submission.zip "
                             "in the project root)")
    parser.add_argument("--skip-verify", action="store_true",
                        help="skip scripts/verify_all.py --static-only")
    args = parser.parse_args()

    pdf = PAPER / PDF
    if not pdf.is_file():
        raise SystemExit(f"{pdf} not found: build it with LuaLaTeX + BibTeX first")
    if pdf.stat().st_mtime < (PAPER / TEX).stat().st_mtime:
        raise SystemExit(f"{pdf} is older than {TEX}: rebuild the PDF first")
    check_freeze()

    commit = _git_head()
    (PROJECT / "GIT_COMMIT").write_text(commit + "\n", encoding="utf-8")
    print(f"GIT_COMMIT = {commit}")

    if not args.skip_verify:
        print("running scripts/verify_all.py --static-only ...")
        r = subprocess.run([sys.executable, str(PROJECT / "scripts" / "verify_all.py"),
                            "--static-only"], cwd=PROJECT)
        if r.returncode != 0:
            print("static verification FAILED; refusing to package", file=sys.stderr)
            return 1

    out = Path(args.output) if args.output else PROJECT / "l12_text2sql_submission.zip"
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        first_sup, n_pages = split_pdf(pdf, workdir)
        print(f"{PDF}: {n_pages} pages; main text + references = p.1-{first_sup - 1}, "
              f"supplement = p.{first_sup}-{n_pages}")

        entries = build_entries(workdir)
        sums = "\n".join(f"{sha256(src)}  {name}" for name, src in sorted(entries.items())) + "\n"
        (workdir / "SHA256SUMS.txt").write_text(sums, encoding="utf-8")
        entries["SHA256SUMS.txt"] = workdir / "SHA256SUMS.txt"

        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
            for name in sorted(entries):
                zf.write(entries[name], name)

        for d in (D1, D2, D3):
            n = sum(1 for k in entries if k.startswith(d + "/"))
            print(f"  {d}: {n} files")
        print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(entries)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
