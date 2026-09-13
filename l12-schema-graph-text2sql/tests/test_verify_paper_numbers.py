"""Regression tests for the default TeX selection of verify_paper_numbers.py.

The script lives in ``scripts/`` in the repository and in ``paper_scripts/``
next to ``sql_package/`` in the distribution layout; it is skipped when
neither copy is present (SQL-only package).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_CANDIDATES = [
    ROOT / "scripts" / "verify_paper_numbers.py",
    ROOT.parent / "paper_scripts" / "verify_paper_numbers.py",
]
_SCRIPT = next((p for p in _CANDIDATES if p.exists()), None)
if _SCRIPT is None:
    pytest.skip("verify_paper_numbers.py is not shipped with this package",
                allow_module_level=True)

_spec = importlib.util.spec_from_file_location("verify_paper_numbers", _SCRIPT)
assert _spec is not None and _spec.loader is not None
vpn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vpn)

PAPER_DATA = {"dataset": {"n_queries": 245}, "results": {"recall_pct": 86.1}}
TEX_EN = "\\begin{document}\n\\maketitle\n245 queries, 86.1\\% recall.\n\\end{document}\n"
TEX_JA = "\\begin{document}\n\\maketitle\n245問、86.1\\%。\n\\end{document}\n"


def _make_paper_dir(tmp_path: Path, tex_files: dict[str, str]) -> Path:
    paper_dir = tmp_path / "paper"
    paper_dir.mkdir()
    (paper_dir / "paper_data.json").write_text(json.dumps(PAPER_DATA), encoding="utf-8")
    for name, text in tex_files.items():
        (paper_dir / name).write_text(text, encoding="utf-8")
    return paper_dir


def test_default_audits_every_tex_in_paper_dir(tmp_path, capsys):
    paper_dir = _make_paper_dir(tmp_path, {"stam-m.tex": TEX_EN, "stam-m_ja.tex": TEX_JA})
    rc = vpn.main(["--paper-dir", str(paper_dir)])
    out, err = capsys.readouterr()
    assert rc == 0
    assert "TeX files audited: 2 (stam-m.tex, stam-m_ja.tex)" in out
    assert "WARNING" not in err


def test_default_with_single_tex_does_not_name_absent_file(tmp_path, capsys):
    paper_dir = _make_paper_dir(tmp_path, {"stam-m.tex": TEX_EN})
    rc = vpn.main(["--paper-dir", str(paper_dir)])
    out, err = capsys.readouterr()
    assert rc == 0
    assert "TeX files audited: 1 (stam-m.tex)" in out
    assert "stam-m_ja" not in err
    assert "stam-m_ja" not in out

    rc = vpn.main(["--paper-dir", str(paper_dir), "--tex", "stam-m_ja.tex"])
    _, err = capsys.readouterr()
    assert rc == 1
    assert "stam-m_ja.tex not found" in err
