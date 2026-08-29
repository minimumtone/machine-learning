"""Provenance metadata for evaluation result files.

Every saved evaluation result embeds the SHA-256 of the question
dataset, the gold SQL corpus, and the generation prompt template, plus
the git commit and a timestamp, so a stored result can be mechanically
matched to the exact revision of the inputs it was produced from.
"""
from __future__ import annotations

import datetime
import hashlib
import os
import subprocess
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
PROMPT_TEMPLATE = (
    PROJECT / "llm" / "prompt_templates" / "sql_generation_prompt.md"
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_gold_dir(gold_dir: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(gold_dir.glob("*.sql")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def _git_commit() -> str:
    env_commit = os.getenv("GIT_COMMIT")
    if env_commit:
        return env_commit
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_provenance(dataset_path: Path,
                     gold_dir: Path | None = None,
                     prompt_path: Path | None = None) -> dict[str, str]:
    """Provenance block to embed in an evaluation result JSON."""
    if gold_dir is None:
        gold_dir = PROJECT / "evaluation" / "gold_sql"
    if prompt_path is None:
        prompt_path = PROMPT_TEMPLATE
    return {
        "dataset_file": dataset_path.name,
        "dataset_sha256": _sha256_file(dataset_path),
        "gold_dir": gold_dir.name,
        "gold_sha256": _sha256_gold_dir(gold_dir),
        "prompt_file": prompt_path.name,
        "prompt_sha256": _sha256_file(prompt_path),
        "git_commit": _git_commit(),
        "generated_at": datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="seconds"),
    }
