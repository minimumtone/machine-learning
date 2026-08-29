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


def _sha256_json_dir(directory: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(directory.glob("*.json")):
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
                     prompt_path: Path | None = None,
                     expected_dir: Path | None = None) -> dict[str, str]:
    """Provenance block to embed in an evaluation result JSON."""
    if gold_dir is None:
        gold_dir = PROJECT / "evaluation" / "gold_sql"
    if prompt_path is None:
        prompt_path = PROMPT_TEMPLATE
    if expected_dir is None:
        expected_dir = PROJECT / "evaluation" / "expected_results"
    return {
        "dataset_file": dataset_path.name,
        "dataset_sha256": _sha256_file(dataset_path),
        "gold_dir": gold_dir.name,
        "gold_sha256": _sha256_gold_dir(gold_dir),
        "prompt_template_file": prompt_path.name,
        "prompt_template_sha256": _sha256_file(prompt_path),
        "prompt_template_note": (
            "SHA-256 of the static prompt template only; the prompt "
            "actually sent per query additionally contains dynamically "
            "injected few-shot examples, schema/JOIN listings and column "
            "hints, so it is not covered by this hash"),
        "expected_dir": expected_dir.name,
        "expected_sha256": _sha256_json_dir(expected_dir),
        "git_commit": _git_commit(),
        "generated_at": datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="seconds"),
    }


_HASH_KEY_ALIASES = {
    "prompt_sha256": "prompt_template_sha256",
    "prompt_file": "prompt_template_file",
}

# Keys that pin a run: input hashes plus the model and the code
# revision.  Timestamps are excluded.
_INPUT_HASH_KEYS = ("dataset_sha256", "gold_sha256",
                    "prompt_template_sha256", "model", "git_commit")


def _canon(prov: dict[str, str]) -> dict[str, str]:
    return {_HASH_KEY_ALIASES.get(k, k): v for k, v in prov.items()}


def assert_resumable(stored: dict[str, str] | None,
                     current: dict[str, str],
                     *, force: bool = False,
                     what: str = "stored results") -> None:
    """Refuse to resume from stale results.

    Compares dataset / gold / prompt-template hashes plus model and
    git commit of a previously saved evaluation file against the
    current run and raises RuntimeError on any difference, unless
    ``force`` is True.
    """
    stored_c = _canon(stored or {})
    current_c = _canon(current)
    diffs = [
        f"{k}: stored={stored_c.get(k, '<missing>')} current={current_c.get(k, '<missing>')}"
        for k in _INPUT_HASH_KEYS
        if stored_c.get(k) != current_c.get(k)
    ]
    if not diffs:
        return
    msg = (f"refusing to resume from {what}: provenance differs from the "
           "current dataset/gold/prompt/model/commit ("
           + "; ".join(diffs) + "). "
           "Delete the stale output or pass --force-stale-resume.")
    if force:
        print(f"WARNING (--force-stale-resume): {msg}")
        return
    raise RuntimeError(msg)


def sha256_file(path: Path) -> str:
    """Public helper: SHA-256 of one file (for source_result_sha256)."""
    return _sha256_file(path)
