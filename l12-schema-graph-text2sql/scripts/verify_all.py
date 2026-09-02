#!/usr/bin/env python3
"""Single read-only verification entry point for the L12 SQL package.

This script deliberately does *not* regenerate expected results or any other
artifact.  It combines the package's static consistency checks and the
database-backed verification/audit scripts behind one exit status.

Formal reviewer mode (default):
    python scripts/verify_all.py

Fast package-only preflight (no DB/dependency requirement):
    python scripts/verify_all.py --static-only

Useful options:
    --no-pytest          skip the final FULL_DB_TEST pytest run
    --no-scoring         skip database-backed scoring self-check
    --allow-db-skip      development only: allow missing DB-suite DSNs
    --warnings-as-errors fail on warnings as well as errors
    --json-report PATH   write a machine-readable report outside/inside the
                         package only when explicitly requested
    --fail-fast          stop after the first failed check

Default formal verification is intentionally strict:
* all 4 DB suites are required;
* no missing/orphan/malformed canonical query artifacts are accepted;
* generated-SQL manifests/files must be complete and agree with their source
  evaluation JSON;
* current provenance hashes must match stored provenance;
* duplicate "main" result artifacts with the same inputs may not contain
  different generated SQL;
* every child verifier must exit 0;
* FULL_DB_TEST pytest must not silently skip the DB-integrity tests.

The child verification commands are run with PYTHONPATH rooted at this
package.  The scoring audit is forced read-only through PGOPTIONS and writes
its report only to a temporary file, never to evaluation/scoring_audit.json.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import unquote, urlparse

PROJECT = Path(__file__).resolve().parent.parent
EVAL = PROJECT / "evaluation"
SCRIPTS = PROJECT / "scripts"
PROMPTS = PROJECT / "llm" / "prompt_templates"

CANONICAL_SUITES = {
    "main": {
        "dataset": "main_evaluation_dataset.jsonl",
        "count": 245,
        "gold_dir": "gold_sql",
        "expected_dir": "expected_results",
    },
    "transfer": {
        "dataset": "transfer_evaluation_dataset.jsonl",
        "count": 20,
        "gold_dir": "gold_sql",
        "expected_dir": "expected_results",
    },
    "transfer_obfuscated": {
        "dataset": "transfer_obfuscated_evaluation_dataset.jsonl",
        "count": 20,
        "gold_dir": "gold_sql_obfuscated",
        "expected_dir": "expected_results_obfuscated",
    },
    "mp_transfer": {
        "dataset": "mp_transfer_evaluation_dataset.jsonl",
        "count": 15,
        "gold_dir": "gold_sql_mp",
        "expected_dir": "expected_results_mp_transfer",
    },
}

GENERATED_SOURCES = {
    "main": "main_eval_with_sql.json",
    "independent": "independent_eval_results.json",
    "cte": "cte_eval_results.json",
    "llm_only": "llm_only_results.json",
    "prototype": "prototype_eval_results.json",
    "transfer": "transfer_eval_results.json",
    "transfer_obfuscated": "transfer_obfuscated_eval_results.json",
    "mp_transfer": "mp_transfer_eval_results.json",
}

REQUIRED_FILES = [
    "README_SQL.md",
    "GIT_COMMIT",
    "requirements-repro.txt",
    "db/001_schema.sql",
    "db/006_integrity_checks.sql",
    "db/007_initialization_marker.sql",
    "scripts/run_gold_verification.py",
    "scripts/audit_order_totality.py",
    "scripts/audit_semantics.py",
    "scripts/audit_vocabulary.py",
    "scripts/fixture_guard.py",
    "scripts/transfer_guard.py",
    "scripts/mp_guard.py",
]
REQUIRED_DIRS = ["db", "evaluation", "scripts", "tests", "llm"]


class VerifyError(RuntimeError):
    pass


@dataclass
class CheckResult:
    name: str
    status: str  # PASS / FAIL / WARN / SKIP
    seconds: float
    summary: str
    details: list[str]
    output: str = ""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_gold_dir(path: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(path.glob("*.sql")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def _sha256_json_dir(path: Path) -> str:
    # Must stay byte-identical to scripts/provenance.py::_sha256_json_dir.
    h = hashlib.sha256()
    for p in sorted(path.glob("*.json")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise VerifyError(f"{path.relative_to(PROJECT)}: invalid JSON: {exc}") from exc


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise VerifyError(f"{path.name}:{i}: row is not an object")
            rows.append(value)
    except VerifyError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise VerifyError(f"{path.relative_to(PROJECT)}: invalid JSONL: {exc}") from exc
    return rows


def _result_rows(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict):
        for key in ("results", "queries"):
            value = data.get(key)
            if isinstance(value, list) and (not value or isinstance(value[0], dict)):
                return value
        # Historical result formats: find a unique qid-bearing list.
        candidates = [
            v for v in data.values()
            if isinstance(v, list) and v and isinstance(v[0], dict)
            and ("qid" in v[0] or "id" in v[0])
        ]
        if len(candidates) == 1:
            return candidates[0]
    if isinstance(data, list) and (not data or isinstance(data[0], dict)):
        return data
    return []


def _qid(row: dict[str, Any]) -> str | None:
    value = row.get("qid", row.get("id"))
    return str(value) if value is not None else None


def _sql_from_row(row: dict[str, Any]) -> str | None:
    for key in ("sql", "gen_sql", "generated_sql"):
        value = row.get(key)
        if isinstance(value, str):
            return value
    return None


def _dedupe(ids: Iterable[str]) -> tuple[set[str], list[str]]:
    seen: set[str] = set()
    dup: list[str] = []
    for qid in ids:
        if qid in seen:
            dup.append(qid)
        seen.add(qid)
    return seen, sorted(set(dup))


def check_package_structure() -> tuple[str, list[str]]:
    missing = [p for p in REQUIRED_FILES if not (PROJECT / p).is_file()]
    missing += [p + "/" for p in REQUIRED_DIRS if not (PROJECT / p).is_dir()]
    if missing:
        raise VerifyError("missing required package paths: " + ", ".join(missing))
    return "required package structure present", []


def check_python_syntax() -> tuple[str, list[str]]:
    failures: list[str] = []
    count = 0
    for path in PROJECT.rglob("*.py"):
        if ".pytest_cache" in path.parts or "__pycache__" in path.parts:
            continue
        count += 1
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{path.relative_to(PROJECT)}: {exc}")
    if failures:
        raise VerifyError(f"{len(failures)} Python syntax error(s): " + "; ".join(failures[:8]))
    return f"{count} Python files compile", []


def _canonical_dataset_rows() -> tuple[dict[str, list[dict[str, Any]]], set[str]]:
    suites: dict[str, list[dict[str, Any]]] = {}
    all_ids: list[str] = []
    for name, spec in CANONICAL_SUITES.items():
        rows = _jsonl(EVAL / spec["dataset"])
        if len(rows) != spec["count"]:
            raise VerifyError(
                f"{spec['dataset']}: expected {spec['count']} rows, found {len(rows)}")
        ids = [_qid(r) for r in rows]
        if any(q is None for q in ids):
            raise VerifyError(f"{spec['dataset']}: row without id/qid")
        _, dup = _dedupe(q for q in ids if q is not None)
        if dup:
            raise VerifyError(f"{spec['dataset']}: duplicate ids {dup[:10]}")
        suites[name] = rows
        all_ids.extend(q for q in ids if q is not None)
    all_set, dup_all = _dedupe(all_ids)
    if dup_all:
        raise VerifyError(f"canonical suites have duplicate qids: {dup_all[:20]}")
    return suites, all_set


def check_canonical_catalog() -> tuple[str, list[str]]:
    suites, canonical_ids = _canonical_dataset_rows()
    if len(canonical_ids) != 300:
        raise VerifyError(f"canonical corpus must contain 300 unique qids, found {len(canonical_ids)}")

    missing_paths: list[str] = []
    for name, rows in suites.items():
        spec = CANONICAL_SUITES[name]
        for row in rows:
            qid = _qid(row)
            assert qid is not None
            gold_rel = row.get("gold_sql_path")
            exp_rel = row.get("expected_result_path")
            if not isinstance(gold_rel, str):
                gold_rel = f"{spec['gold_dir']}/{qid}.sql"
            if not isinstance(exp_rel, str):
                exp_rel = f"{spec['expected_dir']}/{qid}.json"
            if not (EVAL / gold_rel).is_file():
                missing_paths.append(f"{qid}: {gold_rel}")
            if not (EVAL / exp_rel).is_file():
                missing_paths.append(f"{qid}: {exp_rel}")
    if missing_paths:
        raise VerifyError("canonical dataset references missing artifact(s): "
                          + "; ".join(missing_paths[:20]))

    # The physical canonical directories must not contain unreferenced
    # gold/expected artifacts either.
    referenced_by_dir: dict[tuple[str, str], set[str]] = {}
    for name, rows in suites.items():
        spec = CANONICAL_SUITES[name]
        key = (spec["gold_dir"], spec["expected_dir"])
        referenced_by_dir.setdefault(key, set()).update(
            q for q in (_qid(r) for r in rows) if q is not None
        )
    for (gold_name, exp_name), wanted in referenced_by_dir.items():
        actual_gold = {q.stem for q in (EVAL / gold_name).glob("*.sql")}
        actual_exp = {q.stem for q in (EVAL / exp_name).glob("*.json")}
        if actual_gold != wanted:
            raise VerifyError(
                f"{gold_name}: canonical file-set mismatch "
                f"(wanted={len(wanted)}, actual={len(actual_gold)}, "
                f"missing={sorted(wanted-actual_gold)[:20]}, "
                f"orphan={sorted(actual_gold-wanted)[:20]})")
        if actual_exp != wanted:
            raise VerifyError(
                f"{exp_name}: canonical file-set mismatch "
                f"(wanted={len(wanted)}, actual={len(actual_exp)}, "
                f"missing={sorted(wanted-actual_exp)[:20]}, "
                f"orphan={sorted(actual_exp-wanted)[:20]})")

    catalog = _json(EVAL / "query_catalog.json")
    if not isinstance(catalog, dict) or not isinstance(catalog.get("queries"), list):
        raise VerifyError("query_catalog.json has no queries list")
    queries = catalog["queries"]
    catalog_ids = [str(q.get("qid")) for q in queries if q.get("qid") is not None]
    catalog_set, dup = _dedupe(catalog_ids)
    problems: list[str] = []
    if dup:
        problems.append(f"duplicate catalog qids: {dup[:20]}")
    if len(queries) != 300 or catalog.get("n_total") != 300:
        problems.append(f"catalog rows/n_total must both be 300 "
                        f"(rows={len(queries)}, n_total={catalog.get('n_total')})")
    if catalog_set != canonical_ids:
        problems.append(
            f"catalog/canonical qid mismatch: missing={sorted(canonical_ids-catalog_set)[:20]} "
            f"extra={sorted(catalog_set-canonical_ids)[:20]}")
    for q in queries:
        if not q.get("gold_sql_available", False):
            problems.append(f"{q.get('qid')}: catalog says gold unavailable")
        if not q.get("expected_result_available", False):
            problems.append(f"{q.get('qid')}: catalog says expected result unavailable")
    if problems:
        raise VerifyError("; ".join(problems[:30]))
    return "300 unique canonical queries; catalog/gold/expected paths agree", []


def check_expected_json_schema() -> tuple[str, list[str]]:
    dirs = [
        EVAL / "expected_results",
        EVAL / "expected_results_obfuscated",
        EVAL / "expected_results_mp_transfer",
    ]
    errors: list[str] = []
    n = 0
    for directory in dirs:
        for p in sorted(directory.glob("*.json")):
            n += 1
            data = _json(p)
            rel = p.relative_to(PROJECT)
            if not isinstance(data, dict):
                errors.append(f"{rel}: not an object")
                continue
            cols, rows, ordered = data.get("columns"), data.get("rows"), data.get("ordered")
            if not isinstance(cols, list) or not all(isinstance(c, str) for c in cols):
                errors.append(f"{rel}: columns must be list[str]")
                continue
            if not isinstance(ordered, bool):
                errors.append(f"{rel}: ordered must be bool")
            sem = data.get("semantic_ordered")
            if not isinstance(sem, bool):
                errors.append(f"{rel}: semantic_ordered must be bool")
            elif sem and ordered is not True:
                errors.append(
                    f"{rel}: semantic_ordered=true requires ordered=true "
                    f"(gold must store the required order deterministically)")
            if not isinstance(rows, list) or not all(isinstance(r, list) for r in rows):
                errors.append(f"{rel}: rows must be list[list]")
                continue
            bad_width = [i for i, row in enumerate(rows) if len(row) != len(cols)]
            if bad_width:
                errors.append(f"{rel}: row width mismatch at rows {bad_width[:5]}")
            if data.get("expected_empty") is True and rows:
                errors.append(f"{rel}: expected_empty=true but rows are non-empty")
    if errors:
        raise VerifyError(f"{len(errors)} malformed expected result(s): " + "; ".join(errors[:20]))
    return f"{n} expected-result JSON files have a valid structural contract", []


def check_generated_sql_consistency() -> tuple[str, list[str]]:
    base = EVAL / "generated_sql"
    errors: list[str] = []
    checked = 0
    for subdir, eval_name in GENERATED_SOURCES.items():
        directory = base / subdir
        source_path = EVAL / eval_name
        if not directory.is_dir():
            errors.append(f"{subdir}: generated_sql directory missing")
            continue
        if not source_path.is_file():
            errors.append(f"{subdir}: source evaluation file {eval_name} missing")
            continue
        source = _json(source_path)
        source_rows = _result_rows(source)
        source_by_qid = {_qid(r): r for r in source_rows if _qid(r)}
        if not source_by_qid:
            errors.append(f"{subdir}: cannot locate qid-bearing results in {eval_name}")
            continue

        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            errors.append(f"{subdir}: manifest.json missing")
            continue
        manifest = _json(manifest_path)
        if not isinstance(manifest, dict) or not isinstance(manifest.get("queries"), list):
            errors.append(f"{subdir}: malformed manifest")
            continue
        manifest_rows = manifest["queries"]
        if manifest.get("eval_file") != eval_name:
            errors.append(
                f"{subdir}: manifest eval_file={manifest.get('eval_file')!r} "
                f"but canonical source is {eval_name!r}")

        source_result_file = manifest.get("source_result_file")
        if isinstance(source_result_file, str):
            source_result_path = EVAL / source_result_file
            if not source_result_path.is_file():
                errors.append(
                    f"{subdir}: manifest source_result_file missing: "
                    f"{source_result_file}")
            else:
                actual_source_sha = _sha256_file(source_result_path)
                if manifest.get("source_result_sha256") != actual_source_sha:
                    errors.append(
                        f"{subdir}: source_result_sha256 mismatch "
                        f"(stored={manifest.get('source_result_sha256')}, "
                        f"actual={actual_source_sha})")
                source_result_obj = _json(source_result_path)
                source_prov = (
                    source_result_obj.get("provenance")
                    if isinstance(source_result_obj, dict) else None
                )
                manifest_prov = manifest.get("provenance")
                if (isinstance(source_prov, dict)
                        and isinstance(manifest_prov, dict)
                        and source_prov != manifest_prov):
                    errors.append(
                        f"{subdir}: manifest provenance differs from "
                        f"{source_result_file}")
        manifest_ids = [str(r.get("qid")) for r in manifest_rows if r.get("qid") is not None]
        manifest_set, dup = _dedupe(manifest_ids)
        if dup:
            errors.append(f"{subdir}: duplicate qids in manifest {dup[:10]}")
        if manifest.get("n_queries") != len(manifest_rows):
            errors.append(
                f"{subdir}: manifest n_queries={manifest.get('n_queries')} "
                f"but contains {len(manifest_rows)} rows")
        if set(source_by_qid) != manifest_set:
            errors.append(
                f"{subdir}: source-result/manifest qid mismatch "
                f"(source={len(source_by_qid)}, manifest={len(manifest_set)}, "
                f"missing={sorted(set(source_by_qid)-manifest_set)[:10]}, "
                f"extra={sorted(manifest_set-set(source_by_qid))[:10]})")

        actual_files = {p.stem: p for p in directory.glob("*.sql")}
        if actual_files.keys() != manifest_set:
            errors.append(
                f"{subdir}: SQL-file/manifest mismatch "
                f"(sql_files={len(actual_files)}, manifest={len(manifest_set)}, "
                f"missing={sorted(manifest_set-set(actual_files))[:10]}, "
                f"orphan={sorted(set(actual_files)-manifest_set)[:10]})")

        for mrow in manifest_rows:
            qid = str(mrow.get("qid"))
            source_file = mrow.get("source_file")
            if source_file is not None and source_file != eval_name:
                errors.append(
                    f"{subdir}:{qid}: manifest source_file={source_file!r} "
                    f"!= {eval_name!r}")
            sql_path_raw = mrow.get("sql_path")
            if isinstance(sql_path_raw, str):
                declared = PROJECT / sql_path_raw
                if not declared.is_file():
                    errors.append(f"{subdir}:{qid}: manifest sql_path missing: {sql_path_raw}")
            p = actual_files.get(qid)
            src_row = source_by_qid.get(qid)
            if p is None or src_row is None:
                continue
            src_sql = _sql_from_row(src_row)
            if src_sql is not None and p.read_text(encoding="utf-8").strip() != src_sql.strip():
                errors.append(f"{subdir}:{qid}: SQL file differs from source evaluation JSON")
        checked += 1

    if errors:
        raise VerifyError(
            f"generated-SQL artifact inconsistency ({len(errors)} issue(s))\n"
            + "\n".join(errors[:50]))
    return f"{checked} generated-SQL suites agree with manifests/source results", []


def _find_prompt(name: str) -> Path | None:
    candidates = [
        PROMPTS / name,
        PROJECT / name,
        PROJECT / "llm" / name,
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def check_provenance() -> tuple[str, list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    package_commit = (PROJECT / "GIT_COMMIT").read_text(encoding="utf-8").strip()
    n_blocks = 0

    # Result JSONs with model outputs are required to carry provenance.
    for p in sorted(EVAL.glob("*.json")):
        data = _json(p)
        if not isinstance(data, dict):
            continue
        looks_like_model_result = (
            isinstance(data.get("model"), str)
            and bool(_result_rows(data))
        )
        prov = data.get("provenance")
        if looks_like_model_result and not isinstance(prov, dict):
            errors.append(f"{p.name}: model result has no provenance block")
            continue
        if not isinstance(prov, dict):
            continue
        n_blocks += 1

        dataset_file = prov.get("dataset_file")
        dataset_rows: list[dict] = []
        if isinstance(dataset_file, str):
            dataset = EVAL / dataset_file
            if not dataset.is_file():
                errors.append(f"{p.name}: provenance dataset missing: {dataset_file}")
            else:
                if prov.get("dataset_sha256") != _sha256_file(dataset):
                    errors.append(f"{p.name}: dataset_sha256 mismatch")
                dataset_rows = [
                    json.loads(line)
                    for line in dataset.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]

        # The recorded gold/expected directories must be the ones the
        # dataset itself points at, not merely directories whose hashes
        # check out (guards against build_provenance() being called with
        # default directories for a non-default dataset).
        for path_key, dir_key in (("gold_sql_path", "gold_dir"),
                                  ("expected_result_path", "expected_dir")):
            dirs = {Path(row[path_key]).parent.as_posix()
                    for row in dataset_rows if isinstance(row.get(path_key), str)}
            recorded = prov.get(dir_key)
            if dirs and isinstance(recorded, str) and recorded not in dirs:
                errors.append(
                    f"{p.name}: provenance {dir_key} '{recorded}' does not match "
                    f"dataset {path_key} parent dir(s) {sorted(dirs)}")

        gold_dir_name = prov.get("gold_dir")
        if isinstance(gold_dir_name, str):
            gold_dir = EVAL / gold_dir_name
            if not gold_dir.is_dir():
                errors.append(f"{p.name}: provenance gold dir missing: {gold_dir_name}")
            elif prov.get("gold_sha256") != _sha256_gold_dir(gold_dir):
                errors.append(f"{p.name}: gold_sha256 mismatch")

        prompt_file = prov.get("prompt_template_file")
        prompt_hash_key = "prompt_template_sha256"
        if not isinstance(prompt_file, str):
            # Explicit backward-compatible branch for pre-R22 provenance.
            prompt_file = prov.get("prompt_file")
            prompt_hash_key = "prompt_sha256"
        if isinstance(prompt_file, str):
            prompt = _find_prompt(prompt_file)
            if prompt is None:
                errors.append(f"{p.name}: provenance prompt missing: {prompt_file}")
            elif prov.get(prompt_hash_key) != _sha256_file(prompt):
                errors.append(f"{p.name}: {prompt_hash_key} mismatch")

        expected_dir_name = prov.get("expected_dir")
        if isinstance(expected_dir_name, str):
            expected_dir = EVAL / expected_dir_name
            if not expected_dir.is_dir():
                errors.append(
                    f"{p.name}: provenance expected dir missing: {expected_dir_name}")
            elif prov.get("expected_sha256") != _sha256_json_dir(expected_dir):
                errors.append(f"{p.name}: expected_sha256 mismatch")

        stored_commit = prov.get("git_commit")
        if isinstance(stored_commit, str) and stored_commit not in ("", "unknown", package_commit):
            # A tracked artifact can never embed the commit that will later
            # contain it (the commit hash covers the artifact's own bytes),
            # so after a deterministic re-scoring pass
            # (scripts/rescore_stored_results.py) the input identity is
            # pinned by the dataset/gold/expected/prompt SHA-256 checks
            # above -- which are hard errors -- and the commit label is
            # informational only.  Without a rescore_note the difference
            # still warns, because then nothing ties the stored scores to
            # the packaged inputs' revision.
            if not isinstance(prov.get("rescore_note"), str):
                warnings.append(
                    f"{p.name}: evaluation git_commit {stored_commit[:12]}... "
                    f"differs from package GIT_COMMIT {package_commit[:12]}...")

    # Every generated-SQL manifest must carry provenance as README claims.
    for directory in sorted((EVAL / "generated_sql").iterdir()):
        if not directory.is_dir():
            continue
        manifest = directory / "manifest.json"
        if not manifest.is_file():
            # Missing manifests are a hard error in generated_sql check too.
            continue
        data = _json(manifest)
        if not isinstance(data, dict) or not isinstance(data.get("provenance"), dict):
            errors.append(
                f"generated_sql/{directory.name}/manifest.json: missing provenance")

    if errors:
        raise VerifyError(
            f"provenance errors ({len(errors)})\n" + "\n".join(errors[:50]))
    return f"{n_blocks} evaluation provenance blocks match current input hashes", warnings


def _canonical_hash(prov: dict, key: str) -> str | None:
    aliases = {"prompt_template_sha256": "prompt_sha256"}
    v = prov.get(key)
    if v is None and key in aliases:
        v = prov.get(aliases[key])
    return v


def check_main_run_single_source() -> tuple[str, list[str]]:
    a_path = EVAL / "multiaxis_results.json"
    b_path = EVAL / "main_eval_with_sql.json"
    if not (a_path.is_file() and b_path.is_file()):
        return "no duplicate main-run pair to compare", []
    a, b = _json(a_path), _json(b_path)
    if not isinstance(a, dict) or not isinstance(b, dict):
        raise VerifyError("main result artifact is not a JSON object")
    pa, pb = a.get("provenance", {}), b.get("provenance", {})
    same_inputs = (
        a.get("model") == b.get("model")
        and isinstance(pa, dict) and isinstance(pb, dict)
        and all(_canonical_hash(pa, k) == _canonical_hash(pb, k) for k in
                ("dataset_sha256", "gold_sha256", "prompt_template_sha256"))
    )
    if not same_inputs:
        return "main result artifacts identify different inputs; no single-run equivalence asserted", []

    ar = {_qid(r): r for r in _result_rows(a) if _qid(r)}
    br = {_qid(r): r for r in _result_rows(b) if _qid(r)}
    if set(ar) != set(br):
        raise VerifyError(
            f"same-input main artifacts have different qid sets "
            f"({len(ar)} vs {len(br)})")
    sql_diff: list[str] = []
    score_diff: list[str] = []
    for qid in sorted(ar):
        asql, bsql = _sql_from_row(ar[qid]), _sql_from_row(br[qid])
        if asql is not None and bsql is not None and asql.strip() != bsql.strip():
            sql_diff.append(qid)
        ascore = ar[qid].get("recall", ar[qid].get("execution_recall"))
        bscore = br[qid].get("recall", br[qid].get("execution_recall"))
        if isinstance(ascore, (int, float)) and isinstance(bscore, (int, float)):
            if abs(float(ascore) - float(bscore)) > 1e-12:
                score_diff.append(qid)
    if sql_diff or score_diff:
        raise VerifyError(
            "same-input main result artifacts diverge\n"
            f"sql_diff={len(sql_diff)}: {sql_diff[:20]}\n"
            f"score_diff={len(score_diff)}: {score_diff[:20]}\n"
            "Choose one canonical inference run and regenerate all derivatives from it.")
    return f"duplicate main artifacts agree on {len(ar)} query outputs", []



def check_question_gold_contract() -> tuple[str, list[str]]:
    """Conservative static lint for common hidden gold constraints.

    This is intentionally not a complete natural-language semantic parser.
    It only rejects patterns that previously caused concrete benchmark
    inconsistencies: unstated top-N caps, unstated minimum group sizes,
    unstated stable/on-hull filters, and L12-only filters hidden behind
    A/B-site questions.
    """
    suites, _ = _canonical_dataset_rows()
    problems: list[str] = []
    checked = 0

    stable_question_re = re.compile(
        r"安定|凸包上|stable|on[- ]?hull|ハル上", re.IGNORECASE)
    l12_question_re = re.compile(r"L12|L1[₂2]", re.IGNORECASE)
    site_question_re = re.compile(
        r"(?:A|B)[-\s]?site|(?:A|B)サイト", re.IGNORECASE)
    order_request_re = re.compile(
        r"順|上位|下位|トップ|ワースト|ランキング|ランク|昇順|降順|TOP\s*\d",
        re.IGNORECASE)

    for suite_name, rows in suites.items():
        spec = CANONICAL_SUITES[suite_name]
        for row in rows:
            qid = _qid(row)
            if qid is None:
                continue
            question = str(row.get("question", ""))
            gold_rel = row.get("gold_sql_path")
            if not isinstance(gold_rel, str):
                gold_rel = f"{spec['gold_dir']}/{qid}.sql"
            sql_path = EVAL / gold_rel
            sql = sql_path.read_text(encoding="utf-8")
            checked += 1

            # LIMIT 1 is often a harmless uniqueness guard. LIMIT 10000 is
            # the package-wide safety cap. Any other small cap must be
            # visible in the question as the same integer.
            for raw_n in re.findall(r"\bLIMIT\s+(\d+)\b", sql, re.IGNORECASE):
                n = int(raw_n)
                if n in (1, 10000) or n >= 10000:
                    continue
                if re.search(rf"(?<!\d){n}(?!\d)", question) is None:
                    problems.append(
                        f"{qid}: gold has LIMIT {n}, but the question does "
                        f"not state {n}")

            # Hidden minimum group-size thresholds are not allowed.
            for raw_n in re.findall(
                    r"\bHAVING\b[^\n;]*\bCOUNT\s*\([^;]*?"
                    r"\)\s*>=\s*(\d+)",
                    sql, re.IGNORECASE):
                n = int(raw_n)
                if n <= 1:
                    continue
                if re.search(rf"(?<!\d){n}(?!\d)", question) is None:
                    problems.append(
                        f"{qid}: gold has HAVING COUNT >= {n}, but the "
                        f"question does not state {n}")

            # Stable/on-hull filters must be explicit in the question.
            stable_filter = any(re.search(pat, sql, re.IGNORECASE) for pat in (
                r"(?:WHERE|AND)\s+[^\n;]*energy_above_hull\s*<=\s*0\.001",
                r"(?:WHERE|AND)\s+[^\n;]*\bis_stable\s*=\s*(?:TRUE|1)",
                r"(?:WHERE|AND)\s+[^\n;]*\bon_hull\s*=\s*(?:TRUE|1)",
            ))
            if stable_filter and not stable_question_re.search(question):
                problems.append(
                    f"{qid}: gold applies a stable/on-hull filter that is "
                    f"not stated in the question")

            # Site terminology does not imply an L12-only scope. If the gold
            # makes that restriction, the question must say L12/L1₂.
            l12_filter = bool(re.search(
                r"prototype\s*=\s*'L12'|strukturbericht\s*=\s*'L12'",
                sql, re.IGNORECASE))
            if (l12_filter and site_question_re.search(question)
                    and not l12_question_re.search(question)):
                problems.append(
                    f"{qid}: gold restricts an A/B-site question to L12, "
                    f"but the question does not state L12")

            # If the question explicitly asks for an ordered answer, the
            # expected result must be annotated semantic_ordered=true so the
            # canonical exact metric enforces the sequence.
            exp_rel = row.get("expected_result_path")
            if not isinstance(exp_rel, str):
                exp_rel = f"{spec['expected_dir']}/{qid}.json"
            expected = _json(EVAL / exp_rel)
            sem = (bool(expected.get("semantic_ordered"))
                   if isinstance(expected, dict) else False)
            if order_request_re.search(question) and not sem:
                problems.append(
                    f"{qid}: question requests an explicit order but "
                    f"expected semantic_ordered is false")

    if problems:
        raise VerifyError(
            f"question/gold contract lint found {len(problems)} issue(s)\n"
            + "\n".join(problems[:80]))
    return (
        f"{checked} canonical question/gold pairs passed hidden-constraint lint",
        [],
    )


def check_dependencies() -> tuple[str, list[str]]:
    required = ["psycopg", "sqlglot", "pytest"]
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        raise VerifyError(
            "missing runtime verification dependencies: "
            + ", ".join(missing)
            + " (install requirements-repro.txt)")
    return "verification dependencies importable", []


def check_db_environment(allow_skip: bool) -> tuple[str, list[str]]:
    required = ["L12_DSN", "TRANSFER_DSN", "OBF_TRANSFER_DSN", "MP_DSN"]
    missing = [name for name in required if not os.environ.get(name)]
    if missing and not allow_skip:
        raise VerifyError(
            "formal verification requires all four DB DSNs: missing "
            + ", ".join(missing))
    warnings = [f"database suite may be skipped because {name} is unset" for name in missing]
    return "all DB-suite DSNs set" if not missing else f"{len(missing)} DB DSN(s) unset", warnings


def _run_subprocess(
    name: str,
    argv: list[str],
    *,
    extra_env: dict[str, str] | None = None,
    timeout: int = 1800,
) -> tuple[str, list[str], str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run(
        argv,
        cwd=PROJECT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    output = proc.stdout or ""
    if proc.returncode != 0:
        tail = "\n".join(output.splitlines()[-25:])
        raise VerifyError(
            f"exit={proc.returncode}; command={' '.join(argv)}\n{tail}")
    return f"exit=0: {' '.join(argv)}", [], output


def _postgres_env_from_l12_dsn() -> dict[str, str]:
    """Populate POSTGRES_* for legacy audit scripts from L12_DSN when
    possible, without overriding explicit POSTGRES_* variables."""
    env: dict[str, str] = {}
    dsn = os.environ.get("L12_DSN", "")
    if not dsn.startswith(("postgresql://", "postgres://")):
        return env
    try:
        u = urlparse(dsn)
        values = {
            "POSTGRES_HOST": u.hostname,
            "POSTGRES_PORT": str(u.port) if u.port else None,
            "POSTGRES_USER": unquote(u.username) if u.username else None,
            "POSTGRES_PASSWORD": unquote(u.password) if u.password else None,
        }
        for key, value in values.items():
            if value and not os.environ.get(key):
                env[key] = value
    except Exception:
        pass
    return env


def check_scoring_audit() -> tuple[str, list[str], str]:
    with tempfile.TemporaryDirectory(prefix="l12_verify_scoring_") as td:
        out = Path(td) / "scoring_audit.json"
        env = _postgres_env_from_l12_dsn()
        # Defense in depth: even though this is a verifier, the historical
        # scoring script executes generated SQL.  Force server-side read-only
        # transactions without modifying the package file.
        env["PGOPTIONS"] = (
            (os.environ.get("PGOPTIONS", "") + " "
             "-c default_transaction_read_only=on "
             "-c statement_timeout=30000").strip()
        )
        summary, warnings, output = _run_subprocess(
            "scoring_audit",
            [sys.executable, str(SCRIPTS / "audit_scoring.py"),
             "--datasets", "main", "independent", "transfer",
             "transfer_obfuscated", "--out", str(out)],
            extra_env=env,
            timeout=1800,
        )
        if not out.is_file():
            raise VerifyError("audit_scoring.py exited 0 but produced no report")
        data = _json(out)
        required = {"main", "independent", "transfer", "transfer_obfuscated"}
        missing = sorted(required - set(data)) if isinstance(data, dict) else sorted(required)
        if missing:
            raise VerifyError(f"scoring audit skipped required datasets: {missing}")
        bad = []
        for key in sorted(required):
            n = data[key].get("selfcheck", {}).get("n_mismatch")
            if n != 0:
                bad.append(f"{key}: n_mismatch={n}")
        if bad:
            raise VerifyError("scoring self-check mismatch: " + "; ".join(bad))
        return "4 scoring datasets reproduced with selfcheck n_mismatch=0", warnings, output


def _execute_callable(
    name: str,
    fn: Callable[[], tuple[str, list[str]]],
) -> CheckResult:
    started = time.monotonic()
    try:
        summary, warnings = fn()
        status = "WARN" if warnings else "PASS"
        return CheckResult(name, status, time.monotonic() - started,
                           summary, warnings)
    except Exception as exc:  # noqa: BLE001
        return CheckResult(name, "FAIL", time.monotonic() - started,
                           str(exc).splitlines()[0], str(exc).splitlines())


def _execute_subprocess(
    name: str,
    argv: list[str],
    *,
    extra_env: dict[str, str] | None = None,
    timeout: int = 1800,
) -> CheckResult:
    started = time.monotonic()
    try:
        summary, warnings, output = _run_subprocess(
            name, argv, extra_env=extra_env, timeout=timeout)
        return CheckResult(
            name, "WARN" if warnings else "PASS",
            time.monotonic() - started, summary, warnings, output)
    except Exception as exc:  # noqa: BLE001
        return CheckResult(name, "FAIL", time.monotonic() - started,
                           str(exc).splitlines()[0], str(exc).splitlines())


def _print_result(result: CheckResult, verbose: bool) -> None:
    print(f"[{result.status:4s}] {result.name:30s} "
          f"{result.seconds:7.2f}s  {result.summary}")
    if result.status in {"FAIL", "WARN"}:
        for detail in result.details[:20]:
            print(f"       {detail}")
    if verbose and result.output:
        for line in result.output.rstrip().splitlines():
            print(f"       | {line}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--static-only", action="store_true",
                    help="run only package/artifact consistency checks")
    ap.add_argument("--no-pytest", action="store_true",
                    help="skip FULL_DB_TEST pytest in full mode")
    ap.add_argument("--no-scoring", action="store_true",
                    help="skip database-backed scoring self-check")
    ap.add_argument("--allow-db-skip", action="store_true",
                    help="development only: allow missing DB-suite DSNs")
    ap.add_argument("--warnings-as-errors", action="store_true")
    ap.add_argument("--fail-fast", action="store_true")
    ap.add_argument("--verbose", action="store_true",
                    help="show successful child-script output")
    ap.add_argument("--json-report", type=Path)
    args = ap.parse_args()

    results: list[CheckResult] = []

    static_checks: list[tuple[str, Callable[[], tuple[str, list[str]]]]] = [
        ("package_structure", check_package_structure),
        ("python_syntax", check_python_syntax),
        ("canonical_catalog", check_canonical_catalog),
        ("expected_json_schema", check_expected_json_schema),
        ("question_gold_contract", check_question_gold_contract),
        ("generated_sql_consistency", check_generated_sql_consistency),
        ("provenance", check_provenance),
        ("main_run_single_source", check_main_run_single_source),
    ]

    print("L12 SQL PACKAGE VERIFICATION")
    print(f"project={PROJECT}")
    print(f"mode={'static-only' if args.static_only else 'formal-full'}")
    print()

    for name, fn in static_checks:
        result = _execute_callable(name, fn)
        results.append(result)
        _print_result(result, args.verbose)
        if args.fail_fast and result.status == "FAIL":
            break

    static_failed = any(r.status == "FAIL" for r in results)

    if not args.static_only and not (args.fail_fast and static_failed):
        for name, fn in [
            ("dependencies", check_dependencies),
            ("db_environment", lambda: check_db_environment(args.allow_db_skip)),
        ]:
            result = _execute_callable(name, fn)
            results.append(result)
            _print_result(result, args.verbose)
            if args.fail_fast and result.status == "FAIL":
                break

        prereq_failed = any(
            r.status == "FAIL" and r.name in {"dependencies", "db_environment"}
            for r in results
        )
        if not prereq_failed:
            py = sys.executable
            child_checks: list[tuple[str, list[str]]] = [
                ("gold_verification",
                 [py, str(SCRIPTS / "run_gold_verification.py")]
                 + (["--allow-skip"] if args.allow_db_skip else [])),
                ("expected_results_crosscheck",
                 [py, str(SCRIPTS / "check_expected_results.py")]
                 + (["--allow-missing-transfer"] if args.allow_db_skip else [])),
                ("order_totality",
                 [py, str(SCRIPTS / "audit_order_totality.py")]),
                ("semantic_audit",
                 [py, str(SCRIPTS / "audit_semantics.py")]),
                ("vocabulary_audit",
                 [py, str(SCRIPTS / "audit_vocabulary.py")]
                 + (["--allow-skip"] if args.allow_db_skip else [])),
                ("safety_cases",
                 [py, str(SCRIPTS / "verify_safety_cases.py")]),
            ]
            for name, argv in child_checks:
                result = _execute_subprocess(name, argv)
                results.append(result)
                _print_result(result, args.verbose)
                if args.fail_fast and result.status == "FAIL":
                    break

            if not args.no_scoring:
                # Do not pretend scoring is reproducible when the static
                # generated-SQL contract is already broken.
                gen_ok = next(
                    (r.status != "FAIL" for r in results
                     if r.name == "generated_sql_consistency"), False)
                if gen_ok:
                    started = time.monotonic()
                    try:
                        summary, warnings, output = check_scoring_audit()
                        result = CheckResult(
                            "scoring_selfcheck",
                            "WARN" if warnings else "PASS",
                            time.monotonic() - started,
                            summary, warnings, output)
                    except Exception as exc:  # noqa: BLE001
                        result = CheckResult(
                            "scoring_selfcheck", "FAIL",
                            time.monotonic() - started,
                            str(exc).splitlines()[0],
                            str(exc).splitlines())
                else:
                    result = CheckResult(
                        "scoring_selfcheck", "FAIL", 0.0,
                        "blocked: generated_sql_consistency failed",
                        ["Fix generated-SQL/manifest completeness before "
                         "re-running the scoring self-check."])
                results.append(result)
                _print_result(result, args.verbose)

            if not args.no_pytest:
                pytest_env = {"FULL_DB_TEST": "1"}
                result = _execute_subprocess(
                    "pytest_full_db",
                    [py, "-m", "pytest", "-q", "-ra"],
                    extra_env=pytest_env,
                    timeout=1800,
                )
                results.append(result)
                _print_result(result, args.verbose)

    n_pass = sum(r.status == "PASS" for r in results)
    n_warn = sum(r.status == "WARN" for r in results)
    n_fail = sum(r.status == "FAIL" for r in results)
    n_skip = sum(r.status == "SKIP" for r in results)

    print()
    print("=" * 78)
    print(f"PASS={n_pass} WARN={n_warn} FAIL={n_fail} SKIP={n_skip}")
    failed = n_fail > 0 or (args.warnings_as_errors and n_warn > 0)
    print("VERIFICATION " + ("FAILED" if failed else "PASSED"))

    if args.json_report:
        report = {
            "project": str(PROJECT),
            "mode": "static-only" if args.static_only else "formal-full",
            "passed": not failed,
            "summary": {
                "pass": n_pass, "warn": n_warn, "fail": n_fail, "skip": n_skip,
            },
            "checks": [asdict(r) for r in results],
        }
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"report={args.json_report}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
