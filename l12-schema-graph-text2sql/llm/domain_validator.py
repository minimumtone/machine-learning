"""Materials-domain post-execution validation for SQL results.

Applies domain-specific rules to detect semantically incorrect SQL results
that pass syntactic/execution checks but violate materials science constraints.
This catches errors that SQLGuard and execution checks cannot detect.

Domain rules:
- L1₂ ↔ A3B formula type consistency
- B2 ↔ AB formula type consistency
- γ' phase → Ni/Co base required
- Lattice constant physical bounds (2.5–6.0 Å for intermetallics)
- Energy above hull: stable ≤ 0.001, metastable ≤ 0.05
- Site label validity (A-site, B-site only)
"""
from __future__ import annotations

from typing import Any


# ── Prototype ↔ formula_type consistency ──

_PROTO_FORMULA_MAP: dict[str, str] = {
    "L12": "A3B",
    "BiF3": "AB3",
    "B2": "AB",
    "NaCl": "AB",
    "NiAs": "AB",
}

# γ' phase base elements (A-site dominant in Ni/Co-based superalloys)
_GAMMA_PRIME_ELEMENTS = {"Ni", "Co"}

# Physical bounds for lattice constants in intermetallics (Å)
_LATTICE_A_MIN = 2.5
_LATTICE_A_MAX = 6.0

# Energy above hull thresholds (eV/atom)
_EHULL_STABLE = 0.001
_EHULL_METASTABLE = 0.05

# Valid site labels
_VALID_SITE_LABELS = {"A-site", "B-site"}


def validate_result_rows(
    rows: list[list[Any]],
    columns: list[str],
    conditions: dict[str, Any],
    query: str = "",
) -> dict[str, Any]:
    """Validate SQL result rows against materials domain constraints.

    Parameters
    ----------
    rows : list[list]
        Result rows from SQL execution.
    columns : list[str]
        Column names corresponding to row values.
    conditions : dict
        Extracted conditions from entity_extractor.
    query : str
        Original user query (for context-aware rules).

    Returns
    -------
    dict with:
      - valid: bool (True if no domain violations found)
      - warnings: list[dict] with keys: rule, severity, message, affected_rows
      - score_penalty: int (0-30, deducted from n-best score)
    """
    if not rows or not columns:
        return {"valid": True, "warnings": [], "score_penalty": 0}

    col_lower = [c.lower() for c in columns]
    col_idx = {c: i for i, c in enumerate(col_lower)}
    warnings: list[dict[str, Any]] = []

    # Rule 1: Prototype ↔ formula_type consistency
    _check_proto_formula(rows, col_idx, conditions, warnings)

    # Rule 2: Lattice constant physical bounds
    _check_lattice_bounds(rows, col_idx, warnings)

    # Rule 3: Energy above hull consistency with stability claims
    _check_ehull_stability(rows, col_idx, conditions, warnings)

    # Rule 4: γ' phase element requirements
    _check_gamma_prime(rows, col_idx, conditions, query, warnings)

    # Rule 5: Site label validity
    _check_site_labels(rows, col_idx, warnings)

    score_penalty = sum(w.get("penalty", 0) for w in warnings)
    return {
        "valid": len(warnings) == 0,
        "warnings": warnings,
        "score_penalty": min(score_penalty, 30),
    }


def _check_proto_formula(
    rows: list[list[Any]],
    col_idx: dict[str, int],
    conditions: dict[str, Any],
    warnings: list[dict[str, Any]],
) -> None:
    """Check prototype ↔ formula_type consistency."""
    proto_idx = col_idx.get("prototype")
    ft_idx = col_idx.get("formula_type")
    if proto_idx is None or ft_idx is None:
        return

    bad_rows = []
    for i, row in enumerate(rows):
        proto = str(row[proto_idx]) if row[proto_idx] else ""
        ft = str(row[ft_idx]) if row[ft_idx] else ""
        expected_ft = _PROTO_FORMULA_MAP.get(proto, "")
        if expected_ft and ft and ft != expected_ft:
            bad_rows.append(i)

    if bad_rows:
        warnings.append({
            "rule": "proto_formula_mismatch",
            "severity": "error",
            "message": (
                f"Prototype-formula_type mismatch in {len(bad_rows)} rows. "
                f"E.g., L12 should be A3B, B2 should be AB."
            ),
            "affected_rows": bad_rows[:5],
            "penalty": 10,
        })


def _check_lattice_bounds(
    rows: list[list[Any]],
    col_idx: dict[str, int],
    warnings: list[dict[str, Any]],
) -> None:
    """Check lattice constants are within physical bounds."""
    lat_idx = col_idx.get("lattice_a")
    if lat_idx is None:
        return

    out_of_range = []
    for i, row in enumerate(rows):
        val = row[lat_idx]
        if val is None:
            continue
        try:
            lat = float(val)
            if lat < _LATTICE_A_MIN or lat > _LATTICE_A_MAX:
                out_of_range.append(i)
        except (ValueError, TypeError):
            pass

    if out_of_range:
        warnings.append({
            "rule": "lattice_out_of_range",
            "severity": "warning",
            "message": (
                f"Lattice constant out of physical range "
                f"({_LATTICE_A_MIN}–{_LATTICE_A_MAX} Å) in {len(out_of_range)} rows."
            ),
            "affected_rows": out_of_range[:5],
            "penalty": 5,
        })


def _check_ehull_stability(
    rows: list[list[Any]],
    col_idx: dict[str, int],
    conditions: dict[str, Any],
    warnings: list[dict[str, Any]],
) -> None:
    """Check energy_above_hull consistency with stability conditions."""
    ehull_idx = col_idx.get("energy_above_hull")
    stable_idx = col_idx.get("is_stable")
    if ehull_idx is None:
        return

    stability = conditions.get("stability", "")
    inconsistent = []

    for i, row in enumerate(rows):
        ehull = row[ehull_idx]
        if ehull is None:
            continue
        try:
            ehull_val = float(ehull)
        except (ValueError, TypeError):
            continue

        if stable_idx is not None:
            is_stable = row[stable_idx]
            if is_stable and ehull_val > _EHULL_STABLE * 10:
                inconsistent.append(i)
            elif not is_stable and ehull_val <= 0:
                inconsistent.append(i)

        if stability == "stable" and ehull_val > _EHULL_STABLE * 10:
            inconsistent.append(i)
        elif stability == "metastable" and ehull_val > _EHULL_METASTABLE * 10:
            inconsistent.append(i)

    if inconsistent:
        unique = sorted(set(inconsistent))
        warnings.append({
            "rule": "ehull_stability_mismatch",
            "severity": "warning",
            "message": (
                f"Energy above hull inconsistent with stability claim "
                f"in {len(unique)} rows."
            ),
            "affected_rows": unique[:5],
            "penalty": 5,
        })


def _check_gamma_prime(
    rows: list[list[Any]],
    col_idx: dict[str, int],
    conditions: dict[str, Any],
    query: str,
    warnings: list[dict[str, Any]],
) -> None:
    """Check γ' phase candidates contain Ni or Co base elements."""
    q_lower = query.lower()
    is_gamma_prime = any(
        kw in q_lower
        for kw in ["γ'", "gamma prime", "γ′", "ガンマプライム", "γ'相"]
    )
    if not is_gamma_prime:
        return

    formula_idx = col_idx.get("formula")
    if formula_idx is None:
        return

    no_base = []
    for i, row in enumerate(rows):
        formula = str(row[formula_idx]) if row[formula_idx] else ""
        has_base = any(elem in formula for elem in _GAMMA_PRIME_ELEMENTS)
        if formula and not has_base:
            no_base.append(i)

    if no_base and len(no_base) > len(rows) * 0.5:
        warnings.append({
            "rule": "gamma_prime_no_base",
            "severity": "warning",
            "message": (
                f"γ' phase query but {len(no_base)}/{len(rows)} results "
                f"lack Ni/Co base elements."
            ),
            "affected_rows": no_base[:5],
            "penalty": 10,
        })


def _check_site_labels(
    rows: list[list[Any]],
    col_idx: dict[str, int],
    warnings: list[dict[str, Any]],
) -> None:
    """Check site labels are valid (A-site or B-site)."""
    site_idx = col_idx.get("site_label")
    if site_idx is None:
        return

    invalid = []
    for i, row in enumerate(rows):
        site = str(row[site_idx]).strip() if row[site_idx] else ""
        if site and site not in _VALID_SITE_LABELS:
            invalid.append(i)

    if invalid:
        warnings.append({
            "rule": "invalid_site_label",
            "severity": "error",
            "message": (
                f"Invalid site labels in {len(invalid)} rows. "
                f"Expected: {_VALID_SITE_LABELS}"
            ),
            "affected_rows": invalid[:5],
            "penalty": 10,
        })
