"""Output schema specifier: determine which columns the SQL should return.

Analyzes the question and extracted conditions to generate a hint
for the LLM about what columns to include in the SELECT clause.
This prevents the LLM from adding unnecessary columns (entry_id,
reduced_formula, etc.) that cause result set mismatches.
"""
from __future__ import annotations

import re
from typing import Any


# Maps question keywords to relevant column hints
_COLUMN_HINTS: list[tuple[re.Pattern[str], list[str]]] = [
    # Stability-related
    (re.compile(r"安定|stable|energy.above.hull|Ehull", re.I),
     ["formula", "energy_above_hull", "is_stable"]),
    # Formation energy
    (re.compile(r"形成エネルギー|formation.energy", re.I),
     ["formula", "formation_energy_per_atom"]),
    # Lattice constant
    (re.compile(r"格子定数|lattice.constant|lattice_a", re.I),
     ["formula", "lattice_a"]),
    # Bulk modulus
    (re.compile(r"バルクモジュラス|bulk.modulus|体積弾性率", re.I),
     ["formula", "value AS bulk_modulus"]),
    # Band gap
    (re.compile(r"バンドギャップ|band.gap|bandgap", re.I),
     ["formula", "band_gap"]),
    # Elastic tensor (C11/C12/C44 are output aliases over calculated_property.value)
    (re.compile(r"弾性|elastic|C11|C12|C44", re.I),
     ["formula", "C11", "C12", "C44"]),
    # Volume
    (re.compile(r"体積|volume", re.I),
     ["formula", "volume_per_atom"]),
    # Site information
    (re.compile(r"サイト|site.label|A.site|B.site", re.I),
     ["element", "site_label"]),
    # Element composition
    (re.compile(r"元素|組成|composition|含む|containing", re.I),
     ["formula", "element", "atomic_fraction"]),
    # Lattice mismatch (Ni3Al reference)
    (re.compile(r"Ni3Al|ミスマッチ|mismatch|格子整合|3\.57", re.I),
     ["formula", "lattice_a", "ABS(lattice_a - 3.57) AS lattice_diff"]),
]


def specify_output_schema(
    query: str,
    conditions: dict[str, Any],
    allowed_columns: list[str],
) -> str:
    """Generate a SELECT column hint for the LLM prompt.

    Returns a string instruction about which columns to include,
    or empty string if no specific hint can be determined.
    """
    # Collect column hints from question keywords
    matched_cols: set[str] = set()
    for pat, cols in _COLUMN_HINTS:
        if pat.search(query):
            matched_cols.update(cols)

    if not matched_cols:
        return "Return only the columns directly relevant to answering the question. Do NOT add entry_id, reduced_formula, or other auxiliary columns unless explicitly requested."

    # Keep only hints that reference columns actually present in this schema.
    # allowed_columns arrive as "table.column"; the base token is the column name.
    allowed_bases = {
        c.rsplit(".", 1)[-1].strip().lower() for c in allowed_columns
    }

    # Pivoted property aliases whose SELECT item is an aggregate over "value"
    # (e.g. MAX(CASE WHEN cp.property_name = 'C11' THEN cp.value END) AS C11).
    _PIVOT_ALIASES = {"c11", "c12", "c44"}

    def _hint_is_allowed(hint: str) -> bool:
        """Return True if *hint* is supported by this schema."""
        # Separate the expression from its output alias (case-insensitive AS).
        expr = re.split(r"\s+as\s+", hint, flags=re.I, maxsplit=1)[0].strip()
        if not expr or expr == "*":
            return False
        # Find all identifier tokens in the expression.
        tokens = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", expr.lower()))
        if tokens & allowed_bases:
            return True
        # Bare aliases such as C11 are acceptable when a "value" column exists
        # (calculated_property.value in the L1_2 schema), because few-shot examples
        # show how to pivot them with CASE/MAX.
        single = expr.lower()
        if single in _PIVOT_ALIASES and "value" in allowed_bases:
            return True
        return False

    available: set[str] = {col for col in matched_cols if _hint_is_allowed(col)}

    if not available:
        return "Return only the columns directly relevant to answering the question. Do NOT add entry_id, reduced_formula, or other auxiliary columns unless explicitly requested."

    # Always include a formula identifier if the schema has one.
    if "formula" in allowed_bases:
        available.add("formula")

    # Build instruction
    col_list = ", ".join(sorted(available))
    return (
        f"Focus the SELECT clause on these relevant columns: {col_list}. "
        "Do NOT add unnecessary columns like entry_id, reduced_formula, chemical_system, "
        "prototype, or strukturbericht unless the question specifically asks for them."
    )
