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
     ["formula", "band_gap_value"]),
    # Elastic tensor
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
    allowed_lower = {c.lower() for c in allowed_columns}
    available: set[str] = set()
    for col in matched_cols:
        # Normalize "table.column" and "expr AS alias" forms to a base token.
        base = col.split(" as ", 1)[0].strip()
        base = base.rsplit(".", 1)[-1].strip()
        if base in ("*", "formula", "element", "atomic_fraction", "lattice_a",
                    "lattice_b", "lattice_c", "band_gap", "volume", "energy_above_hull",
                    "energy_per_atom", "is_stable", "crystal_system", "spacegroup_symbol",
                    "chemsys", "nelements", "formula", "atomic_number", "name"):
            available.add(col)
        elif f"mp_entries.{base}" in allowed_lower or f"mp_element_ratios.{base}" in allowed_lower:
            available.add(col)
        # Complex expressions are kept only if a single keyword matches a column.

    if not available:
        return "Return only the columns directly relevant to answering the question. Do NOT add entry_id, reduced_formula, or other auxiliary columns unless explicitly requested."

    # Always include formula as identifier
    available.add("formula")

    # Build instruction
    col_list = ", ".join(sorted(available))
    return (
        f"Focus the SELECT clause on these relevant columns: {col_list}. "
        "Do NOT add unnecessary columns like entry_id, reduced_formula, chemical_system, "
        "prototype, or strukturbericht unless the question specifically asks for them."
    )
