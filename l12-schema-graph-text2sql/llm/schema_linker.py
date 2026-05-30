"""Link extracted conditions to required tables, columns, and JOIN paths."""
from __future__ import annotations

from typing import Any

from llm.condition_mapper import map_conditions

CONDITION_TABLE_MAP: dict[str, list[str]] = {
    "prototype": ["structure"],
    "contains_elements": ["composition"],
    "stability": ["phase_stability"],
    "lattice_reference": ["structure"],
    "lattice_constant": ["structure"],
    "formation_energy": ["phase_stability"],
    "bulk_modulus": ["calculated_property", "calculation"],
    "shear_modulus": ["calculated_property", "calculation"],
    "formula": ["material_entry"],
    "chemical_system": ["material_entry"],
}

CONDITION_COLUMN_MAP: dict[str, list[str]] = {
    "prototype": ["structure.prototype", "structure.strukturbericht"],
    "contains_elements": ["composition.element"],
    "stability": [
        "phase_stability.energy_above_hull",
    ],
    "lattice_reference": ["structure.lattice_a"],
    "lattice_constant": ["structure.lattice_a"],
    "formation_energy": ["phase_stability.formation_energy_per_atom"],
    "bulk_modulus": ["calculated_property.property_name", "calculated_property.value"],
    "shear_modulus": ["calculated_property.property_name", "calculated_property.value"],
    "formula": ["material_entry.formula"],
    "chemical_system": ["material_entry.chemical_system"],
}

BASE_TABLE = "material_entry"
BASE_COLUMNS = ["material_entry.entry_id", "material_entry.formula"]


def link_schema(conditions: dict[str, Any]) -> dict[str, Any]:
    """Given extracted conditions, return required tables, columns, and mapped SQL fragments."""
    required_tables: set[str] = {BASE_TABLE}
    required_columns: set[str] = set(BASE_COLUMNS)

    for key in conditions:
        if key in CONDITION_TABLE_MAP:
            required_tables.update(CONDITION_TABLE_MAP[key])
        if key in CONDITION_COLUMN_MAP:
            required_columns.update(CONDITION_COLUMN_MAP[key])

    if "sort_by" in conditions:
        col = conditions["sort_by"]
        if "." in col:
            table = col.split(".")[0]
            required_tables.add(table)
            required_columns.add(col)

    if "properties" in conditions:
        for prop in conditions["properties"]:
            if "." in prop:
                table = prop.split(".")[0]
                required_tables.add(table)
                required_columns.add(prop)

    mapped_fragments = map_conditions(conditions)

    return {
        "required_tables": sorted(required_tables),
        "required_columns": sorted(required_columns),
        "sql_fragments": mapped_fragments,
    }
