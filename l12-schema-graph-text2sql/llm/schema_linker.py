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
    "bulk_modulus": ["elastic_tensor"],
    "shear_modulus": ["elastic_tensor"],
    "youngs_modulus": ["elastic_tensor"],
    "poisson_ratio": ["elastic_tensor"],
    "total_magnetization": ["magnetic_property"],
    "magnetic_ordering": ["magnetic_property"],
    "curie_temperature": ["magnetic_property"],
    "magnetic_anisotropy": ["magnetic_property"],
    "debye_temperature": ["thermal_property"],
    "thermal_conductivity": ["thermal_property"],
    "gruneisen_parameter": ["thermal_property"],
    "dos_at_fermi": ["density_of_states"],
    "is_metallic": ["density_of_states"],
    "spin_polarized": ["density_of_states"],
    "direct_gap": ["band_structure"],
    "surface_energy": ["surface_energy"],
    "work_function": ["surface_energy"],
    "grain_boundary_energy": ["grain_boundary"],
    "vacancy_formation": ["material_defect", "defect_type"],
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
    "bulk_modulus": ["elastic_tensor.bulk_modulus_vrh"],
    "shear_modulus": ["elastic_tensor.shear_modulus_vrh"],
    "youngs_modulus": ["elastic_tensor.youngs_modulus"],
    "poisson_ratio": ["elastic_tensor.poisson_ratio"],
    "total_magnetization": ["magnetic_property.total_magnetization"],
    "magnetic_ordering": ["magnetic_property.magnetic_ordering"],
    "curie_temperature": ["magnetic_property.curie_temperature_k"],
    "magnetic_anisotropy": ["magnetic_property.magnetic_anisotropy_energy"],
    "debye_temperature": ["thermal_property.debye_temperature_k"],
    "thermal_conductivity": ["thermal_property.thermal_conductivity"],
    "gruneisen_parameter": ["thermal_property.gruneisen_parameter"],
    "dos_at_fermi": ["density_of_states.total_dos_at_fermi"],
    "is_metallic": ["density_of_states.is_metallic"],
    "spin_polarized": ["density_of_states.spin_polarized"],
    "direct_gap": ["band_structure.is_direct_gap"],
    "surface_energy": ["surface_energy.surface_energy_j_m2"],
    "work_function": ["surface_energy.work_function"],
    "grain_boundary_energy": ["grain_boundary.gb_energy_j_m2"],
    "vacancy_formation": ["material_defect.formation_energy"],
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
        if key.startswith("_"):
            continue
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

    # Numeric conditions add their tables/columns
    if "numeric_conditions" in conditions:
        for nc in conditions["numeric_conditions"]:
            col = nc["column"]
            if "." in col:
                table = col.split(".")[0]
                required_tables.add(table)
                required_columns.add(col)

    # Formula conditions
    if "formula" in conditions:
        required_columns.add("material_entry.formula")
        required_columns.add("material_entry.reduced_formula")

    mapped_fragments = map_conditions(conditions)

    return {
        "required_tables": sorted(required_tables),
        "required_columns": sorted(required_columns),
        "sql_fragments": mapped_fragments,
    }
