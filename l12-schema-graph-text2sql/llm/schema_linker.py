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
    "lattice_c": ["structure"],
    "volume": ["structure"],
    "crystal_system": ["structure"],
    "space_group": ["structure"],
    "formation_energy": ["phase_stability"],
    "band_gap": ["phase_stability"],
    "bulk_modulus": ["calculation", "calculated_property"],
    "shear_modulus": ["calculation", "calculated_property"],
    "youngs_modulus": ["calculation", "calculated_property"],
    "poisson_ratio": ["elastic_tensor"],
    "elastic_stability": ["elastic_tensor"],
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
    "miller_index": ["surface_energy"],
    "surface_reconstruction": ["surface_energy"],
    "grain_boundary_energy": ["grain_boundary"],
    "vacancy_formation": ["material_defect", "defect_type"],
    "interstitial": ["material_defect", "defect_type"],
    "defect": ["material_defect", "defect_type"],
    "dopant": ["material_defect", "element"],
    "formula": ["material_entry"],
    "chemical_system": ["material_entry"],
    "number_of_elements": ["material_entry"],
    "source_db": ["material_entry"],
    "atomic_number": ["composition", "element"],
    "electronegativity": ["composition", "element"],
    "element_property": ["composition", "element"],
    "synthesis": ["material_synthesis", "synthesis_method"],
    "ball_milling": ["material_synthesis", "synthesis_method"],
    "arc_melting": ["material_synthesis", "synthesis_method"],
    "experimental": ["material_synthesis"],
    "doi": ["material_reference", "literature_reference"],
    "literature": ["material_reference", "literature_reference"],
    "reference": ["material_reference", "literature_reference"],
    "application": ["material_application", "application_domain"],
    "functional": ["calculation"],
    "site_label": ["composition"],
    "calculation_method": ["calculation"],
    "phase_diagram": ["phase_diagram_entry"],
    "alloy_system": ["material_alloy_system", "alloy_system"],
}

CONDITION_COLUMN_MAP: dict[str, list[str]] = {
    "prototype": ["structure.prototype", "structure.strukturbericht"],
    "contains_elements": ["composition.element", "composition.atomic_fraction"],
    "site_label": ["composition.site_label"],
    "stability": [
        "phase_stability.energy_above_hull",
        "phase_stability.is_stable",
    ],
    "lattice_reference": ["structure.lattice_a"],
    "lattice_constant": ["structure.lattice_a", "structure.lattice_b", "structure.lattice_c"],
    "lattice_c": ["structure.lattice_c"],
    "volume": ["structure.volume_per_atom"],
    "crystal_system": ["structure.crystal_system"],
    "space_group": ["structure.space_group_number", "structure.space_group"],
    "formation_energy": ["phase_stability.formation_energy_per_atom"],
    "band_gap": ["phase_stability.band_gap"],
    "bulk_modulus": ["calculated_property.property_name", "calculated_property.value"],
    "shear_modulus": ["calculated_property.property_name", "calculated_property.value"],
    "youngs_modulus": ["calculated_property.property_name", "calculated_property.value"],
    "poisson_ratio": ["elastic_tensor.poisson_ratio"],
    "elastic_stability": ["elastic_tensor.is_stable"],
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
    "surface_energy": ["surface_energy.surface_energy_j_m2", "surface_energy.miller_index"],
    "work_function": ["surface_energy.work_function"],
    "miller_index": ["surface_energy.miller_index"],
    "surface_reconstruction": ["surface_energy.is_reconstructed"],
    "grain_boundary_energy": ["grain_boundary.gb_energy_j_m2"],
    "vacancy_formation": ["material_defect.formation_energy", "defect_type.category"],
    "interstitial": ["material_defect.formation_energy", "defect_type.category"],
    "defect": ["material_defect.formation_energy", "defect_type.defect_name", "defect_type.category"],
    "dopant": ["material_defect.dopant_element_id", "element.symbol"],
    "formula": ["material_entry.formula"],
    "chemical_system": ["material_entry.chemical_system"],
    "number_of_elements": ["material_entry.number_of_elements"],
    "source_db": ["material_entry.source_db"],
    "atomic_number": ["element.atomic_number", "element.symbol", "composition.element"],
    "electronegativity": ["element.electronegativity", "element.symbol"],
    "element_property": ["element_property.property_name", "element_property.value"],
    "synthesis": ["synthesis_method.method_name", "material_synthesis.success"],
    "ball_milling": ["synthesis_method.method_name"],
    "arc_melting": ["synthesis_method.method_name"],
    "experimental": ["material_synthesis.success"],
    "doi": ["literature_reference.doi", "literature_reference.title"],
    "literature": ["literature_reference.doi", "literature_reference.title", "literature_reference.year"],
    "reference": ["literature_reference.doi", "literature_reference.title"],
    "application": ["application_domain.domain_name"],
    "functional": ["calculation.functional"],
    "calculation_method": ["calculation.method"],
    "phase_diagram": ["phase_diagram_entry.is_on_hull", "phase_diagram_entry.hull_distance"],
    "alloy_system": ["alloy_system.system_name", "alloy_system.num_components"],
}

# Multi-hop JOIN definitions (table -> prerequisite tables + join clause)
MULTI_HOP_JOINS: dict[str, list[str]] = {
    "element": ["composition"],
    "element_property": ["composition", "element"],
    "calculated_property": ["calculation"],
    "synthesis_method": ["material_synthesis"],
    "defect_type": ["material_defect"],
    "literature_reference": ["material_reference"],
    "application_domain": ["material_application"],
    "measured_property": ["experimental_measurement"],
    "alloy_system": ["material_alloy_system"],
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

    # Resolve multi-hop dependencies (after all tables are collected)
    added = True
    while added:
        added = False
        for table in list(required_tables):
            if table in MULTI_HOP_JOINS:
                for prereq in MULTI_HOP_JOINS[table]:
                    if prereq not in required_tables:
                        required_tables.add(prereq)
                        added = True

    mapped_fragments = map_conditions(conditions)

    return {
        "required_tables": sorted(required_tables),
        "required_columns": sorted(required_columns),
        "sql_fragments": mapped_fragments,
    }
