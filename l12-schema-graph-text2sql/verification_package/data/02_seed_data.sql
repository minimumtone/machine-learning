-- Seed data loading for 7 core tables
-- Uses server-side COPY (files mounted at /seed/ in container)

COPY material_entry(entry_id, source_db, source_material_id, formula, reduced_formula, chemical_system, number_of_elements)
FROM '/seed/seed_l12_entries.csv' WITH (FORMAT csv, HEADER true);

COPY composition(composition_id, entry_id, element, atomic_fraction, site_label)
FROM '/seed/seed_composition.csv' WITH (FORMAT csv, HEADER true);

COPY structure(structure_id, entry_id, prototype, strukturbericht, formula_type, space_group_number, crystal_system, lattice_a, lattice_b, lattice_c, volume_per_atom, space_group)
FROM '/seed/seed_structure.csv' WITH (FORMAT csv, HEADER true);

COPY phase_stability(stability_id, entry_id, formation_energy_per_atom, energy_above_hull, is_stable, band_gap)
FROM '/seed/seed_phase_stability.csv' WITH (FORMAT csv, HEADER true);

COPY calculation(calculation_id, entry_id, method, functional, calculation_type)
FROM '/seed/seed_calculation.csv' WITH (FORMAT csv, HEADER true);

COPY calculated_property(property_id, calculation_id, property_name, value, unit)
FROM '/seed/seed_properties.csv' WITH (FORMAT csv, HEADER true);

COPY prototype_definition(prototype_id, prototype_name, strukturbericht, formula_type, description)
FROM '/seed/seed_prototype_definition.csv' WITH (FORMAT csv, HEADER true);
