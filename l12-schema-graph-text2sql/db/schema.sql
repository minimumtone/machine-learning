CREATE TABLE material_entry (
    entry_id TEXT PRIMARY KEY,
    source_db TEXT,
    source_material_id TEXT,
    formula TEXT NOT NULL,
    reduced_formula TEXT,
    chemical_system TEXT,
    number_of_elements INTEGER
);

CREATE TABLE composition (
    composition_id TEXT PRIMARY KEY,
    entry_id TEXT NOT NULL REFERENCES material_entry(entry_id),
    element TEXT NOT NULL,
    atomic_fraction DOUBLE PRECISION NOT NULL,
    site_label TEXT
);

CREATE TABLE structure (
    structure_id TEXT PRIMARY KEY,
    entry_id TEXT NOT NULL REFERENCES material_entry(entry_id),
    prototype TEXT,
    strukturbericht TEXT,
    formula_type TEXT,
    space_group_number INTEGER,
    crystal_system TEXT,
    lattice_a DOUBLE PRECISION,
    lattice_b DOUBLE PRECISION,
    lattice_c DOUBLE PRECISION,
    volume_per_atom DOUBLE PRECISION
);

CREATE TABLE calculation (
    calculation_id TEXT PRIMARY KEY,
    entry_id TEXT NOT NULL REFERENCES material_entry(entry_id),
    method TEXT,
    functional TEXT,
    calculation_type TEXT
);

CREATE TABLE calculated_property (
    property_id TEXT PRIMARY KEY,
    calculation_id TEXT NOT NULL REFERENCES calculation(calculation_id),
    property_name TEXT NOT NULL,
    value DOUBLE PRECISION,
    unit TEXT
);

CREATE TABLE phase_stability (
    stability_id TEXT PRIMARY KEY,
    entry_id TEXT NOT NULL REFERENCES material_entry(entry_id),
    formation_energy_per_atom DOUBLE PRECISION,
    energy_above_hull DOUBLE PRECISION,
    is_stable BOOLEAN
);

CREATE TABLE prototype_definition (
    prototype_id TEXT PRIMARY KEY,
    prototype_name TEXT,
    strukturbericht TEXT,
    formula_type TEXT,
    description TEXT
);

CREATE INDEX idx_composition_entry_id ON composition(entry_id);
CREATE INDEX idx_composition_element ON composition(element);
CREATE INDEX idx_structure_entry_id ON structure(entry_id);
CREATE INDEX idx_structure_prototype ON structure(prototype);
CREATE INDEX idx_structure_strukturbericht ON structure(strukturbericht);
CREATE INDEX idx_phase_stability_entry_id ON phase_stability(entry_id);
CREATE INDEX idx_phase_energy_above_hull ON phase_stability(energy_above_hull);
CREATE INDEX idx_calculation_entry_id ON calculation(entry_id);
CREATE INDEX idx_property_calculation_id ON calculated_property(calculation_id);
CREATE INDEX idx_property_name ON calculated_property(property_name);
