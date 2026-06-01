-- ============================================================
-- Extended Schema for Schema-Graph Scalability Experiment
-- 20 tables + many-to-many + hierarchical + self-referencing FK
-- Based on Materials Project / AFLOW / NOMAD patterns
-- ============================================================

-- === Core Entity Tables ===

CREATE TABLE material_entry (
    entry_id SERIAL PRIMARY KEY,
    formula VARCHAR(200) NOT NULL,
    reduced_formula VARCHAR(200),
    chemical_system VARCHAR(100),
    num_elements INTEGER,
    source_database VARCHAR(50) DEFAULT 'OQMD',
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE composition (
    composition_id SERIAL PRIMARY KEY,
    entry_id INTEGER NOT NULL REFERENCES material_entry(entry_id),
    element VARCHAR(10) NOT NULL,
    atomic_fraction NUMERIC(8,6),
    site_label VARCHAR(50)
);

CREATE TABLE structure (
    structure_id SERIAL PRIMARY KEY,
    entry_id INTEGER NOT NULL REFERENCES material_entry(entry_id),
    prototype VARCHAR(100),
    strukturbericht VARCHAR(20),
    space_group VARCHAR(20),
    space_group_number INTEGER,
    lattice_a NUMERIC(10,6),
    lattice_b NUMERIC(10,6),
    lattice_c NUMERIC(10,6),
    alpha NUMERIC(8,4) DEFAULT 90.0,
    beta NUMERIC(8,4) DEFAULT 90.0,
    gamma NUMERIC(8,4) DEFAULT 90.0,
    volume NUMERIC(12,4),
    crystal_system VARCHAR(30)
);

CREATE TABLE phase_stability (
    stability_id SERIAL PRIMARY KEY,
    entry_id INTEGER NOT NULL REFERENCES material_entry(entry_id),
    formation_energy_per_atom NUMERIC(10,6),
    energy_above_hull NUMERIC(10,6),
    is_stable BOOLEAN,
    band_gap NUMERIC(8,4),
    is_metal BOOLEAN
);

-- === Calculation & Properties (parent-child) ===

CREATE TABLE calculation (
    calculation_id SERIAL PRIMARY KEY,
    entry_id INTEGER NOT NULL REFERENCES material_entry(entry_id),
    method VARCHAR(50) DEFAULT 'GGA+PBE',
    pseudopotential VARCHAR(50),
    energy_cutoff NUMERIC(10,2),
    k_points VARCHAR(30),
    convergence_threshold NUMERIC(12,8),
    software VARCHAR(30) DEFAULT 'VASP'
);

CREATE TABLE calculated_property (
    property_id SERIAL PRIMARY KEY,
    calculation_id INTEGER NOT NULL REFERENCES calculation(calculation_id),
    property_name VARCHAR(100) NOT NULL,
    value NUMERIC(15,6),
    unit VARCHAR(30),
    tensor_component VARCHAR(20)
);

-- === Prototype & Space Group Master Tables ===

CREATE TABLE prototype_definition (
    prototype_id SERIAL PRIMARY KEY,
    prototype_name VARCHAR(100) NOT NULL UNIQUE,
    strukturbericht VARCHAR(20),
    space_group_number INTEGER,
    pearson_symbol VARCHAR(20),
    num_atoms_per_cell INTEGER,
    description TEXT
);

CREATE TABLE space_group (
    space_group_id SERIAL PRIMARY KEY,
    space_group_number INTEGER NOT NULL UNIQUE,
    hermann_mauguin VARCHAR(30),
    crystal_system VARCHAR(30),
    point_group VARCHAR(20),
    laue_class VARCHAR(20),
    is_centrosymmetric BOOLEAN
);

-- === Element & Periodic Table (hierarchical) ===

CREATE TABLE element (
    element_id SERIAL PRIMARY KEY,
    symbol VARCHAR(5) NOT NULL UNIQUE,
    name VARCHAR(50),
    atomic_number INTEGER NOT NULL,
    atomic_mass NUMERIC(10,4),
    electronegativity NUMERIC(5,3),
    atomic_radius NUMERIC(6,2),
    group_number INTEGER,
    period_number INTEGER,
    block VARCHAR(5),
    category VARCHAR(50)
);

CREATE TABLE element_property (
    element_property_id SERIAL PRIMARY KEY,
    element_id INTEGER NOT NULL REFERENCES element(element_id),
    property_name VARCHAR(100) NOT NULL,
    value NUMERIC(15,6),
    unit VARCHAR(30),
    temperature_k NUMERIC(8,2),
    source VARCHAR(100)
);

-- === Many-to-Many: Applications ===

CREATE TABLE application_domain (
    domain_id SERIAL PRIMARY KEY,
    domain_name VARCHAR(100) NOT NULL,
    description TEXT,
    parent_domain_id INTEGER REFERENCES application_domain(domain_id)  -- self-referencing hierarchy
);

CREATE TABLE material_application (
    material_application_id SERIAL PRIMARY KEY,
    entry_id TEXT NOT NULL REFERENCES material_entry(entry_id),
    domain_id INTEGER NOT NULL REFERENCES application_domain(domain_id),
    relevance_score NUMERIC(5,3),
    notes TEXT
);

-- === Many-to-Many: Literature References ===

CREATE TABLE literature_reference (
    reference_id SERIAL PRIMARY KEY,
    doi VARCHAR(200),
    title TEXT,
    authors TEXT,
    journal VARCHAR(200),
    year INTEGER,
    volume VARCHAR(20),
    pages VARCHAR(50)
);

CREATE TABLE material_reference (
    material_reference_id SERIAL PRIMARY KEY,
    entry_id TEXT NOT NULL REFERENCES material_entry(entry_id),
    reference_id INTEGER NOT NULL REFERENCES literature_reference(reference_id),
    context VARCHAR(100)  -- 'experimental_validation', 'theoretical_prediction', 'review'
);

-- === Experimental Data (separate from DFT) ===

CREATE TABLE experimental_measurement (
    measurement_id SERIAL PRIMARY KEY,
    entry_id TEXT NOT NULL REFERENCES material_entry(entry_id),
    reference_id INTEGER REFERENCES literature_reference(reference_id),
    method VARCHAR(100),  -- 'XRD', 'neutron_diffraction', 'calorimetry'
    temperature_k NUMERIC(8,2),
    pressure_gpa NUMERIC(8,3)
);

CREATE TABLE measured_property (
    measured_property_id SERIAL PRIMARY KEY,
    measurement_id INTEGER NOT NULL REFERENCES experimental_measurement(measurement_id),
    property_name VARCHAR(100) NOT NULL,
    value NUMERIC(15,6),
    uncertainty NUMERIC(15,6),
    unit VARCHAR(30)
);

-- === Synthesis & Processing ===

CREATE TABLE synthesis_method (
    synthesis_id SERIAL PRIMARY KEY,
    method_name VARCHAR(100) NOT NULL,
    category VARCHAR(50),  -- 'arc_melting', 'ball_milling', 'sputtering', 'czochralski'
    description TEXT
);

CREATE TABLE material_synthesis (
    material_synthesis_id SERIAL PRIMARY KEY,
    entry_id TEXT NOT NULL REFERENCES material_entry(entry_id),
    synthesis_id INTEGER NOT NULL REFERENCES synthesis_method(synthesis_id),
    reference_id INTEGER REFERENCES literature_reference(reference_id),
    temperature_k NUMERIC(8,2),
    duration_hours NUMERIC(10,2),
    atmosphere VARCHAR(50),
    success BOOLEAN DEFAULT TRUE
);

-- === Defect & Dopant Information ===

CREATE TABLE defect_type (
    defect_type_id SERIAL PRIMARY KEY,
    defect_name VARCHAR(100) NOT NULL,
    category VARCHAR(50),  -- 'vacancy', 'interstitial', 'antisite', 'substitutional'
    description TEXT
);

CREATE TABLE material_defect (
    material_defect_id SERIAL PRIMARY KEY,
    entry_id TEXT NOT NULL REFERENCES material_entry(entry_id),
    defect_type_id INTEGER NOT NULL REFERENCES defect_type(defect_type_id),
    formation_energy NUMERIC(10,6),
    concentration NUMERIC(15,8),
    site VARCHAR(50),
    dopant_element_id INTEGER REFERENCES element(element_id)
);

-- === Indexes for performance ===
CREATE INDEX idx_composition_entry ON composition(entry_id);
CREATE INDEX idx_composition_element ON composition(element);
CREATE INDEX idx_structure_entry ON structure(entry_id);
CREATE INDEX idx_structure_prototype ON structure(prototype);
CREATE INDEX idx_phase_stability_entry ON phase_stability(entry_id);
CREATE INDEX idx_phase_stability_hull ON phase_stability(energy_above_hull);
CREATE INDEX idx_calculation_entry ON calculation(entry_id);
CREATE INDEX idx_calc_property_calc ON calculated_property(calculation_id);
CREATE INDEX idx_element_symbol ON element(symbol);
CREATE INDEX idx_element_property_elem ON element_property(element_id);
CREATE INDEX idx_material_app_entry ON material_application(entry_id);
CREATE INDEX idx_material_app_domain ON material_application(domain_id);
CREATE INDEX idx_material_ref_entry ON material_reference(entry_id);
CREATE INDEX idx_exp_measurement_entry ON experimental_measurement(entry_id);
CREATE INDEX idx_measured_prop_meas ON measured_property(measurement_id);
CREATE INDEX idx_material_synthesis_entry ON material_synthesis(entry_id);
CREATE INDEX idx_material_defect_entry ON material_defect(entry_id);
