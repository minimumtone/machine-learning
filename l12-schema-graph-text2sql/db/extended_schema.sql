-- ============================================================
-- Extended Schema for Schema-Graph Scalability Experiment
-- 30 tables + many-to-many + hierarchical + self-referencing FK
-- Based on OQMD (~30 tables) / Materials Project (~20 collections) patterns
-- ER図設計において実際のRDB構造に近い規模を想定
-- ============================================================

-- === Core Entity Tables ===

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
    atomic_fraction DOUBLE PRECISION,
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
    volume_per_atom DOUBLE PRECISION,
    space_group TEXT
);

CREATE TABLE phase_stability (
    stability_id TEXT PRIMARY KEY,
    entry_id TEXT NOT NULL REFERENCES material_entry(entry_id),
    formation_energy_per_atom DOUBLE PRECISION,
    energy_above_hull DOUBLE PRECISION,
    is_stable BOOLEAN,
    band_gap DOUBLE PRECISION
);

-- === Calculation & Properties (parent-child) ===

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
    unit TEXT,
    tensor_component TEXT
);

-- === Prototype & Space Group Master Tables ===

CREATE TABLE prototype_definition (
    prototype_id TEXT PRIMARY KEY,
    prototype_name TEXT,
    strukturbericht TEXT,
    formula_type TEXT,
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

-- === Electronic Structure Tables (Tables 21-22) ===

CREATE TABLE band_structure (
    band_structure_id SERIAL PRIMARY KEY,
    calculation_id TEXT NOT NULL REFERENCES calculation(calculation_id),
    entry_id TEXT NOT NULL REFERENCES material_entry(entry_id),
    is_direct_gap BOOLEAN,
    cbm_energy DOUBLE PRECISION,
    vbm_energy DOUBLE PRECISION,
    band_gap_type VARCHAR(20),
    num_bands INTEGER,
    num_kpoints INTEGER
);

CREATE TABLE density_of_states (
    dos_id SERIAL PRIMARY KEY,
    calculation_id TEXT NOT NULL REFERENCES calculation(calculation_id),
    entry_id TEXT NOT NULL REFERENCES material_entry(entry_id),
    total_dos_at_fermi DOUBLE PRECISION,
    efermi DOUBLE PRECISION,
    is_metallic BOOLEAN,
    spin_polarized BOOLEAN
);

-- === Mechanical/Physical Property Tables (Tables 23-25) ===

CREATE TABLE elastic_tensor (
    elastic_id SERIAL PRIMARY KEY,
    entry_id TEXT NOT NULL REFERENCES material_entry(entry_id),
    calculation_id TEXT NOT NULL REFERENCES calculation(calculation_id),
    bulk_modulus_vrh DOUBLE PRECISION,
    shear_modulus_vrh DOUBLE PRECISION,
    youngs_modulus DOUBLE PRECISION,
    poisson_ratio DOUBLE PRECISION,
    is_stable BOOLEAN
);

CREATE TABLE magnetic_property (
    magnetic_id SERIAL PRIMARY KEY,
    entry_id TEXT NOT NULL REFERENCES material_entry(entry_id),
    total_magnetization DOUBLE PRECISION,
    magnetic_ordering VARCHAR(30),
    curie_temperature_k DOUBLE PRECISION,
    magnetic_anisotropy_energy DOUBLE PRECISION
);

CREATE TABLE thermal_property (
    thermal_id SERIAL PRIMARY KEY,
    entry_id TEXT NOT NULL REFERENCES material_entry(entry_id),
    calculation_id TEXT NOT NULL REFERENCES calculation(calculation_id),
    debye_temperature_k DOUBLE PRECISION,
    thermal_conductivity DOUBLE PRECISION,
    specific_heat_cv DOUBLE PRECISION,
    gruneisen_parameter DOUBLE PRECISION,
    temperature_k DOUBLE PRECISION DEFAULT 300.0
);

-- === Surface/Interface Tables (Tables 26-27) ===

CREATE TABLE surface_energy (
    surface_id SERIAL PRIMARY KEY,
    entry_id TEXT NOT NULL REFERENCES material_entry(entry_id),
    miller_index VARCHAR(10) NOT NULL,
    surface_energy_j_m2 DOUBLE PRECISION,
    work_function DOUBLE PRECISION,
    is_reconstructed BOOLEAN DEFAULT FALSE
);

CREATE TABLE grain_boundary (
    grain_boundary_id SERIAL PRIMARY KEY,
    entry_id TEXT NOT NULL REFERENCES material_entry(entry_id),
    sigma_value INTEGER,
    rotation_axis VARCHAR(10),
    tilt_angle DOUBLE PRECISION,
    gb_energy_j_m2 DOUBLE PRECISION,
    excess_volume DOUBLE PRECISION
);

-- === Phase Diagram & Alloy Tables (Tables 28-30) ===

CREATE TABLE phase_diagram_entry (
    phase_entry_id SERIAL PRIMARY KEY,
    entry_id TEXT NOT NULL REFERENCES material_entry(entry_id),
    chemical_system TEXT NOT NULL,
    is_on_hull BOOLEAN,
    decomposition_products TEXT,
    hull_distance DOUBLE PRECISION
);

CREATE TABLE alloy_system (
    alloy_system_id SERIAL PRIMARY KEY,
    system_name VARCHAR(100) NOT NULL,
    num_components INTEGER,
    category VARCHAR(30),
    description TEXT
);

CREATE TABLE material_alloy_system (
    material_alloy_id SERIAL PRIMARY KEY,
    entry_id TEXT NOT NULL REFERENCES material_entry(entry_id),
    alloy_system_id INTEGER NOT NULL REFERENCES alloy_system(alloy_system_id),
    phase VARCHAR(30),
    composition_type VARCHAR(30)
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
CREATE INDEX idx_bs_entry ON band_structure(entry_id);
CREATE INDEX idx_bs_calc ON band_structure(calculation_id);
CREATE INDEX idx_dos_entry ON density_of_states(entry_id);
CREATE INDEX idx_dos_calc ON density_of_states(calculation_id);
CREATE INDEX idx_elastic_entry ON elastic_tensor(entry_id);
CREATE INDEX idx_magnetic_entry ON magnetic_property(entry_id);
CREATE INDEX idx_thermal_entry ON thermal_property(entry_id);
CREATE INDEX idx_surface_entry ON surface_energy(entry_id);
CREATE INDEX idx_gb_entry ON grain_boundary(entry_id);
CREATE INDEX idx_phase_entry ON phase_diagram_entry(entry_id);
CREATE INDEX idx_mat_alloy_entry ON material_alloy_system(entry_id);
CREATE INDEX idx_mat_alloy_system ON material_alloy_system(alloy_system_id);
