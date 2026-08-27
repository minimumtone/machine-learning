-- ============================================================
-- 001_schema.sql — Schema definition (DDL only)
-- 32 tables: 31 entity tables + property_definition dictionary
-- Load order: 001_schema -> 002_reference_data -> 003_material_data
--             -> 004_views -> 005_roles
--
-- Design rules enforced at the DDL level:
--   * Every categorical/master value is FK-constrained
--     (composition.element, structure.prototype, structure.space_group_number,
--      EAV property names/units via property_definition).
--   * Cardinality is explicit: 1:1 relations carry UNIQUE(entry_id) /
--     UNIQUE(calculation_id); 1:N relations carry a composite natural key.
--   * phase_stability.is_stable is a generated column derived from
--     energy_above_hull (operational definition: stable <=> E_hull <= 0.001).
--   * Calculation-derived tables reference calculation only; entry_id is
--     reached via calculation.entry_id (no redundant, unconstrained copies).
--   * Physical-range CHECK constraints on fractions, energies, scores.
-- ============================================================

-- === Core Entity Tables ===

CREATE TABLE material_entry (
    entry_id TEXT PRIMARY KEY,
    source_db TEXT,
    source_material_id TEXT,
    formula TEXT NOT NULL,
    reduced_formula TEXT,
    chemical_system TEXT,
    number_of_elements INTEGER CHECK (number_of_elements > 0)
);

-- === Element & Periodic Table ===

CREATE TABLE element (
    element_id SERIAL PRIMARY KEY,
    symbol VARCHAR(5) NOT NULL UNIQUE,
    name VARCHAR(50),
    atomic_number INTEGER NOT NULL CHECK (atomic_number > 0),
    atomic_mass NUMERIC(10,4) CHECK (atomic_mass > 0),
    electronegativity NUMERIC(5,3) CHECK (electronegativity >= 0),
    atomic_radius NUMERIC(6,2) CHECK (atomic_radius > 0),
    group_number INTEGER CHECK (group_number BETWEEN 1 AND 18),
    period_number INTEGER CHECK (period_number BETWEEN 1 AND 7),
    block VARCHAR(5),
    category VARCHAR(50)
);

-- === Property dictionary (canonical names & units for EAV tables) ===

CREATE TABLE property_definition (
    property_def_id SERIAL PRIMARY KEY,
    canonical_name VARCHAR(100) NOT NULL UNIQUE,
    canonical_unit VARCHAR(30),
    value_type VARCHAR(20) NOT NULL DEFAULT 'float'
        CHECK (value_type IN ('float', 'integer', 'text', 'boolean')),
    applies_to VARCHAR(30) NOT NULL
        CHECK (applies_to IN ('calculated', 'measured', 'element')),
    description TEXT,
    UNIQUE (canonical_name, canonical_unit)
);

CREATE TABLE composition (
    composition_id TEXT PRIMARY KEY,
    entry_id TEXT NOT NULL REFERENCES material_entry(entry_id),
    element TEXT NOT NULL REFERENCES element(symbol),
    atomic_fraction DOUBLE PRECISION
        CHECK (atomic_fraction > 0 AND atomic_fraction <= 1),
    site_label TEXT,
    UNIQUE (entry_id, element)
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
    space_group_number INTEGER NOT NULL UNIQUE
        CHECK (space_group_number BETWEEN 1 AND 230),
    hermann_mauguin VARCHAR(30),
    crystal_system VARCHAR(30),
    point_group VARCHAR(20),
    laue_class VARCHAR(20),
    is_centrosymmetric BOOLEAN
);

CREATE TABLE structure (
    structure_id TEXT PRIMARY KEY,
    entry_id TEXT NOT NULL UNIQUE REFERENCES material_entry(entry_id),
    prototype TEXT REFERENCES prototype_definition(prototype_id),
    strukturbericht TEXT,
    formula_type TEXT,
    space_group_number INTEGER REFERENCES space_group(space_group_number),
    crystal_system TEXT,
    lattice_a DOUBLE PRECISION CHECK (lattice_a > 0),
    lattice_b DOUBLE PRECISION CHECK (lattice_b > 0),
    lattice_c DOUBLE PRECISION CHECK (lattice_c > 0),
    volume_per_atom DOUBLE PRECISION CHECK (volume_per_atom > 0),
    space_group TEXT
);

-- Operational stability definition (paper / gold SQL / DB single source):
--   stable <=> energy_above_hull <= 0.001 eV/atom
CREATE TABLE phase_stability (
    stability_id TEXT PRIMARY KEY,
    entry_id TEXT NOT NULL UNIQUE REFERENCES material_entry(entry_id),
    formation_energy_per_atom DOUBLE PRECISION,
    energy_above_hull DOUBLE PRECISION CHECK (energy_above_hull >= 0),
    is_stable BOOLEAN GENERATED ALWAYS AS (energy_above_hull <= 0.001) STORED,
    band_gap DOUBLE PRECISION CHECK (band_gap >= 0)
);

-- === Calculation & Properties (parent-child) ===

CREATE TABLE calculation (
    calculation_id TEXT PRIMARY KEY,
    entry_id TEXT NOT NULL REFERENCES material_entry(entry_id),
    method TEXT,
    functional TEXT,
    calculation_type TEXT,
    -- One calculation per (type, method, functional) combination; the same
    -- entry may hold e.g. PBE and HSE06 band-structure calculations.
    UNIQUE NULLS NOT DISTINCT (entry_id, calculation_type, method, functional)
);

CREATE TABLE calculated_property (
    property_id TEXT PRIMARY KEY,
    calculation_id TEXT NOT NULL REFERENCES calculation(calculation_id),
    property_name TEXT NOT NULL,
    value DOUBLE PRECISION,
    unit TEXT,
    tensor_component TEXT,
    UNIQUE (calculation_id, property_name),
    -- Single-column FK closes the composite-FK NULL loophole (a NULL unit
    -- would otherwise skip the FK check entirely); the composite FK then
    -- additionally pins the unit to the canonical one when unit is present.
    FOREIGN KEY (property_name)
        REFERENCES property_definition(canonical_name),
    FOREIGN KEY (property_name, unit)
        REFERENCES property_definition(canonical_name, canonical_unit)
);

CREATE TABLE element_property (
    element_property_id SERIAL PRIMARY KEY,
    element_id INTEGER NOT NULL REFERENCES element(element_id),
    property_name VARCHAR(100) NOT NULL,
    value NUMERIC(15,6),
    unit VARCHAR(30),
    temperature_k NUMERIC(8,2) CHECK (temperature_k >= 0),
    source VARCHAR(100),
    UNIQUE (element_id, property_name),
    FOREIGN KEY (property_name)
        REFERENCES property_definition(canonical_name),
    FOREIGN KEY (property_name, unit)
        REFERENCES property_definition(canonical_name, canonical_unit)
);

-- === Many-to-Many: Applications ===

CREATE TABLE application_domain (
    domain_id SERIAL PRIMARY KEY,
    domain_name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    parent_domain_id INTEGER REFERENCES application_domain(domain_id)  -- self-referencing hierarchy
);

CREATE TABLE material_application (
    material_application_id SERIAL PRIMARY KEY,
    entry_id TEXT NOT NULL REFERENCES material_entry(entry_id),
    domain_id INTEGER NOT NULL REFERENCES application_domain(domain_id),
    relevance_score NUMERIC(5,3) CHECK (relevance_score BETWEEN 0 AND 1),
    notes TEXT,
    UNIQUE (entry_id, domain_id)
);

-- === Many-to-Many: Literature References ===

CREATE TABLE literature_reference (
    reference_id SERIAL PRIMARY KEY,
    doi VARCHAR(200),
    title TEXT,
    authors TEXT,
    journal VARCHAR(200),
    year INTEGER CHECK (year BETWEEN 1800 AND 2100),
    volume VARCHAR(20),
    pages VARCHAR(50)
);

CREATE TABLE material_reference (
    material_reference_id SERIAL PRIMARY KEY,
    entry_id TEXT NOT NULL REFERENCES material_entry(entry_id),
    reference_id INTEGER NOT NULL REFERENCES literature_reference(reference_id),
    context VARCHAR(100),  -- 'experimental_validation', 'theoretical_prediction', 'review'
    UNIQUE (entry_id, reference_id)
);

-- === Experimental Data (separate from DFT) ===

CREATE TABLE experimental_measurement (
    measurement_id SERIAL PRIMARY KEY,
    entry_id TEXT NOT NULL REFERENCES material_entry(entry_id),
    reference_id INTEGER REFERENCES literature_reference(reference_id),
    method VARCHAR(100),  -- 'XRD', 'neutron_diffraction', 'calorimetry'
    temperature_k NUMERIC(8,2) CHECK (temperature_k >= 0),
    pressure_gpa NUMERIC(8,3) CHECK (pressure_gpa >= 0)
);

CREATE TABLE measured_property (
    measured_property_id SERIAL PRIMARY KEY,
    measurement_id INTEGER NOT NULL REFERENCES experimental_measurement(measurement_id),
    property_name VARCHAR(100) NOT NULL,
    value NUMERIC(15,6),
    uncertainty NUMERIC(15,6) CHECK (uncertainty >= 0),
    unit VARCHAR(30),
    UNIQUE (measurement_id, property_name),
    FOREIGN KEY (property_name)
        REFERENCES property_definition(canonical_name),
    FOREIGN KEY (property_name, unit)
        REFERENCES property_definition(canonical_name, canonical_unit)
);

-- === Synthesis & Processing ===

CREATE TABLE synthesis_method (
    synthesis_id SERIAL PRIMARY KEY,
    method_name VARCHAR(100) NOT NULL UNIQUE,
    category VARCHAR(50),  -- 'arc_melting', 'ball_milling', 'sputtering', 'czochralski'
    description TEXT
);

CREATE TABLE material_synthesis (
    material_synthesis_id SERIAL PRIMARY KEY,
    entry_id TEXT NOT NULL REFERENCES material_entry(entry_id),
    synthesis_id INTEGER NOT NULL REFERENCES synthesis_method(synthesis_id),
    reference_id INTEGER REFERENCES literature_reference(reference_id),
    temperature_k NUMERIC(8,2) CHECK (temperature_k >= 0),
    duration_hours NUMERIC(10,2) CHECK (duration_hours >= 0),
    atmosphere VARCHAR(50),
    success BOOLEAN DEFAULT TRUE
    -- No UNIQUE(entry_id, synthesis_id): the same material may be synthesized
    -- by the same method under different conditions (temperature, duration,
    -- atmosphere, reference); rows are identified by the surrogate PK.
);

-- === Defect & Dopant Information ===

CREATE TABLE defect_type (
    defect_type_id SERIAL PRIMARY KEY,
    defect_name VARCHAR(100) NOT NULL UNIQUE,
    category VARCHAR(50),  -- 'vacancy', 'interstitial', 'antisite', 'substitutional'
    description TEXT
);

CREATE TABLE material_defect (
    material_defect_id SERIAL PRIMARY KEY,
    entry_id TEXT NOT NULL REFERENCES material_entry(entry_id),
    defect_type_id INTEGER NOT NULL REFERENCES defect_type(defect_type_id),
    formation_energy NUMERIC(10,6),
    concentration NUMERIC(15,8) CHECK (concentration >= 0),
    site VARCHAR(50),
    dopant_element_id INTEGER REFERENCES element(element_id),
    -- A defect record is identified by type + site + dopant: the same entry
    -- may host e.g. an Al-site and a Ni-site vacancy, or several dopants.
    UNIQUE NULLS NOT DISTINCT (entry_id, defect_type_id, site, dopant_element_id)
);

-- === Electronic Structure Tables ===
-- entry_id is reached via calculation.entry_id (calculation is the parent).

CREATE TABLE band_structure (
    band_structure_id SERIAL PRIMARY KEY,
    calculation_id TEXT NOT NULL UNIQUE REFERENCES calculation(calculation_id),
    is_direct_gap BOOLEAN,
    cbm_energy DOUBLE PRECISION,
    vbm_energy DOUBLE PRECISION,
    band_gap_type VARCHAR(20),
    num_bands INTEGER CHECK (num_bands > 0),
    num_kpoints INTEGER CHECK (num_kpoints > 0)
);

CREATE TABLE density_of_states (
    dos_id SERIAL PRIMARY KEY,
    calculation_id TEXT NOT NULL UNIQUE REFERENCES calculation(calculation_id),
    total_dos_at_fermi DOUBLE PRECISION CHECK (total_dos_at_fermi >= 0),
    efermi DOUBLE PRECISION,
    is_metallic BOOLEAN,
    spin_polarized BOOLEAN
);

-- === Mechanical/Physical Property Tables ===

CREATE TABLE elastic_tensor (
    elastic_id SERIAL PRIMARY KEY,
    calculation_id TEXT NOT NULL UNIQUE REFERENCES calculation(calculation_id),
    bulk_modulus_vrh DOUBLE PRECISION CHECK (bulk_modulus_vrh > 0),
    shear_modulus_vrh DOUBLE PRECISION CHECK (shear_modulus_vrh > 0),
    youngs_modulus DOUBLE PRECISION CHECK (youngs_modulus > 0),
    poisson_ratio DOUBLE PRECISION CHECK (poisson_ratio > -1 AND poisson_ratio < 0.5),
    is_stable BOOLEAN  -- mechanical (Born) stability, distinct from phase stability
);

CREATE TABLE magnetic_property (
    magnetic_id SERIAL PRIMARY KEY,
    entry_id TEXT NOT NULL UNIQUE REFERENCES material_entry(entry_id),
    total_magnetization DOUBLE PRECISION CHECK (total_magnetization >= 0),
    magnetic_ordering VARCHAR(30),
    curie_temperature_k DOUBLE PRECISION CHECK (curie_temperature_k >= 0),
    magnetic_anisotropy_energy DOUBLE PRECISION
);

CREATE TABLE thermal_property (
    thermal_id SERIAL PRIMARY KEY,
    calculation_id TEXT NOT NULL REFERENCES calculation(calculation_id),
    debye_temperature_k DOUBLE PRECISION CHECK (debye_temperature_k > 0),
    thermal_conductivity DOUBLE PRECISION CHECK (thermal_conductivity >= 0),
    specific_heat_cv DOUBLE PRECISION CHECK (specific_heat_cv >= 0),
    gruneisen_parameter DOUBLE PRECISION,
    temperature_k DOUBLE PRECISION DEFAULT 300.0 CHECK (temperature_k >= 0),
    UNIQUE (calculation_id, temperature_k)
);

-- === Surface/Interface Tables ===

CREATE TABLE surface_energy (
    surface_id SERIAL PRIMARY KEY,
    entry_id TEXT NOT NULL REFERENCES material_entry(entry_id),
    miller_index VARCHAR(10) NOT NULL,
    surface_energy_j_m2 DOUBLE PRECISION CHECK (surface_energy_j_m2 > 0),
    work_function DOUBLE PRECISION CHECK (work_function > 0),
    is_reconstructed BOOLEAN DEFAULT FALSE,
    -- Reconstructed and unreconstructed variants of the same facet coexist.
    UNIQUE NULLS NOT DISTINCT (entry_id, miller_index, is_reconstructed)
);

CREATE TABLE grain_boundary (
    grain_boundary_id SERIAL PRIMARY KEY,
    entry_id TEXT NOT NULL REFERENCES material_entry(entry_id),
    sigma_value INTEGER CHECK (sigma_value > 0),
    rotation_axis VARCHAR(10),
    tilt_angle DOUBLE PRECISION CHECK (tilt_angle >= 0 AND tilt_angle <= 180),
    gb_energy_j_m2 DOUBLE PRECISION CHECK (gb_energy_j_m2 > 0),
    excess_volume DOUBLE PRECISION CHECK (excess_volume >= 0),
    UNIQUE (entry_id, sigma_value, rotation_axis)
);

-- === Phase Diagram & Alloy Tables ===

CREATE TABLE phase_diagram_entry (
    phase_entry_id SERIAL PRIMARY KEY,
    entry_id TEXT NOT NULL REFERENCES material_entry(entry_id),
    chemical_system TEXT NOT NULL,
    is_on_hull BOOLEAN,
    decomposition_products TEXT,
    hull_distance DOUBLE PRECISION CHECK (hull_distance >= 0),
    UNIQUE (entry_id, chemical_system)
);

CREATE TABLE alloy_system (
    alloy_system_id SERIAL PRIMARY KEY,
    system_name VARCHAR(100) NOT NULL UNIQUE,
    num_components INTEGER CHECK (num_components > 1),
    category VARCHAR(30),
    description TEXT
);

CREATE TABLE material_alloy_system (
    material_alloy_id SERIAL PRIMARY KEY,
    entry_id TEXT NOT NULL REFERENCES material_entry(entry_id),
    alloy_system_id INTEGER NOT NULL REFERENCES alloy_system(alloy_system_id),
    phase VARCHAR(30),
    composition_type VARCHAR(30),
    UNIQUE (entry_id, alloy_system_id)
);

-- === Pure Element Reference (ground-state DFT energies, OQMD) ===

CREATE TABLE pure_element_reference (
    pure_ref_id SERIAL PRIMARY KEY,
    element_symbol VARCHAR(5) NOT NULL REFERENCES element(symbol),
    -- Reference energies are only comparable within one energy convention;
    -- the calculation-condition axis (method / functional / reference_set)
    -- makes that convention explicit. formation_enthalpy pins one set.
    method TEXT NOT NULL DEFAULT 'DFT',
    functional TEXT NOT NULL DEFAULT 'PBE',
    reference_set TEXT NOT NULL DEFAULT 'OQMD-PBE',
    oqmd_entry_id INTEGER,
    ground_state_spacegroup VARCHAR(30),
    energy_per_atom DOUBLE PRECISION,  -- eV/atom (delta_e from OQMD)
    volume_per_atom DOUBLE PRECISION CHECK (volume_per_atom > 0),  -- Angstrom^3/atom
    stability DOUBLE PRECISION CHECK (stability >= 0),  -- eV/atom above hull
    band_gap DOUBLE PRECISION CHECK (band_gap >= 0),  -- eV
    n_polymorphs INTEGER CHECK (n_polymorphs > 0),
    source TEXT DEFAULT 'OQMD',
    UNIQUE (element_symbol, reference_set)
);

-- === Indexes ===
-- UNIQUE / PRIMARY KEY constraints already create their own indexes;
-- only non-covered access paths get explicit indexes here.
CREATE INDEX idx_composition_element ON composition(element);
CREATE INDEX idx_structure_prototype ON structure(prototype);
CREATE INDEX idx_phase_stability_hull ON phase_stability(energy_above_hull);
CREATE INDEX idx_material_app_domain ON material_application(domain_id);
CREATE INDEX idx_exp_measurement_entry ON experimental_measurement(entry_id);
CREATE INDEX idx_material_synthesis_method ON material_synthesis(synthesis_id);
CREATE INDEX idx_material_defect_type ON material_defect(defect_type_id);
CREATE INDEX idx_mat_alloy_system ON material_alloy_system(alloy_system_id);

-- === Consistency trigger: structure vs. master tables ===
-- structure keeps human-readable copies (strukturbericht, formula_type,
-- crystal_system, space_group) of master-table attributes for query
-- convenience; this trigger rejects any row whose copies contradict
-- prototype_definition / space_group, so the DB never holds two truths.

CREATE FUNCTION check_structure_master_consistency() RETURNS trigger AS $$
DECLARE
    p_sb TEXT;
    p_ft TEXT;
    g_cs VARCHAR(30);
    g_hm VARCHAR(30);
BEGIN
    IF NEW.prototype IS NOT NULL THEN
        SELECT strukturbericht, formula_type INTO p_sb, p_ft
        FROM prototype_definition WHERE prototype_id = NEW.prototype;
        IF NEW.strukturbericht IS DISTINCT FROM p_sb
           OR NEW.formula_type IS DISTINCT FROM p_ft THEN
            RAISE EXCEPTION
                'structure %: strukturbericht/formula_type (%, %) contradict prototype_definition % (%, %)',
                NEW.structure_id, NEW.strukturbericht, NEW.formula_type,
                NEW.prototype, p_sb, p_ft;
        END IF;
    ELSIF NEW.strukturbericht IS NOT NULL OR NEW.formula_type IS NOT NULL THEN
        RAISE EXCEPTION
            'structure %: strukturbericht/formula_type set without prototype reference',
            NEW.structure_id;
    END IF;

    IF NEW.space_group_number IS NOT NULL THEN
        SELECT crystal_system, hermann_mauguin INTO g_cs, g_hm
        FROM space_group WHERE space_group_number = NEW.space_group_number;
        IF NEW.crystal_system IS DISTINCT FROM g_cs
           OR NEW.space_group IS DISTINCT FROM g_hm THEN
            RAISE EXCEPTION
                'structure %: crystal_system/space_group (%, %) contradict space_group % (%, %)',
                NEW.structure_id, NEW.crystal_system, NEW.space_group,
                NEW.space_group_number, g_cs, g_hm;
        END IF;
    ELSIF NEW.crystal_system IS NOT NULL OR NEW.space_group IS NOT NULL THEN
        RAISE EXCEPTION
            'structure %: crystal_system/space_group set without space_group_number reference',
            NEW.structure_id;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_structure_master_consistency
    BEFORE INSERT OR UPDATE ON structure
    FOR EACH ROW EXECUTE FUNCTION check_structure_master_consistency();
