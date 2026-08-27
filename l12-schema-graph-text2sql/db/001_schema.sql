-- ============================================================
-- 001_schema.sql — Schema definition (DDL only)
-- 33 tables: 31 entity tables + property_definition dictionary
--            + reference_energy_set (energy-convention master)
-- Load order: 001_schema -> 002_reference_data -> 003_material_data
--             -> 004_views -> 005_roles -> 006_integrity_checks
--             -> 007_initialization_marker
--
-- Design rules enforced at the DDL level:
--   * Core controlled vocabularies used for joins and evaluation are
--     FK-constrained (composition.element, structure.prototype,
--     structure.space_group_number, EAV property names/units via
--     property_definition, pure_element_reference.reference_set via
--     reference_energy_set); free-text descriptors (e.g. source_db,
--     category, atmosphere) intentionally remain unconstrained.
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
    number_of_elements INTEGER NOT NULL CHECK (number_of_elements > 0)
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
    -- Storage contract: all EAV value columns are numeric
    -- (DOUBLE PRECISION / NUMERIC), so the dictionary only declares types
    -- the storage can actually hold. text/boolean properties are outside
    -- this verification schema.
    value_type VARCHAR(20) NOT NULL DEFAULT 'float'
        CHECK (value_type IN ('float', 'integer')),
    applies_to VARCHAR(30) NOT NULL
        CHECK (applies_to IN ('calculated', 'measured', 'element')),
    description TEXT,
    UNIQUE (canonical_name, canonical_unit)
);

CREATE TABLE composition (
    composition_id TEXT PRIMARY KEY,
    entry_id TEXT NOT NULL REFERENCES material_entry(entry_id),
    element TEXT NOT NULL REFERENCES element(symbol),
    atomic_fraction DOUBLE PRECISION NOT NULL
        CHECK (atomic_fraction > 0 AND atomic_fraction <= 1),
    site_label TEXT,
    -- Site-resolved composition: the same element may occupy several
    -- crystallographic sites (rows differ by site_label).
    UNIQUE NULLS NOT DISTINCT (entry_id, element, site_label)
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
    is_centrosymmetric BOOLEAN NOT NULL
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

-- Energy-convention master: each reference_set pins one (method, functional,
-- source, fit) combination, so the convention is defined in exactly one
-- place. Child rows (phase_stability, pure_element_reference) carry only
-- reference_set; the convention details are obtained by joining this master.
CREATE TABLE reference_energy_set (
    reference_set TEXT PRIMARY KEY,
    method TEXT NOT NULL,
    functional TEXT NOT NULL,
    source TEXT NOT NULL,
    -- Name of the elemental-reference fit the formation energies are
    -- relative to (e.g. the OQMD standard fit); 'OQMD-PBE' alone would not
    -- identify which reference fit was used.
    fit_name TEXT NOT NULL,
    description TEXT
);

-- Operational stability definition (paper / gold SQL / DB single source):
--   stable <=> energy_above_hull <= 0.001 eV/atom
CREATE TABLE phase_stability (
    stability_id TEXT PRIMARY KEY,
    entry_id TEXT NOT NULL UNIQUE REFERENCES material_entry(entry_id),
    -- Formation energy per atom relative to the elemental reference states
    -- of reference_set below. NOT NULL: every stability row must carry it
    -- (a NULL here could not be distinguished from a computation gap).
    formation_energy_per_atom DOUBLE PRECISION NOT NULL,
    -- Energy convention of formation_energy_per_atom. Formation energies
    -- from different source databases (OQMD / Materials Project / AFLOW)
    -- use different pseudopotentials, corrections and elemental reference
    -- fits, so they are only comparable within one reference_set; views
    -- must join elemental references on the SAME reference_set.
    reference_set TEXT NOT NULL
        REFERENCES reference_energy_set(reference_set),
    -- NOT NULL keeps the generated is_stable strictly two-valued
    -- (a NULL hull energy would make is_stable NULL, i.e. three-valued).
    energy_above_hull DOUBLE PRECISION NOT NULL CHECK (energy_above_hull >= 0),
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
    -- Intentional simplification for this verification DB: numerical
    -- parameters (cutoff, k-mesh, pseudopotential, U) are not part of the
    -- key, so only one calculation per combination is stored (see README).
    UNIQUE NULLS NOT DISTINCT (entry_id, calculation_type, method, functional)
);

CREATE TABLE calculated_property (
    property_id TEXT PRIMARY KEY,
    calculation_id TEXT NOT NULL REFERENCES calculation(calculation_id),
    property_name TEXT NOT NULL,
    value DOUBLE PRECISION,
    unit TEXT,
    tensor_component TEXT,
    -- Tensor-valued properties store one row per component (e.g. C11, C12,
    -- C44 of the elastic tensor); scalar properties leave it NULL.
    UNIQUE NULLS NOT DISTINCT (calculation_id, property_name, tensor_component),
    -- Single-column FK closes the composite-FK NULL loophole (a NULL unit
    -- would otherwise skip the FK check entirely); the composite FK then
    -- additionally pins the unit to the canonical one when unit is present.
    -- unit is deliberately denormalized (dictionary holds the canonical
    -- unit) so Text-to-SQL queries can read it without an extra JOIN; the
    -- composite FK prevents it from ever contradicting the dictionary.
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
    -- Temperature-dependent properties keep one row per temperature, and
    -- values from different literature sources may coexist.
    UNIQUE NULLS NOT DISTINCT (element_id, property_name, temperature_k, source),
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
    pressure_gpa NUMERIC(8,3) CHECK (pressure_gpa >= 0),
    -- One measurement per (entry, reference, method, T, P); replicate
    -- measurements are outside this verification schema, so accidental
    -- double-loading of the same measurement is rejected.
    -- NULLS NOT DISTINCT consequence (documented limitation): NULL condition
    -- values do NOT represent independent measurements — only one
    -- unknown-condition measurement per (entry, reference, method) is
    -- representable. Distinct real-world measurements must carry their
    -- actual conditions to coexist.
    UNIQUE NULLS NOT DISTINCT
        (entry_id, reference_id, method, temperature_k, pressure_gpa)
);

CREATE TABLE measured_property (
    measured_property_id SERIAL PRIMARY KEY,
    measurement_id INTEGER NOT NULL REFERENCES experimental_measurement(measurement_id),
    property_name VARCHAR(100) NOT NULL,
    value NUMERIC(15,6),
    uncertainty NUMERIC(15,6) CHECK (uncertainty >= 0),
    unit VARCHAR(30),
    -- Intentional simplification: one scalar value per (measurement,
    -- property); component-resolved experimental properties are outside
    -- this verification schema.
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
    success BOOLEAN NOT NULL DEFAULT TRUE
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
    is_direct_gap BOOLEAN NOT NULL,
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
    -- NULL means metallicity was not determined for this DOS record;
    -- three-valued on purpose, unlike the other two-state flags.
    is_metallic BOOLEAN,
    spin_polarized BOOLEAN NOT NULL
);

-- === Mechanical/Physical Property Tables ===

CREATE TABLE elastic_tensor (
    elastic_id SERIAL PRIMARY KEY,
    calculation_id TEXT NOT NULL UNIQUE REFERENCES calculation(calculation_id),
    bulk_modulus_vrh DOUBLE PRECISION CHECK (bulk_modulus_vrh > 0),
    shear_modulus_vrh DOUBLE PRECISION CHECK (shear_modulus_vrh > 0),
    youngs_modulus DOUBLE PRECISION CHECK (youngs_modulus > 0),
    poisson_ratio DOUBLE PRECISION CHECK (poisson_ratio > -1 AND poisson_ratio < 0.5),
    is_stable BOOLEAN NOT NULL  -- mechanical (Born) stability, distinct from phase stability
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
    temperature_k DOUBLE PRECISION NOT NULL DEFAULT 300.0 CHECK (temperature_k >= 0),
    UNIQUE (calculation_id, temperature_k)
);

-- === Surface/Interface Tables ===

CREATE TABLE surface_energy (
    surface_id SERIAL PRIMARY KEY,
    entry_id TEXT NOT NULL REFERENCES material_entry(entry_id),
    miller_index VARCHAR(10) NOT NULL,
    surface_energy_j_m2 DOUBLE PRECISION CHECK (surface_energy_j_m2 > 0),
    work_function DOUBLE PRECISION CHECK (work_function > 0),
    is_reconstructed BOOLEAN NOT NULL DEFAULT FALSE,
    -- Reconstructed and unreconstructed variants of the same facet coexist.
    UNIQUE (entry_id, miller_index, is_reconstructed)
);

CREATE TABLE grain_boundary (
    grain_boundary_id SERIAL PRIMARY KEY,
    entry_id TEXT NOT NULL REFERENCES material_entry(entry_id),
    sigma_value INTEGER CHECK (sigma_value > 0),
    rotation_axis VARCHAR(10),
    tilt_angle DOUBLE PRECISION CHECK (tilt_angle >= 0 AND tilt_angle <= 180),
    gb_energy_j_m2 DOUBLE PRECISION CHECK (gb_energy_j_m2 > 0),
    excess_volume DOUBLE PRECISION CHECK (excess_volume >= 0),
    -- The same sigma / rotation axis admits distinct boundary geometries
    -- (e.g. sigma-5 [001] 36.87 deg vs 53.13 deg), so tilt_angle is part of
    -- the natural key. A full description would also need the GB plane,
    -- which this simplified model does not store.
    UNIQUE NULLS NOT DISTINCT (entry_id, sigma_value, rotation_axis, tilt_angle)
);

-- === Phase Diagram & Alloy Tables ===

CREATE TABLE phase_diagram_entry (
    phase_entry_id SERIAL PRIMARY KEY,
    entry_id TEXT NOT NULL REFERENCES material_entry(entry_id),
    chemical_system TEXT NOT NULL,
    is_on_hull BOOLEAN NOT NULL,
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

-- === Pure Element Reference (ground-state OQMD delta_e values) ===

CREATE TABLE pure_element_reference (
    pure_ref_id SERIAL PRIMARY KEY,
    element_symbol VARCHAR(5) NOT NULL REFERENCES element(symbol),
    -- Reference energies are only comparable within one energy convention;
    -- reference_set points at the reference_energy_set master, which fixes
    -- method / functional / source once per set (join the master to read
    -- them). formation_enthalpy pins one set.
    -- No DEFAULT on purpose: the energy convention is required scientific
    -- metadata, so omitting it must fail instead of silently becoming
    -- some default set.
    reference_set TEXT NOT NULL
        REFERENCES reference_energy_set(reference_set),
    oqmd_entry_id INTEGER,
    ground_state_spacegroup VARCHAR(30),
    -- OQMD delta_e (formation energy, eV/atom) of the element's ground-state
    -- entry, relative to the fitted elemental reference states of
    -- reference_set. This is NOT a total DFT energy and NOT the reference
    -- energy itself (OQMD's ReferenceEnergy.value). Subtracting
    -- SUM(x_i * delta_e_i) from a compound's formation_energy_per_atom of
    -- the SAME reference_set re-references it to the stored pure-element
    -- ground states (the fitted reference energies cancel); it must never
    -- be combined with formation energies of a different reference_set.
    delta_e DOUBLE PRECISION NOT NULL,
    volume_per_atom DOUBLE PRECISION CHECK (volume_per_atom > 0),  -- Angstrom^3/atom
    stability DOUBLE PRECISION CHECK (stability >= 0),  -- eV/atom above hull
    band_gap DOUBLE PRECISION CHECK (band_gap >= 0),  -- eV
    n_polymorphs INTEGER CHECK (n_polymorphs > 0),
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

-- Master-side companions: updating prototype_definition / space_group
-- propagates the new values to structure's derived copies, so a master
-- UPDATE cannot silently reintroduce two truths.
CREATE FUNCTION sync_structure_from_prototype() RETURNS trigger AS $$
BEGIN
    UPDATE structure
    SET strukturbericht = NEW.strukturbericht,
        formula_type = NEW.formula_type
    WHERE prototype = NEW.prototype_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_prototype_sync_structure
    AFTER UPDATE OF strukturbericht, formula_type ON prototype_definition
    FOR EACH ROW EXECUTE FUNCTION sync_structure_from_prototype();

CREATE FUNCTION sync_structure_from_space_group() RETURNS trigger AS $$
BEGIN
    UPDATE structure
    SET crystal_system = NEW.crystal_system,
        space_group = NEW.hermann_mauguin
    WHERE space_group_number = NEW.space_group_number;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_space_group_sync_structure
    AFTER UPDATE OF crystal_system, hermann_mauguin ON space_group
    FOR EACH ROW EXECUTE FUNCTION sync_structure_from_space_group();

-- === Dictionary scope enforcement: property_definition.applies_to ===
-- Each EAV child table may only reference properties whose applies_to
-- matches the table (calculated / measured / element); the FK alone does
-- not check this classification.
CREATE FUNCTION check_property_applies_to() RETURNS trigger AS $$
DECLARE
    expected TEXT := TG_ARGV[0];
    actual TEXT;
BEGIN
    SELECT applies_to INTO actual
    FROM property_definition WHERE canonical_name = NEW.property_name;
    IF actual IS DISTINCT FROM expected THEN
        RAISE EXCEPTION
            '%: property % has applies_to=%, expected %',
            TG_TABLE_NAME, NEW.property_name, actual, expected;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_calculated_property_applies_to
    BEFORE INSERT OR UPDATE OF property_name ON calculated_property
    FOR EACH ROW EXECUTE FUNCTION check_property_applies_to('calculated');

CREATE TRIGGER trg_measured_property_applies_to
    BEFORE INSERT OR UPDATE OF property_name ON measured_property
    FOR EACH ROW EXECUTE FUNCTION check_property_applies_to('measured');

CREATE TRIGGER trg_element_property_applies_to
    BEFORE INSERT OR UPDATE OF property_name ON element_property
    FOR EACH ROW EXECUTE FUNCTION check_property_applies_to('element');

-- Master-side guard: applies_to may not be changed while child rows still
-- reference the property, so the scope guarantee cannot be broken from the
-- dictionary side either. (Dictionary mutation and child-property writes
-- are not intended to occur concurrently in this verification DB.)
-- canonical_name renames are rejected by the child FKs themselves
-- (NO ACTION, no ON UPDATE CASCADE) once a property is referenced.
CREATE FUNCTION prevent_invalid_property_scope_change() RETURNS trigger AS $$
BEGIN
    IF NEW.applies_to IS NOT DISTINCT FROM OLD.applies_to THEN
        RETURN NEW;
    END IF;
    IF NEW.applies_to <> 'calculated' AND EXISTS (
        SELECT 1 FROM calculated_property
        WHERE property_name = OLD.canonical_name
    ) THEN
        RAISE EXCEPTION
            'property % is referenced by calculated_property and cannot change applies_to to %',
            OLD.canonical_name, NEW.applies_to;
    END IF;
    IF NEW.applies_to <> 'measured' AND EXISTS (
        SELECT 1 FROM measured_property
        WHERE property_name = OLD.canonical_name
    ) THEN
        RAISE EXCEPTION
            'property % is referenced by measured_property and cannot change applies_to to %',
            OLD.canonical_name, NEW.applies_to;
    END IF;
    IF NEW.applies_to <> 'element' AND EXISTS (
        SELECT 1 FROM element_property
        WHERE property_name = OLD.canonical_name
    ) THEN
        RAISE EXCEPTION
            'property % is referenced by element_property and cannot change applies_to to %',
            OLD.canonical_name, NEW.applies_to;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_property_definition_scope_change
    BEFORE UPDATE OF applies_to ON property_definition
    FOR EACH ROW EXECUTE FUNCTION prevent_invalid_property_scope_change();

-- === Canonical-unit enforcement ===
-- The composite FK (property_name, unit) pins unit to the canonical one
-- only when unit IS NOT NULL (MATCH SIMPLE skips NULLs). This trigger
-- closes the NULL loophole: when the dictionary declares a canonical unit,
-- child rows must carry exactly that unit (a unit-less value for a
-- unit-bearing property is rejected).
CREATE FUNCTION check_property_unit() RETURNS trigger AS $$
DECLARE
    cu TEXT;
BEGIN
    SELECT canonical_unit INTO cu
    FROM property_definition WHERE canonical_name = NEW.property_name;
    IF cu IS NOT NULL AND NEW.unit IS DISTINCT FROM cu THEN
        RAISE EXCEPTION
            '%: property % requires unit % (got %)',
            TG_TABLE_NAME, NEW.property_name, cu, COALESCE(NEW.unit, 'NULL');
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_calculated_property_unit
    BEFORE INSERT OR UPDATE OF property_name, unit ON calculated_property
    FOR EACH ROW EXECUTE FUNCTION check_property_unit();

CREATE TRIGGER trg_measured_property_unit
    BEFORE INSERT OR UPDATE OF property_name, unit ON measured_property
    FOR EACH ROW EXECUTE FUNCTION check_property_unit();

CREATE TRIGGER trg_element_property_unit
    BEFORE INSERT OR UPDATE OF property_name, unit ON element_property
    FOR EACH ROW EXECUTE FUNCTION check_property_unit();
