-- ============================================================
-- 001_schema.sql — Schema definition (DDL only)
-- 35 tables: 31 entity tables + property_definition dictionary
--            + property_scope (property -> storage-scope relation)
--            + reference_energy_set (energy-convention master)
--            + fixture_source_reference_set (source_db -> reference_set map)
-- Load order: 001_schema -> 002_reference_data -> 003_material_data
--             -> 004_views -> 005_roles -> 006_integrity_checks
--             -> 007_initialization_marker
--
-- Design rules enforced at the DDL level:
--   * Core controlled vocabularies used for joins and evaluation are
--     FK-constrained (composition.element, structure.prototype,
--     structure.space_group_number, EAV property names/units via
--     property_definition, pure_element_reference.reference_set via
--     reference_energy_set, element.category via a CHECK-constrained
--     snake_case vocabulary); free-text descriptors (e.g. source_db,
--     atmosphere) intentionally remain unconstrained.
--   * Cardinality is explicit: 1:1 relations carry UNIQUE(entry_id) /
--     UNIQUE(calculation_id); 1:N relations carry a composite natural key.
--   * phase_stability.is_stable is a generated column derived from
--     energy_above_hull (operational definition: stable <=> E_hull <= 0.001).
--   * Calculation-derived tables reference calculation only; entry_id is
--     reached via calculation.entry_id (no redundant, unconstrained copies).
--   * Physical-range CHECK constraints on fractions, energies, scores.
--   * Physical quantities carry finiteness CHECKs: PostgreSQL floats
--     accept NaN / Infinity / -Infinity, which are never valid physical
--     values here, so the CHECKs reject them (NaN fails `x < 'Infinity'`
--     because NaN sorts above Infinity in PostgreSQL).
-- ============================================================

-- === Core Entity Tables ===

CREATE TABLE material_entry (
    entry_id TEXT PRIMARY KEY,
    -- Provenance identity: source_material_id is the entry's ID inside the
    -- source database (a synthetic label in this fixture), so the same
    -- external record may not be ingested twice under different entry_ids.
    source_db TEXT NOT NULL,
    source_material_id TEXT NOT NULL,
    UNIQUE (source_db, source_material_id),
    formula TEXT NOT NULL,
    reduced_formula TEXT NOT NULL,
    chemical_system TEXT,
    number_of_elements INTEGER NOT NULL CHECK (number_of_elements > 0)
);

-- === Element & Periodic Table ===

CREATE TABLE element (
    element_id SERIAL PRIMARY KEY,
    symbol VARCHAR(5) NOT NULL UNIQUE,
    name VARCHAR(50),
    -- Periodic-table master: atomic numbers are unique and bounded by the
    -- currently known elements (1..118).
    atomic_number INTEGER NOT NULL UNIQUE
        CHECK (atomic_number BETWEEN 1 AND 118),
    -- NUMERIC admits NaN (which compares greater than any number), so
    -- one-sided lower bounds alone would not reject it; hence <> 'NaN'.
    atomic_mass NUMERIC(10,4) CHECK (atomic_mass > 0 AND atomic_mass <> 'NaN'),
    electronegativity NUMERIC(5,3)
        CHECK (electronegativity >= 0 AND electronegativity <> 'NaN'),
    atomic_radius NUMERIC(6,2)
        CHECK (atomic_radius > 0 AND atomic_radius <> 'NaN'),
    group_number INTEGER CHECK (group_number BETWEEN 1 AND 18),
    period_number INTEGER CHECK (period_number BETWEEN 1 AND 7),
    -- block/group_number/period_number: block is required for every
    -- fixture element; group_number is NULL for f-block interior elements
    -- (no IUPAC group). category is a required controlled vocabulary
    -- because gold SQL filters on it.
    -- Taxonomy convention (fixed for this fixture): transition_metal
    -- covers Sc–Zn, Y–Cd and Hf–Hg, i.e. the group-12 elements
    -- Zn/Cd/Hg are classified as transition_metal (some taxonomies
    -- place them in post_transition_metal instead; this fixture
    -- deliberately uses the d-block definition, and the natural-language
    -- questions reference this database category explicitly).
    block VARCHAR(5) NOT NULL,
    category VARCHAR(50) NOT NULL CHECK (
        category IN (
            'alkali_metal',
            'alkaline_earth_metal',
            'transition_metal',
            'post_transition_metal',
            'metalloid',
            'nonmetal',
            'halogen',
            'noble_gas',
            'lanthanide',
            'actinide'
        )
    )
);

-- === Property dictionary (canonical names & units for EAV tables) ===

CREATE TABLE property_definition (
    property_def_id SERIAL PRIMARY KEY,
    canonical_name VARCHAR(100) NOT NULL UNIQUE,
    canonical_unit VARCHAR(30),
    -- Storage contract: all EAV value columns are numeric
    -- (DOUBLE PRECISION / NUMERIC) and no integer-typed property is used
    -- by this verification DB, so the dictionary only declares the one
    -- type the DB actually enforces. Allowing 'integer' here without a
    -- trunc(value) trigger would make the declared type an unenforced
    -- self-report; text/boolean properties are likewise outside this
    -- schema.
    value_type VARCHAR(20) NOT NULL DEFAULT 'float'
        CHECK (value_type = 'float'),
    -- Shape contract for calculated_property.tensor_component:
    --   'scalar'    -> child rows must have tensor_component IS NULL
    --   'component' -> child rows must carry a tensor_component
    -- (enforced by trg_calculated_property_shape below).
    value_shape VARCHAR(20) NOT NULL DEFAULT 'scalar'
        CHECK (value_shape IN ('scalar', 'component')),
    description TEXT,
    UNIQUE (canonical_name, canonical_unit)
);

-- Scopes a property may be used in (many-to-many): one property can be
-- legitimately stored as calculated AND measured (e.g. a lattice
-- parameter). A single-valued applies_to column could not represent
-- that, so the scope classification lives in this relation and the EAV
-- child-table triggers below consult it.
-- value_shape='component' is only representable in calculated_property
-- (the only child table with a component column), so component-shaped
-- properties may only carry the 'calculated' scope (enforced by
-- trg_property_scope_shape / trg_property_definition_shape_change).
CREATE TABLE property_scope (
    property_name VARCHAR(100) NOT NULL
        REFERENCES property_definition(canonical_name),
    applies_to VARCHAR(30) NOT NULL
        CHECK (applies_to IN ('calculated', 'measured', 'element')),
    PRIMARY KEY (property_name, applies_to)
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
    -- Required: the semantic audit derives prototype stoichiometry from it.
    formula_type TEXT NOT NULL,
    -- Atoms in the conventional unit cell (L12=4, B2=2, NaCl=8, D03=16);
    -- NULL for per-element ground-state prototypes whose cells vary.
    conventional_cell_atoms INTEGER
        CHECK (conventional_cell_atoms > 0),
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
    -- Fixture contract: every structure row carries its prototype,
    -- formula type, space group and crystal system (only strukturbericht
    -- may be NULL — pure-element ground states have none).
    prototype TEXT NOT NULL REFERENCES prototype_definition(prototype_id),
    strukturbericht TEXT,
    formula_type TEXT NOT NULL,
    space_group_number INTEGER NOT NULL
        REFERENCES space_group(space_group_number),
    crystal_system TEXT NOT NULL,
    -- Finite-only physical values: NaN sorts above every number in
    -- PostgreSQL, so 'x > 0' alone would accept NaN and +Infinity;
    -- the upper bound rejects both.
    -- Lattice geometry NULL policy: compound entries carry full lattice
    -- parameters; OQMD pure-element ground states carry only their
    -- per-atom volume (their conventional cells vary), so lattice_a/b/c
    -- may be unknown — but only as a whole. The table CHECK below rejects
    -- partially-known geometry (a NULL hiding inside a non-NULL set).
    lattice_a DOUBLE PRECISION CHECK (lattice_a > 0 AND lattice_a < 'Infinity'),
    lattice_b DOUBLE PRECISION CHECK (lattice_b > 0 AND lattice_b < 'Infinity'),
    lattice_c DOUBLE PRECISION CHECK (lattice_c > 0 AND lattice_c < 'Infinity'),
    -- volume_per_atom is present for every fixture structure row.
    volume_per_atom DOUBLE PRECISION NOT NULL
        CHECK (volume_per_atom > 0 AND volume_per_atom < 'Infinity'),
    space_group TEXT NOT NULL,
    CHECK (
        (lattice_a IS NULL AND lattice_b IS NULL AND lattice_c IS NULL)
        OR (lattice_a IS NOT NULL AND lattice_b IS NOT NULL
            AND lattice_c IS NOT NULL)
    )
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
    -- relative to (e.g. the OQMD standard fit); the set name alone would
    -- not identify which reference fit was used. Once a set is referenced
    -- by loaded energies, these convention fields are immutable (see
    -- trg_reference_energy_set_change below).
    fit_name TEXT NOT NULL,
    description TEXT
);

-- Which energy convention each material source is allowed to declare.
-- material_entry.source_db stays free text, but 006 asserts that every
-- (source_db, phase_stability.reference_set) pair actually loaded is
-- present in this map, so a source cannot silently declare a convention
-- it was never mapped to.
-- Scope: this table does NOT assert physical compatibility with the real
-- external database named by source_db (source_db is a synthetic
-- provenance label in this fixture, not the origin of the energy values).
-- It only declares which (label, reference_set) combinations this
-- fixture is allowed to load.
CREATE TABLE fixture_source_reference_set (
    source_db TEXT,
    reference_set TEXT REFERENCES reference_energy_set(reference_set),
    PRIMARY KEY (source_db, reference_set)
);

-- Operational stability definition (paper / gold SQL / DB single source):
--   stable <=> energy_above_hull <= 0.001 eV/atom
CREATE TABLE phase_stability (
    stability_id TEXT PRIMARY KEY,
    entry_id TEXT NOT NULL UNIQUE REFERENCES material_entry(entry_id),
    -- Formation energy per atom relative to the elemental reference states
    -- of reference_set below. NOT NULL: every stability row must carry it
    -- (a NULL here could not be distinguished from a computation gap).
    formation_energy_per_atom DOUBLE PRECISION NOT NULL
        CHECK (formation_energy_per_atom > '-Infinity'
           AND formation_energy_per_atom < 'Infinity'),
    -- Energy convention of formation_energy_per_atom. Formation energies
    -- from different source databases use different pseudopotentials,
    -- corrections and elemental reference fits, so they are only
    -- comparable within one reference_set; views must join elemental
    -- references on the SAME reference_set, and 006 asserts that every
    -- loaded (source_db, reference_set) pair is declared in
    -- fixture_source_reference_set.
    reference_set TEXT NOT NULL
        REFERENCES reference_energy_set(reference_set),
    -- NOT NULL keeps the generated is_stable strictly two-valued
    -- (a NULL hull energy would make is_stable NULL, i.e. three-valued).
    energy_above_hull DOUBLE PRECISION NOT NULL
        CHECK (energy_above_hull >= 0 AND energy_above_hull < 'Infinity'),
    is_stable BOOLEAN GENERATED ALWAYS AS (energy_above_hull <= 0.001) STORED,
    -- NOT NULL: every fixture stability row carries a gap value (0 for
    -- metals); "gap unknown" is not a fixture state, which keeps the
    -- band-structure and metallicity integrity checks free of NULL holes.
    band_gap DOUBLE PRECISION NOT NULL
        CHECK (band_gap >= 0 AND band_gap < 'Infinity')
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
    -- NOT NULL: a property row without a value would make COUNT(*) and
    -- COUNT(value) diverge; unknown values are represented by absence.
    value DOUBLE PRECISION NOT NULL
        CHECK (value > '-Infinity' AND value < 'Infinity'),
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
    -- NOT NULL: unknown values are represented by absence (see
    -- calculated_property.value). NUMERIC still admits NaN, hence the CHECK.
    value NUMERIC(15,6) NOT NULL CHECK (value <> 'NaN'),
    unit VARCHAR(30),
    temperature_k NUMERIC(8,2)
        CHECK (temperature_k >= 0 AND temperature_k <> 'NaN'),
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
    relevance_score NUMERIC(5,3)
        CHECK (relevance_score BETWEEN 0 AND 1 AND relevance_score <> 'NaN'),
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
    temperature_k NUMERIC(8,2)
        CHECK (temperature_k >= 0 AND temperature_k <> 'NaN'),
    pressure_gpa NUMERIC(8,3)
        CHECK (pressure_gpa >= 0 AND pressure_gpa <> 'NaN'),
    -- One measurement run per (entry, reference, method, T, P,
    -- measurement_run); measurement_run distinguishes replicates, so
    -- accidental double-loading of the same run is rejected while real
    -- replicate measurements remain representable.
    -- NULLS NOT DISTINCT consequence (documented limitation): NULL condition
    -- values do NOT represent independent measurements — only one
    -- unknown-condition measurement per (entry, reference, method) is
    -- representable. Distinct real-world measurements must carry their
    -- actual conditions to coexist.
    -- measurement_run distinguishes independent measurements made under
    -- the same (reference, method, T, P); the current fixture only loads
    -- run 1, but the natural key does not forbid real replicates.
    measurement_run INTEGER NOT NULL DEFAULT 1 CHECK (measurement_run > 0),
    UNIQUE NULLS NOT DISTINCT
        (entry_id, reference_id, method, temperature_k, pressure_gpa,
         measurement_run)
);

CREATE TABLE measured_property (
    measured_property_id SERIAL PRIMARY KEY,
    measurement_id INTEGER NOT NULL REFERENCES experimental_measurement(measurement_id),
    property_name VARCHAR(100) NOT NULL,
    -- NOT NULL: unknown values are represented by absence (see
    -- calculated_property.value). NUMERIC still admits NaN, hence the CHECK.
    value NUMERIC(15,6) NOT NULL CHECK (value <> 'NaN'),
    uncertainty NUMERIC(15,6)
        CHECK (uncertainty >= 0 AND uncertainty <> 'NaN'),
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
    temperature_k NUMERIC(8,2)
        CHECK (temperature_k >= 0 AND temperature_k <> 'NaN'),
    duration_hours NUMERIC(10,2)
        CHECK (duration_hours >= 0 AND duration_hours <> 'NaN'),
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
    category VARCHAR(50),  -- 'point', 'line', 'planar'
    description TEXT
);

CREATE TABLE material_defect (
    material_defect_id SERIAL PRIMARY KEY,
    entry_id TEXT NOT NULL REFERENCES material_entry(entry_id),
    defect_type_id INTEGER NOT NULL REFERENCES defect_type(defect_type_id),
    formation_energy NUMERIC(10,6) CHECK (formation_energy <> 'NaN'),
    concentration NUMERIC(15,8)
        CHECK (concentration >= 0 AND concentration <> 'NaN'),
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
    -- Single truth for gap directness: band_gap_type is the stored value
    -- and is_direct_gap is derived from it, so the two can never disagree.
    band_gap_type VARCHAR(20) NOT NULL
        CHECK (band_gap_type IN ('direct', 'indirect')),
    is_direct_gap BOOLEAN GENERATED ALWAYS AS
        (band_gap_type = 'direct') STORED,
    -- Sign-free energies: finite-only (NaN / +-Infinity rejected).
    -- NOT NULL: a band-structure row asserts both band edges, so the
    -- integrity check band_gap = cbm - vbm has no NULL escape hatch.
    cbm_energy DOUBLE PRECISION NOT NULL
        CHECK (cbm_energy > '-Infinity' AND cbm_energy < 'Infinity'),
    vbm_energy DOUBLE PRECISION NOT NULL
        CHECK (vbm_energy > '-Infinity' AND vbm_energy < 'Infinity'),
    num_bands INTEGER CHECK (num_bands > 0),
    num_kpoints INTEGER CHECK (num_kpoints > 0)
);

CREATE TABLE density_of_states (
    dos_id SERIAL PRIMARY KEY,
    calculation_id TEXT NOT NULL UNIQUE REFERENCES calculation(calculation_id),
    total_dos_at_fermi DOUBLE PRECISION
        CHECK (total_dos_at_fermi >= 0 AND total_dos_at_fermi < 'Infinity'),
    efermi DOUBLE PRECISION
        CHECK (efermi > '-Infinity' AND efermi < 'Infinity'),
    -- NULL means metallicity was not determined for this DOS record;
    -- three-valued on purpose, unlike the other two-state flags.
    is_metallic BOOLEAN,
    spin_polarized BOOLEAN NOT NULL
);

-- === Mechanical/Physical Property Tables ===

-- Intentional denormalized duplicate: the scalar moduli below are also
-- mirrored into calculated_property (EAV) so the benchmark can pose both
-- wide-table and EAV navigation questions against the same physics. Both
-- copies are written from one generated value and their equality is
-- asserted by validate_fixture_integrity(); the fixture is immutable, so
-- partial post-load updates are unsupported.
CREATE TABLE elastic_tensor (
    elastic_id SERIAL PRIMARY KEY,
    calculation_id TEXT NOT NULL UNIQUE REFERENCES calculation(calculation_id),
    bulk_modulus_vrh DOUBLE PRECISION
        CHECK (bulk_modulus_vrh > 0 AND bulk_modulus_vrh < 'Infinity'),
    shear_modulus_vrh DOUBLE PRECISION
        CHECK (shear_modulus_vrh > 0 AND shear_modulus_vrh < 'Infinity'),
    youngs_modulus DOUBLE PRECISION
        CHECK (youngs_modulus > 0 AND youngs_modulus < 'Infinity'),
    poisson_ratio DOUBLE PRECISION CHECK (poisson_ratio > -1 AND poisson_ratio < 0.5),
    is_stable BOOLEAN NOT NULL  -- mechanical (Born) stability, distinct from phase stability
);

CREATE TABLE magnetic_property (
    magnetic_id SERIAL PRIMARY KEY,
    entry_id TEXT NOT NULL UNIQUE REFERENCES material_entry(entry_id),
    total_magnetization DOUBLE PRECISION
        CHECK (total_magnetization >= 0 AND total_magnetization < 'Infinity'),
    magnetic_ordering VARCHAR(30),
    curie_temperature_k DOUBLE PRECISION
        CHECK (curie_temperature_k >= 0 AND curie_temperature_k < 'Infinity'),
    magnetic_anisotropy_energy DOUBLE PRECISION
        CHECK (magnetic_anisotropy_energy > '-Infinity'
           AND magnetic_anisotropy_energy < 'Infinity')
);

CREATE TABLE thermal_property (
    thermal_id SERIAL PRIMARY KEY,
    calculation_id TEXT NOT NULL REFERENCES calculation(calculation_id),
    debye_temperature_k DOUBLE PRECISION
        CHECK (debye_temperature_k > 0 AND debye_temperature_k < 'Infinity'),
    thermal_conductivity DOUBLE PRECISION
        CHECK (thermal_conductivity >= 0 AND thermal_conductivity < 'Infinity'),
    specific_heat_cv DOUBLE PRECISION
        CHECK (specific_heat_cv >= 0 AND specific_heat_cv < 'Infinity'),
    gruneisen_parameter DOUBLE PRECISION
        CHECK (gruneisen_parameter > '-Infinity'
           AND gruneisen_parameter < 'Infinity'),
    temperature_k DOUBLE PRECISION NOT NULL DEFAULT 300.0
        CHECK (temperature_k >= 0 AND temperature_k < 'Infinity'),
    UNIQUE (calculation_id, temperature_k)
);

-- === Surface/Interface Tables ===

CREATE TABLE surface_energy (
    surface_id SERIAL PRIMARY KEY,
    entry_id TEXT NOT NULL REFERENCES material_entry(entry_id),
    miller_index VARCHAR(10) NOT NULL,
    surface_energy_j_m2 DOUBLE PRECISION
        CHECK (surface_energy_j_m2 > 0 AND surface_energy_j_m2 < 'Infinity'),
    work_function DOUBLE PRECISION
        CHECK (work_function > 0 AND work_function < 'Infinity'),
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
    gb_energy_j_m2 DOUBLE PRECISION
        CHECK (gb_energy_j_m2 > 0 AND gb_energy_j_m2 < 'Infinity'),
    excess_volume DOUBLE PRECISION
        CHECK (excess_volume >= 0 AND excess_volume < 'Infinity'),
    -- The same sigma / rotation axis admits distinct boundary geometries
    -- (e.g. sigma-5 [001] 36.87 deg vs 53.13 deg), so tilt_angle is part of
    -- the natural key. A full description would also need the GB plane,
    -- which this simplified model does not store.
    UNIQUE NULLS NOT DISTINCT (entry_id, sigma_value, rotation_axis, tilt_angle)
);

-- === Phase Diagram & Alloy Tables ===

CREATE TABLE phase_diagram_entry (
    phase_entry_id SERIAL PRIMARY KEY,
    -- One phase-diagram entry per material (its composition is fixed, so
    -- it belongs to exactly one chemical system). chemical_system must
    -- match material_entry.chemical_system and hull_distance must equal
    -- phase_stability.energy_above_hull; both cross-table copies are
    -- asserted by validate_fixture_integrity() (006).
    entry_id TEXT NOT NULL UNIQUE REFERENCES material_entry(entry_id),
    chemical_system TEXT NOT NULL,
    -- Single stability truth: is_on_hull is DERIVED from hull_distance
    -- with the same operational threshold as phase_stability.is_stable
    -- (on hull <=> hull_distance <= 0.001 eV/atom), so the two columns
    -- can never contradict each other.
    hull_distance DOUBLE PRECISION NOT NULL
        CHECK (hull_distance >= 0 AND hull_distance < 'Infinity'),
    is_on_hull BOOLEAN GENERATED ALWAYS AS (hull_distance <= 0.001) STORED,
    decomposition_products TEXT
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
    delta_e DOUBLE PRECISION NOT NULL
        CHECK (delta_e > '-Infinity' AND delta_e < 'Infinity'),
    volume_per_atom DOUBLE PRECISION
        CHECK (volume_per_atom > 0 AND volume_per_atom < 'Infinity'),  -- Angstrom^3/atom
    stability DOUBLE PRECISION
        CHECK (stability >= 0 AND stability < 'Infinity'),  -- eV/atom above hull
    band_gap DOUBLE PRECISION
        CHECK (band_gap >= 0 AND band_gap < 'Infinity'),  -- eV
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
CREATE INDEX idx_material_synthesis_entry ON material_synthesis(entry_id);
CREATE INDEX idx_material_synthesis_ref ON material_synthesis(reference_id);
CREATE INDEX idx_exp_measurement_ref ON experimental_measurement(reference_id);
CREATE INDEX idx_material_reference_ref ON material_reference(reference_id);
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

-- === Dictionary scope enforcement: property_scope ===
-- Each EAV child table may only reference properties that declare the
-- matching scope (calculated / measured / element) in property_scope;
-- the FK alone does not check this classification.
CREATE FUNCTION check_property_applies_to() RETURNS trigger AS $$
DECLARE
    expected TEXT := TG_ARGV[0];
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM property_scope
        WHERE property_name = NEW.property_name
          AND applies_to = expected
    ) THEN
        RAISE EXCEPTION
            '%: property % does not declare scope % in property_scope',
            TG_TABLE_NAME, NEW.property_name, expected;
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

-- === Dictionary shape enforcement: property_definition.value_shape ===
-- scalar properties must not carry a tensor_component; component-valued
-- properties must carry one. Without this, the tensor_component column is
-- an unchecked free-text axis.
CREATE FUNCTION check_property_value_shape() RETURNS trigger AS $$
DECLARE
    shape TEXT;
BEGIN
    SELECT value_shape INTO shape
    FROM property_definition WHERE canonical_name = NEW.property_name;
    IF shape = 'scalar' AND NEW.tensor_component IS NOT NULL THEN
        RAISE EXCEPTION
            'calculated_property: scalar property % may not carry tensor_component %',
            NEW.property_name, NEW.tensor_component;
    ELSIF shape = 'component' AND NEW.tensor_component IS NULL THEN
        RAISE EXCEPTION
            'calculated_property: component property % requires a tensor_component',
            NEW.property_name;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_calculated_property_shape
    BEFORE INSERT OR UPDATE OF property_name, tensor_component
    ON calculated_property
    FOR EACH ROW EXECUTE FUNCTION check_property_value_shape();

-- Master-side companion: value_shape may not change while child rows exist
-- in the opposite shape.
CREATE FUNCTION prevent_invalid_value_shape_change() RETURNS trigger AS $$
BEGIN
    IF NEW.value_shape IS NOT DISTINCT FROM OLD.value_shape THEN
        RETURN NEW;
    END IF;
    IF NEW.value_shape = 'scalar' AND EXISTS (
        SELECT 1 FROM calculated_property
        WHERE property_name = OLD.canonical_name
          AND tensor_component IS NOT NULL
    ) THEN
        RAISE EXCEPTION
            'property %: cannot become scalar while component rows exist',
            OLD.canonical_name;
    END IF;
    IF NEW.value_shape = 'component' AND EXISTS (
        SELECT 1 FROM calculated_property
        WHERE property_name = OLD.canonical_name
          AND tensor_component IS NULL
    ) THEN
        RAISE EXCEPTION
            'property %: cannot become component-valued while scalar rows exist',
            OLD.canonical_name;
    END IF;
    IF NEW.value_shape = 'component' AND EXISTS (
        SELECT 1 FROM property_scope
        WHERE property_name = OLD.canonical_name
          AND applies_to <> 'calculated'
    ) THEN
        RAISE EXCEPTION
            'property %: cannot become component-valued while scoped to measured/element storage',
            OLD.canonical_name;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_property_definition_shape_change
    BEFORE UPDATE OF value_shape ON property_definition
    FOR EACH ROW EXECUTE FUNCTION prevent_invalid_value_shape_change();

-- Master-side guard: a property_scope row may not be removed (or renamed
-- away) while child rows still use that scope, so the scope guarantee
-- cannot be broken from the dictionary side either. (Dictionary mutation
-- and child-property writes are not intended to occur concurrently in
-- this verification DB.) canonical_name renames are rejected by the
-- child FKs themselves (NO ACTION, no ON UPDATE CASCADE) once a property
-- is referenced.
CREATE FUNCTION prevent_invalid_property_scope_change() RETURNS trigger AS $$
BEGIN
    IF TG_OP = 'UPDATE'
       AND NEW.property_name = OLD.property_name
       AND NEW.applies_to = OLD.applies_to THEN
        RETURN NEW;
    END IF;
    IF OLD.applies_to = 'calculated' AND EXISTS (
        SELECT 1 FROM calculated_property
        WHERE property_name = OLD.property_name
    ) THEN
        RAISE EXCEPTION
            'property_scope (%, calculated) is in use by calculated_property and cannot be removed',
            OLD.property_name;
    END IF;
    IF OLD.applies_to = 'measured' AND EXISTS (
        SELECT 1 FROM measured_property
        WHERE property_name = OLD.property_name
    ) THEN
        RAISE EXCEPTION
            'property_scope (%, measured) is in use by measured_property and cannot be removed',
            OLD.property_name;
    END IF;
    IF OLD.applies_to = 'element' AND EXISTS (
        SELECT 1 FROM element_property
        WHERE property_name = OLD.property_name
    ) THEN
        RAISE EXCEPTION
            'property_scope (%, element) is in use by element_property and cannot be removed',
            OLD.property_name;
    END IF;
    IF TG_OP = 'DELETE' THEN
        RETURN OLD;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_property_scope_change
    BEFORE UPDATE OR DELETE ON property_scope
    FOR EACH ROW EXECUTE FUNCTION prevent_invalid_property_scope_change();

-- Shape/scope compatibility: only calculated_property has a component
-- column (tensor_component), so a component-shaped property may not be
-- scoped to measured/element storage that cannot record which component
-- a value belongs to.
CREATE FUNCTION check_property_scope_shape() RETURNS trigger AS $$
DECLARE
    shape TEXT;
BEGIN
    SELECT value_shape INTO shape
    FROM property_definition WHERE canonical_name = NEW.property_name;
    IF shape = 'component' AND NEW.applies_to <> 'calculated' THEN
        RAISE EXCEPTION
            'property % is component-shaped and may only carry the calculated scope (got %)',
            NEW.property_name, NEW.applies_to;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_property_scope_shape
    BEFORE INSERT OR UPDATE ON property_scope
    FOR EACH ROW EXECUTE FUNCTION check_property_scope_shape();

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

-- Master-side companion of check_property_unit: changing canonical_unit
-- while incompatible child rows exist would create dictionary/child
-- contradictions that the child-side trigger can no longer see (the
-- composite FK skips NULL child units under MATCH SIMPLE).
CREATE FUNCTION prevent_invalid_canonical_unit_change() RETURNS trigger AS $$
DECLARE
    tbl TEXT;
BEGIN
    IF NEW.canonical_unit IS NOT DISTINCT FROM OLD.canonical_unit THEN
        RETURN NEW;
    END IF;
    IF NEW.canonical_unit IS NOT NULL THEN
        IF EXISTS (SELECT 1 FROM calculated_property
                   WHERE property_name = OLD.canonical_name
                     AND unit IS DISTINCT FROM NEW.canonical_unit) THEN
            tbl := 'calculated_property';
        ELSIF EXISTS (SELECT 1 FROM measured_property
                      WHERE property_name = OLD.canonical_name
                        AND unit IS DISTINCT FROM NEW.canonical_unit) THEN
            tbl := 'measured_property';
        ELSIF EXISTS (SELECT 1 FROM element_property
                      WHERE property_name = OLD.canonical_name
                        AND unit IS DISTINCT FROM NEW.canonical_unit) THEN
            tbl := 'element_property';
        END IF;
        IF tbl IS NOT NULL THEN
            RAISE EXCEPTION
                'cannot change canonical unit for property % to %: % contains incompatible rows',
                OLD.canonical_name, NEW.canonical_unit, tbl;
        END IF;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_property_definition_unit_change
    BEFORE UPDATE OF canonical_unit ON property_definition
    FOR EACH ROW EXECUTE FUNCTION prevent_invalid_canonical_unit_change();

-- === Energy-convention master mutation guard ===
-- A reference_set row defines what every child energy value MEANS; once
-- phase_stability or pure_element_reference rows reference it, rewriting
-- method / functional / source / fit_name would silently change the
-- meaning of all loaded energies without breaking any FK. Reject it.
CREATE FUNCTION prevent_referenced_convention_change() RETURNS trigger AS $$
BEGIN
    IF NEW.method IS NOT DISTINCT FROM OLD.method
       AND NEW.functional IS NOT DISTINCT FROM OLD.functional
       AND NEW.source IS NOT DISTINCT FROM OLD.source
       AND NEW.fit_name IS NOT DISTINCT FROM OLD.fit_name THEN
        RETURN NEW;
    END IF;
    IF EXISTS (SELECT 1 FROM phase_stability
               WHERE reference_set = OLD.reference_set)
       OR EXISTS (SELECT 1 FROM pure_element_reference
                  WHERE reference_set = OLD.reference_set) THEN
        RAISE EXCEPTION
            'reference_set % is referenced by loaded energies; its convention (method/functional/source/fit_name) is immutable',
            OLD.reference_set;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_reference_energy_set_change
    BEFORE UPDATE ON reference_energy_set
    FOR EACH ROW EXECUTE FUNCTION prevent_referenced_convention_change();
