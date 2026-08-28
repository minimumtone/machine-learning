-- Transfer-test schema: an OQMD-flavored relational layout with table and
-- column names that differ from the L1_2 schema. Used to test whether the
-- text-to-SQL pipeline generalizes to an unseen schema (zero adaptation).

CREATE TABLE oqmd_elements (
    symbol          VARCHAR(5) PRIMARY KEY,
    element_name    TEXT,
    atomic_number   INTEGER
        CHECK (atomic_number BETWEEN 1 AND 118),
    atomic_mass     DOUBLE PRECISION
        CHECK (atomic_mass > 0 AND atomic_mass < 'Infinity')
);

CREATE TABLE oqmd_entries (
    entry_key           TEXT PRIMARY KEY,
    composition_formula TEXT NOT NULL,
    prototype_label     TEXT,
    spacegroup_number   INTEGER,
    crystal_system      TEXT,
    lattice_param_a     DOUBLE PRECISION
        CHECK (lattice_param_a > 0 AND lattice_param_a < 'Infinity'),
    cell_volume_pa      DOUBLE PRECISION
        CHECK (cell_volume_pa > 0 AND cell_volume_pa < 'Infinity')
);

CREATE TABLE oqmd_formation_energies (
    fe_key        TEXT PRIMARY KEY,
    -- One formation-energy row per entry (mirrors the main schema's
    -- phase_stability.entry_id UNIQUE): duplicate energies for one entry
    -- would silently multiply every join through this table.
    entry_key     TEXT NOT NULL UNIQUE REFERENCES oqmd_entries(entry_key),
    -- Every row of this table IS a formation energy (copied from the main
    -- schema's NOT NULL formation_energy_per_atom), so NULL is not a valid
    -- state; the CHECK also rejects NaN / +-Infinity (NaN sorts above
    -- Infinity in PostgreSQL, so it fails `< 'Infinity'`).
    delta_e       DOUBLE PRECISION NOT NULL
        CHECK (delta_e > '-Infinity' AND delta_e < 'Infinity'),
    hull_distance DOUBLE PRECISION NOT NULL
        CHECK (hull_distance >= 0 AND hull_distance < 'Infinity'),
    -- Stability truth (single source): on_hull is DERIVED from
    -- hull_distance with the same operational definition as the main
    -- schema (stable <=> hull_distance <= 0.001 eV/atom), so the two
    -- columns can never contradict each other. Transfer gold SQL may use
    -- either; both are the same truth.
    on_hull       BOOLEAN GENERATED ALWAYS AS (hull_distance <= 0.001) STORED,
    gap_ev        DOUBLE PRECISION CHECK (gap_ev >= 0 AND gap_ev < 'Infinity')
);

CREATE TABLE oqmd_element_ratios (
    ratio_key    TEXT PRIMARY KEY,
    entry_key    TEXT NOT NULL REFERENCES oqmd_entries(entry_key),
    symbol       VARCHAR(5) NOT NULL REFERENCES oqmd_elements(symbol),
    atomic_ratio DOUBLE PRECISION NOT NULL
        CHECK (atomic_ratio > 0 AND atomic_ratio <= 1),
    wyckoff_site TEXT,
    -- Same natural key as the main schema's composition table: one row per
    -- (entry, element, site); per-entry ratio sums are asserted after load
    -- by db/transfer_integrity_checks.sql (a per-row CHECK cannot express
    -- the sum invariant).
    UNIQUE NULLS NOT DISTINCT (entry_key, symbol, wyckoff_site)
);

-- One elemental ground-state reference row per element. This transfer DB
-- carries a single energy convention (the main schema's L12-FIXTURE-PBE-v1
-- set), so symbol alone is the natural key; reference_delta_e is the
-- elemental delta_e of that convention copied from the main schema's
-- pure_element_reference.delta_e — it is NOT a total/raw DFT energy.
-- Subtracting SUM(atomic_ratio * reference_delta_e) from delta_e therefore
-- re-references a formation energy to the elemental ground states, exactly
-- like the main schema's enthalpy_vs_element_ground_states.
CREATE TABLE oqmd_reference_states (
    ref_key            TEXT PRIMARY KEY,
    symbol             VARCHAR(5) NOT NULL UNIQUE
        REFERENCES oqmd_elements(symbol),
    gs_spacegroup      TEXT,
    reference_delta_e  DOUBLE PRECISION NOT NULL
        CHECK (reference_delta_e > '-Infinity'
           AND reference_delta_e < 'Infinity'),
    volume_pa          DOUBLE PRECISION
        CHECK (volume_pa > 0 AND volume_pa < 'Infinity'),
    polymorph_count    INTEGER CHECK (polymorph_count >= 1)
);

CREATE INDEX idx_tr_fe_entry ON oqmd_formation_energies(entry_key);
CREATE INDEX idx_tr_ratio_entry ON oqmd_element_ratios(entry_key);
CREATE INDEX idx_tr_ratio_symbol ON oqmd_element_ratios(symbol);
