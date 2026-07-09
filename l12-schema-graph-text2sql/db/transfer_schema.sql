-- Transfer-test schema: an OQMD-flavored relational layout with table and
-- column names that differ from the L1_2 schema. Used to test whether the
-- text-to-SQL pipeline generalizes to an unseen schema (zero adaptation).

CREATE TABLE oqmd_elements (
    symbol          VARCHAR(5) PRIMARY KEY,
    element_name    TEXT,
    atomic_number   INTEGER,
    atomic_mass     DOUBLE PRECISION
);

CREATE TABLE oqmd_entries (
    entry_key           TEXT PRIMARY KEY,
    composition_formula TEXT NOT NULL,
    prototype_label     TEXT,
    spacegroup_number   INTEGER,
    crystal_system      TEXT,
    lattice_param_a     DOUBLE PRECISION,
    cell_volume_pa      DOUBLE PRECISION
);

CREATE TABLE oqmd_formation_energies (
    fe_key        TEXT PRIMARY KEY,
    entry_key     TEXT NOT NULL REFERENCES oqmd_entries(entry_key),
    delta_e       DOUBLE PRECISION,
    hull_distance DOUBLE PRECISION,
    on_hull       BOOLEAN,
    gap_ev        DOUBLE PRECISION
);

CREATE TABLE oqmd_element_ratios (
    ratio_key    TEXT PRIMARY KEY,
    entry_key    TEXT NOT NULL REFERENCES oqmd_entries(entry_key),
    symbol       VARCHAR(5) NOT NULL REFERENCES oqmd_elements(symbol),
    atomic_ratio DOUBLE PRECISION,
    wyckoff_site TEXT
);

CREATE TABLE oqmd_reference_states (
    ref_key            TEXT PRIMARY KEY,
    symbol             VARCHAR(5) NOT NULL REFERENCES oqmd_elements(symbol),
    gs_spacegroup      TEXT,
    energy_pa          DOUBLE PRECISION,
    volume_pa          DOUBLE PRECISION,
    polymorph_count    INTEGER
);

CREATE INDEX idx_tr_fe_entry ON oqmd_formation_energies(entry_key);
CREATE INDEX idx_tr_ratio_entry ON oqmd_element_ratios(entry_key);
CREATE INDEX idx_tr_ratio_symbol ON oqmd_element_ratios(symbol);
