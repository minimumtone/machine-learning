-- ============================================================
-- 006_integrity_checks.sql — Post-load integrity assertions
-- Cross-row invariants that a per-row CHECK cannot express.
-- Each block RAISEs (aborting the load) if the invariant is violated,
-- so a database that finishes loading is guaranteed to satisfy them.
-- ============================================================

-- Composition must be normalized: atomic fractions sum to 1 per entry.
-- formation_enthalpy multiplies fractions by reference energies directly,
-- so an un-normalized composition would corrupt the corrected enthalpy.
DO $$
DECLARE
    n_bad BIGINT;
BEGIN
    SELECT COUNT(*) INTO n_bad
    FROM (
        SELECT entry_id
        FROM composition
        GROUP BY entry_id
        -- NULL fractions are already rejected by the DDL (NOT NULL);
        -- the FILTER term is a second line of defense against regressions.
        HAVING COUNT(*) FILTER (WHERE atomic_fraction IS NULL) > 0
            OR ABS(SUM(atomic_fraction) - 1.0) > 1e-8
    ) t;
    IF n_bad > 0 THEN
        RAISE EXCEPTION
            'composition: % entries whose atomic fractions do not sum to 1',
            n_bad;
    END IF;
END
$$;

-- Every material_entry must have composition rows and its declared
-- element count must equal the distinct composition elements
-- (site-resolved rows may repeat an element). LEFT JOIN + COALESCE
-- makes entries with zero composition rows fail (n = 0 <> declared).
DO $$
DECLARE
    n_bad BIGINT;
BEGIN
    SELECT COUNT(*) INTO n_bad
    FROM material_entry m
    LEFT JOIN (
        SELECT entry_id, COUNT(DISTINCT element) AS n
        FROM composition
        GROUP BY entry_id
    ) c ON c.entry_id = m.entry_id
    WHERE m.number_of_elements <> COALESCE(c.n, 0);
    IF n_bad > 0 THEN
        RAISE EXCEPTION
            'material_entry: % entries whose number_of_elements disagrees with distinct composition elements',
            n_bad;
    END IF;
END
$$;

-- NULL reference energies are now rejected row-by-row by the DDL
-- (pure_element_reference.energy_per_atom NOT NULL); this file keeps only
-- cross-row invariants that a single-row constraint cannot express.

-- The formation_enthalpy view pins reference_set = 'OQMD-PBE'; if the set
-- were renamed the view would not error but silently return NULLs, so its
-- existence and per-element coverage are asserted here.
DO $$
DECLARE
    n_ref BIGINT;
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM reference_energy_set WHERE reference_set = 'OQMD-PBE'
    ) THEN
        RAISE EXCEPTION
            'Required reference energy set OQMD-PBE does not exist';
    END IF;
    SELECT COUNT(*) INTO n_ref
    FROM (
        SELECT DISTINCT element FROM composition
        EXCEPT
        SELECT element_symbol FROM pure_element_reference
        WHERE reference_set = 'OQMD-PBE'
    ) missing;
    IF n_ref > 0 THEN
        RAISE EXCEPTION
            'pure_element_reference(OQMD-PBE) is missing % element(s) used in composition',
            n_ref;
    END IF;
END
$$;

-- structure copies of master attributes must match the master tables
-- (also enforced per-row by trg_structure_master_consistency; this is
-- the set-level assertion for databases loaded before the trigger).
DO $$
DECLARE
    n_bad BIGINT;
BEGIN
    SELECT COUNT(*) INTO n_bad
    FROM structure s
    LEFT JOIN prototype_definition p ON p.prototype_id = s.prototype
    LEFT JOIN space_group g ON g.space_group_number = s.space_group_number
    WHERE (s.prototype IS NOT NULL AND
           (s.strukturbericht IS DISTINCT FROM p.strukturbericht
            OR s.formula_type IS DISTINCT FROM p.formula_type))
       OR (s.space_group_number IS NOT NULL AND
           (s.crystal_system IS DISTINCT FROM g.crystal_system
            OR s.space_group IS DISTINCT FROM g.hermann_mauguin));
    IF n_bad > 0 THEN
        RAISE EXCEPTION
            'structure: % rows whose derived columns contradict the master tables',
            n_bad;
    END IF;
END
$$;

-- Initialization completion marker: created only after every assertion
-- above has passed. A database missing this row (e.g. 001–005 loaded but
-- 006 failed) is a partial initialization and must not be used as a
-- verification DB.
-- NOTE: this is an initialization completion marker only, NOT a current
-- integrity status — it records that the assertions held at load time and
-- is not invalidated by later writes. This package treats the loaded DB
-- as an immutable verification fixture (see README): post-006 entity
-- INSERT/UPDATE/DELETE is unsupported, and queries should run as the
-- read-only role l12_reader.
CREATE TABLE schema_initialization_status (
    version TEXT PRIMARY KEY,
    initialized_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
INSERT INTO schema_initialization_status (version) VALUES ('006');
