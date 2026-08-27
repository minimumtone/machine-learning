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
        HAVING ABS(SUM(atomic_fraction) - 1.0) > 1e-8
    ) t;
    IF n_bad > 0 THEN
        RAISE EXCEPTION
            'composition: % entries whose atomic fractions do not sum to 1',
            n_bad;
    END IF;
END
$$;

-- Every material_entry with composition rows must be fully covered
-- (no entry whose declared element count differs from its distinct
-- composition elements; site-resolved rows may repeat an element).
DO $$
DECLARE
    n_bad BIGINT;
BEGIN
    SELECT COUNT(*) INTO n_bad
    FROM material_entry m
    JOIN (
        SELECT entry_id, COUNT(DISTINCT element) AS n
        FROM composition
        GROUP BY entry_id
    ) c ON c.entry_id = m.entry_id
    WHERE m.number_of_elements IS NOT NULL
      AND m.number_of_elements <> c.n;
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
