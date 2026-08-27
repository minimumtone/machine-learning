-- ============================================================
-- 006_integrity_checks.sql — Post-load integrity assertions
-- Cross-row invariants that a per-row CHECK cannot express.
-- Each block RAISEs (aborting the load) if the invariant is violated,
-- so a database that finishes loading is guaranteed to satisfy them.
-- This file is assertion-only and idempotent: it may be re-run at any
-- time to re-validate the loaded data. The one-time initialization
-- marker lives in 007_initialization_marker.sql.
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

-- NULL elemental delta_e values are rejected row-by-row by the DDL
-- (pure_element_reference.delta_e NOT NULL); this file keeps only
-- cross-row invariants that a single-row constraint cannot express.

-- Set-wise reference coverage: for every energy convention actually used
-- by phase_stability, every element of every material in that convention
-- must have an elemental delta_e in the SAME reference_set. This is a
-- true set difference (per reviewer guidance), not a count comparison,
-- so it also catches per-set gaps when multiple conventions coexist.
DO $$
DECLARE
    n_missing BIGINT;
BEGIN
    SELECT COUNT(*) INTO n_missing
    FROM (
        SELECT DISTINCT ps.reference_set, c.element
        FROM phase_stability ps
        JOIN composition c ON c.entry_id = ps.entry_id
        EXCEPT
        SELECT per.reference_set, per.element_symbol
        FROM pure_element_reference per
    ) missing;
    IF n_missing > 0 THEN
        RAISE EXCEPTION
            'pure_element_reference: % (reference_set, element) pairs used by phase_stability materials have no elemental delta_e in the same set',
            n_missing;
    END IF;
END
$$;

-- Source/convention mapping: every (source_db, reference_set) pair that
-- was actually loaded must be declared in source_energy_convention, so a
-- source can never be silently assigned an energy convention it was not
-- mapped to (e.g. labeling Materials-Project-derived energies with an
-- OQMD convention would fail here unless explicitly declared and
-- documented in the map).
DO $$
DECLARE
    n_bad BIGINT;
BEGIN
    SELECT COUNT(*) INTO n_bad
    FROM material_entry m
    JOIN phase_stability ps ON ps.entry_id = m.entry_id
    LEFT JOIN source_energy_convention sec
        ON sec.source_db IS NOT DISTINCT FROM m.source_db
       AND sec.reference_set = ps.reference_set
    WHERE sec.reference_set IS NULL;
    IF n_bad > 0 THEN
        RAISE EXCEPTION
            'phase_stability: % rows whose (source_db, reference_set) pair is not declared in source_energy_convention',
            n_bad;
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

-- (Initialization completion marker moved to 007_initialization_marker.sql
-- so this assertion file stays re-runnable.)
