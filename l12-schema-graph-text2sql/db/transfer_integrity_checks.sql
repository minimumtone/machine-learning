-- ============================================================
-- transfer_integrity_checks.sql — Post-load assertions for the transfer DB
-- Cross-row invariants a per-row CHECK cannot express (mirrors the main
-- schema's 006_integrity_checks.sql / validate_fixture_integrity()).
--
-- All assertions live in validate_transfer_integrity(), a re-runnable
-- CURRENT-STATE validator: the builder, the gold verifiers and the
-- vocabulary audit all call it (SELECT validate_transfer_integrity()),
-- so "the integrity checks were run once at build time" is replaced by
-- "the database passes the integrity checks NOW".
--
-- The file ends by running the validator and, on success, writing the
-- transfer_initialization_status marker (mirrors the main schema's 007
-- marker). Re-running is a no-op (ON CONFLICT DO NOTHING).
--
-- The obfuscated transfer DB gets the same validator and marker with
-- identifiers renamed by scripts/build_obfuscated_transfer_db.py; the
-- marker table itself and the validator function name are deliberately
-- NOT renamed so the guards can locate them in both databases.
-- ============================================================

CREATE OR REPLACE FUNCTION validate_transfer_integrity() RETURNS void AS $$
DECLARE
    n_bad BIGINT;
BEGIN
    -- Element ratios must be normalized: per-entry ratios sum to 1, so
    -- the weighted reference sums in the transfer gold SQL are
    -- well-defined.
    SELECT COUNT(*) INTO n_bad
    FROM (
        SELECT entry_key
        FROM oqmd_element_ratios
        GROUP BY entry_key
        HAVING ABS(SUM(atomic_ratio) - 1.0) > 1e-8
    ) t;
    IF n_bad > 0 THEN
        RAISE EXCEPTION
            'oqmd_element_ratios: % entries whose atomic ratios do not sum to 1',
            n_bad;
    END IF;

    -- Every element used by a loaded entry must have exactly one
    -- reference state (symbol is UNIQUE by DDL; this asserts coverage).
    SELECT COUNT(*) INTO n_bad
    FROM (
        SELECT DISTINCT r.symbol
        FROM oqmd_element_ratios r
        EXCEPT
        SELECT rs.symbol FROM oqmd_reference_states rs
    ) missing;
    IF n_bad > 0 THEN
        RAISE EXCEPTION
            'oqmd_reference_states: % elements used by loaded entries have no reference state',
            n_bad;
    END IF;

    -- Exactly one formation energy per entry (mirrors the main schema's
    -- one-phase_stability-per-material contract): entry_key is UNIQUE by
    -- DDL (at most one), so asserting no missing row makes it exactly one.
    SELECT COUNT(*) INTO n_bad
    FROM oqmd_entries e
    LEFT JOIN oqmd_formation_energies f ON f.entry_key = e.entry_key
    WHERE f.entry_key IS NULL;
    IF n_bad > 0 THEN
        RAISE EXCEPTION
            'oqmd_entries: % entries have no formation-energy row',
            n_bad;
    END IF;

    -- Every entry with a formation energy must have ratio rows.
    SELECT COUNT(*) INTO n_bad
    FROM oqmd_formation_energies f
    LEFT JOIN oqmd_element_ratios r ON r.entry_key = f.entry_key
    WHERE r.entry_key IS NULL;
    IF n_bad > 0 THEN
        RAISE EXCEPTION
            'oqmd_formation_energies: % entries have no element ratios',
            n_bad;
    END IF;

    -- Composition coverage: every transfer entry must have at least one
    -- composition (ratio) row — an entry without composition would
    -- silently drop out of every composition-joining transfer gold query.
    SELECT COUNT(*) INTO n_bad
    FROM oqmd_entries e
    LEFT JOIN oqmd_element_ratios r ON r.entry_key = e.entry_key
    WHERE r.entry_key IS NULL;
    IF n_bad > 0 THEN
        RAISE EXCEPTION
            'oqmd_entries: % entries have no element ratio rows',
            n_bad;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Gate the marker on the assertions actually passing now.
SELECT validate_transfer_integrity();

CREATE TABLE IF NOT EXISTS transfer_initialization_status (
    version TEXT PRIMARY KEY,
    initialized_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO transfer_initialization_status (version)
VALUES ('001')
ON CONFLICT (version) DO NOTHING;
