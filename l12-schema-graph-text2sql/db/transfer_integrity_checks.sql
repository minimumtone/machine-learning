-- ============================================================
-- transfer_integrity_checks.sql — Post-load assertions for the transfer DB
-- Cross-row invariants a per-row CHECK cannot express (mirrors the main
-- schema's 006_integrity_checks.sql). Assertion-only and re-runnable.
-- ============================================================

-- Element ratios must be normalized: per-entry ratios sum to 1, so the
-- weighted reference sums in the transfer gold SQL are well-defined.
DO $$
DECLARE
    n_bad BIGINT;
BEGIN
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
END
$$;

-- Every element used by a loaded entry must have exactly one reference
-- state (symbol is UNIQUE by DDL; this asserts coverage).
DO $$
DECLARE
    n_missing BIGINT;
BEGIN
    SELECT COUNT(*) INTO n_missing
    FROM (
        SELECT DISTINCT r.symbol
        FROM oqmd_element_ratios r
        EXCEPT
        SELECT rs.symbol FROM oqmd_reference_states rs
    ) missing;
    IF n_missing > 0 THEN
        RAISE EXCEPTION
            'oqmd_reference_states: % elements used by loaded entries have no reference state',
            n_missing;
    END IF;
END
$$;

-- Every entry with a formation energy must have ratio rows.
DO $$
DECLARE
    n_bad BIGINT;
BEGIN
    SELECT COUNT(*) INTO n_bad
    FROM oqmd_formation_energies f
    LEFT JOIN oqmd_element_ratios r ON r.entry_key = f.entry_key
    WHERE r.entry_key IS NULL;
    IF n_bad > 0 THEN
        RAISE EXCEPTION
            'oqmd_formation_energies: % entries have no element ratios',
            n_bad;
    END IF;
END
$$;
