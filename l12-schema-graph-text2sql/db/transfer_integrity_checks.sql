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
-- marker), which also records a schema fingerprint
-- (compute_transfer_schema_fingerprint(), same catalog scope as the main
-- schema's compute_schema_fingerprint()): scripts/transfer_guard.py
-- recomputes it on every guard call, so post-build schema drift (ALTER
-- TABLE / DROP CONSTRAINT / ...) is detected even when the dropped
-- constraint is not one the validator re-checks. Re-running against an
-- unchanged schema is a no-op; re-running after a schema change FAILS
-- (anti-reseal, mirrors 007) — rebuild the transfer DB instead.
--
-- The obfuscated transfer DB gets the same validator and marker with
-- identifiers renamed by scripts/build_obfuscated_transfer_db.py; the
-- marker table, the validator function name and the fingerprint function
-- name are deliberately NOT renamed so the guards can locate them in both
-- databases. The obfuscated DB is sealed with its OWN post-rename
-- fingerprint (the builder clears the marker row copied from the
-- template before re-running this file).
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
    -- one-phase_stability-per-material contract). The "at most one" side
    -- is re-checked here instead of being delegated to the DDL UNIQUE
    -- constraint: a CURRENT-STATE validator must keep holding even if the
    -- constraint was dropped after initialization.
    SELECT COUNT(*) INTO n_bad
    FROM (
        SELECT e.entry_key
        FROM oqmd_entries e
        LEFT JOIN oqmd_formation_energies f ON f.entry_key = e.entry_key
        GROUP BY e.entry_key
        HAVING COUNT(f.entry_key) <> 1
    ) x;
    IF n_bad > 0 THEN
        RAISE EXCEPTION
            'oqmd_entries: % entries do not have exactly one formation-energy row',
            n_bad;
    END IF;

    -- Exactly one reference state per symbol (independent of the DDL
    -- UNIQUE(symbol) constraint, same rationale as above).
    SELECT COUNT(*) INTO n_bad
    FROM (
        SELECT symbol
        FROM oqmd_reference_states
        GROUP BY symbol
        HAVING COUNT(*) <> 1
    ) x;
    IF n_bad > 0 THEN
        RAISE EXCEPTION
            'oqmd_reference_states: % symbols do not have exactly one reference state',
            n_bad;
    END IF;

    -- Ratio natural key: one row per (entry, element, site), independent
    -- of the DDL UNIQUE NULLS NOT DISTINCT constraint.
    SELECT COUNT(*) INTO n_bad
    FROM (
        SELECT entry_key, symbol, wyckoff_site
        FROM oqmd_element_ratios
        GROUP BY entry_key, symbol, wyckoff_site
        HAVING COUNT(*) <> 1
    ) x;
    IF n_bad > 0 THEN
        RAISE EXCEPTION
            'oqmd_element_ratios: % (entry, element, site) keys have duplicate rows',
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

-- Schema fingerprint (mirrors the main schema's
-- compute_schema_fingerprint()): columns with exact types, nullability,
-- defaults and generation expressions, every constraint / view / trigger
-- definition and every public function body. Same scope exclusions as the
-- main fingerprint (indexes, roles, GRANTs, GUCs are out of scope). The
-- marker table itself and this function are excluded so sealing does not
-- change the sealed value.
CREATE OR REPLACE FUNCTION compute_transfer_schema_fingerprint()
RETURNS TEXT AS $$
    WITH parts AS (
        SELECT 'col:' || c.relname || '.' || a.attname
               || ':' || format_type(a.atttypid, a.atttypmod)
               || ':' || a.attnotnull
               || ':' || a.attgenerated::TEXT
               || ':' || COALESCE(pg_get_expr(d.adbin, d.adrelid), '')
               AS part,
               1 AS grp, c.relname AS k1, a.attnum AS k2
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum > 0
             AND NOT a.attisdropped
        LEFT JOIN pg_attrdef d ON d.adrelid = c.oid AND d.adnum = a.attnum
        WHERE n.nspname = 'public' AND c.relkind IN ('r', 'v')
          AND c.relname <> 'transfer_initialization_status'
        UNION ALL
        SELECT 'con:' || c.relname || ':' || con.conname
               || ':' || pg_get_constraintdef(con.oid),
               2, c.relname, 0
        FROM pg_constraint con
        JOIN pg_class c ON c.oid = con.conrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public'
          AND c.relname <> 'transfer_initialization_status'
        UNION ALL
        SELECT 'view:' || c.relname || ':' || pg_get_viewdef(c.oid),
               3, c.relname, 0
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public' AND c.relkind = 'v'
        UNION ALL
        SELECT 'trg:' || c.relname || ':' || t.tgname
               || ':' || pg_get_triggerdef(t.oid),
               4, c.relname, 0
        FROM pg_trigger t
        JOIN pg_class c ON c.oid = t.tgrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public' AND NOT t.tgisinternal
        UNION ALL
        SELECT 'fn:' || p.proname || ':' || pg_get_functiondef(p.oid),
               5, p.proname, 0
        FROM pg_proc p
        JOIN pg_namespace n ON n.oid = p.pronamespace
        WHERE n.nspname = 'public' AND p.prokind = 'f'
          AND p.proname <> 'compute_transfer_schema_fingerprint'
    )
    SELECT encode(sha256(convert_to(
               string_agg(part, ',' ORDER BY grp, k1, k2, part),
               'UTF8')), 'hex')
    FROM parts;
$$ LANGUAGE sql STABLE;

-- Gate the marker on the assertions actually passing now.
SELECT validate_transfer_integrity();

-- Transfer DBs built by a pre-fingerprint package revision lack the
-- schema_fingerprint column; they must be REBUILT via
-- scripts/build_transfer_db.py (no in-place migration — the whole point
-- is that the fingerprint seals the freshly built schema).
CREATE TABLE IF NOT EXISTS transfer_initialization_status (
    version TEXT PRIMARY KEY,
    schema_fingerprint TEXT NOT NULL,
    -- Build provenance, same GUC convention as the main 007 marker
    -- (l12.git_commit; the builders SET it from the GIT_COMMIT file/env).
    git_commit TEXT NOT NULL DEFAULT 'unknown',
    initialized_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Anti-reseal guard (mirrors the main 007 marker): an existing marker
-- whose fingerprint differs from the current schema means the schema was
-- changed AFTER initialization — refuse to re-legitimize it.
DO $$
DECLARE
    old_fp TEXT;
    new_fp TEXT;
BEGIN
    SELECT schema_fingerprint INTO old_fp
    FROM transfer_initialization_status
    WHERE version = '001';
    new_fp := compute_transfer_schema_fingerprint();
    IF old_fp IS NOT NULL AND old_fp <> new_fp THEN
        RAISE EXCEPTION
            'transfer schema fingerprint changed after initialization (recorded %, current %); refusing to re-seal the marker — rebuild the transfer DB via scripts/build_transfer_db.py instead',
            substr(old_fp, 1, 12), substr(new_fp, 1, 12);
    END IF;
END
$$;

INSERT INTO transfer_initialization_status (version, schema_fingerprint,
                                            git_commit)
VALUES ('001', compute_transfer_schema_fingerprint(),
        COALESCE(NULLIF(current_setting('l12.git_commit', true), ''),
                 'unknown'))
ON CONFLICT (version) DO NOTHING;
