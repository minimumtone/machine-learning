-- ============================================================
-- 007_initialization_marker.sql — Initialization completion marker
-- Run LAST, only after 006_integrity_checks.sql has passed. A database
-- missing the '007' row (e.g. 001–005 loaded but 006 failed) is a partial
-- initialization and must not be used as a verification DB.
-- NOTE: this is an initialization completion marker only, NOT a current
-- integrity status — it records that the assertions held at load time and
-- is not invalidated by later writes. This package treats the loaded DB
-- as an immutable verification fixture (see README): post-initialization
-- entity INSERT/UPDATE/DELETE is unsupported, and queries should run as
-- the read-only role l12_reader.
-- Idempotent: re-running only refreshes the timestamp.

-- The marker is gated on the 006 assertions: validate_fixture_integrity()
-- RAISEs on any violated invariant, so this whole file aborts and the
-- marker row is never created for a database that fails validation.
SELECT validate_fixture_integrity();

-- The marker is bound to a fingerprint of the actual schema shape, not
-- just a version string: a marker copied onto a structurally different
-- database is detectable by recomputing the same expression. The hash
-- covers, in deterministic order:
--   * columns with exact types (format_type includes precision/length),
--     nullability, defaults and generation expressions,
--   * every constraint definition (CHECK / FK / UNIQUE / PK) via
--     pg_get_constraintdef,
--   * every view body via pg_get_viewdef,
--   * every trigger definition via pg_get_triggerdef,
--   * every public function body via pg_get_functiondef,
-- so dropping a CHECK/FK/trigger, editing a view or trigger function, or
-- changing NUMERIC precision all change the fingerprint. SHA-256 (core
-- since PostgreSQL 11, no extension dependency) is used as a
-- corruption/mismatch detector.
--
-- Scope: this is a SEMANTIC schema fingerprint — it covers everything
-- that changes query semantics or the integrity contract. It does NOT
-- cover indexes, roles, GRANT/REVOKE, default privileges, or role/GUC
-- settings (default_transaction_read_only, statement_timeout, ...):
-- those affect performance and access control but not what the gold
-- queries compute, and are enforced separately by db/005_roles.sql and
-- the verifiers' own READ ONLY + timeout session settings.
CREATE OR REPLACE FUNCTION compute_schema_fingerprint() RETURNS TEXT AS $$
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
          AND c.relname <> 'schema_initialization_status'
        UNION ALL
        SELECT 'con:' || c.relname || ':' || con.conname
               || ':' || pg_get_constraintdef(con.oid),
               2, c.relname, 0
        FROM pg_constraint con
        JOIN pg_class c ON c.oid = con.conrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public'
          AND c.relname <> 'schema_initialization_status'
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
          AND p.proname <> 'compute_schema_fingerprint'
    )
    SELECT encode(sha256(convert_to(
               string_agg(part, ',' ORDER BY grp, k1, k2, part),
               'UTF8')), 'hex')
    FROM parts;
$$ LANGUAGE sql STABLE;

CREATE TABLE IF NOT EXISTS schema_initialization_status (
    version TEXT PRIMARY KEY,
    schema_fingerprint TEXT NOT NULL,
    -- Build provenance: the git commit of the package that initialized
    -- this database, passed at load time as the custom GUC
    -- l12.git_commit (e.g. PGOPTIONS="-c l12.git_commit=$(cat GIT_COMMIT)"
    -- or the GIT_COMMIT env var in docker/docker-compose.yml). 'unknown'
    -- when the loader did not supply it.
    git_commit TEXT NOT NULL DEFAULT 'unknown',
    initialized_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
-- Databases initialized by an older package revision predate git_commit;
-- make re-running this file against them idempotent.
ALTER TABLE schema_initialization_status
    ADD COLUMN IF NOT EXISTS git_commit TEXT NOT NULL DEFAULT 'unknown';
INSERT INTO schema_initialization_status (version, schema_fingerprint,
                                          git_commit)
VALUES ('007', compute_schema_fingerprint(),
        COALESCE(NULLIF(current_setting('l12.git_commit', true), ''),
                 'unknown'))
ON CONFLICT (version) DO UPDATE
    SET schema_fingerprint = EXCLUDED.schema_fingerprint,
        git_commit = EXCLUDED.git_commit,
        initialized_at = now();
