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

-- The marker is bound to a fingerprint of the actual schema shape
-- (md5 over the ordered table/column/type catalog of the public schema),
-- not just a version string: a marker copied onto a structurally
-- different database is detectable by recomputing the same expression.
CREATE OR REPLACE FUNCTION compute_schema_fingerprint() RETURNS TEXT AS $$
    SELECT md5(string_agg(
        table_name || '.' || column_name || ':' || data_type
        || ':' || is_nullable,
        ',' ORDER BY table_name, ordinal_position))
    FROM information_schema.columns
    WHERE table_schema = 'public'
      AND table_name <> 'schema_initialization_status';
$$ LANGUAGE sql STABLE;

CREATE TABLE IF NOT EXISTS schema_initialization_status (
    version TEXT PRIMARY KEY,
    schema_fingerprint TEXT NOT NULL,
    initialized_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
INSERT INTO schema_initialization_status (version, schema_fingerprint)
VALUES ('007', compute_schema_fingerprint())
ON CONFLICT (version) DO UPDATE
    SET schema_fingerprint = EXCLUDED.schema_fingerprint,
        initialized_at = now();
