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

CREATE TABLE IF NOT EXISTS schema_initialization_status (
    version TEXT PRIMARY KEY,
    initialized_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
INSERT INTO schema_initialization_status (version) VALUES ('007')
ON CONFLICT (version) DO UPDATE SET initialized_at = now();
