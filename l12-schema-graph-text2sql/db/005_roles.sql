-- ============================================================
-- 005_roles.sql — Read-only role for the Text-to-SQL execution layer
-- The Python-side validator is a first line of defence; PostgreSQL
-- privileges are the last line. The pipeline connects as l12_reader,
-- which can only SELECT and whose transactions default to read-only.
-- ============================================================

-- NOTE: 'l12_reader_password' is a LOCAL VERIFICATION-ONLY credential for
-- the throwaway Docker database. For any non-local deployment, create the
-- role with a real secret and point the pipeline at it via the
-- POSTGRES_EVAL_USER / POSTGRES_EVAL_PASSWORD environment variables
-- (see .env.example); nothing in the code requires this literal value.
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'l12_reader') THEN
        CREATE ROLE l12_reader LOGIN PASSWORD 'l12_reader_password';
    END IF;
END
$$;

REVOKE ALL ON SCHEMA public FROM l12_reader;
GRANT USAGE ON SCHEMA public TO l12_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO l12_reader;
-- Scope default privileges to the migration owner explicitly: a bare
-- ALTER DEFAULT PRIVILEGES only covers objects later created by the role
-- that ran it, so name the owner to keep the grant deterministic.
-- NOTE: this names the owner role verbatim, so this local-verification
-- setup requires POSTGRES_USER=l12_user (the docker-compose default).
-- Overriding POSTGRES_USER would make this statement fail; if you must
-- rename the owner, use current_user via a DO block instead.
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'l12_user') THEN
        RAISE EXCEPTION
            'role l12_user not found: POSTGRES_USER must stay l12_user for this local setup';
    END IF;
END
$$;
ALTER DEFAULT PRIVILEGES FOR ROLE l12_user IN SCHEMA public
    GRANT SELECT ON TABLES TO l12_reader;

ALTER ROLE l12_reader SET default_transaction_read_only = on;
ALTER ROLE l12_reader SET statement_timeout = '30s';
ALTER ROLE l12_reader SET search_path = public;
