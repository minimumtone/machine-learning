-- ============================================================
-- 005_roles.sql — Read-only role for the Text-to-SQL execution layer
-- The Python-side validator is a first line of defence; PostgreSQL
-- privileges are the last line. The pipeline connects as l12_reader,
-- which can only SELECT and whose transactions default to read-only.
-- ============================================================

CREATE ROLE l12_reader LOGIN PASSWORD 'l12_reader_password';

REVOKE ALL ON SCHEMA public FROM l12_reader;
GRANT USAGE ON SCHEMA public TO l12_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO l12_reader;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO l12_reader;

ALTER ROLE l12_reader SET default_transaction_read_only = on;
ALTER ROLE l12_reader SET statement_timeout = '30s';
ALTER ROLE l12_reader SET search_path = public;
