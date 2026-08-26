"""Parse PostgreSQL information_schema to extract tables, columns, and FK relationships."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    import psycopg

PUBLIC_SCHEMA = "public"


@dataclass(frozen=True)
class ColumnMetadata:
    """Metadata for a single database column."""

    table_name: str
    column_name: str
    data_type: str
    is_nullable: bool
    is_primary_key: bool


@dataclass(frozen=True)
class ForeignKeyMetadata:
    """A foreign key relationship between two tables."""

    source_table: str
    source_column: str
    target_table: str
    target_column: str


def get_tables(
    conn: psycopg.Connection,
    schema: str = PUBLIC_SCHEMA,
    include_views: bool = True,
) -> list[str]:
    """Return table (and, by default, view) names in the specified schema.

    Views are first-class queryable relations for Text-to-SQL, so they are
    included by default; pass ``include_views=False`` for base tables only.
    """
    types = ("BASE TABLE", "VIEW") if include_views else ("BASE TABLE",)
    query = (
        "SELECT table_name "
        "FROM information_schema.tables "
        "WHERE table_schema = %s AND table_type = ANY(%s) "
        "ORDER BY table_name"
    )
    with conn.cursor() as cur:
        cur.execute(query, (schema, list(types)))
        return [row[0] for row in cur.fetchall()]


def get_columns(
    conn: psycopg.Connection,
    table_name: str,
    schema: str = PUBLIC_SCHEMA,
) -> list[ColumnMetadata]:
    """Return column metadata for the specified table."""
    sql = (
        "SELECT c.column_name, c.data_type, c.is_nullable = 'YES' AS is_nullable, "
        "       EXISTS ("
        "           SELECT 1"
        "           FROM information_schema.table_constraints tc"
        "           JOIN information_schema.key_column_usage kcu"
        "             ON tc.constraint_name = kcu.constraint_name"
        "            AND tc.table_schema = kcu.table_schema"
        "           WHERE tc.constraint_type = 'PRIMARY KEY'"
        "             AND tc.table_schema = c.table_schema"
        "             AND tc.table_name = c.table_name"
        "             AND kcu.column_name = c.column_name"
        "       ) AS is_primary_key"
        "  FROM information_schema.columns c"
        " WHERE c.table_schema = %s"
        "   AND c.table_name = %s"
        " ORDER BY c.ordinal_position"
    )
    with conn.cursor() as cur:
        cur.execute(sql, (schema, table_name))
        rows = cur.fetchall()
    return [
        ColumnMetadata(
            table_name=table_name,
            column_name=row[0],
            data_type=row[1],
            is_nullable=bool(row[2]),
            is_primary_key=bool(row[3]),
        )
        for row in rows
    ]


def get_foreign_keys(
    conn: psycopg.Connection,
    schema: str = PUBLIC_SCHEMA,
) -> list[ForeignKeyMetadata]:
    """Return foreign key relationships."""
    # pg_constraint keeps conkey/confkey positionally aligned, so composite
    # FKs pair source and target columns correctly (information_schema's
    # constraint_column_usage would cross-product them).
    sql = (
        "SELECT src.relname AS source_table,"
        "       sa.attname  AS source_column,"
        "       tgt.relname AS target_table,"
        "       ta.attname  AS target_column"
        "  FROM pg_constraint con"
        "  JOIN pg_class src ON src.oid = con.conrelid"
        "  JOIN pg_class tgt ON tgt.oid = con.confrelid"
        "  JOIN pg_namespace ns ON ns.oid = con.connamespace"
        "  CROSS JOIN LATERAL unnest(con.conkey, con.confkey)"
        "       WITH ORDINALITY AS pairs(src_attnum, tgt_attnum, ord)"
        "  JOIN pg_attribute sa"
        "    ON sa.attrelid = con.conrelid AND sa.attnum = pairs.src_attnum"
        "  JOIN pg_attribute ta"
        "    ON ta.attrelid = con.confrelid AND ta.attnum = pairs.tgt_attnum"
        " WHERE con.contype = 'f' AND ns.nspname = %s"
        " ORDER BY source_table, source_column"
    )
    with conn.cursor() as cur:
        cur.execute(sql, (schema,))
        rows = cur.fetchall()
    return [
        ForeignKeyMetadata(
            source_table=row[0],
            source_column=row[1],
            target_table=row[2],
            target_column=row[3],
        )
        for row in rows
    ]


def introspect_schema(
    conn: psycopg.Connection,
    schema: str = PUBLIC_SCHEMA,
) -> dict[str, Any]:
    """Return a dictionary containing tables, columns, and foreign keys."""
    tables = get_tables(conn, schema)
    columns = {t: get_columns(conn, t, schema) for t in tables}
    foreign_keys = get_foreign_keys(conn, schema)
    return {"tables": tables, "columns": columns, "foreign_keys": foreign_keys}


def iter_columns(
    columns: dict[str, list[ColumnMetadata]],
) -> Iterable[ColumnMetadata]:
    """Flatten a table→columns mapping into a single iterable of ColumnMetadata."""
    for cols in columns.values():
        yield from cols
