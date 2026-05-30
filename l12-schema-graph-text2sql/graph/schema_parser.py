"""Parse PostgreSQL information_schema to extract tables, columns, and FK relationships."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import psycopg

PUBLIC_SCHEMA = "public"


@dataclass(frozen=True)
class ColumnMetadata:
    table_name: str
    column_name: str
    data_type: str
    is_nullable: bool
    is_primary_key: bool


@dataclass(frozen=True)
class ForeignKeyMetadata:
    source_table: str
    source_column: str
    target_table: str
    target_column: str


def get_tables(conn: psycopg.Connection, schema: str = PUBLIC_SCHEMA) -> list[str]:
    """Return table names in the specified schema."""
    query = (
        "SELECT table_name "
        "FROM information_schema.tables "
        "WHERE table_schema = %s AND table_type = 'BASE TABLE' "
        "ORDER BY table_name"
    )
    with conn.cursor() as cur:
        cur.execute(query, (schema,))
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
    sql = (
        "SELECT tc.table_name AS source_table,"
        "       kcu.column_name AS source_column,"
        "       ccu.table_name AS target_table,"
        "       ccu.column_name AS target_column"
        "  FROM information_schema.table_constraints tc"
        "  JOIN information_schema.key_column_usage kcu"
        "    ON tc.constraint_name = kcu.constraint_name"
        "   AND tc.table_schema = kcu.table_schema"
        "  JOIN information_schema.constraint_column_usage ccu"
        "    ON ccu.constraint_name = tc.constraint_name"
        "   AND ccu.table_schema = tc.table_schema"
        " WHERE tc.constraint_type = 'FOREIGN KEY'"
        "   AND tc.table_schema = %s"
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
    for cols in columns.values():
        yield from cols
