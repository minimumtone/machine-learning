"""FastAPI application for L1_2 schema-graph-assisted Text-to-SQL."""
from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import FastAPI, HTTPException

logger = logging.getLogger(__name__)
from pydantic import BaseModel

from llm.entity_extractor import extract_conditions
from llm.few_shot_store import add_example
from llm.schema_linker import link_schema
from llm.sql_generator import build_schema_context_from_db, pipeline
from safety.sql_validator import validate_sql
from safety.sql_guard import execute_sql

app = FastAPI(
    title="L1₂ Schema-Graph Text-to-SQL",
    description="Natural language query interface for L1₂ intermetallic compound database",
    version="0.1.0",
)


class QueryRequest(BaseModel):
    query: str
    execute: bool = True


class ValidateRequest(BaseModel):
    sql: str


class QueryResponse(BaseModel):
    query: str
    conditions: dict[str, Any]
    linked_tables: list[str]
    linked_columns: list[str]
    generated_sql: str
    validation: dict[str, Any]
    results: dict[str, Any] | None = None
    schema_mode: str = "full_30_table"


@app.get("/health")
def health() -> dict[str, Any]:
    ctx = _get_schema_ctx()
    schema_ok = ctx.get("join_list") is not None
    return {
        "status": "ok",
        "schema_context": "full_30_table" if schema_ok else "fallback_5_table",
    }


_schema_ctx: dict[str, list[str]] | None = None


def _get_schema_ctx() -> dict[str, list[str]]:
    """Lazily build and cache schema context from the live DB.

    On connection failure, returns fallback without caching so
    subsequent requests retry (e.g. DB not yet ready at startup).
    """
    global _schema_ctx  # noqa: PLW0603
    if _schema_ctx is not None:
        return _schema_ctx
    try:
        import psycopg
        conn = psycopg.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            dbname=os.getenv("POSTGRES_DB", "l12_materials"),
            user=os.getenv("POSTGRES_USER", "l12_user"),
            password=os.getenv("POSTGRES_PASSWORD", "l12_password"),
        )
        _schema_ctx = build_schema_context_from_db(conn)
        conn.close()
        return _schema_ctx
    except Exception as exc:
        logger.error(
            "DB schema context unavailable — falling back to 5-table mode: %s",
            exc,
        )
        return {"join_list": None, "all_columns": None}


@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest) -> QueryResponse:
    """Process a natural language query and optionally execute the SQL."""
    ctx = _get_schema_ctx()
    result = pipeline(
        req.query,
        join_list=ctx.get("join_list"),
        all_columns=ctx.get("all_columns"),
    )

    if result.get("mode") == "rejected":
        raise HTTPException(
            status_code=403,
            detail=result.get("reason", "Query rejected by intent classifier"),
        )

    validation = validate_sql(result["sql"])
    sql = validation["sql"]

    exec_result = None
    if req.execute and validation["valid"]:
        exec_result = execute_sql(sql, validate=False)
        if (
            result.get("_store_on_success")
            and exec_result
            and exec_result.get("success")
            and exec_result.get("row_count", 0) > 0
        ):
            add_example(
                nl_query=req.query,
                sql=sql,
                conditions=result.get("conditions", {}),
                row_count=exec_result.get("row_count", 0),
                source="api",
            )

    ctx_mode = (
        "full_30_table" if ctx.get("join_list") is not None
        else "fallback_5_table"
    )
    return QueryResponse(
        query=req.query,
        conditions=result.get("conditions", {}),
        linked_tables=result.get("linked_schema", {}).get("required_tables", []),
        linked_columns=result.get("linked_schema", {}).get("required_columns", []),
        generated_sql=sql,
        validation=validation,
        results=exec_result,
        schema_mode=ctx_mode,
    )


@app.post("/extract")
def extract_endpoint(req: QueryRequest) -> dict[str, Any]:
    """Extract conditions from a natural language query."""
    conditions = extract_conditions(req.query)
    linked = link_schema(conditions)
    return {
        "query": req.query,
        "conditions": conditions,
        "linked": linked,
    }


@app.post("/validate")
def validate_endpoint(req: ValidateRequest) -> dict[str, Any]:
    """Validate a SQL string."""
    return validate_sql(req.sql)
