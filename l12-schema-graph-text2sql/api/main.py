"""FastAPI application for L1_2 schema-graph-assisted Text-to-SQL."""
from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from llm.entity_extractor import extract_conditions
from llm.schema_linker import link_schema
from llm.sql_generator import pipeline
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


class QueryResponse(BaseModel):
    query: str
    conditions: dict[str, Any]
    linked_tables: list[str]
    linked_columns: list[str]
    generated_sql: str
    validation: dict[str, Any]
    results: dict[str, Any] | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest) -> QueryResponse:
    """Process a natural language query and optionally execute the SQL."""
    result = pipeline(req.query)

    validation = validate_sql(result["sql"])
    sql = validation["sql"]

    exec_result = None
    if req.execute and validation["valid"]:
        exec_result = execute_sql(sql, validate=False)

    return QueryResponse(
        query=req.query,
        conditions=result.get("conditions", {}),
        linked_tables=result.get("linked_schema", {}).get("required_tables", []),
        linked_columns=result.get("linked_schema", {}).get("required_columns", []),
        generated_sql=sql,
        validation=validation,
        results=exec_result,
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
def validate_endpoint(sql: str) -> dict[str, Any]:
    """Validate a SQL string."""
    return validate_sql(sql)
