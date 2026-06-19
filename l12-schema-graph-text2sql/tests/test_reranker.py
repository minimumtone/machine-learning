"""Tests for LLM reranker module.

These tests verify graceful degradation (no API key) and basic logic.
API-dependent scoring is tested only when OPENAI_API_KEY is set.
"""
import os

import pytest

from llm.reranker import (
    rerank_few_shot_examples,
    rerank_schema_tables,
    rerank_sql_candidates,
)


# ---------------------------------------------------------------------------
# SQL candidate reranking
# ---------------------------------------------------------------------------


def test_rerank_sql_single_candidate():
    """Single candidate should be returned unchanged."""
    cands = [{"sql": "SELECT 1", "score": 50}]
    result = rerank_sql_candidates("test query", cands)
    assert len(result) == 1
    assert result[0]["score"] == 50


def test_rerank_sql_empty_sql_preserved():
    """Candidates with empty SQL should not crash."""
    cands = [
        {"sql": "", "score": 0},
        {"sql": "SELECT 1", "score": 50},
    ]
    result = rerank_sql_candidates("test query", cands)
    assert len(result) == 2


def test_rerank_sql_no_api_key(monkeypatch):
    """Without API key, candidates are returned with original scores."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cands = [
        {"sql": "SELECT a FROM t1", "score": 60},
        {"sql": "SELECT b FROM t2", "score": 40},
    ]
    result = rerank_sql_candidates("find materials", cands)
    assert result[0]["score"] == 60
    assert result[1]["score"] == 40


# ---------------------------------------------------------------------------
# Few-shot reranking
# ---------------------------------------------------------------------------


def test_rerank_fewshot_within_topk():
    """When examples <= top_k, return all without reranking."""
    examples = [
        {"nl_query": "Find L12", "sql": "SELECT ..."},
        {"nl_query": "Find B2", "sql": "SELECT ..."},
    ]
    result = rerank_few_shot_examples("Find L12 materials", examples, top_k=3)
    assert len(result) == 2


def test_rerank_fewshot_no_api_key(monkeypatch):
    """Without API key, return first top_k examples."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    examples = [
        {"nl_query": "Q1", "sql": "S1"},
        {"nl_query": "Q2", "sql": "S2"},
        {"nl_query": "Q3", "sql": "S3"},
        {"nl_query": "Q4", "sql": "S4"},
    ]
    result = rerank_few_shot_examples("test", examples, top_k=2)
    assert len(result) == 2
    assert result[0]["nl_query"] == "Q1"


# ---------------------------------------------------------------------------
# Schema table reranking
# ---------------------------------------------------------------------------


def test_rerank_schema_few_tables():
    """With <= 2 tables, return unchanged."""
    tables = ["material_entry", "composition"]
    result = rerank_schema_tables("find Ni compounds", tables)
    assert result == tables


def test_rerank_schema_no_api_key(monkeypatch):
    """Without API key, return original table list."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    tables = ["material_entry", "composition", "structure", "phase_stability"]
    result = rerank_schema_tables("find stable L12", tables)
    assert result == tables


def test_rerank_schema_preserves_material_entry(monkeypatch):
    """material_entry should always be preserved as hub table."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    tables = ["material_entry", "composition", "structure"]
    result = rerank_schema_tables("find something", tables)
    assert "material_entry" in result
