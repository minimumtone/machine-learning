"""Tests for hybrid reranker module.

Cross-Encoder tests run locally (no API key needed).
LLM-based tests verify graceful degradation without API key.
"""
import os

import pytest

from llm.reranker import (
    rerank_few_shot_examples,
    rerank_schema_tables,
    rerank_sql_candidates,
)


# ---------------------------------------------------------------------------
# SQL candidate reranking (LLM-based)
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
# Few-shot reranking (Cross-Encoder)
# ---------------------------------------------------------------------------


def test_rerank_fewshot_within_topk():
    """When examples <= top_k, return all without reranking."""
    examples = [
        {"nl_query": "Find L12", "sql": "SELECT ..."},
        {"nl_query": "Find B2", "sql": "SELECT ..."},
    ]
    result = rerank_few_shot_examples("Find L12 materials", examples, top_k=3)
    assert len(result) == 2


def test_rerank_fewshot_cross_encoder():
    """Cross-Encoder should rerank by semantic relevance."""
    examples = [
        {"nl_query": "バルクモジュラスが高い化合物", "sql": "SELECT FROM calculated_property"},
        {"nl_query": "L1₂構造を持つ化合物一覧", "sql": "SELECT FROM structure"},
        {"nl_query": "Niを含む化合物", "sql": "SELECT FROM composition"},
        {"nl_query": "格子定数が小さい化合物", "sql": "SELECT FROM structure"},
    ]
    result = rerank_few_shot_examples(
        "L1₂型の化合物を一覧にせよ", examples, top_k=2,
    )
    assert len(result) == 2
    # The L12/structure-related example should rank higher than bulk modulus
    top_queries = [e["nl_query"] for e in result]
    assert "L1₂構造を持つ化合物一覧" in top_queries


# ---------------------------------------------------------------------------
# Schema table reranking (LLM-based, sort-only)
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


def test_rerank_schema_never_drops_tables(monkeypatch):
    """Schema reranker must never remove tables from the list."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    tables = ["material_entry", "composition", "structure"]
    result = rerank_schema_tables("find something", tables)
    assert set(result) == set(tables)
