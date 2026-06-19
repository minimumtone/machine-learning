"""Hybrid reranker: Cross-Encoder for few-shot, LLM for SQL candidates & schema.

Performance-oriented design:
- Few-shot example reranking: Cross-Encoder (ms-marco-MiniLM, <50ms, local)
- SQL candidate reranking: GPT-5.5 (semantic SQL correctness)
- Schema table reranking: GPT-5.5 (domain knowledge, sort-only — never drops tables)

All functions gracefully degrade when dependencies are unavailable.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cross-Encoder singleton (lazy-loaded)
# ---------------------------------------------------------------------------

_cross_encoder = None
_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _get_cross_encoder() -> Any | None:
    """Lazy-load Cross-Encoder model. Returns None if unavailable."""
    global _cross_encoder
    if _cross_encoder is not None:
        return _cross_encoder
    try:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder(_CROSS_ENCODER_MODEL)
        logger.info("Cross-Encoder loaded: %s", _CROSS_ENCODER_MODEL)
        return _cross_encoder
    except Exception as e:
        logger.debug("Cross-Encoder unavailable: %s", e)
        return None


# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

def _get_client() -> Any | None:
    """Return an OpenAI client if API key is available, else None."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key == "your_api_key_here":
        return None
    try:
        import openai
        return openai.OpenAI(api_key=api_key)
    except Exception:
        return None


def _rerank_model() -> str:
    """LLM model used for SQL/schema reranking."""
    return os.getenv("RERANK_MODEL", "gpt-5.5")


# ---------------------------------------------------------------------------
# 1. SQL candidate reranking (LLM-based)
# ---------------------------------------------------------------------------

_SQL_RERANK_PROMPT = """\
You are a PostgreSQL expert for a materials science database.

Given a natural language question and multiple SQL candidates, score each SQL
on how well it answers the question. Consider:
- Semantic correctness (does the SQL answer what was asked?)
- Appropriate table joins and conditions
- Proper column selection
- No unnecessary complexity

Question: {question}

{candidates_block}

For each candidate, respond with ONLY a JSON array of scores (0-100).
Example: [85, 42, 91]
"""


def rerank_sql_candidates(
    question: str,
    candidates: list[dict[str, Any]],
    *,
    weight: float = 0.4,
) -> list[dict[str, Any]]:
    """Rerank SQL candidates using LLM semantic scoring.

    Parameters
    ----------
    question : str
        Original natural language query.
    candidates : list[dict]
        Each dict must have "sql" and "score" keys.
    weight : float
        Weight for reranker score (0-1). Final score =
        (1 - weight) * original_score + weight * reranker_score.

    Returns
    -------
    list[dict]
        Same candidates with updated "score" and added "reranker_score".
    """
    if len(candidates) <= 1:
        return candidates

    valid = [c for c in candidates if c.get("sql")]
    if not valid:
        return candidates

    client = _get_client()
    if client is None:
        logger.debug("Reranker: no API key, skipping SQL reranking")
        return candidates

    candidates_block = "\n".join(
        f"Candidate {i+1}:\n```sql\n{c['sql']}\n```"
        for i, c in enumerate(valid)
    )
    prompt = _SQL_RERANK_PROMPT.format(
        question=question, candidates_block=candidates_block,
    )

    try:
        t0 = time.time()
        model = _rerank_model()
        _is_reasoning = model and any(t in model for t in ("gpt-5", "o1", "o3", "o4"))
        create_kwargs: dict[str, Any] = dict(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        if _is_reasoning:
            create_kwargs["max_completion_tokens"] = 1024
        else:
            create_kwargs["temperature"] = 0.0
            create_kwargs["max_tokens"] = 256

        resp = client.chat.completions.create(**create_kwargs)
        raw = resp.choices[0].message.content or ""
        latency_ms = int((time.time() - t0) * 1000)
        logger.debug("Reranker SQL scoring: %dms, raw=%s", latency_ms, raw)

        import json
        import re
        match = re.search(r"\[[\d\s,]+\]", raw)
        if match:
            scores = json.loads(match.group())
        else:
            return candidates

        for i, c in enumerate(valid):
            if i < len(scores):
                reranker_score = min(max(scores[i], 0), 100)
                c["reranker_score"] = reranker_score
                original = c.get("score", 0)
                c["score"] = (1 - weight) * original + weight * reranker_score

    except Exception as e:
        logger.warning("Reranker SQL scoring failed: %s", e)

    return candidates


# ---------------------------------------------------------------------------
# 2. Few-shot example reranking (Cross-Encoder)
# ---------------------------------------------------------------------------

def rerank_few_shot_examples(
    query: str,
    examples: list[dict[str, Any]],
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Rerank few-shot examples using Cross-Encoder (ms-marco-MiniLM).

    ~50ms local inference, no API calls.
    Falls back to returning first top_k if Cross-Encoder is unavailable.
    """
    if len(examples) <= top_k:
        return examples

    ce = _get_cross_encoder()
    if ce is None:
        return examples[:top_k]

    try:
        t0 = time.time()
        pairs = [
            [query, e["nl_query"] + " " + e.get("sql", "")]
            for e in examples
        ]
        scores = ce.predict(pairs)
        latency_ms = int((time.time() - t0) * 1000)
        logger.debug("Cross-Encoder few-shot scoring: %dms", latency_ms)

        scored = sorted(
            zip(scores, examples),
            key=lambda x: x[0],
            reverse=True,
        )
        return [e for _, e in scored[:top_k]]

    except Exception as e:
        logger.warning("Cross-Encoder few-shot scoring failed: %s", e)
        return examples[:top_k]


# ---------------------------------------------------------------------------
# 3. Schema table reranking (LLM-based, sort-only)
# ---------------------------------------------------------------------------

_SCHEMA_RERANK_PROMPT = """\
You are a materials database schema expert.

Given a natural language query about materials, and a list of required database
tables, score each table on how relevant it is to the query (0-100).
All tables are required — do NOT remove any. Just rank by relevance.

Schema overview:
- material_entry: base table, formula, source_db
- composition: element fractions, chemical composition
- structure: prototype, lattice constants, space group
- phase_stability: formation energy, energy above hull, band gap
- calculation: DFT calculation metadata
- calculated_property: bulk/shear/Young's modulus
- elastic_tensor: full elastic tensor, Poisson ratio
- thermal_property: Debye temperature, thermal conductivity
- magnetic_property: magnetization, Curie temperature
- band_structure: band structure data, direct/indirect gap
- density_of_states: DOS at Fermi level, metallic/insulating
- surface_energy: surface energy, work function
- grain_boundary: grain boundary energy
- material_defect / defect_type: vacancies, interstitials
- element / element_property: atomic properties
- prototype_definition: Strukturbericht designations
- literature_reference / material_reference: citations

Query: {query}

Tables: {tables}

Respond with ONLY a JSON object mapping table name to score (0-100).
Example: {{"material_entry": 95, "composition": 80, "structure": 30}}
"""


def rerank_schema_tables(
    query: str,
    candidate_tables: list[str],
) -> list[str]:
    """Sort tables by relevance to the query. Never removes tables.

    Unlike the previous implementation, this function only reorders tables
    by relevance score — it never drops tables from the list. This ensures
    that required_tables, required_columns, and sql_fragments stay consistent.
    """
    if len(candidate_tables) <= 2:
        return candidate_tables

    client = _get_client()
    if client is None:
        return candidate_tables

    prompt = _SCHEMA_RERANK_PROMPT.format(
        query=query, tables=", ".join(candidate_tables),
    )

    try:
        t0 = time.time()
        model = _rerank_model()
        _is_reasoning = model and any(t in model for t in ("gpt-5", "o1", "o3", "o4"))
        create_kwargs: dict[str, Any] = dict(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        if _is_reasoning:
            create_kwargs["max_completion_tokens"] = 1024
        else:
            create_kwargs["temperature"] = 0.0
            create_kwargs["max_tokens"] = 512

        resp = client.chat.completions.create(**create_kwargs)
        raw = resp.choices[0].message.content or ""
        latency_ms = int((time.time() - t0) * 1000)
        logger.debug("Reranker schema scoring: %dms", latency_ms)

        import json
        import re
        match = re.search(r"\{[^}]+\}", raw, re.DOTALL)
        if match:
            scores = json.loads(match.group())
            scored = [
                (scores.get(t, scores.get(t.lower(), 50)), t)
                for t in candidate_tables
            ]
            scored.sort(key=lambda x: x[0], reverse=True)
            return [t for _, t in scored]
    except Exception as e:
        logger.warning("Reranker schema scoring failed: %s", e)

    return candidate_tables
