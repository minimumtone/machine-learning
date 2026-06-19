"""LLM-based reranker for SQL candidates, few-shot examples, and schema linking.

Uses OpenAI API for semantic scoring. All functions gracefully degrade
when the API key is unavailable — returning input unchanged.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)


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
    """Model used for reranking (lighter/cheaper than main generation model)."""
    return os.getenv("RERANK_MODEL", "gpt-4o-mini")


# ---------------------------------------------------------------------------
# 1. SQL candidate reranking
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

    # Filter out empty SQL candidates
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
        resp = client.chat.completions.create(
            model=_rerank_model(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=256,
        )
        raw = resp.choices[0].message.content or ""
        latency_ms = int((time.time() - t0) * 1000)
        logger.debug("Reranker SQL scoring: %dms, raw=%s", latency_ms, raw)

        # Parse scores from response
        import json
        import re
        match = re.search(r"\[[\d\s,]+\]", raw)
        if match:
            scores = json.loads(match.group())
        else:
            return candidates

        # Apply reranker scores
        for i, c in enumerate(valid):
            if i < len(scores):
                reranker_score = min(max(scores[i], 0), 100)
                c["reranker_score"] = reranker_score
                original = c.get("score", 0)
                # Normalize original to 0-100 scale
                c["score"] = (1 - weight) * original + weight * reranker_score

    except Exception as e:
        logger.warning("Reranker SQL scoring failed: %s", e)

    return candidates


# ---------------------------------------------------------------------------
# 2. Few-shot example reranking
# ---------------------------------------------------------------------------

_FEWSHOT_RERANK_PROMPT = """\
You are a materials database query expert.

Given a new query and several candidate few-shot examples, rank the examples
by how useful they would be as SQL generation guidance for the new query.
Consider: similar table usage, similar conditions, similar query structure.

New query: {query}

{examples_block}

Respond with ONLY a JSON array of scores (0-100), one per example.
Example: [90, 45, 72]
"""


def rerank_few_shot_examples(
    query: str,
    examples: list[dict[str, Any]],
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Rerank few-shot examples by semantic relevance to the query.

    Parameters
    ----------
    query : str
        The new natural language query.
    examples : list[dict]
        Candidate examples from the TF-IDF retrieval stage.
    top_k : int
        Number of examples to return after reranking.

    Returns
    -------
    list[dict]
        Top-k examples sorted by reranker score.
    """
    if len(examples) <= top_k:
        return examples

    client = _get_client()
    if client is None:
        return examples[:top_k]

    examples_block = "\n".join(
        f"Example {i+1}:\n  Query: {e['nl_query']}\n  SQL: {e.get('sql', 'N/A')}"
        for i, e in enumerate(examples)
    )
    prompt = _FEWSHOT_RERANK_PROMPT.format(
        query=query, examples_block=examples_block,
    )

    try:
        t0 = time.time()
        resp = client.chat.completions.create(
            model=_rerank_model(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=256,
        )
        raw = resp.choices[0].message.content or ""
        latency_ms = int((time.time() - t0) * 1000)
        logger.debug("Reranker few-shot scoring: %dms", latency_ms)

        import json
        import re
        match = re.search(r"\[[\d\s,]+\]", raw)
        if match:
            scores = json.loads(match.group())
            scored = []
            for i, e in enumerate(examples):
                s = scores[i] if i < len(scores) else 0
                scored.append((s, e))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [e for _, e in scored[:top_k]]
    except Exception as e:
        logger.warning("Reranker few-shot scoring failed: %s", e)

    return examples[:top_k]


# ---------------------------------------------------------------------------
# 3. Schema linking reranking
# ---------------------------------------------------------------------------

_SCHEMA_RERANK_PROMPT = """\
You are a materials database schema expert.

Given a natural language query about materials, and a list of candidate database
tables, score each table on how likely it is needed to answer the query (0-100).

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
- (+ others for synthesis, applications, alloy systems, etc.)

Query: {query}

Candidate tables: {tables}

Respond with ONLY a JSON object mapping table name to score (0-100).
Example: {{"material_entry": 95, "composition": 80, "structure": 30}}
"""


def rerank_schema_tables(
    query: str,
    candidate_tables: list[str],
    threshold: float = 30.0,
) -> list[str]:
    """Rerank candidate tables by relevance to the query.

    Parameters
    ----------
    query : str
        Natural language query.
    candidate_tables : list[str]
        Tables identified by the rule-based schema linker.
    threshold : float
        Minimum score to keep a table (0-100).

    Returns
    -------
    list[str]
        Tables sorted by relevance, low-scoring ones removed.
        Always includes material_entry as the hub table.
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
        resp = client.chat.completions.create(
            model=_rerank_model(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
        )
        raw = resp.choices[0].message.content or ""
        latency_ms = int((time.time() - t0) * 1000)
        logger.debug("Reranker schema scoring: %dms", latency_ms)

        import json
        import re
        match = re.search(r"\{[^}]+\}", raw, re.DOTALL)
        if match:
            scores = json.loads(match.group())
            scored = [
                (scores.get(t, scores.get(t.lower(), 0)), t)
                for t in candidate_tables
            ]
            scored.sort(key=lambda x: x[0], reverse=True)
            result = [t for s, t in scored if s >= threshold]
            # Always keep material_entry as the hub
            if "material_entry" not in result and "material_entry" in candidate_tables:
                result.append("material_entry")
            return result if result else candidate_tables
    except Exception as e:
        logger.warning("Reranker schema scoring failed: %s", e)

    return candidate_tables
