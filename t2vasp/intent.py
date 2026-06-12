"""
Intent classifier — detect VASP calculation type from natural language.

Maps free-text queries (Japanese / English) to one of the supported
calculation types:  relax, static, dos, band, crystal_field, phonon,
elastic, polarization, dielectric, magnetic, neb, md, sqs.

Design: purely rule-based (no LLM required).  Keyword tables are loaded
from ``config/material_terms.yaml``.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

# Canonical calculation type list (order = priority for tie-breaking)
CALC_TYPES = [
    "polarization", "dielectric",  # very specific — rank high
    "neb", "md", "phonon", "elastic",
    "crystal_field", "sqs",
    "band", "dos",
    "magnetic",
    "static", "relax",  # most generic — rank last
]

_SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")


def _normalize(text: str) -> str:
    return text.translate(_SUBSCRIPT_MAP)


def _load_calc_keywords(path: Optional[Path] = None) -> Dict[str, List[str]]:
    if path is None:
        path = Path(__file__).parent / "config" / "material_terms.yaml"
    with open(path, encoding="utf-8") as f:
        terms = yaml.safe_load(f)
    raw = terms.get("calc_type_keywords", {})
    merged: Dict[str, List[str]] = {}
    for calc_type, lang_map in raw.items():
        keywords: List[str] = []
        for lang_keywords in lang_map.values():
            keywords.extend(lang_keywords)
        merged[calc_type] = keywords
    return merged


@dataclass
class IntentResult:
    """Result of intent classification."""
    calc_type: str                          # primary calculation type
    confidence: float = 1.0                 # 0–1
    matched_keywords: List[str] = field(default_factory=list)
    secondary_types: List[str] = field(default_factory=list)  # multi-step
    is_multi_step: bool = False
    raw_query: str = ""


def classify(query: str, keywords_path: Optional[Path] = None) -> IntentResult:
    """Classify a natural-language query into a VASP calculation type.

    Parameters
    ----------
    query : str
        Free-text query in Japanese or English.
    keywords_path : Path, optional
        Path to material_terms.yaml.  Defaults to bundled config.

    Returns
    -------
    IntentResult
    """
    calc_keywords = _load_calc_keywords(keywords_path)
    q_norm = _normalize(query).lower()

    scores: Dict[str, List[str]] = {ct: [] for ct in CALC_TYPES}

    for calc_type in CALC_TYPES:
        for kw in calc_keywords.get(calc_type, []):
            kw_lower = kw.lower()
            if kw_lower in q_norm:
                scores[calc_type].append(kw)

    # Rank by (number of matched keywords DESC, priority in CALC_TYPES ASC)
    ranked = sorted(
        [(ct, matched) for ct, matched in scores.items() if matched],
        key=lambda x: (-len(x[1]), CALC_TYPES.index(x[0])),
    )

    if not ranked:
        # Default: if query mentions elements/formula, assume relax
        logger.info("No calc-type keywords matched; defaulting to 'relax'")
        return IntentResult(
            calc_type="relax",
            confidence=0.3,
            raw_query=query,
        )

    primary_type, primary_kws = ranked[0]

    # Detect multi-step: e.g. "構造最適化して、DOSも計算して"
    secondary: List[str] = []
    for ct, kws in ranked[1:]:
        if kws:
            secondary.append(ct)

    confidence = min(1.0, len(primary_kws) * 0.4 + 0.2)

    result = IntentResult(
        calc_type=primary_type,
        confidence=confidence,
        matched_keywords=primary_kws,
        secondary_types=secondary,
        is_multi_step=len(secondary) > 0,
        raw_query=query,
    )
    logger.info("Intent: %s (confidence=%.2f, keywords=%s, secondary=%s)",
                result.calc_type, result.confidence,
                result.matched_keywords, result.secondary_types)
    return result
