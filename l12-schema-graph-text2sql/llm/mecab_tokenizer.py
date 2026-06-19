"""MeCab-based tokenizer with materials science custom dictionary.

Provides tokenization for Japanese text that correctly handles
domain-specific terms (e.g., バルクモジュラス, 生成エンタルピー).

Usage:
    from llm.mecab_tokenizer import tokenize
    tokens = tokenize("バルクモジュラスが高いL1₂構造の化合物")
    # -> ['バルクモジュラス', 'が', '高い', 'L', '1₂', '構造', 'の', '化合物']
"""
from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any

_PROJECT = Path(__file__).resolve().parent.parent
_DICT_PATH = Path(__file__).resolve().parent / "mecab_materials.dic"


@functools.lru_cache(maxsize=1)
def _get_tagger():
    """Lazy-load MeCab tagger with materials dictionary."""
    try:
        import MeCab
        import ipadic
    except ImportError:
        return None

    if _DICT_PATH.exists():
        return MeCab.Tagger(f"{ipadic.MECAB_ARGS} -u {_DICT_PATH}")
    return MeCab.Tagger(ipadic.MECAB_ARGS)


def tokenize(text: str) -> list[str]:
    """Tokenize Japanese text using MeCab with materials dictionary.

    Falls back to simple character-based splitting if MeCab is unavailable.
    """
    tagger = _get_tagger()
    if tagger is None:
        # Fallback: split on whitespace and punctuation
        import re
        return [t for t in re.split(r'[\s、。,.\t]+', text) if t]

    result = tagger.parse(text)
    if result is None:
        import re
        return [t for t in re.split(r'[\s、。,.\t]+', text) if t]

    tokens = []
    for line in result.strip().split("\n"):
        if "\t" in line:
            surface = line.split("\t")[0]
            if surface:
                tokens.append(surface)
    return tokens


def tokenize_for_tfidf(text: str) -> str:
    """Tokenize and return space-separated string for TF-IDF vectorizer."""
    return " ".join(tokenize(text))
