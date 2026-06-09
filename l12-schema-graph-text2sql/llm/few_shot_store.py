"""SQL-as-Few-Shot-Examples: store, retrieve, and inject successful NL→SQL pairs.

This module implements a RAG-like feedback loop where successfully executed
NL→SQL translations are stored and later retrieved as few-shot examples
for new queries.  Retrieval uses TF-IDF cosine similarity over the stored
natural-language queries.

Additionally, ``extract_examples_from_paper()`` can parse LaTeX manuscripts
to discover implicit material queries (e.g. "B2-type FeAl", "stable L1₂
Ni₃Al") and register them as seed few-shot examples.
"""
from __future__ import annotations

import fcntl
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Persistent JSON store
# ---------------------------------------------------------------------------

_DEFAULT_STORE_PATH = Path(__file__).parent.parent / "few_shot_examples.json"


def _store_path() -> Path:
    return Path(os.getenv("FEW_SHOT_STORE", str(_DEFAULT_STORE_PATH)))


def load_store() -> list[dict[str, Any]]:
    p = _store_path()
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def save_store(examples: list[dict[str, Any]]) -> None:
    p = _store_path()
    with p.open("w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)


def add_example(
    nl_query: str,
    sql: str,
    conditions: dict[str, Any],
    row_count: int,
    source: str = "runtime",
) -> dict[str, Any]:
    """Register a successful NL→SQL pair in the store (file-lock protected)."""
    p = _store_path()
    lock_path = p.with_suffix(".lock")
    entry = {
        "nl_query": nl_query,
        "sql": sql,
        "conditions": conditions,
        "row_count": row_count,
        "source": source,
    }
    with lock_path.open("w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            examples = load_store()
            examples = [e for e in examples if e["nl_query"] != nl_query]
            examples.append(entry)
            save_store(examples)
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)
    return entry


# ---------------------------------------------------------------------------
# TF-IDF based retrieval
# ---------------------------------------------------------------------------

_TOKENIZE_RE = re.compile(
    r"[A-Za-z]\d+[_]?\d*"       # Strukturbericht: L12, B2, A15, D0_3
    r"|[A-Z][a-z]?\d*"          # Chemical formula tokens: Ni3, Al, Fe2
    r"|[\u3040-\u309F]+"        # Hiragana
    r"|[\u30A0-\u30FF]+"        # Katakana
    r"|[\u4E00-\u9FFF]+"        # Kanji
    r"|[a-z]+"                  # English words
    r"|\d+"                     # Numbers
)


def _tokenize(text: str) -> list[str]:
    return [m.group().lower() for m in _TOKENIZE_RE.finditer(text)]


def _tf(tokens: list[str]) -> dict[str, float]:
    c = Counter(tokens)
    total = len(tokens) or 1
    return {t: n / total for t, n in c.items()}


def _idf(corpus: list[list[str]]) -> dict[str, float]:
    n = len(corpus) or 1
    df: Counter[str] = Counter()
    for doc in corpus:
        df.update(set(doc))
    return {t: math.log((n + 1) / (cnt + 1)) + 1 for t, cnt in df.items()}


def _cosine(v1: dict[str, float], v2: dict[str, float]) -> float:
    common = set(v1) & set(v2)
    if not common:
        return 0.0
    dot = sum(v1[k] * v2[k] for k in common)
    n1 = math.sqrt(sum(x * x for x in v1.values()))
    n2 = math.sqrt(sum(x * x for x in v2.values()))
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)


def retrieve_similar(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    """Return the top-k most similar stored examples for a given query."""
    examples = load_store()
    if not examples:
        return []
    corpus_tokens = [_tokenize(e["nl_query"]) for e in examples]
    query_tokens = _tokenize(query)
    idf = _idf(corpus_tokens + [query_tokens])

    def _tfidf(tokens: list[str]) -> dict[str, float]:
        tf = _tf(tokens)
        return {t: tf[t] * idf.get(t, 1.0) for t in tf}

    q_vec = _tfidf(query_tokens)
    scored = []
    for i, doc_tokens in enumerate(corpus_tokens):
        d_vec = _tfidf(doc_tokens)
        sim = _cosine(q_vec, d_vec)
        scored.append((sim, i))
    scored.sort(reverse=True)
    return [
        {**examples[idx], "similarity": sim}
        for sim, idx in scored[:top_k]
        if sim > 0.05
    ]


# ---------------------------------------------------------------------------
# Prompt injection
# ---------------------------------------------------------------------------

def format_few_shot_block(examples: list[dict[str, Any]]) -> str:
    """Format retrieved examples as a few-shot block for the LLM prompt."""
    if not examples:
        return ""
    lines = ["", "Here are similar successful queries for reference:", ""]
    for i, ex in enumerate(examples, 1):
        lines.append(f"Example {i}:")
        lines.append(f"  Query: {ex['nl_query']}")
        lines.append(f"  SQL: {ex['sql']}")
        if ex.get("conditions"):
            lines.append(f"  Conditions: {json.dumps(ex['conditions'], ensure_ascii=False)}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Paper-based example extraction
# ---------------------------------------------------------------------------

_PAPER_PATTERNS = [
    # B2-type compounds: "B2 FeAl", "B2-type NiAl"
    (r"B2[- ](?:type\s+)?([A-Z][a-z]?[A-Z][a-z]?)", "B2"),
    # L1_2 compounds: "L1$_2$ Ni$_3$Al", "L12 Co3Ti"
    (r"L1[\$_{}2₂]+[- ]?(?:type\s+)?([A-Z][a-z]?)[\$_{}3₃]*([A-Z][a-z]?)", "L12"),
    # Explicit compound mentions with prototype context
    (r"(\w{2,6})\s+(?:の|の格子定数|compound|化合物)", None),
]

_COMPOUND_RE = re.compile(r"([A-Z][a-z]?)(\d*)([A-Z][a-z]?)(\d*)")

# Metallic elements commonly found in intermetallic compounds.
# Single-letter symbols (B, C, N, O, etc.) are excluded to avoid LaTeX false positives.
_VALID_ELEMENTS = {
    "Li", "Be", "Na", "Mg", "Al", "Si", "Ca", "Sc", "Ti", "V", "Cr", "Mn",
    "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Rb",
    "Sr", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In",
    "Sn", "Sb", "Te", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta",
    "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi",
}


def _is_valid_formula(elements: list[str]) -> bool:
    """Check if all elements in a formula are valid chemical symbols."""
    return all(e in _VALID_ELEMENTS for e in elements)


def extract_examples_from_paper(
    tex_path: str | Path,
) -> list[dict[str, Any]]:
    """Parse a LaTeX manuscript and extract implicit material queries as seed examples.

    Returns a list of dicts with keys: nl_query, sql, conditions, source.
    """
    tex_path = Path(tex_path)
    if not tex_path.exists():
        return []
    text = tex_path.read_text(encoding="utf-8")
    # Remove LaTeX commands but keep content
    text_clean = re.sub(r"\\text\{([^}]*)\}", r"\1", text)  # \text{Ni} → Ni
    text_clean = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", text_clean)
    text_clean = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", text_clean)
    # Remove subscript/superscript delimiters: _{3} → 3, ^{2} → 2
    text_clean = re.sub(r"[_^]\{([^}]*)\}", r"\1", text_clean)
    # Remove remaining underscores used as subscript markers: L1_2 → L12
    text_clean = re.sub(r"_(\d)", r"\1", text_clean)
    text_clean = re.sub(r"[\\${}]", "", text_clean)

    found_compounds: dict[str, str] = {}  # formula -> prototype

    # Find B2 compound mentions
    for m in re.finditer(r"B2[^A-Za-z]*([A-Z][a-z]?)([A-Z][a-z]?)", text_clean):
        elems = [m.group(1), m.group(2)]
        if _is_valid_formula(elems):
            formula = f"{m.group(1)}{m.group(2)}"
            found_compounds[formula] = "B2"

    # Find L12 compound mentions
    for m in re.finditer(
        r"L1[_2₂]+[^A-Za-z]*([A-Z][a-z]?)[_3₃]*([A-Z][a-z]?)", text_clean
    ):
        elems = [m.group(1), m.group(2)]
        if _is_valid_formula(elems):
            formula = f"{m.group(1)}3{m.group(2)}"
            found_compounds[formula] = "L12"

    # Find generic compound mentions like "Ni3Al" near lattice/stable/etc keywords
    for m in re.finditer(r"([A-Z][a-z]?)(\d)([A-Z][a-z]?)", text_clean):
        formula = f"{m.group(1)}{m.group(2)}{m.group(3)}"
        elem1 = m.group(1)
        elem2 = m.group(3)
        num = m.group(2)
        if not _is_valid_formula([elem1, elem2]):
            continue
        if num == "3" and formula not in found_compounds:
            # Check surrounding context for L12-related keywords
            start = max(0, m.start() - 100)
            end = min(len(text_clean), m.end() + 100)
            ctx = text_clean[start:end].lower()
            if any(kw in ctx for kw in ["l12", "l1_2", "fcc", "gamma"]):
                found_compounds[formula] = "L12"
            elif any(kw in ctx for kw in ["b2", "bcc", "cscl"]):
                found_compounds[formula] = "B2"

    examples: list[dict[str, Any]] = []
    seen_queries: set[str] = set()

    for formula, proto in found_compounds.items():
        # Extract elements from formula
        elems = re.findall(r"[A-Z][a-z]?", formula)
        if not elems:
            continue

        # Generate several query variants
        queries = []
        proto_label = "L1₂" if proto == "L12" else proto

        # Query 1: specific compound lookup
        q1 = f"{formula}の{proto_label}化合物の物性を出して"
        if q1 not in seen_queries:
            conds1 = {"prototype": proto, "contains_elements": elems}
            sql1 = _build_seed_sql(proto, elems)
            queries.append((q1, sql1, conds1))
            seen_queries.add(q1)

        # Query 2: element-based search
        if len(elems) >= 1:
            q2 = f"{elems[0]}を含む{proto_label}化合物を出して"
            if q2 not in seen_queries:
                conds2 = {"prototype": proto, "contains_elements": [elems[0]]}
                sql2 = _build_seed_sql(proto, [elems[0]])
                queries.append((q2, sql2, conds2))
                seen_queries.add(q2)

        for nl, sql, conds in queries:
            examples.append({
                "nl_query": nl,
                "sql": sql,
                "conditions": conds,
                "row_count": -1,  # unknown until executed
                "source": f"paper:{tex_path.name}",
            })

    # Add general queries derived from paper themes
    general_queries = [
        (
            "OQMDのB2化合物の形成エネルギーを出して",
            _build_seed_sql("B2", [], sort_col="ps.formation_energy_per_atom"),
            {"prototype": "B2", "sort_by": "phase_stability.formation_energy_per_atom", "sort_order": "asc"},
        ),
        (
            "Materials ProjectのL1₂化合物の格子定数を出して",
            _build_seed_sql("L12", []),
            {"prototype": "L12", "properties": ["structure.lattice_a"]},
        ),
        (
            "安定なB2化合物をリストして",
            _build_seed_sql("B2", [], stability="stable"),
            {"prototype": "B2", "stability": "stable"},
        ),
        (
            "安定なL1₂化合物を形成エネルギーが低い順に出して",
            _build_seed_sql("L12", [], stability="stable", sort_col="ps.formation_energy_per_atom"),
            {"prototype": "L12", "stability": "stable", "sort_by": "phase_stability.formation_energy_per_atom", "sort_order": "asc"},
        ),
    ]
    for nl, sql, conds in general_queries:
        if nl not in seen_queries:
            examples.append({
                "nl_query": nl,
                "sql": sql,
                "conditions": conds,
                "row_count": -1,
                "source": f"paper:{tex_path.name}",
            })
            seen_queries.add(nl)

    return examples


def _build_seed_sql(
    proto: str,
    elements: list[str],
    stability: str | None = None,
    sort_col: str | None = None,
) -> str:
    """Build a seed SQL query for a paper-extracted example."""
    select = ["m.entry_id", "m.formula", "s.prototype", "s.lattice_a"]
    joins = ["JOIN structure s ON s.entry_id = m.entry_id"]
    where = [f"(s.prototype = '{proto}' OR s.strukturbericht = '{proto}')"]

    if len(elements) == 1:
        joins.append("JOIN composition c ON c.entry_id = m.entry_id")
        where.append(f"c.element = '{elements[0]}'")
    elif len(elements) > 1:
        for e in elements:
            where.append(
                f"EXISTS (SELECT 1 FROM composition c_{e.lower()}"
                f" WHERE c_{e.lower()}.entry_id = m.entry_id"
                f" AND c_{e.lower()}.element = '{e}')"
            )

    if stability == "stable":
        joins.append("JOIN phase_stability ps ON ps.entry_id = m.entry_id")
        where.append("ps.energy_above_hull <= 0.001")
        select.extend(["ps.formation_energy_per_atom", "ps.energy_above_hull"])
    elif stability == "metastable":
        joins.append("JOIN phase_stability ps ON ps.entry_id = m.entry_id")
        where.append("ps.energy_above_hull <= 0.05")
        select.extend(["ps.formation_energy_per_atom", "ps.energy_above_hull"])

    if sort_col and ("phase_stability" in sort_col or sort_col.startswith("ps.")) and stability is None:
        joins.append("JOIN phase_stability ps ON ps.entry_id = m.entry_id")
        select.extend(["ps.formation_energy_per_atom", "ps.energy_above_hull"])

    order = ""
    if sort_col:
        order = f"\nORDER BY {sort_col} ASC"

    sql = f"SELECT DISTINCT\n    {', '.join(select)}"
    sql += "\nFROM material_entry m"
    for j in joins:
        sql += f"\n    {j}"
    sql += "\nWHERE\n    " + "\n    AND ".join(where)
    if order:
        sql += order
    sql += "\nLIMIT 100;"
    return sql


def seed_from_paper(tex_path: str | Path) -> int:
    """Extract examples from paper and add them to the store. Returns count added."""
    paper_examples = extract_examples_from_paper(tex_path)
    store = load_store()
    existing_nls = {e["nl_query"] for e in store}
    added = 0
    for ex in paper_examples:
        if ex["nl_query"] not in existing_nls:
            store.append(ex)
            existing_nls.add(ex["nl_query"])
            added += 1
    save_store(store)
    return added
