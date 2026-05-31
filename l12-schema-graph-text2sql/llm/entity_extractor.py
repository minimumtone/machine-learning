"""Extract material entities and conditions from natural language queries.

Includes:
- Element, prototype, stability, property extraction
- Numeric condition parser (band_gap > 1.0 eV, etc.)
- Chemical formula parser (NiAl, Ni3Al, AlNi₃, etc.)
- Coverage score computation for fallback policy
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

# All standard chemical element symbols for formula parsing / unknown element detection
_ALL_ELEMENTS = {
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm",
}


def _load_terms(path: Path | None = None) -> dict[str, Any]:
    if path is None:
        path = Path(__file__).parent / "material_terms.yaml"
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


_SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")


def _normalize(text: str) -> str:
    return text.translate(_SUBSCRIPT_MAP)


def extract_prototype(query: str, terms: dict[str, Any] | None = None) -> str | list[str] | None:
    if terms is None:
        terms = _load_terms()
    q = _normalize(query)
    found: list[str] = []
    for proto, aliases in terms.get("prototype_aliases", {}).items():
        for alias in aliases:
            if _normalize(alias).lower() in q.lower():
                if proto not in found:
                    found.append(proto)
                break
    if not found:
        return None
    if len(found) == 1:
        return found[0]
    return found


def _is_ascii(s: str) -> bool:
    return all(ord(c) < 128 for c in s)


_NON_ALNUM = r'(?:(?<=\s)|(?<=[\u3000-\u9FFF\uFF00-\uFFEF])|(?<=^)|(?<=[^A-Za-z0-9]))'
_NON_ALNUM_END = r'(?=\s|[\u3000-\u9FFF\uFF00-\uFFEF]|$|[^A-Za-z0-9])'


def extract_elements(query: str, terms: dict[str, Any] | None = None) -> list[str]:
    if terms is None:
        terms = _load_terms()
    found: list[str] = []
    for elem, info in terms.get("elements", {}).items():
        for alias in info.get("aliases", []):
            if _is_ascii(alias):
                pattern = _NON_ALNUM + re.escape(alias) + _NON_ALNUM_END
                if re.search(pattern, query, re.IGNORECASE):
                    if elem not in found:
                        found.append(elem)
                    break
            else:
                if alias in query:
                    if elem not in found:
                        found.append(elem)
                    break
    return found


def extract_stability(query: str, terms: dict[str, Any] | None = None) -> str | None:
    if terms is None:
        terms = _load_terms()
    q = _normalize(query).lower()
    stab_terms = terms.get("stability_terms", {})
    if "準安定" in query or "metastable" in q:
        return "metastable"
    if "安定" in query or "stable" in q:
        return "stable"
    return None


def extract_properties(
    query: str, terms: dict[str, Any] | None = None,
) -> list[str]:
    if terms is None:
        terms = _load_terms()
    q = _normalize(query).lower()
    found: list[str] = []
    for prop_key, info in terms.get("property_terms", {}).items():
        for alias in info.get("aliases", []):
            if alias.lower() in q:
                col = info.get("column", info.get("property_name", prop_key))
                if col not in found:
                    found.append(col)
                break
    return found


def extract_sort(query: str, terms: dict[str, Any] | None = None) -> dict[str, str] | None:
    if terms is None:
        terms = _load_terms()
    q = _normalize(query).lower()
    prop_terms = terms.get("property_terms", {})
    for prop_key, info in prop_terms.items():
        for alias in info.get("aliases", []):
            if alias.lower() in q:
                col = info.get("column", info.get("property_name", prop_key))
                order = "asc"
                if any(w in q for w in ["低い順", "小さい順", "ascending", "asc"]):
                    order = "asc"
                elif any(w in q for w in ["高い順", "大きい順", "descending", "desc"]):
                    order = "desc"
                if "順" in query or "order" in q or "ランキング" in query or "ranking" in q:
                    return {"column": col, "order": order}
    return None


def extract_lattice_reference(query: str) -> dict[str, float] | None:
    """Detect 'near Ni3Al lattice constant'-type queries."""
    q = _normalize(query)
    pattern = r"([A-Z][a-z]?)3([A-Z][a-z]?).*(?:格子定数|lattice)"
    m = re.search(pattern, q, re.IGNORECASE)
    if m:
        ref_formulas = {
            "Ni3Al": 3.572,
            "Co3Ti": 3.550,
            "Al3Sc": 4.090,
        }
        formula = f"{m.group(1)}3{m.group(2)}"
        ref_val = ref_formulas.get(formula)
        if ref_val:
            return {"reference_formula": formula, "reference_lattice_a": ref_val}
    if "ni3al" in q.lower() and ("格子定数" in query or "lattice" in q.lower()):
        return {"reference_formula": "Ni3Al", "reference_lattice_a": 3.572}
    return None


# ---------------------------------------------------------------------------
# Numeric condition parser
# ---------------------------------------------------------------------------

_UNIT_CONVERSION: dict[str, tuple[str, float]] = {
    "ev": ("eV", 1.0),
    "ev/atom": ("eV/atom", 1.0),
    "mev": ("eV", 0.001),
    "mev/atom": ("eV/atom", 0.001),
    "å": ("Å", 1.0),
    "angstrom": ("Å", 1.0),
    "nm": ("Å", 10.0),
}

_PROPERTY_COLUMN_MAP: dict[str, str] = {
    "band_gap": "phase_stability.band_gap",
    "band gap": "phase_stability.band_gap",
    "bandgap": "phase_stability.band_gap",
    "バンドギャップ": "phase_stability.band_gap",
    "energy_above_hull": "phase_stability.energy_above_hull",
    "energy above hull": "phase_stability.energy_above_hull",
    "ehull": "phase_stability.energy_above_hull",
    "ハル上エネルギー": "phase_stability.energy_above_hull",
    "formation_energy": "phase_stability.formation_energy_per_atom",
    "formation energy": "phase_stability.formation_energy_per_atom",
    "形成エネルギー": "phase_stability.formation_energy_per_atom",
    "生成エネルギー": "phase_stability.formation_energy_per_atom",
    "lattice_a": "structure.lattice_a",
    "lattice constant": "structure.lattice_a",
    "格子定数": "structure.lattice_a",
    "volume_per_atom": "structure.volume_per_atom",
    "volume per atom": "structure.volume_per_atom",
    "原子あたり体積": "structure.volume_per_atom",
}

# Regex patterns for numeric comparison in natural language
_NUMERIC_PATTERNS: list[tuple[str, str]] = [
    # "band gap > 1.0" / "band_gap >= 1.0 eV"
    (r"({props})\s*(?:が|は)?\s*([><=!]+)\s*(-?\d+\.?\d*)\s*({units})?", "symbol"),
    # "band gapが1 eV以上"
    (r"({props})\s*(?:が|は)?\s*(-?\d+\.?\d*)\s*({units})?\s*(以上|以下|より大きい|より小さい|未満|超)", "ja_post"),
    # "band gapが1 eVより大きい"
    (r"({props})\s*(?:が|は)?\s*(-?\d+\.?\d*)\s*({units})?\s*より\s*(大きい|小さい|高い|低い)", "ja_yori"),
    # "band gapが負の" / "band gapが正の"
    (r"({props})\s*(?:が|は)?\s*(負|正|positive|negative)", "sign"),
    # "band gap between 1.0 and 2.0"
    (r"({props})\s*(?:が|は)?\s*(?:between|が)\s*(-?\d+\.?\d*)\s*(?:and|から|〜|~|--)\s*(-?\d+\.?\d*)\s*({units})?", "between"),
]

_JA_POST_OP: dict[str, str] = {
    "以上": ">=",
    "以下": "<=",
    "より大きい": ">",
    "より小さい": "<",
    "未満": "<",
    "超": ">",
}

_JA_YORI_OP: dict[str, str] = {
    "大きい": ">",
    "小さい": "<",
    "高い": ">",
    "低い": "<",
}


def _build_numeric_regex() -> list[tuple[re.Pattern[str], str]]:
    props_alt = "|".join(re.escape(p) for p in sorted(_PROPERTY_COLUMN_MAP, key=len, reverse=True))
    units_alt = "|".join(re.escape(u) for u in sorted(_UNIT_CONVERSION, key=len, reverse=True))
    compiled = []
    for pat_template, kind in _NUMERIC_PATTERNS:
        pat = pat_template.replace("{props}", props_alt).replace("{units}", units_alt)
        compiled.append((re.compile(pat, re.IGNORECASE), kind))
    return compiled


def extract_numeric_conditions(query: str) -> list[dict[str, Any]]:
    """Extract numeric comparison conditions from a query.

    Returns a list of dicts with keys:
      column, operator, value, unit (optional), raw_match
    """
    q = _normalize(query)
    results: list[dict[str, Any]] = []
    compiled = _build_numeric_regex()

    for regex, kind in compiled:
        for m in regex.finditer(q):
            if kind == "symbol":
                prop_str, op, val_str, unit_str = m.group(1), m.group(2), m.group(3), m.group(4)
                column = _PROPERTY_COLUMN_MAP.get(prop_str.lower())
                if not column:
                    continue
                value = float(val_str)
                if unit_str:
                    canon_unit, factor = _UNIT_CONVERSION.get(unit_str.lower(), (unit_str, 1.0))
                    value *= factor
                results.append({
                    "column": column,
                    "operator": op,
                    "value": value,
                    "raw_match": m.group(0),
                })

            elif kind == "ja_post":
                prop_str, val_str, unit_str, post = m.group(1), m.group(2), m.group(3), m.group(4)
                column = _PROPERTY_COLUMN_MAP.get(prop_str.lower())
                if not column:
                    continue
                value = float(val_str)
                if unit_str:
                    canon_unit, factor = _UNIT_CONVERSION.get(unit_str.lower(), (unit_str, 1.0))
                    value *= factor
                op = _JA_POST_OP.get(post, ">=")
                results.append({"column": column, "operator": op, "value": value, "raw_match": m.group(0)})

            elif kind == "ja_yori":
                prop_str, val_str, unit_str, adj = m.group(1), m.group(2), m.group(3), m.group(4)
                column = _PROPERTY_COLUMN_MAP.get(prop_str.lower())
                if not column:
                    continue
                value = float(val_str)
                if unit_str:
                    canon_unit, factor = _UNIT_CONVERSION.get(unit_str.lower(), (unit_str, 1.0))
                    value *= factor
                op = _JA_YORI_OP.get(adj, ">")
                results.append({"column": column, "operator": op, "value": value, "raw_match": m.group(0)})

            elif kind == "sign":
                prop_str, sign = m.group(1), m.group(2)
                column = _PROPERTY_COLUMN_MAP.get(prop_str.lower())
                if not column:
                    continue
                if sign in ("負", "negative"):
                    results.append({"column": column, "operator": "<", "value": 0, "raw_match": m.group(0)})
                else:
                    results.append({"column": column, "operator": ">", "value": 0, "raw_match": m.group(0)})

            elif kind == "between":
                prop_str, lo_str, hi_str = m.group(1), m.group(2), m.group(3)
                unit_str = m.group(4) if m.lastindex and m.lastindex >= 4 else None
                column = _PROPERTY_COLUMN_MAP.get(prop_str.lower())
                if not column:
                    continue
                lo, hi = float(lo_str), float(hi_str)
                if unit_str:
                    canon_unit, factor = _UNIT_CONVERSION.get(unit_str.lower(), (unit_str, 1.0))
                    lo *= factor
                    hi *= factor
                results.append({
                    "column": column, "operator": "BETWEEN",
                    "value": [lo, hi], "raw_match": m.group(0),
                })

    return results


# ---------------------------------------------------------------------------
# Chemical formula parser
# ---------------------------------------------------------------------------

_FORMULA_RE = re.compile(
    r"\b([A-Z][a-z]?)([₀-₉\d]*)([A-Z][a-z]?)([₀-₉\d]*)(?:([A-Z][a-z]?)([₀-₉\d]*))?\b"
)


def _parse_formula_token(token: str) -> dict[str, float] | None:
    """Parse a chemical formula like Ni3Al, AlNi₃, NiAl into {element: count}."""
    t = _normalize(token).strip()
    parts: list[tuple[str, float]] = []
    pos = 0
    while pos < len(t):
        m = re.match(r"([A-Z][a-z]?)(\d*\.?\d*)", t[pos:])
        if not m or not m.group(1):
            break
        elem = m.group(1)
        if elem not in _ALL_ELEMENTS:
            return None
        count = float(m.group(2)) if m.group(2) else 1.0
        parts.append((elem, count))
        pos += m.end()
    if not parts or pos < len(t):
        return None
    return {elem: count for elem, count in parts}


def extract_formula(query: str) -> dict[str, Any] | None:
    """Detect chemical-formula-like tokens and return parsed info.

    Returns dict with:
      formula_str: original token
      composition: {element: count}
      elements: sorted list of elements
      interpretation: 'exact_formula' | 'contains_elements'
    """
    q = _normalize(query)
    candidates: list[str] = []
    for m in re.finditer(r"\b([A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*){1,5})\b", q):
        token = m.group(1)
        parsed = _parse_formula_token(token)
        if parsed and len(parsed) >= 2:
            candidates.append(token)

    if not candidates:
        return None

    token = candidates[0]
    parsed = _parse_formula_token(token)
    if not parsed:
        return None

    elements = sorted(parsed.keys())
    has_counts = any(v != 1.0 for v in parsed.values())
    interpretation = "exact_formula" if has_counts else "contains_elements"

    return {
        "formula_str": token,
        "composition": parsed,
        "elements": elements,
        "interpretation": interpretation,
    }


# ---------------------------------------------------------------------------
# Coverage score & unrecognized term detection
# ---------------------------------------------------------------------------

_STOPWORDS_JA = {
    "を", "の", "に", "は", "が", "で", "と", "も", "から", "まで",
    "て", "な", "した", "する", "ある", "いる", "ない", "ください",
    "出して", "教えて", "見せて", "出力して", "表示して", "ほしい",
    "全", "もの", "化合物", "合金", "材料", "データ",
    "リスト", "全リスト", "全データ", "一覧",
}

_STOPWORDS_EN = {
    "the", "a", "an", "of", "in", "for", "with", "and", "or", "to",
    "is", "are", "was", "were", "that", "this", "show", "list", "give",
    "me", "all", "find", "get", "containing", "compounds", "alloys",
    "materials", "data", "display", "output", "please",
}


def _tokenize_for_coverage(query: str) -> list[str]:
    """Tokenize query into meaningful tokens for coverage analysis."""
    tokens: list[str] = []
    for m in re.finditer(
        r"[A-Z][a-z]?\d*|[a-z]+|[\u3040-\u309F]+|[\u30A0-\u30FF]+|[\u4E00-\u9FFF]+|\d+\.?\d*",
        query,
    ):
        t = m.group()
        if t.lower() not in _STOPWORDS_EN and t not in _STOPWORDS_JA and len(t) > 1:
            tokens.append(t)
    return tokens


def _detect_element_like_tokens(query: str, known_elements: set[str]) -> list[str]:
    """Find tokens that look like element symbols but are not in the dictionary.

    Excludes symbols that are part of prototype aliases (e.g. B in B2).
    """
    unknown: list[str] = []
    for m in re.finditer(r"(?:(?<=\s)|(?<=^)|(?<=[^A-Za-z]))([A-Z][a-z]?)(?=\s|[^A-Za-z0-9]|$)", query):
        sym = m.group(1)
        if sym in _ALL_ELEMENTS and sym not in known_elements:
            start = m.start(1)
            after = query[start:]
            if re.match(r"[A-Z][a-z]?\d", after):
                continue
            if sym not in unknown:
                unknown.append(sym)
    return unknown


def compute_coverage(
    query: str,
    conditions: dict[str, Any],
    terms: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute a coverage score indicating how much of the query was recognized.

    Returns:
      recognized_constraints: dict of recognized condition types
      unrecognized_terms: list of tokens that were not matched
      unknown_elements: list of element-like symbols not in dictionary
      coverage_score: float 0.0-1.0
      action: recommended action based on coverage
    """
    if terms is None:
        terms = _load_terms()

    tokens = _tokenize_for_coverage(query)
    if not tokens:
        return {
            "recognized_constraints": conditions,
            "unrecognized_terms": [],
            "unknown_elements": [],
            "coverage_score": 1.0,
            "action": "execute_rule_based",
        }

    known_element_syms = set(terms.get("elements", {}).keys())
    all_known_aliases: set[str] = set()
    for elem_info in terms.get("elements", {}).values():
        for alias in elem_info.get("aliases", []):
            all_known_aliases.add(alias.lower())
    for proto, aliases in terms.get("prototype_aliases", {}).items():
        for alias in aliases:
            all_known_aliases.add(alias.lower())
    for prop_info in terms.get("property_terms", {}).values():
        for alias in prop_info.get("aliases", []):
            all_known_aliases.add(alias.lower())
    for stab_key in terms.get("stability_terms", {}):
        all_known_aliases.add(stab_key)

    recognized_count = 0
    unrecognized: list[str] = []
    for t in tokens:
        if t.lower() in all_known_aliases:
            recognized_count += 1
        elif t in _ALL_ELEMENTS and t in known_element_syms:
            recognized_count += 1
        elif t in _ALL_ELEMENTS:
            unrecognized.append(t)
        elif re.match(r"\d+\.?\d*", t):
            recognized_count += 1
        else:
            matched = False
            for alias in all_known_aliases:
                if t.lower() in alias or alias in t.lower():
                    matched = True
                    break
            if matched:
                recognized_count += 1
            else:
                unrecognized.append(t)

    unknown_elems = _detect_element_like_tokens(query, known_element_syms)
    coverage = recognized_count / len(tokens) if tokens else 1.0

    if coverage >= 0.8 and not unknown_elems:
        action = "execute_rule_based"
    elif coverage >= 0.5 or unknown_elems:
        action = "fallback_to_llm"
    else:
        action = "clarification_required"

    return {
        "recognized_constraints": conditions,
        "unrecognized_terms": unrecognized,
        "unknown_elements": unknown_elems,
        "coverage_score": round(coverage, 3),
        "action": action,
    }


def extract_conditions(query: str) -> dict[str, Any]:
    """Extract all structured conditions from a natural language query.

    Returns a dict with recognized condition keys and a '_coverage' sub-dict
    containing coverage_score, unrecognized_terms, unknown_elements, and
    recommended action.
    """
    terms = _load_terms()
    result: dict[str, Any] = {}

    proto = extract_prototype(query, terms)
    if proto:
        result["prototype"] = proto

    elements = extract_elements(query, terms)
    if elements:
        result["contains_elements"] = elements

    stability = extract_stability(query, terms)
    if stability:
        result["stability"] = stability

    props = extract_properties(query, terms)
    if props:
        result["properties"] = props

    sort = extract_sort(query, terms)
    if sort:
        result["sort_by"] = sort["column"]
        result["sort_order"] = sort["order"]

    lattice_ref = extract_lattice_reference(query)
    if lattice_ref:
        result["lattice_reference"] = lattice_ref

    # Numeric conditions
    numeric_conds = extract_numeric_conditions(query)
    if numeric_conds:
        result["numeric_conditions"] = numeric_conds

    # Chemical formula detection
    formula = extract_formula(query)
    if formula:
        result["formula"] = formula
        if not elements and formula["interpretation"] == "contains_elements":
            result["contains_elements"] = formula["elements"]

    # Coverage score
    coverage = compute_coverage(query, result, terms)
    result["_coverage"] = coverage

    return result
