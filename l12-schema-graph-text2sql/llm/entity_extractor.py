"""Extract material entities and conditions from natural language queries.

Includes:
- Element, prototype, stability, property extraction
- Numeric condition parser (band_gap > 1.0 eV, etc.)
- Chemical formula parser (NiAl, Ni3Al, AlNi₃, etc.)
- Coverage score computation for fallback policy
"""
from __future__ import annotations

import functools
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


@functools.lru_cache(maxsize=1)
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


# CJK aliases that are substrings of common compound words (false positives)
_CJK_FALSE_POSITIVE_CONTEXTS: dict[str, re.Pattern[str]] = {
    "金": re.compile(r"合金|金属|金型|超合金"),
}

# ASCII element symbols that appear in domain-specific terms (false positives)
_ASCII_ELEMENT_FALSE_POSITIVE: dict[str, re.Pattern[str]] = {
    "B": re.compile(r"Bサイト|B[\-\s]?site", re.IGNORECASE),
}


def extract_elements(query: str, terms: dict[str, Any] | None = None) -> list[str]:
    if terms is None:
        terms = _load_terms()
    found: list[str] = []
    for elem, info in terms.get("elements", {}).items():
        for alias in info.get("aliases", []):
            if _is_ascii(alias):
                pattern = _NON_ALNUM + re.escape(alias) + _NON_ALNUM_END
                if re.search(pattern, query, re.IGNORECASE):
                    # Check for false positive ASCII element contexts
                    fp_pat = _ASCII_ELEMENT_FALSE_POSITIVE.get(elem)
                    if fp_pat and fp_pat.search(query):
                        # Only skip if the element ONLY appears in the false-positive context
                        cleaned = fp_pat.sub("", query)
                        if not re.search(pattern, cleaned, re.IGNORECASE):
                            continue
                    if elem not in found:
                        found.append(elem)
                    break
            else:
                if alias in query:
                    # Check for false positive CJK contexts
                    ctx_pat = _CJK_FALSE_POSITIVE_CONTEXTS.get(alias)
                    if ctx_pat and ctx_pat.search(query):
                        continue
                    if elem not in found:
                        found.append(elem)
                    break
    return found


def extract_stability(query: str, terms: dict[str, Any] | None = None) -> str | list[str] | None:
    if terms is None:
        terms = _load_terms()
    q = _normalize(query).lower()
    found: list[str] = []
    if "準安定" in query or "metastable" in q:
        found.append("metastable")
    if "安定" in query or "stable" in q:
        # Only add "stable" if we didn't just match it as part of "metastable"
        if "metastable" not in found:
            found.append("stable")
        elif re.search(r"(?<!meta)(?<!準)安定|(?<!meta)stable\b", q):
            found.append("stable")
    if not found:
        return None
    if len(found) == 1:
        return found[0]
    return found


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
    """Detect 'near Ni3Al lattice constant'-type queries.

    Also extracts user-specified tolerance (e.g. '0.05 Å以内') if present.
    """
    q = _normalize(query)
    pattern = r"([A-Z][a-z]?)3([A-Z][a-z]?).*(?:格子定数|lattice)"
    m = re.search(pattern, q, re.IGNORECASE)
    result: dict[str, float] | None = None
    if m:
        ref_formulas = {
            "Ni3Al": 3.572,
            "Co3Ti": 3.550,
            "Al3Sc": 4.090,
        }
        formula = f"{m.group(1)}3{m.group(2)}"
        ref_val = ref_formulas.get(formula)
        if ref_val:
            result = {"reference_formula": formula, "reference_lattice_a": ref_val}
    if result is None and "ni3al" in q.lower() and ("格子定数" in query or "lattice" in q.lower()):
        result = {"reference_formula": "Ni3Al", "reference_lattice_a": 3.572}
    if result is not None:
        # Extract user-specified tolerance: "0.05 Å以内" or "±0.05" or "within 0.05"
        tol_m = re.search(r"(?:±|以内|within)\s*(\d+\.?\d*)\s*(?:Å|A\b|angstrom)?|(\d+\.?\d*)\s*(?:Å|A\b)\s*以内", q, re.IGNORECASE)
        if tol_m:
            tol_val = float(tol_m.group(1) or tol_m.group(2))
            if 0 < tol_val < 1.0:
                result["tolerance"] = tol_val
    return result


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
    # Elastic properties → calculated_property (gold SQL path)
    "bulk_modulus": "calculated_property.value",
    "bulk modulus": "calculated_property.value",
    "バルクモジュラス": "calculated_property.value",
    "体積弾性率": "calculated_property.value",
    "bulk_modulus_vrh": "calculated_property.value",
    "shear_modulus": "calculated_property.value",
    "shear modulus": "calculated_property.value",
    "せん断弾性率": "calculated_property.value",
    "shear_modulus_vrh": "calculated_property.value",
    "弾性係数": "calculated_property.value",
    # Thermal properties
    "debye_temperature": "thermal_property.debye_temperature_k",
    "debye temperature": "thermal_property.debye_temperature_k",
    "デバイ温度": "thermal_property.debye_temperature_k",
}

_UNIT_CONVERSION.update({
    "gpa": ("GPa", 1.0),
    "mpa": ("GPa", 0.001),
    "k": ("K", 1.0),
})

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


@functools.lru_cache(maxsize=1)
def _build_numeric_regex() -> tuple[tuple[re.Pattern[str], str], ...]:
    props_alt = "|".join(re.escape(p) for p in sorted(_PROPERTY_COLUMN_MAP, key=len, reverse=True))
    units_alt = "|".join(re.escape(u) for u in sorted(_UNIT_CONVERSION, key=len, reverse=True))
    compiled = []
    for pat_template, kind in _NUMERIC_PATTERNS:
        pat = pat_template.replace("{props}", props_alt).replace("{units}", units_alt)
        compiled.append((re.compile(pat, re.IGNORECASE), kind))
    return tuple(compiled)


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

    # Deduplicate: same (column, operator, value) may be matched by multiple patterns
    seen: set[tuple[str, str, float | str]] = set()
    deduped: list[dict[str, Any]] = []
    for r in results:
        v = r["value"] if not isinstance(r["value"], list) else tuple(r["value"])
        key = (r["column"], r["operator"], v)
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    return deduped


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
    # Use Unicode-aware boundaries: Japanese chars, whitespace, punctuation, start/end
    _FB = r'(?:(?<=\s)|(?<=[\u3000-\u9FFF\u3040-\u309F\u30A0-\u30FF\uFF00-\uFFEF])|(?<=^)|(?<=[^A-Za-z0-9]))'
    _FE = r'(?=\s|[\u3000-\u9FFF\u3040-\u309F\u30A0-\u30FF\uFF00-\uFFEF]|$|[^A-Za-z0-9])'
    for m in re.finditer(_FB + r'([A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*){1,5})' + _FE, q):
        token = m.group(1)
        parsed = _parse_formula_token(token)
        if parsed and len(parsed) >= 2:
            candidates.append(token)

    if not candidates:
        return None

    # Parse all formula candidates (supports multi-formula queries like "Ni3AlとCo3Ti")
    all_formulas: list[dict[str, Any]] = []
    for token in candidates:
        parsed = _parse_formula_token(token)
        if not parsed:
            continue
        elements = sorted(parsed.keys())
        has_counts = any(v != 1.0 for v in parsed.values())
        interpretation = "exact_formula" if has_counts else "contains_elements"
        all_formulas.append({
            "formula_str": token,
            "composition": parsed,
            "elements": elements,
            "interpretation": interpretation,
        })

    if not all_formulas:
        return None

    # Return first formula as primary (backward-compatible), all in "all_formulas"
    result = all_formulas[0]
    if len(all_formulas) > 1:
        result["all_formulas"] = all_formulas
    return result


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

    # Standalone lattice range: "格子定数が3.56 Å±0.03 Å" → BETWEEN 3.53 AND 3.59
    if "lattice_reference" not in result:
        lat_range_m = re.search(
            r"格子定数[がの]?\s*(\d+\.?\d*)\s*(?:Å|A\b)?\s*[±]\s*(\d+\.?\d*)\s*(?:Å|A\b)?",
            query,
        )
        if lat_range_m is None:
            lat_range_m = re.search(
                r"lattice.*?(\d+\.?\d*)\s*(?:Å|A\b)?\s*[±]\s*(\d+\.?\d*)\s*(?:Å|A\b)?",
                query, re.IGNORECASE,
            )
        if lat_range_m:
            center = float(lat_range_m.group(1))
            half_range = float(lat_range_m.group(2))
            result["lattice_range"] = {
                "center": center,
                "half_range": half_range,
                "low": round(center - half_range, 4),
                "high": round(center + half_range, 4),
            }

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

    # Extended keyword detection for 30-table schema
    ql = query.lower()
    _EXTENDED_KEYWORDS = {
        "atomic_number": ["原子番号", "atomic number", "atomic_number"],
        "number_of_elements": ["元素数", "3元素", "4元素", "5元素", "多元素", "number of elements"],
        "source_db": ["oqmd", "materials project", "aflow", "source"],
        "synthesis": ["合成", "synthesis", "作製"],
        "ball_milling": ["ボールミリング", "ball milling", "ball_milling"],
        "arc_melting": ["アーク溶解", "arc melting"],
        "experimental": ["実験", "experimental", "合成実績"],
        "doi": ["doi", "文献", "論文", "paper"],
        "literature": ["文献", "参考文献", "literature", "reference"],
        "application": ["応用", "用途", "application", "超合金"],
        "defect": ["欠陥", "defect", "vacancy", "空孔"],
        "interstitial": ["格子間", "interstitial"],
        "dopant": ["ドーパント", "dopant", "添加"],
        "surface_energy": ["表面エネルギー", "surface energy"],
        "miller_index": ["面", "(100)", "(110)", "(111)", "miller"],
        "surface_reconstruction": ["再構成", "reconstruction", "reconstructed"],
        "grain_boundary_energy": ["粒界", "grain boundary"],
        "elastic_stability": ["弾性的に不安定", "elastic.*stable", "is_stable.*false"],
        "crystal_system": ["結晶系", "crystal system", "cubic", "hexagonal", "tetragonal"],
        "space_group": ["空間群", "space group"],
        "volume": ["体積", "volume"],
        "band_gap": ["バンドギャップ", "band gap", "bandgap"],
        "functional": ["汎関数", "functional", "gga", "pbe", "lda"],
        "site_label": ["aサイト", "bサイト", "a-site", "b-site", "a site", "b site", "サイト元素", "サイト"],
        "calculation_method": ["計算手法", "calculation method"],
        "phase_diagram": ["相図", "phase diagram", "hull"],
        "alloy_system": ["合金系", "alloy system"],
        "lattice_c": ["格子定数c", "lattice_c", "c軸"],
    }
    for key, keywords in _EXTENDED_KEYWORDS.items():
        if key not in result:
            for kw in keywords:
                if re.search(kw, ql):
                    result[key] = True
                    break

    # Site label extraction (A-site / B-site)
    _site_a = re.search(r'[Aa]サイト|A[\-\s]?site', query, re.IGNORECASE)
    _site_b = re.search(r'[Bb]サイト|B[\-\s]?site', query, re.IGNORECASE)
    if _site_a or _site_b:
        sites = []
        if _site_a:
            sites.append('A-site')
        if _site_b:
            sites.append('B-site')
        result['site_label'] = sites if len(sites) > 1 else sites[0]
    elif result.get('site_label') is True:
        # Generic 'サイト' keyword matched but no A/B prefix found;
        # remove the boolean to avoid downstream crash in condition_mapper
        del result['site_label']

    # Coverage score
    coverage = compute_coverage(query, result, terms)
    result["_coverage"] = coverage

    return result
