"""Extract material entities and conditions from natural language queries."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml


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


def extract_conditions(query: str) -> dict[str, Any]:
    """Extract all structured conditions from a natural language query."""
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

    return result
