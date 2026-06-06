"""
Entity extractor — extract material entities from natural language.

Extracts:
  - Chemical elements (symbol, Japanese name, English name)
  - Chemical formulas (Ni3Al, BaTiO₃, etc.)
  - Structure prototypes (L1₂, perovskite, etc.)
  - Numeric VASP parameters mentioned in query
  - Supercell size hints

Design: purely rule-based (no LLM required).
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

_SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_SUPERSCRIPT_MAP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻", "0123456789+-")

# Standard element symbols for formula parsing
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
}


def _normalize(text: str) -> str:
    return text.translate(_SUBSCRIPT_MAP).translate(_SUPERSCRIPT_MAP)


def _load_terms(path: Optional[Path] = None) -> dict:
    if path is None:
        path = Path(__file__).parent / "config" / "material_terms.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Formula parser ────────────────────────────────────────────────────

_FORMULA_RE = re.compile(
    r"([A-Z][a-z]?)(\d*\.?\d*)"
)


def parse_formula(formula_str: str) -> Dict[str, float]:
    """Parse a chemical formula string into {element: count}.

    Examples
    --------
    >>> parse_formula("Ni3Al")
    {'Ni': 3.0, 'Al': 1.0}
    >>> parse_formula("BaTiO3")
    {'Ba': 1.0, 'Ti': 1.0, 'O': 3.0}
    >>> parse_formula("(CrFeCoNi)0.25")
    {'Cr': 0.25, 'Fe': 0.25, 'Co': 0.25, 'Ni': 0.25}
    """
    s = _normalize(formula_str).strip()

    # Handle parenthesized formulas: (AB...)x
    paren_match = re.match(r"\((.+?)\)(\d*\.?\d+)", s)
    if paren_match:
        inner = paren_match.group(1)
        multiplier = float(paren_match.group(2))
        inner_comp = parse_formula(inner)
        return {el: cnt * multiplier for el, cnt in inner_comp.items()}

    composition: Dict[str, float] = {}
    for m in _FORMULA_RE.finditer(s):
        elem = m.group(1)
        count_str = m.group(2)
        if elem not in _ALL_ELEMENTS:
            continue
        count = float(count_str) if count_str else 1.0
        composition[elem] = composition.get(elem, 0.0) + count
    return composition


def _extract_formula_from_query(query: str) -> Optional[str]:
    """Try to extract a chemical formula substring from free text."""
    q = _normalize(query)
    # Pattern: sequences of Element+optional_number, at least 2 characters,
    # possibly wrapped in parentheses
    patterns = [
        r"\((?:[A-Z][a-z]?\d*\.?\d*){2,}\)\d*\.?\d+",   # (CrFeCoNi)0.25
        r"(?:[A-Z][a-z]?\d*\.?\d*){1,}",                 # Ni3Al, BaTiO3
    ]
    for pat in patterns:
        for m in re.finditer(pat, q):
            candidate = m.group()
            comp = parse_formula(candidate)
            if comp and len(candidate) >= 2:
                return candidate
    return None


# ── Main extraction ──────────────────────────────────────────────────

@dataclass
class EntityResult:
    """Extracted entities from a query."""
    elements: List[str] = field(default_factory=list)
    composition: Dict[str, float] = field(default_factory=dict)
    formula_str: Optional[str] = None
    prototype: Optional[str] = None
    spin_polarized: Optional[bool] = None
    encut: Optional[int] = None
    kpoints: Optional[Tuple[int, int, int]] = None
    supercell_size: Optional[int] = None
    raw_query: str = ""

    @property
    def species_list(self) -> List[str]:
        """Ordered list of unique element symbols."""
        if self.composition:
            return list(self.composition.keys())
        return self.elements


def extract(query: str, terms_path: Optional[Path] = None) -> EntityResult:
    """Extract material entities from a natural-language query.

    Parameters
    ----------
    query : str
        Free-text in Japanese or English.
    terms_path : Path, optional
        Path to material_terms.yaml.

    Returns
    -------
    EntityResult
    """
    terms = _load_terms(terms_path)
    q_norm = _normalize(query)
    q_lower = q_norm.lower()

    result = EntityResult(raw_query=query)

    # 1. Extract formula
    formula_str = _extract_formula_from_query(query)
    if formula_str:
        result.formula_str = formula_str
        result.composition = parse_formula(formula_str)
        result.elements = list(result.composition.keys())
        logger.info("Extracted formula: %s → %s", formula_str, result.composition)

    # 2. Extract elements from Japanese/English names if no formula found
    if not result.elements:
        elem_terms = terms.get("elements", {})
        found: List[str] = []
        for elem, info in elem_terms.items():
            for alias in info.get("aliases", []):
                alias_lower = alias.lower()
                # For single-letter elements or short aliases in ASCII,
                # require word boundary
                if len(alias) <= 2 and alias.isascii():
                    pattern = r"(?<![A-Za-z])" + re.escape(alias) + r"(?![A-Za-z])"
                    if re.search(pattern, q_norm):
                        if elem not in found:
                            found.append(elem)
                        break
                else:
                    if alias_lower in q_lower or alias in query:
                        if elem not in found:
                            found.append(elem)
                        break
        result.elements = found

    # 3. Extract prototype
    proto_aliases = terms.get("prototype_aliases", {})
    for proto, aliases in proto_aliases.items():
        for alias in aliases:
            alias_check = _normalize(alias).lower()
            if alias_check in q_lower:
                result.prototype = proto
                logger.info("Extracted prototype: %s (matched '%s')", proto, alias)
                break
        if result.prototype:
            break

    # 4. Extract VASP parameters
    # ENCUT
    encut_m = re.search(r"ENCUT\s*[=:]\s*(\d+)", q_norm, re.IGNORECASE)
    if encut_m:
        result.encut = int(encut_m.group(1))

    # k-points: "k点12×12×12" or "kpoints 8 8 8"
    kpt_m = re.search(r"[kK](?:点|points?)?\s*(\d+)\s*[×x×]\s*(\d+)\s*[×x×]\s*(\d+)", q_norm)
    if kpt_m:
        result.kpoints = (int(kpt_m.group(1)), int(kpt_m.group(2)), int(kpt_m.group(3)))

    # Spin polarization hints
    if any(kw in q_lower for kw in ["スピン偏極", "spin polariz", "ispin=2", "磁性", "magnetic"]):
        result.spin_polarized = True

    # Supercell size
    sc_m = re.search(r"(\d+)\s*(?:原子|atoms?|atom)\s*(?:セル|cell|supercell)?", q_norm, re.IGNORECASE)
    if sc_m:
        result.supercell_size = int(sc_m.group(1))

    logger.info("Entities: elements=%s, proto=%s, spin=%s",
                result.elements, result.prototype, result.spin_polarized)
    return result
