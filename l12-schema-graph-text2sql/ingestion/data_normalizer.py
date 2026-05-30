"""Normalize raw material data before ingestion."""
from __future__ import annotations

import re


def normalize_formula(formula: str) -> str:
    """Normalize subscript characters in formulas (e.g. Ni₃Al -> Ni3Al)."""
    subscript_map = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    return formula.translate(subscript_map).strip()


def chemical_system_from_formula(formula: str) -> str:
    """Extract a sorted hyphen-separated chemical system from a formula string."""
    elements = sorted(set(re.findall(r"[A-Z][a-z]?", normalize_formula(formula))))
    return "-".join(elements)


def normalize_prototype(prototype: str) -> str:
    """Normalize prototype designations to canonical form."""
    mapping = {
        "l12": "L12", "l1_2": "L12", "cu3au": "L12",
        "b2": "B2", "cscl": "B2",
        "a15": "A15", "cr3si": "A15",
    }
    key = re.sub(r"[\s_₂₃]", "", prototype).lower()
    key = key.replace("\u2082", "2").replace("\u2083", "3")
    return mapping.get(key, prototype)
