"""Map extracted conditions to SQL WHERE-clause fragments."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _load_terms(path: Path | None = None) -> dict[str, Any]:
    if path is None:
        path = Path(__file__).parent / "material_terms.yaml"
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def map_prototype_condition(prototype: str | list[str]) -> dict[str, Any]:
    if isinstance(prototype, list):
        parts = " OR ".join(
            f"s.prototype = '{p}' OR s.strukturbericht = '{p}'" for p in prototype
        )
        sql_fragment = f"({parts})"
    else:
        sql_fragment = f"(s.prototype = '{prototype}' OR s.strukturbericht = '{prototype}')"
    return {
        "type": "prototype",
        "sql_fragment": sql_fragment,
        "tables": ["structure"],
        "columns": ["structure.prototype", "structure.strukturbericht"],
    }


def map_element_condition(elements: list[str]) -> list[dict[str, Any]]:
    conditions: list[dict[str, Any]] = []
    if len(elements) == 1:
        conditions.append({
            "type": "element",
            "sql_fragment": f"c.element = '{elements[0]}'",
            "tables": ["composition"],
            "columns": ["composition.element"],
        })
    elif len(elements) > 1:
        for elem in elements:
            conditions.append({
                "type": "element_exists",
                "sql_fragment": (
                    f"EXISTS (SELECT 1 FROM composition c_{elem.lower()}"
                    f" WHERE c_{elem.lower()}.entry_id = m.entry_id"
                    f" AND c_{elem.lower()}.element = '{elem}')"
                ),
                "tables": ["composition"],
                "columns": ["composition.element"],
            })
    return conditions


def map_stability_condition(
    stability: str,
    terms: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if terms is None:
        terms = _load_terms()
    stab_info = terms.get("stability_terms", {}).get(stability)
    if not stab_info:
        return None
    cond = stab_info["condition"]
    col_parts = cond["column"].split(".")
    alias = "ps" if col_parts[0] == "phase_stability" else col_parts[0][:2]
    return {
        "type": "stability",
        "sql_fragment": f"{alias}.{col_parts[1]} {cond['operator']} {cond['value']}",
        "tables": [col_parts[0]],
        "columns": [cond["column"]],
    }


def map_lattice_reference_condition(
    ref: dict[str, float],
    tolerance: float = 0.03,
) -> dict[str, Any]:
    val = ref["reference_lattice_a"]
    return {
        "type": "lattice_reference",
        "sql_fragment": f"ABS(s.lattice_a - {val}) <= {tolerance}",
        "tables": ["structure"],
        "columns": ["structure.lattice_a"],
    }


def map_sort_condition(sort_by: str, sort_order: str) -> dict[str, Any]:
    col_parts = sort_by.split(".")
    if len(col_parts) == 2:
        table, col = col_parts
        alias_map = {
            "phase_stability": "ps",
            "structure": "s",
            "material_entry": "m",
            "composition": "c",
        }
        alias = alias_map.get(table, table[:2])
        order_sql = f"{alias}.{col} {sort_order.upper()}"
    else:
        order_sql = f"{sort_by} {sort_order.upper()}"
    return {
        "type": "sort",
        "sql_fragment": f"ORDER BY {order_sql}",
        "tables": [col_parts[0]] if len(col_parts) == 2 else [],
        "columns": [sort_by],
    }


def map_numeric_condition(cond: dict[str, Any]) -> dict[str, Any]:
    """Map a numeric comparison condition to a SQL fragment."""
    column = cond["column"]
    op = cond["operator"]
    value = cond["value"]

    col_parts = column.split(".")
    alias_map = {
        "phase_stability": "ps",
        "structure": "s",
        "material_entry": "m",
        "composition": "c",
        "calculation": "calc",
        "calculated_property": "cp",
    }
    if len(col_parts) == 2:
        table, col = col_parts
        alias = alias_map.get(table, table[:2])
        col_ref = f"{alias}.{col}"
    else:
        col_ref = column
        table = ""

    if op == "BETWEEN" and isinstance(value, list) and len(value) == 2:
        sql_fragment = f"{col_ref} BETWEEN {value[0]} AND {value[1]}"
    else:
        sql_fragment = f"{col_ref} {op} {value}"

    return {
        "type": "numeric",
        "sql_fragment": sql_fragment,
        "tables": [table] if table else [],
        "columns": [column],
    }


def map_formula_condition(formula_info: dict[str, Any]) -> list[dict[str, Any]]:
    """Map a chemical formula condition to SQL fragments."""
    interpretation = formula_info.get("interpretation", "contains_elements")
    fragments: list[dict[str, Any]] = []

    if interpretation == "exact_formula":
        formula_str = formula_info["formula_str"]
        fragments.append({
            "type": "formula",
            "sql_fragment": (
                f"(m.formula = '{formula_str}'"
                f" OR m.reduced_formula = '{formula_str}')"
            ),
            "tables": ["material_entry"],
            "columns": ["material_entry.formula", "material_entry.reduced_formula"],
        })
    else:
        elements = formula_info.get("elements", [])
        if elements:
            fragments.extend(map_element_condition(elements))

    return fragments


def map_conditions(conditions: dict[str, Any]) -> list[dict[str, Any]]:
    """Map a conditions dict (from entity_extractor) to SQL fragments."""
    mapped: list[dict[str, Any]] = []

    if "prototype" in conditions:
        mapped.append(map_prototype_condition(conditions["prototype"]))

    if "contains_elements" in conditions:
        mapped.extend(map_element_condition(conditions["contains_elements"]))

    if "stability" in conditions:
        stab = map_stability_condition(conditions["stability"])
        if stab:
            mapped.append(stab)

    if "lattice_reference" in conditions:
        mapped.append(
            map_lattice_reference_condition(conditions["lattice_reference"])
        )

    if "numeric_conditions" in conditions:
        for nc in conditions["numeric_conditions"]:
            mapped.append(map_numeric_condition(nc))

    if "formula" in conditions and "contains_elements" not in conditions:
        mapped.extend(map_formula_condition(conditions["formula"]))

    if "sort_by" in conditions:
        mapped.append(
            map_sort_condition(
                conditions["sort_by"],
                conditions.get("sort_order", "asc"),
            )
        )

    return mapped
