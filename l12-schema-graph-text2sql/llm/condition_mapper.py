"""Map extracted conditions to SQL WHERE-clause fragments."""
from __future__ import annotations

import functools
import re
from pathlib import Path
from typing import Any

import yaml


@functools.lru_cache(maxsize=1)
def _load_terms(path: Path | None = None) -> dict[str, Any]:
    if path is None:
        path = Path(__file__).parent / "material_terms.yaml"
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _escape_sql_value(value: str) -> str:
    """Escape single quotes in SQL string values to prevent injection."""
    return value.replace("'", "''")


def map_prototype_condition(prototype: str | list[str]) -> dict[str, Any]:
    if isinstance(prototype, list):
        parts = " OR ".join(
            f"s.prototype = '{_escape_sql_value(p)}' OR s.strukturbericht = '{_escape_sql_value(p)}'"
            for p in prototype
        )
        sql_fragment = f"({parts})"
    else:
        safe = _escape_sql_value(prototype)
        sql_fragment = f"(s.prototype = '{safe}' OR s.strukturbericht = '{safe}')"
    return {
        "type": "prototype",
        "sql_fragment": sql_fragment,
        "tables": ["structure"],
        "columns": ["structure.prototype", "structure.strukturbericht"],
    }


def map_element_condition(
    elements: list[str],
    logic: str = "AND",
) -> list[dict[str, Any]]:
    conditions: list[dict[str, Any]] = []
    if len(elements) == 1:
        safe = _escape_sql_value(elements[0])
        conditions.append({
            "type": "element",
            "sql_fragment": f"c.element = '{safe}'",
            "tables": ["composition"],
            "columns": ["composition.element"],
        })
    elif len(elements) > 1 and logic == "OR":
        # OR: single EXISTS with IN clause
        safe_elems = ", ".join(f"'{_escape_sql_value(e)}'" for e in elements)
        conditions.append({
            "type": "element_or",
            "sql_fragment": (
                f"EXISTS (SELECT 1 FROM composition c_or"
                f" WHERE c_or.entry_id = m.entry_id"
                f" AND c_or.element IN ({safe_elems}))"
            ),
            "tables": ["composition"],
            "columns": ["composition.element"],
        })
    elif len(elements) > 1:
        # AND: separate EXISTS per element (all must be present)
        for elem in elements:
            safe = _escape_sql_value(elem)
            alias = re.sub(r"[^a-z0-9]", "", elem.lower())
            conditions.append({
                "type": "element_exists",
                "sql_fragment": (
                    f"EXISTS (SELECT 1 FROM composition c_{alias}"
                    f" WHERE c_{alias}.entry_id = m.entry_id"
                    f" AND c_{alias}.element = '{safe}')"
                ),
                "tables": ["composition"],
                "columns": ["composition.element"],
            })
    return conditions


def map_stability_condition(
    stability: str | list[str],
    terms: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if terms is None:
        terms = _load_terms()

    # Handle list of stability values (e.g. ["stable", "metastable"])
    if isinstance(stability, list):
        fragments: list[str] = []
        tables: set[str] = set()
        columns: set[str] = set()
        for s in stability:
            stab_info = terms.get("stability_terms", {}).get(s)
            if not stab_info:
                continue
            cond = stab_info["condition"]
            col_parts = cond["column"].split(".")
            alias = "ps" if col_parts[0] == "phase_stability" else col_parts[0][:2]
            fragments.append(f"{alias}.{col_parts[1]} {cond['operator']} {cond['value']}")
            tables.add(col_parts[0])
            columns.add(cond["column"])
        if not fragments:
            return None
        sql_fragment = "(" + " OR ".join(fragments) + ")" if len(fragments) > 1 else fragments[0]
        return {
            "type": "stability",
            "sql_fragment": sql_fragment,
            "tables": list(tables),
            "columns": list(columns),
        }

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
    tol = ref.get("tolerance", tolerance)
    return {
        "type": "lattice_reference",
        "sql_fragment": f"ABS(s.lattice_a - {val}) <= {tol}",
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
            "calculation": "calc",
            "calculated_property": "cp",
            "elastic_tensor": "et",
            "thermal_property": "tp",
            "magnetic_property": "mp",
            "density_of_states": "dos",
            "band_structure": "bs",
            "surface_energy": "se",
            "grain_boundary": "gb",
            "literature_reference": "lr",
            "material_defect": "md",
            "defect_type": "dt",
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
        "elastic_tensor": "et",
        "thermal_property": "tp",
        "magnetic_property": "mp",
        "density_of_states": "dos",
        "band_structure": "bs",
        "surface_energy": "se",
        "grain_boundary": "gb",
        "literature_reference": "lr",
        "material_defect": "md",
        "defect_type": "dt",
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


def map_site_label_condition(site_label: str | list[str]) -> dict[str, Any]:
    """Map a site_label condition to a SQL fragment."""
    if isinstance(site_label, list):
        values = ", ".join(f"'{_escape_sql_value(s)}'" for s in site_label)
        sql_fragment = f"c.site_label IN ({values})"
    else:
        safe = _escape_sql_value(site_label)
        sql_fragment = f"c.site_label = '{safe}'"
    return {
        "type": "site_label",
        "sql_fragment": sql_fragment,
        "tables": ["composition"],
        "columns": ["composition.site_label"],
    }


def map_conditions(conditions: dict[str, Any]) -> list[dict[str, Any]]:
    """Map a conditions dict (from entity_extractor) to SQL fragments."""
    mapped: list[dict[str, Any]] = []

    if "prototype" in conditions:
        mapped.append(map_prototype_condition(conditions["prototype"]))

    if "contains_elements" in conditions:
        logic = conditions.get("element_logic", "AND")
        mapped.extend(map_element_condition(conditions["contains_elements"], logic))

    if "stability" in conditions:
        stab = map_stability_condition(conditions["stability"])
        if stab:
            mapped.append(stab)

    if "lattice_reference" in conditions:
        mapped.append(
            map_lattice_reference_condition(conditions["lattice_reference"])
        )

    if "lattice_range" in conditions:
        lr = conditions["lattice_range"]
        mapped.append({
            "type": "lattice_range",
            "sql_fragment": f"s.lattice_a BETWEEN {lr['low']} AND {lr['high']}",
            "tables": ["structure"],
            "columns": ["structure.lattice_a"],
        })

    if "numeric_conditions" in conditions:
        for nc in conditions["numeric_conditions"]:
            mapped.append(map_numeric_condition(nc))

    if "formula" in conditions and "contains_elements" not in conditions:
        if "lattice_reference" not in conditions:
            mapped.extend(map_formula_condition(conditions["formula"]))

    if "site_label" in conditions:
        mapped.append(map_site_label_condition(conditions["site_label"]))

    if "sort_by" in conditions:
        mapped.append(
            map_sort_condition(
                conditions["sort_by"],
                conditions.get("sort_order", "asc"),
            )
        )

    return mapped
