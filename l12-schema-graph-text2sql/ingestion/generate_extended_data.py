#!/usr/bin/env python3
"""Generate data for the 32-table extended schema.

Produces 1,470 compound entries across 5 prototypes (L12, B2, NaCl, NiAs,
BiF3) plus 89 OQMD pure-element ground-state entries. Output is split by
role to match the schema file layout:

    db/002_reference_data.sql  -- master/reference tables
    db/003_material_data.sql   -- material entries and dependent rows

Usage:
    python ingestion/generate_extended_data.py
    # DB rebuild: (cd docker && docker compose down -v && docker compose up -d)
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import TextIO

random.seed(42)

PROJECT = Path(__file__).resolve().parents[1]

# ============================================================
# Compound Definitions
# ============================================================

PROTOTYPES = {
    "L12": {"strukturbericht": "L12", "formula_type": "A3B", "crystal_system": "cubic",
             "sg_number": 221, "sg_symbol": "Pm-3m"},
    "B2":  {"strukturbericht": "B2", "formula_type": "AB", "crystal_system": "cubic",
             "sg_number": 221, "sg_symbol": "Pm-3m"},
    "NaCl": {"strukturbericht": "B1", "formula_type": "AB", "crystal_system": "cubic",
              "sg_number": 225, "sg_symbol": "Fm-3m"},
    "NiAs": {"strukturbericht": "B81", "formula_type": "AB", "crystal_system": "hexagonal",
              "sg_number": 194, "sg_symbol": "P6_3/mmc"},
    "BiF3": {"strukturbericht": "D03", "formula_type": "AB3", "crystal_system": "cubic",
              "sg_number": 225, "sg_symbol": "Fm-3m"},
}

A_ELEMENTS = [
    "Ni", "Co", "Fe", "Cu", "Pd", "Pt", "Ir", "Rh", "Ru", "Ag", "Au",
    "Ti", "Zr", "Hf", "V", "Nb", "Ta", "Cr", "Mo", "W", "Mn",
]
B_ELEMENTS = [
    "Al", "Ga", "Ge", "Ti", "Nb", "Ta", "Sc", "Y", "Hf", "Si", "Sn",
    "Zn", "V", "Zr", "Mn", "In", "Sb", "Be", "Mg", "Cu", "Fe",
]

# Operational stability definition used throughout the paper, the gold SQL
# and the phase_stability.is_stable label. In the DDL, is_stable is a
# GENERATED column (energy_above_hull <= STABLE_EAH_THRESHOLD), so INSERTs
# never write it directly.
STABLE_EAH_THRESHOLD = 0.001  # eV/atom

# Known L12 compounds (a_elem, b_elem, lattice_a, formation_energy,
# energy_above_hull, bulk_modulus, shear_modulus); is_stable is derived
# from energy_above_hull, never hand-assigned
KNOWN_L12 = [
    ("Ni", "Al", 3.572, -0.420, 0.000, 180.0, 85.0),
    ("Ni", "Ga", 3.660, -0.380, 0.010, 165.0, 74.0),
    ("Ni", "Ge", 3.610, -0.350, 0.040, 175.0, 78.0),
    ("Co", "Ti", 3.550, -0.450, 0.000, 190.0, 90.0),
    ("Al", "Sc", 4.090, -0.500, 0.000, 155.0, 65.0),
    ("Al", "Ti", 3.980, -0.410, 0.020, 160.0, 70.0),
    ("Pt", "Al", 3.900, -0.550, 0.000, 210.0, 95.0),
    ("Ir", "Nb", 3.870, -0.480, 0.030, 220.0, 100.0),
    ("Co", "Al", 3.670, -0.360, 0.010, 180.0, 82.0),
    ("Co", "W",  3.740, -0.330, 0.050, 200.0, 88.0),
    ("Co", "Ta", 3.720, -0.340, 0.040, 198.0, 86.0),
]

# Base 62-element attribute set: symbol -> (Z, mass, electronegativity, radius)
ELEMENT_DATA = {
    "H": (1, 1.008, 2.20, 25), "He": (2, 4.003, 0.00, 31),
    "Li": (3, 6.941, 0.98, 152), "Be": (4, 9.012, 1.57, 112),
    "B": (5, 10.81, 2.04, 87), "C": (6, 12.01, 2.55, 77),
    "N": (7, 14.01, 3.04, 75), "O": (8, 16.00, 3.44, 73),
    "F": (9, 19.00, 3.98, 71), "Ne": (10, 20.18, 0.00, 69),
    "Na": (11, 22.99, 0.93, 186), "Mg": (12, 24.31, 1.31, 160),
    "Al": (13, 26.98, 1.61, 143), "Si": (14, 28.09, 1.90, 117),
    "P": (15, 30.97, 2.19, 110), "S": (16, 32.07, 2.58, 104),
    "Cl": (17, 35.45, 3.16, 99), "Ar": (18, 39.95, 0.00, 97),
    "K": (19, 39.10, 0.82, 227), "Ca": (20, 40.08, 1.00, 197),
    "Sc": (21, 44.96, 1.36, 162), "Ti": (22, 47.87, 1.54, 147),
    "V": (23, 50.94, 1.63, 134), "Cr": (24, 52.00, 1.66, 128),
    "Mn": (25, 54.94, 1.55, 127), "Fe": (26, 55.85, 1.83, 126),
    "Co": (27, 58.93, 1.88, 125), "Ni": (28, 58.69, 1.91, 124),
    "Cu": (29, 63.55, 1.90, 128), "Zn": (30, 65.38, 1.65, 134),
    "Ga": (31, 69.72, 1.81, 135), "Ge": (32, 72.63, 2.01, 122),
    "As": (33, 74.92, 2.18, 119), "Se": (34, 78.96, 2.55, 120),
    "Br": (35, 79.90, 2.96, 120), "Kr": (36, 83.80, 3.00, 116),
    "Rb": (37, 85.47, 0.82, 248), "Sr": (38, 87.62, 0.95, 215),
    "Y": (39, 88.91, 1.22, 180), "Zr": (40, 91.22, 1.33, 160),
    "Nb": (41, 92.91, 1.60, 146), "Mo": (42, 95.96, 2.16, 139),
    "Ru": (44, 101.1, 2.20, 134), "Rh": (45, 102.9, 2.28, 134),
    "Pd": (46, 106.4, 2.20, 137), "Ag": (47, 107.9, 1.93, 144),
    "In": (49, 114.8, 1.78, 167), "Sn": (50, 118.7, 1.96, 140),
    "Sb": (51, 121.8, 2.05, 140), "Te": (52, 127.6, 2.10, 142),
    "I": (53, 126.9, 2.66, 139), "Xe": (54, 131.3, 2.60, 140),
    "Cs": (55, 132.9, 0.79, 265), "Ba": (56, 137.3, 0.89, 222),
    "Hf": (72, 178.5, 1.30, 159), "Ta": (73, 180.9, 1.50, 146),
    "W": (74, 183.8, 2.36, 139), "Re": (75, 186.2, 1.90, 137),
    "Os": (76, 190.2, 2.20, 135), "Ir": (77, 192.2, 2.20, 136),
    "Pt": (78, 195.1, 2.28, 139), "Au": (79, 197.0, 2.54, 144),
}

_PERIOD_STARTS = [1, 3, 11, 19, 37, 55, 87]


def _element_period_group_block(anum: int) -> tuple[int, int | None, str]:
    """Derive (period, group, block) from the atomic number."""
    period = max(p for p, start in enumerate(_PERIOD_STARTS, 1) if anum >= start)
    if anum in (1, 2):
        return 1, {1: 1, 2: 18}[anum], "s"
    offset = anum - _PERIOD_STARTS[period - 1]
    if period in (2, 3):
        group = offset + 1 if offset <= 1 else offset + 11
        return period, group, "s" if offset <= 1 else "p"
    if period in (4, 5):
        group = offset + 1
        block = "s" if offset <= 1 else ("d" if offset <= 11 else "p")
        return period, group, block
    # periods 6, 7 include the f-block (14 elements after group 3)
    if offset <= 1:
        return period, offset + 1, "s"
    if offset == 2:
        return period, 3, "f"  # La/Ac treated as f-block heads
    if offset <= 16:
        return period, None, "f"
    group = offset - 13
    return period, group, "d" if group <= 12 else "p"


# Elements present only in the OQMD pure-element reference set:
# symbol -> (name, Z, mass, electronegativity, radius, group, period, block, category)
EXTRA_ELEMENT_DATA = {
    "Ac": ("Actinium", 89, 227.0, 1.10, 195, 3, 7, "f", "actinide"),
    "Bi": ("Bismuth", 83, 208.98, 2.02, 156, 15, 6, "p", "post-transition metal"),
    "Cd": ("Cadmium", 48, 112.41, 1.69, 151, 12, 5, "d", "transition metal"),
    "Ce": ("Cerium", 58, 140.12, 1.12, 182, None, 6, "f", "lanthanide"),
    "Dy": ("Dysprosium", 66, 162.50, 1.22, 178, None, 6, "f", "lanthanide"),
    "Er": ("Erbium", 68, 167.26, 1.24, 176, None, 6, "f", "lanthanide"),
    "Eu": ("Europium", 63, 151.96, 1.20, 180, None, 6, "f", "lanthanide"),
    "Gd": ("Gadolinium", 64, 157.25, 1.20, 180, None, 6, "f", "lanthanide"),
    "Hg": ("Mercury", 80, 200.59, 2.00, 151, 12, 6, "d", "transition metal"),
    "Ho": ("Holmium", 67, 164.93, 1.23, 177, None, 6, "f", "lanthanide"),
    "La": ("Lanthanum", 57, 138.91, 1.10, 187, 3, 6, "f", "lanthanide"),
    "Lu": ("Lutetium", 71, 174.97, 1.27, 174, 3, 6, "f", "lanthanide"),
    "Nd": ("Neodymium", 60, 144.24, 1.14, 181, None, 6, "f", "lanthanide"),
    "Np": ("Neptunium", 93, 237.0, 1.36, 155, None, 7, "f", "actinide"),
    "Pa": ("Protactinium", 91, 231.04, 1.50, 163, None, 7, "f", "actinide"),
    "Pb": ("Lead", 82, 207.2, 2.33, 175, 14, 6, "p", "post-transition metal"),
    "Pm": ("Promethium", 61, 145.0, 1.13, 183, None, 6, "f", "lanthanide"),
    "Pr": ("Praseodymium", 59, 140.91, 1.13, 182, None, 6, "f", "lanthanide"),
    "Pu": ("Plutonium", 94, 244.0, 1.28, 159, None, 7, "f", "actinide"),
    "Sm": ("Samarium", 62, 150.36, 1.17, 180, None, 6, "f", "lanthanide"),
    "Tb": ("Terbium", 65, 158.93, 1.20, 177, None, 6, "f", "lanthanide"),
    "Tc": ("Technetium", 43, 98.0, 1.90, 136, 7, 5, "d", "transition metal"),
    "Th": ("Thorium", 90, 232.04, 1.30, 180, None, 7, "f", "actinide"),
    "Tl": ("Thallium", 81, 204.38, 1.62, 170, 13, 6, "p", "post-transition metal"),
    "Tm": ("Thulium", 69, 168.93, 1.25, 176, None, 6, "f", "lanthanide"),
    "U": ("Uranium", 92, 238.03, 1.38, 156, None, 7, "f", "actinide"),
    "Yb": ("Ytterbium", 70, 173.05, 1.10, 176, None, 6, "f", "lanthanide"),
}

SYNTHESIS_METHODS = [
    ("Arc Melting", "melt"),
    ("Ball Milling", "mechanical"),
    ("Spark Plasma Sintering", "sintering"),
    ("Induction Melting", "melt"),
    ("Magnetron Sputtering", "deposition"),
    ("Molecular Beam Epitaxy", "deposition"),
    ("Chemical Vapor Deposition", "deposition"),
    ("Sol-Gel", "chemical"),
    ("Electrodeposition", "chemical"),
    ("Powder Metallurgy", "sintering"),
]

DEFECT_TYPES = [
    ("vacancy", "point", "Missing atom at lattice site"),
    ("antisite", "point", "Atom on wrong sublattice"),
    ("interstitial", "point", "Extra atom in interstitial position"),
    ("edge_dislocation", "line", "Edge dislocation"),
    ("screw_dislocation", "line", "Screw dislocation"),
    ("stacking_fault", "planar", "Stacking fault"),
]

APPLICATION_DOMAINS = [
    ("High-temperature superalloy", "aerospace"),
    ("Shape memory alloy", "biomedical"),
    ("Hydrogen storage", "energy"),
    ("Catalysis", "chemical"),
    ("Thermoelectric", "energy"),
    ("Magnetic recording", "electronics"),
    ("Nuclear structural material", "nuclear"),
    ("Wear-resistant coating", "industrial"),
]

ALLOY_SYSTEMS = [
    ("Ni-Al", 2, "binary"),
    ("Co-Ti", 2, "binary"),
    ("Ni-Al-Ti", 3, "ternary"),
    ("Co-Ni-Al", 3, "ternary"),
    ("Ni-Co-Al-Ti", 4, "quaternary"),
    ("Fe-Ni-Co-Al-Ti", 5, "high-entropy"),
]

# Canonical property dictionary (property_definition rows).
# EAV tables (calculated_property, measured_property, element_property)
# carry a composite FK (property_name, unit) to this dictionary, so free-text
# property names or mismatched units are rejected by the DB.
PROPERTY_DEFINITIONS = [
    # (canonical_name, canonical_unit, applies_to, description)
    ("bulk_modulus", "GPa", "calculated", "Voigt-Reuss-Hill bulk modulus"),
    ("shear_modulus", "GPa", "calculated", "Voigt-Reuss-Hill shear modulus"),
    ("youngs_modulus", "GPa", "calculated", "Young's modulus"),
    ("hardness", "GPa", "measured", "Indentation hardness"),
    ("lattice_a", "A", "measured", "Measured lattice parameter a"),
    ("density", "g/cm3", "measured", "Mass density"),
    ("resistivity", "uOhm.cm", "measured", "Electrical resistivity"),
    ("melting_point", "K", "element", "Elemental melting point"),
    ("boiling_point", "K", "element", "Elemental boiling point"),
]

# Canonical unit and physically plausible value range per measured property
MEASURED_PROPERTY_SPECS = {
    "hardness": ("GPa", 0.5, 30.0),
    "lattice_a": ("A", 2.5, 7.0),
    "density": ("g/cm3", 2.0, 20.0),
    "resistivity": ("uOhm.cm", 1.0, 200.0),
}

SPACE_GROUPS = [
    # (number, hermann_mauguin, crystal_system, point_group, is_centrosymmetric)
    (194, "P6_3/mmc", "hexagonal", "6/mmm", True),
    (221, "Pm-3m", "cubic", "m-3m", True),
    (225, "Fm-3m", "cubic", "m-3m", True),
]


def _esc(s: str) -> str:
    """Escape single quotes for SQL."""
    return s.replace("'", "''")


def _sql_str(s: str | None) -> str:
    return "NULL" if s is None else f"'{_esc(str(s))}'"


def _sql_num(v: float | int | None) -> str:
    return "NULL" if v is None else str(v)


def _load_pure_elements() -> dict[str, dict]:
    with open(PROJECT / "db" / "pure_element_data.json") as f:
        return json.load(f)["ground_states"]


def write_reference_data(out: TextIO, pure: dict[str, dict]) -> dict[str, int]:
    """Write db/002_reference_data.sql content. Returns element symbol -> id."""
    out.write("-- Auto-generated reference/master data (002)\n")
    out.write("-- Load after 001_schema.sql\n")
    out.write("BEGIN;\n\n")

    # --- Elements: base 62 + 27 OQMD-only elements ---
    out.write("-- Elements\n")
    elem_ids: dict[str, int] = {}
    for i, (sym, (anum, mass, eneg, radius)) in enumerate(ELEMENT_DATA.items(), 1):
        elem_ids[sym] = i
        period, group, block = _element_period_group_block(anum)
        out.write(
            f"INSERT INTO element (element_id, symbol, name, atomic_number, atomic_mass, "
            f"electronegativity, atomic_radius, group_number, period_number, block) VALUES "
            f"({i}, '{sym}', '{sym}', {anum}, {mass}, {eneg}, {radius}, "
            f"{_sql_num(group)}, {period}, '{block}');\n"
        )
    next_id = len(ELEMENT_DATA)
    for sym, (name, anum, mass, eneg, radius, group, period, block, cat) in sorted(
        EXTRA_ELEMENT_DATA.items(), key=lambda x: x[1][1]
    ):
        next_id += 1
        elem_ids[sym] = next_id
        out.write(
            f"INSERT INTO element (element_id, symbol, name, atomic_number, atomic_mass, "
            f"electronegativity, atomic_radius, group_number, period_number, block, category) "
            f"VALUES ({next_id}, '{sym}', '{_esc(name)}', {anum}, {mass}, "
            f"{_sql_num(eneg)}, {_sql_num(radius)}, {_sql_num(group)}, {_sql_num(period)}, "
            f"{_sql_str(block)}, {_sql_str(cat)});\n"
        )
    out.write(
        f"SELECT setval('element_element_id_seq', {next_id});\n"
    )

    # --- Property dictionary ---
    out.write("\n-- Property definitions (canonical names & units)\n")
    for name, unit, applies_to, desc in PROPERTY_DEFINITIONS:
        out.write(
            f"INSERT INTO property_definition (canonical_name, canonical_unit, "
            f"value_type, applies_to, description) VALUES "
            f"('{name}', '{unit}', 'float', '{applies_to}', '{_esc(desc)}');\n"
        )

    # --- Element properties ---
    out.write("\n-- Element properties\n")
    ep_id = 0
    for sym in ELEMENT_DATA:
        eid = elem_ids[sym]
        anum = ELEMENT_DATA[sym][0]
        ep_id += 1
        out.write(
            f"INSERT INTO element_property (element_property_id, element_id, property_name, value, unit) "
            f"VALUES ({ep_id}, {eid}, 'melting_point', {800 + anum * 20 + random.uniform(-50, 50):.1f}, 'K');\n"
        )
        ep_id += 1
        out.write(
            f"INSERT INTO element_property (element_property_id, element_id, property_name, value, unit) "
            f"VALUES ({ep_id}, {eid}, 'boiling_point', {1600 + anum * 40 + random.uniform(-100, 100):.1f}, 'K');\n"
        )
    out.write(f"SELECT setval('element_property_element_property_id_seq', {ep_id});\n")

    # --- Prototype definitions: 5 compound prototypes + per-element ground states ---
    out.write("\n-- Prototype definitions\n")
    for pname, pinfo in PROTOTYPES.items():
        out.write(
            f"INSERT INTO prototype_definition (prototype_id, prototype_name, strukturbericht, "
            f"formula_type, description) VALUES "
            f"('{pname}', '{pname}', '{pinfo['strukturbericht']}', "
            f"'{pinfo['formula_type']}', '{pname} ordered intermetallic');\n"
        )
    for sym in sorted(pure):
        out.write(
            f"INSERT INTO prototype_definition (prototype_id, prototype_name, strukturbericht, "
            f"formula_type, description) VALUES "
            f"('{sym}_gs', '{sym}_gs', NULL, 'A', "
            f"'OQMD ground-state structure of {sym}');\n"
        )

    # --- Space groups ---
    out.write("\n-- Space groups\n")
    for sgn, hm, cs, pg, centro in SPACE_GROUPS:
        out.write(
            f"INSERT INTO space_group (space_group_number, hermann_mauguin, crystal_system, "
            f"point_group, is_centrosymmetric) VALUES "
            f"({sgn}, '{hm}', '{cs}', '{pg}', {'TRUE' if centro else 'FALSE'});\n"
        )

    # --- Synthesis methods ---
    out.write("\n-- Synthesis methods\n")
    for sid, (name, category) in enumerate(SYNTHESIS_METHODS, 1):
        out.write(
            f"INSERT INTO synthesis_method (synthesis_id, method_name, category, description) "
            f"VALUES ({sid}, '{name}', '{category}', '{name} synthesis');\n"
        )
    out.write(f"SELECT setval('synthesis_method_synthesis_id_seq', {len(SYNTHESIS_METHODS)});\n")

    # --- Defect types ---
    out.write("\n-- Defect types\n")
    for did, (name, category, desc) in enumerate(DEFECT_TYPES, 1):
        out.write(
            f"INSERT INTO defect_type (defect_type_id, defect_name, category, description) "
            f"VALUES ({did}, '{name}', '{category}', '{_esc(desc)}');\n"
        )
    out.write(f"SELECT setval('defect_type_defect_type_id_seq', {len(DEFECT_TYPES)});\n")

    # --- Application domains ---
    out.write("\n-- Application domains\n")
    for dom_id, (name, _sector) in enumerate(APPLICATION_DOMAINS, 1):
        out.write(
            f"INSERT INTO application_domain (domain_id, domain_name, description) "
            f"VALUES ({dom_id}, '{_esc(name)}', '{_esc(name)} applications');\n"
        )
    out.write(f"SELECT setval('application_domain_domain_id_seq', {len(APPLICATION_DOMAINS)});\n")

    # --- Alloy systems ---
    out.write("\n-- Alloy systems\n")
    for aid, (name, ncomp, cat) in enumerate(ALLOY_SYSTEMS, 1):
        out.write(
            f"INSERT INTO alloy_system (alloy_system_id, system_name, num_components, "
            f"category, description) VALUES "
            f"({aid}, '{name}', {ncomp}, '{cat}', '{name} system');\n"
        )
    out.write(f"SELECT setval('alloy_system_alloy_system_id_seq', {len(ALLOY_SYSTEMS)});\n")

    # --- Pure element reference energies (OQMD) ---
    out.write("\n-- Pure element reference energies (OQMD ground states)\n")
    for sym in sorted(pure):
        info = pure[sym]
        out.write(
            f"INSERT INTO pure_element_reference "
            f"(element_symbol, oqmd_entry_id, ground_state_spacegroup, "
            f"energy_per_atom, volume_per_atom, stability, band_gap, n_polymorphs) "
            f"VALUES ('{sym}', {info['oqmd_entry_id']}, {_sql_str(info['spacegroup'])}, "
            f"{_sql_num(info['delta_e_per_atom'])}, {_sql_num(info['volume_per_atom'])}, "
            f"{_sql_num(max(0.0, info['stability']) if info['stability'] is not None else None)}, "
            f"{_sql_num(info['band_gap'])}, {info['n_polymorphs']});\n"
        )

    out.write("\nCOMMIT;\n")
    return elem_ids


def write_material_data(out: TextIO, pure: dict[str, dict], elem_ids: dict[str, int]) -> int:
    """Write db/003_material_data.sql content. Returns compound entry count."""
    out.write("-- Auto-generated material data (003)\n")
    out.write("-- 1,470 compound entries + 89 OQMD pure-element entries\n")
    out.write("BEGIN;\n\n")

    synth_ids = {name: sid for sid, (name, _cat) in enumerate(SYNTHESIS_METHODS, 1)}
    defect_ids = {name: did for did, (name, _c, _d) in enumerate(DEFECT_TYPES, 1)}
    domain_ids = {name: dom for dom, (name, _s) in enumerate(APPLICATION_DOMAINS, 1)}
    alloy_ids = {name: aid for aid, (name, _n, _c) in enumerate(ALLOY_SYSTEMS, 1)}

    entry_count = 0
    calc_count = 0
    prop_count = 0
    ref_count = 0

    def _gen_entry(
        proto_key: str, a_elem: str, b_elem: str,
        lattice_a: float, fe: float, eah: float,
        bulk_mod: float, shear_mod: float,
        source: str = "OQMD",
    ) -> None:
        nonlocal entry_count, calc_count, prop_count, ref_count
        entry_count += 1
        eid = f"entry_{entry_count:05d}"
        pinfo = PROTOTYPES[proto_key]

        if pinfo["formula_type"] == "A3B":
            formula = f"{a_elem}3{b_elem}"
            a_frac, b_frac = 0.75, 0.25
        elif pinfo["formula_type"] == "AB":
            formula = f"{a_elem}{b_elem}"
            a_frac, b_frac = 0.50, 0.50
        elif pinfo["formula_type"] == "AB3":
            formula = f"{a_elem}{b_elem}3"
            a_frac, b_frac = 0.25, 0.75
        else:  # AB2
            formula = f"{a_elem}{b_elem}2"
            a_frac, b_frac = 0.333, 0.667

        chem_sys = "-".join(sorted([a_elem, b_elem]))
        volume = lattice_a**3 / (4 if "cubic" in pinfo["crystal_system"] else 2)

        # material_entry
        out.write(
            f"INSERT INTO material_entry (entry_id, source_db, source_material_id, "
            f"formula, reduced_formula, chemical_system, number_of_elements) VALUES "
            f"('{eid}', '{source}', '{source.lower()}_{entry_count}', "
            f"'{formula}', '{formula}', '{chem_sys}', 2);\n"
        )

        # composition (2 rows)
        out.write(
            f"INSERT INTO composition (composition_id, entry_id, element, atomic_fraction, site_label) VALUES "
            f"('comp_{entry_count:05d}_a', '{eid}', '{a_elem}', {a_frac}, 'A-site');\n"
        )
        out.write(
            f"INSERT INTO composition (composition_id, entry_id, element, atomic_fraction, site_label) VALUES "
            f"('comp_{entry_count:05d}_b', '{eid}', '{b_elem}', {b_frac}, 'B-site');\n"
        )

        # structure
        lat_b = lattice_a if pinfo["crystal_system"] == "cubic" else lattice_a * 0.95
        lat_c = lattice_a if pinfo["crystal_system"] == "cubic" else lattice_a * 1.63
        out.write(
            f"INSERT INTO structure (structure_id, entry_id, prototype, strukturbericht, "
            f"formula_type, space_group_number, crystal_system, lattice_a, lattice_b, lattice_c, "
            f"volume_per_atom, space_group) VALUES "
            f"('struct_{entry_count:05d}', '{eid}', '{proto_key}', '{pinfo['strukturbericht']}', "
            f"'{pinfo['formula_type']}', {pinfo['sg_number']}, '{pinfo['crystal_system']}', "
            f"{lattice_a:.4f}, {lat_b:.4f}, {lat_c:.4f}, {volume/4:.4f}, '{pinfo['sg_symbol']}');\n"
        )

        # phase_stability (is_stable is a generated column, never inserted)
        out.write(
            f"INSERT INTO phase_stability (stability_id, entry_id, formation_energy_per_atom, "
            f"energy_above_hull, band_gap) VALUES "
            f"('stab_{entry_count:05d}', '{eid}', {fe:.4f}, {eah:.4f}, "
            f"{random.uniform(0, 0.5):.3f});\n"
        )

        # calculation
        calc_count += 1
        cid = f"calc_{calc_count:05d}"
        out.write(
            f"INSERT INTO calculation (calculation_id, entry_id, method, functional, "
            f"calculation_type) VALUES "
            f"('{cid}', '{eid}', 'DFT', 'GGA-PBE', 'relaxation');\n"
        )

        # calculated_property (bulk_modulus, shear_modulus, youngs_modulus)
        for pname, pval, punit in [
            ("bulk_modulus", bulk_mod, "GPa"),
            ("shear_modulus", shear_mod, "GPa"),
            ("youngs_modulus", bulk_mod * 1.5 + random.uniform(-10, 10), "GPa"),
        ]:
            prop_count += 1
            out.write(
                f"INSERT INTO calculated_property (property_id, calculation_id, "
                f"property_name, value, unit) VALUES "
                f"('prop_{prop_count:05d}', '{cid}', '{pname}', {pval:.2f}, '{punit}');\n"
            )

        # elastic_tensor (50% of entries) — child of calculation
        if random.random() < 0.5:
            c44 = shear_mod * 0.9 + random.uniform(-5, 5)
            out.write(
                f"INSERT INTO elastic_tensor (calculation_id, "
                f"bulk_modulus_vrh, shear_modulus_vrh, is_stable) VALUES "
                f"('{cid}', "
                f"{bulk_mod:.1f}, {shear_mod:.1f}, {'TRUE' if c44 > 0 else 'FALSE'});\n"
            )

        # magnetic_property (30% of entries)
        if random.random() < 0.3:
            mag = random.uniform(0, 5.0)
            ordering = random.choice(["ferromagnetic", "antiferromagnetic", "paramagnetic"])
            out.write(
                f"INSERT INTO magnetic_property (entry_id, total_magnetization, "
                f"magnetic_ordering, curie_temperature_k) VALUES "
                f"('{eid}', {mag:.3f}, '{ordering}', "
                f"{random.uniform(200, 1400):.1f});\n"
            )

        # thermal_property (40% of entries) — child of calculation
        if random.random() < 0.4:
            out.write(
                f"INSERT INTO thermal_property (calculation_id, "
                f"debye_temperature_k, thermal_conductivity, specific_heat_cv, "
                f"gruneisen_parameter, temperature_k) VALUES "
                f"('{cid}', {random.uniform(200, 800):.1f}, "
                f"{random.uniform(5, 400):.1f}, {random.uniform(20, 50):.2f}, "
                f"{random.uniform(1.0, 3.0):.3f}, 300.0);\n"
            )

        # band_structure (25% of entries) — child of calculation
        if random.random() < 0.25:
            out.write(
                f"INSERT INTO band_structure (calculation_id, "
                f"is_direct_gap, cbm_energy, vbm_energy) VALUES "
                f"('{cid}', "
                f"{'TRUE' if random.random() > 0.5 else 'FALSE'}, "
                f"{random.uniform(0, 3):.3f}, {random.uniform(-5, -1):.3f});\n"
            )

        # density_of_states (25% of entries) — child of calculation
        if random.random() < 0.25:
            out.write(
                f"INSERT INTO density_of_states (calculation_id, "
                f"total_dos_at_fermi, spin_polarized) VALUES "
                f"('{cid}', {random.uniform(0, 50):.3f}, "
                f"{'TRUE' if random.random() > 0.5 else 'FALSE'});\n"
            )

        # surface_energy (20% of entries, multiple surfaces)
        if random.random() < 0.2:
            for miller in ["100", "110", "111"]:
                if random.random() < 0.7:
                    out.write(
                        f"INSERT INTO surface_energy (entry_id, miller_index, "
                        f"surface_energy_j_m2, work_function, is_reconstructed) VALUES "
                        f"('{eid}', '{miller}', {random.uniform(0.5, 4.0):.3f}, "
                        f"{random.uniform(3.5, 6.0):.3f}, "
                        f"{'TRUE' if random.random() < 0.2 else 'FALSE'});\n"
                    )

        # grain_boundary (10% of entries)
        if random.random() < 0.1:
            out.write(
                f"INSERT INTO grain_boundary (entry_id, sigma_value, rotation_axis, "
                f"tilt_angle, gb_energy_j_m2, excess_volume) VALUES "
                f"('{eid}', {random.choice([3, 5, 7, 9, 11])}, "
                f"'{random.choice(['001', '011', '111'])}', "
                f"{random.uniform(10, 90):.1f}, {random.uniform(0.3, 2.5):.3f}, "
                f"{random.uniform(0.01, 0.5):.4f});\n"
            )

        # material_synthesis (30% of entries)
        if random.random() < 0.3:
            method = random.choice(SYNTHESIS_METHODS)
            mid = synth_ids[method[0]]
            out.write(
                f"INSERT INTO material_synthesis (entry_id, synthesis_id, "
                f"temperature_k, duration_hours, success) VALUES "
                f"('{eid}', {mid}, {random.uniform(800, 2000):.0f}, "
                f"{random.uniform(1, 48):.1f}, "
                f"{'TRUE' if random.random() > 0.2 else 'FALSE'});\n"
            )

        # material_defect (15% of entries)
        if random.random() < 0.15:
            defect = random.choice(DEFECT_TYPES)
            did = defect_ids[defect[0]]
            out.write(
                f"INSERT INTO material_defect (entry_id, defect_type_id, "
                f"formation_energy, concentration, dopant_element_id) VALUES "
                f"('{eid}', {did}, {random.uniform(0.5, 5.0):.3f}, "
                f"{random.uniform(1e-6, 0.01):.6f}, "
                f"{elem_ids.get(random.choice(list(ELEMENT_DATA.keys())[:30]), 1)});\n"
            )

        # material_application (20% of entries)
        if random.random() < 0.2:
            domain = random.choice(APPLICATION_DOMAINS)
            dom_id = domain_ids[domain[0]]
            out.write(
                f"INSERT INTO material_application (entry_id, domain_id, "
                f"relevance_score, notes) VALUES "
                f"('{eid}', {dom_id}, {random.uniform(0.3, 1.0):.3f}, "
                f"'Candidate for {_esc(domain[0])}');\n"
            )

        # literature_reference + material_reference (25% of entries)
        if random.random() < 0.25:
            ref_count += 1
            year = random.randint(2000, 2025)
            doi = f"10.1000/test.{entry_count:05d}"
            out.write(
                f"INSERT INTO literature_reference (reference_id, doi, title, "
                f"authors, journal, year) VALUES "
                f"({ref_count}, '{doi}', 'Study of {formula} properties', "
                f"'Author et al.', 'J. Mater. Sci.', {year});\n"
            )
            out.write(
                f"INSERT INTO material_reference (entry_id, reference_id) VALUES "
                f"('{eid}', {ref_count});\n"
            )

        # phase_diagram_entry (40% of entries)
        if random.random() < 0.4:
            out.write(
                f"INSERT INTO phase_diagram_entry (entry_id, chemical_system, "
                f"is_on_hull, decomposition_products, hull_distance) VALUES "
                f"('{eid}', '{chem_sys}', {'TRUE' if eah <= STABLE_EAH_THRESHOLD else 'FALSE'}, "
                f"'{formula} -> {a_elem} + {b_elem}', {eah:.4f});\n"
            )

        # material_alloy_system (35% of entries)
        if random.random() < 0.35:
            matching = [k for k in alloy_ids if a_elem in k or b_elem in k]
            if matching:
                asys = random.choice(matching)
                out.write(
                    f"INSERT INTO material_alloy_system (entry_id, alloy_system_id, "
                    f"phase, composition_type) VALUES "
                    f"('{eid}', {alloy_ids[asys]}, '{proto_key}', 'stoichiometric');\n"
                )

    # Generate entries per prototype
    # L12: 392 entries (known + generated) — paper Table 3
    out.write("\n-- L12 entries (known + generated)\n")
    for a, b, lat, fe, eah, bulk, shear in KNOWN_L12:
        _gen_entry("L12", a, b, lat, fe, eah, bulk, shear)

    for _ in range(381):
        a = random.choice(A_ELEMENTS)
        b = random.choice(B_ELEMENTS)
        while b == a:
            b = random.choice(B_ELEMENTS)
        lat = random.uniform(3.4, 4.2)
        fe = random.uniform(-0.6, 0.1)
        eah = abs(random.gauss(0.05, 0.04))
        bulk = random.uniform(100, 250)
        shear = random.uniform(40, 120)
        _gen_entry("L12", a, b, lat, fe, eah, bulk, shear)

    # B2: ~636 entries — paper Table 3
    out.write("\n-- B2 entries\n")
    for _ in range(636):
        a = random.choice(A_ELEMENTS)
        b = random.choice(B_ELEMENTS)
        while b == a:
            b = random.choice(B_ELEMENTS)
        lat = random.uniform(2.8, 3.5)
        fe = random.uniform(-0.5, 0.2)
        eah = abs(random.gauss(0.06, 0.05))
        bulk = random.uniform(80, 220)
        shear = random.uniform(30, 100)
        _gen_entry("B2", a, b, lat, fe, eah, bulk, shear, "Materials Project")

    # NaCl: ~355 entries — paper Table 3
    out.write("\n-- NaCl entries\n")
    for _ in range(355):
        a = random.choice(A_ELEMENTS[:15])
        b = random.choice(B_ELEMENTS[:12])
        while b == a:
            b = random.choice(B_ELEMENTS[:12])
        lat = random.uniform(4.5, 5.5)
        fe = random.uniform(-0.4, 0.1)
        eah = abs(random.gauss(0.07, 0.05))
        bulk = random.uniform(120, 280)
        shear = random.uniform(50, 130)
        _gen_entry("NaCl", a, b, lat, fe, eah, bulk, shear, "AFLOW")

    # NiAs: ~74 entries — paper Table 3
    out.write("\n-- NiAs entries\n")
    for _ in range(74):
        a = random.choice(A_ELEMENTS[:12])
        b = random.choice(B_ELEMENTS[:10])
        while b == a:
            b = random.choice(B_ELEMENTS[:10])
        lat = random.uniform(5.0, 5.8)
        fe = random.uniform(-0.45, 0.15)
        eah = abs(random.gauss(0.06, 0.04))
        bulk = random.uniform(90, 200)
        shear = random.uniform(35, 95)
        _gen_entry("NiAs", a, b, lat, fe, eah, bulk, shear, "OQMD")

    # BiF3: ~13 entries — paper Table 3
    out.write("\n-- BiF3 entries\n")
    for _ in range(13):
        a = random.choice(A_ELEMENTS[:14])
        b = random.choice(B_ELEMENTS[:14])
        while b == a:
            b = random.choice(B_ELEMENTS[:14])
        lat = random.uniform(6.5, 7.8)
        fe = random.uniform(-0.35, 0.2)
        eah = abs(random.gauss(0.08, 0.06))
        bulk = random.uniform(70, 180)
        shear = random.uniform(25, 85)
        _gen_entry("BiF3", a, b, lat, fe, eah, bulk, shear, "Materials Project")

    compound_count = entry_count

    # --- Pure element ground-state entries (OQMD) ---
    out.write("\n-- Pure element ground-state entries (OQMD)\n")
    for sym in sorted(pure):
        info = pure[sym]
        peid = f"elem_{sym.lower()}_{info['oqmd_entry_id']}"
        eah = max(0.0, info["stability"]) if info["stability"] is not None else None
        out.write(
            f"INSERT INTO material_entry (entry_id, source_db, source_material_id, "
            f"formula, reduced_formula, chemical_system, number_of_elements) VALUES "
            f"('{peid}', 'OQMD', 'oqmd_{info['oqmd_entry_id']}', "
            f"'{sym}', '{sym}', '{sym}', 1);\n"
        )
        out.write(
            f"INSERT INTO composition (composition_id, entry_id, element, atomic_fraction) VALUES "
            f"('comp_{peid}', '{peid}', '{sym}', 1.0);\n"
        )
        out.write(
            f"INSERT INTO structure (structure_id, entry_id, prototype, strukturbericht, "
            f"space_group, volume_per_atom) VALUES "
            f"('struct_elem_{sym.lower()}', '{peid}', '{sym}_gs', NULL, "
            f"{_sql_str(info['spacegroup'])}, {_sql_num(info['volume_per_atom'])});\n"
        )
        out.write(
            f"INSERT INTO phase_stability (stability_id, entry_id, formation_energy_per_atom, "
            f"energy_above_hull, band_gap) VALUES "
            f"('stab_elem_{sym.lower()}', '{peid}', {_sql_num(info['delta_e_per_atom'])}, "
            f"{_sql_num(eah)}, {_sql_num(info['band_gap'])});\n"
        )
        out.write(
            f"INSERT INTO calculation (calculation_id, entry_id, method, functional, "
            f"calculation_type) VALUES "
            f"('calc_elem_{sym.lower()}', '{peid}', 'DFT', 'PBE', 'ground_state');\n"
        )

    # --- Experimental measurements (100 random compound entries) ---
    out.write("\n-- Experimental measurements\n")
    for i in range(1, 101):
        eid = f"entry_{random.randint(1, compound_count):05d}"
        prop = random.choice(sorted(MEASURED_PROPERTY_SPECS))
        unit, lo, hi = MEASURED_PROPERTY_SPECS[prop]
        out.write(
            f"INSERT INTO experimental_measurement (measurement_id, entry_id, "
            f"method, temperature_k) VALUES "
            f"({i}, '{eid}', "
            f"'{random.choice(['XRD', 'TEM', 'SEM', 'DSC', 'Nanoindentation'])}', "
            f"{random.uniform(77, 1200):.1f});\n"
        )
        out.write(
            f"INSERT INTO measured_property (measurement_id, property_name, value, unit, uncertainty) VALUES "
            f"({i}, '{prop}', {random.uniform(lo, hi):.3f}, "
            f"'{unit}', {random.uniform(0.01, hi * 0.02):.3f});\n"
        )
    out.write("SELECT setval('experimental_measurement_measurement_id_seq', 100);\n")
    out.write(f"SELECT setval('literature_reference_reference_id_seq', {ref_count});\n")

    out.write("\nCOMMIT;\n")
    out.write(f"-- Total compound entries: {compound_count}\n")
    return compound_count


def generate() -> None:
    """Generate 002_reference_data.sql and 003_material_data.sql."""
    pure = _load_pure_elements()
    ref_path = PROJECT / "db" / "002_reference_data.sql"
    mat_path = PROJECT / "db" / "003_material_data.sql"
    with open(ref_path, "w") as f:
        elem_ids = write_reference_data(f, pure)
    with open(mat_path, "w") as f:
        n = write_material_data(f, pure, elem_ids)
    print(f"-- Generated {n} compound entries + {len(pure)} pure-element entries",
          file=sys.stderr)
    print(f"-- Wrote {ref_path} and {mat_path}", file=sys.stderr)


if __name__ == "__main__":
    generate()
