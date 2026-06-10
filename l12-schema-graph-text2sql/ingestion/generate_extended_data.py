#!/usr/bin/env python3
"""Generate data for the 30-table extended schema.

Produces ~1,471 material entries across 5 prototypes (L12, B2, NaCl, NiAs, BiF3)
with full coverage of all 30 tables. Output: a single SQL file that can be
loaded after the schema (extended_schema.sql) is applied.

Usage:
    python ingestion/generate_extended_data.py > db/insert_data.sql
    # or pipe directly:
    docker exec -i l12_postgres psql -U l12_user l12_materials < db/insert_data.sql
"""
from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field
from typing import TextIO

random.seed(42)

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

# Known stable L12 compounds
KNOWN_L12 = [
    ("Ni", "Al", 3.572, -0.420, 0.000, True, 180.0, 85.0),
    ("Ni", "Ga", 3.660, -0.380, 0.010, True, 165.0, 74.0),
    ("Ni", "Ge", 3.610, -0.350, 0.040, False, 175.0, 78.0),
    ("Co", "Ti", 3.550, -0.450, 0.000, True, 190.0, 90.0),
    ("Al", "Sc", 4.090, -0.500, 0.000, True, 155.0, 65.0),
    ("Al", "Ti", 3.980, -0.410, 0.020, True, 160.0, 70.0),
    ("Pt", "Al", 3.900, -0.550, 0.000, True, 210.0, 95.0),
    ("Ir", "Nb", 3.870, -0.480, 0.030, False, 220.0, 100.0),
    ("Co", "Al", 3.670, -0.360, 0.010, True, 180.0, 82.0),
    ("Co", "W",  3.740, -0.330, 0.050, False, 200.0, 88.0),
    ("Co", "Ta", 3.720, -0.340, 0.040, False, 198.0, 86.0),
]

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


def _esc(s: str) -> str:
    """Escape single quotes for SQL."""
    return s.replace("'", "''")


def generate(out: TextIO = sys.stdout) -> None:
    """Generate full INSERT statements for 30-table schema."""
    out.write("-- Auto-generated data for 30-table extended schema\n")
    out.write("-- ~1,471 material entries across 5 prototypes (L12, B2, NaCl, NiAs, BiF3)\n")
    out.write("BEGIN;\n\n")

    # --- 1. Element table ---
    out.write("-- Elements\n")
    elem_ids: dict[str, int] = {}
    for i, (sym, (anum, mass, eneg, radius)) in enumerate(ELEMENT_DATA.items(), 1):
        elem_ids[sym] = i
        out.write(
            f"INSERT INTO element (element_id, symbol, name, atomic_number, atomic_mass, "
            f"electronegativity, atomic_radius) VALUES "
            f"({i}, '{sym}', '{sym}', {anum}, {mass}, {eneg}, {radius});\n"
        )

    # --- 2. Element properties ---
    out.write("\n-- Element properties\n")
    ep_id = 0
    for sym, eid in elem_ids.items():
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

    # --- 3. Prototype definitions ---
    out.write("\n-- Prototype definitions\n")
    for pid, (pname, pinfo) in enumerate(PROTOTYPES.items(), 1):
        out.write(
            f"INSERT INTO prototype_definition (prototype_id, prototype_name, strukturbericht, "
            f"formula_type, description) VALUES "
            f"('{pname}', '{pname}', '{pinfo['strukturbericht']}', "
            f"'{pinfo['formula_type']}', '{pname} ordered intermetallic');\n"
        )

    # --- 4. Space groups ---
    out.write("\n-- Space groups\n")
    sg_done: set[int] = set()
    for pinfo in PROTOTYPES.values():
        sgn = pinfo["sg_number"]
        if sgn not in sg_done:
            sg_done.add(sgn)
            out.write(
                f"INSERT INTO space_group (space_group_number, hermann_mauguin, crystal_system, "
                f"point_group, is_centrosymmetric) VALUES "
                f"({sgn}, '{pinfo['sg_symbol']}', '{pinfo['crystal_system']}', "
                f"'m-3m', TRUE);\n"
            )

    # --- 5. Synthesis methods ---
    out.write("\n-- Synthesis methods\n")
    synth_ids: dict[str, int] = {}
    for sid, (name, category) in enumerate(SYNTHESIS_METHODS, 1):
        synth_ids[name] = sid
        out.write(
            f"INSERT INTO synthesis_method (synthesis_id, method_name, category, description) "
            f"VALUES ({sid}, '{name}', '{category}', '{name} synthesis');\n"
        )

    # --- 6. Defect types ---
    out.write("\n-- Defect types\n")
    defect_ids: dict[str, int] = {}
    for did, (name, category, desc) in enumerate(DEFECT_TYPES, 1):
        defect_ids[name] = did
        out.write(
            f"INSERT INTO defect_type (defect_type_id, defect_name, category, description) "
            f"VALUES ({did}, '{name}', '{category}', '{_esc(desc)}');\n"
        )

    # --- 7. Application domains ---
    out.write("\n-- Application domains\n")
    domain_ids: dict[str, int] = {}
    for dom_id, (name, sector) in enumerate(APPLICATION_DOMAINS, 1):
        domain_ids[name] = dom_id
        out.write(
            f"INSERT INTO application_domain (domain_id, domain_name, description) "
            f"VALUES ({dom_id}, '{_esc(name)}', '{_esc(name)} applications');\n"
        )

    # --- 8. Alloy systems ---
    out.write("\n-- Alloy systems\n")
    alloy_ids: dict[str, int] = {}
    for aid, (name, ncomp, cat) in enumerate(ALLOY_SYSTEMS, 1):
        alloy_ids[name] = aid
        out.write(
            f"INSERT INTO alloy_system (alloy_system_id, system_name, num_components, "
            f"category, description) VALUES "
            f"({aid}, '{name}', {ncomp}, '{cat}', '{name} system');\n"
        )

    # --- 9. Generate material entries ---
    out.write("\n-- Material entries + dependent tables\n")
    entry_count = 0
    calc_count = 0
    prop_count = 0
    ref_count = 0

    def _gen_entry(
        proto_key: str, a_elem: str, b_elem: str,
        lattice_a: float, fe: float, eah: float, is_stable: bool,
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

        # phase_stability
        out.write(
            f"INSERT INTO phase_stability (stability_id, entry_id, formation_energy_per_atom, "
            f"energy_above_hull, is_stable, band_gap) VALUES "
            f"('stab_{entry_count:05d}', '{eid}', {fe:.4f}, {eah:.4f}, "
            f"{'TRUE' if is_stable else 'FALSE'}, {random.uniform(0, 0.5):.3f});\n"
        )

        # calculation
        calc_count += 1
        cid = f"calc_{calc_count:05d}"
        out.write(
            f"INSERT INTO calculation (calculation_id, entry_id, method, functional, "
            f"calculation_type) VALUES "
            f"('{cid}', '{eid}', 'DFT', 'GGA-PBE', 'relaxation');\n"
        )

        # calculated_property (bulk_modulus, shear_modulus)
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

        # elastic_tensor (50% of entries)
        if random.random() < 0.5:
            c44 = shear_mod * 0.9 + random.uniform(-5, 5)
            out.write(
                f"INSERT INTO elastic_tensor (entry_id, "
                f"bulk_modulus_vrh, shear_modulus_vrh, is_stable) VALUES "
                f"('{eid}', "
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

        # thermal_property (40% of entries)
        if random.random() < 0.4:
            out.write(
                f"INSERT INTO thermal_property (entry_id, calculation_id, "
                f"debye_temperature_k, thermal_conductivity, specific_heat_cv, "
                f"gruneisen_parameter, temperature_k) VALUES "
                f"('{eid}', '{cid}', {random.uniform(200, 800):.1f}, "
                f"{random.uniform(5, 400):.1f}, {random.uniform(20, 50):.2f}, "
                f"{random.uniform(1.0, 3.0):.3f}, 300.0);\n"
            )

        # band_structure (25% of entries)
        if random.random() < 0.25:
            out.write(
                f"INSERT INTO band_structure (entry_id, calculation_id, "
                f"is_direct_gap, cbm_energy, vbm_energy) VALUES "
                f"('{eid}', '{cid}', "
                f"{'TRUE' if random.random() > 0.5 else 'FALSE'}, "
                f"{random.uniform(0, 3):.3f}, {random.uniform(-5, -1):.3f});\n"
            )

        # density_of_states (25% of entries)
        if random.random() < 0.25:
            out.write(
                f"INSERT INTO density_of_states (entry_id, calculation_id, "
                f"total_dos_at_fermi, spin_polarized) VALUES "
                f"('{eid}', '{cid}', {random.uniform(0, 50):.3f}, "
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
                f"('{eid}', '{chem_sys}', {'TRUE' if eah < 0.01 else 'FALSE'}, "
                f"'{formula} -> {a_elem} + {b_elem}', {eah:.4f});\n"
            )

        # material_alloy_system (35% of entries)
        if random.random() < 0.35:
            # Find matching alloy system
            matching = [k for k in alloy_ids if a_elem in k or b_elem in k]
            if matching:
                asys = random.choice(matching)
                out.write(
                    f"INSERT INTO material_alloy_system (entry_id, alloy_system_id, "
                    f"phase, composition_type) VALUES "
                    f"('{eid}', {alloy_ids[asys]}, '{proto_key}', 'stoichiometric');\n"
                )

    # Generate entries per prototype
    # L12: ~393 entries (known + generated) — paper Table 3
    out.write("\n-- L12 entries (known + generated)\n")
    for a, b, lat, fe, eah, stab, bulk, shear in KNOWN_L12:
        _gen_entry("L12", a, b, lat, fe, eah, stab, bulk, shear)

    for _ in range(381):
        a = random.choice(A_ELEMENTS)
        b = random.choice(B_ELEMENTS)
        while b == a:
            b = random.choice(B_ELEMENTS)
        lat = random.uniform(3.4, 4.2)
        fe = random.uniform(-0.6, 0.1)
        eah = abs(random.gauss(0.05, 0.04))
        stab = eah < 0.01
        bulk = random.uniform(100, 250)
        shear = random.uniform(40, 120)
        _gen_entry("L12", a, b, lat, fe, eah, stab, bulk, shear)

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
        stab = eah < 0.01
        bulk = random.uniform(80, 220)
        shear = random.uniform(30, 100)
        _gen_entry("B2", a, b, lat, fe, eah, stab, bulk, shear, "Materials Project")

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
        stab = eah < 0.01
        bulk = random.uniform(120, 280)
        shear = random.uniform(50, 130)
        _gen_entry("NaCl", a, b, lat, fe, eah, stab, bulk, shear, "AFLOW")

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
        stab = eah < 0.01
        bulk = random.uniform(90, 200)
        shear = random.uniform(35, 95)
        _gen_entry("NiAs", a, b, lat, fe, eah, stab, bulk, shear, "OQMD")

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
        stab = eah < 0.01
        bulk = random.uniform(70, 180)
        shear = random.uniform(25, 85)
        _gen_entry("BiF3", a, b, lat, fe, eah, stab, bulk, shear, "Materials Project")

    # Experimental measurements (100 random entries)
    out.write("\n-- Experimental measurements\n")
    for i in range(1, 101):
        eid = f"entry_{random.randint(1, entry_count):05d}"
        out.write(
            f"INSERT INTO experimental_measurement (measurement_id, entry_id, "
            f"method, temperature_k) VALUES "
            f"({i}, '{eid}', "
            f"'{random.choice(['XRD', 'TEM', 'SEM', 'DSC', 'Nanoindentation'])}', "
            f"{random.uniform(77, 1200):.1f});\n"
        )
        out.write(
            f"INSERT INTO measured_property (measurement_id, property_name, value, unit, uncertainty) VALUES "
            f"({i}, '{random.choice(['hardness', 'lattice_a', 'density', 'resistivity'])}', "
            f"{random.uniform(1, 500):.3f}, '{random.choice(['GPa', 'A', 'g/cm3', 'uOhm.cm'])}', "
            f"{random.uniform(0.01, 5):.3f});\n"
        )

    out.write(f"\nCOMMIT;\n")
    out.write(f"-- Total entries: {entry_count}\n")
    print(f"-- Generated {entry_count} material entries", file=sys.stderr)


if __name__ == "__main__":
    generate()
