#!/usr/bin/env python3
"""
Generate SQL INSERT statements for pure element ground-state data from OQMD.
Creates entries in:
  - element (for elements not already in DB)
  - pure_element_reference (ground-state energies)
  - material_entry (pure element entries)
  - composition (single element, fraction=1.0)
  - structure (ground-state crystal structure)
  - phase_stability (delta_e, stability)
  - calculation (DFT source)

Output: db/insert_pure_elements.sql
"""
import json

# Basic element data for those missing from the element table
ELEMENT_DATA = {
    'Ac': ('Actinium', 89, 227.0, 1.10, 195, 3, 7, 'f', 'actinide'),
    'Bi': ('Bismuth', 83, 208.98, 2.02, 156, 15, 6, 'p', 'post-transition metal'),
    'Cd': ('Cadmium', 48, 112.41, 1.69, 151, 12, 5, 'd', 'transition metal'),
    'Ce': ('Cerium', 58, 140.12, 1.12, 182, None, 6, 'f', 'lanthanide'),
    'Dy': ('Dysprosium', 66, 162.50, 1.22, 178, None, 6, 'f', 'lanthanide'),
    'Er': ('Erbium', 68, 167.26, 1.24, 176, None, 6, 'f', 'lanthanide'),
    'Eu': ('Europium', 63, 151.96, 1.20, 180, None, 6, 'f', 'lanthanide'),
    'Gd': ('Gadolinium', 64, 157.25, 1.20, 180, None, 6, 'f', 'lanthanide'),
    'Hg': ('Mercury', 80, 200.59, 2.00, 151, 12, 6, 'd', 'transition metal'),
    'Ho': ('Holmium', 67, 164.93, 1.23, 177, None, 6, 'f', 'lanthanide'),
    'La': ('Lanthanum', 57, 138.91, 1.10, 187, 3, 6, 'f', 'lanthanide'),
    'Lu': ('Lutetium', 71, 174.97, 1.27, 174, 3, 6, 'f', 'lanthanide'),
    'Nd': ('Neodymium', 60, 144.24, 1.14, 181, None, 6, 'f', 'lanthanide'),
    'Np': ('Neptunium', 93, 237.0, 1.36, 155, None, 7, 'f', 'actinide'),
    'Pa': ('Protactinium', 91, 231.04, 1.50, 163, None, 7, 'f', 'actinide'),
    'Pb': ('Lead', 82, 207.2, 2.33, 175, 14, 6, 'p', 'post-transition metal'),
    'Pm': ('Promethium', 61, 145.0, 1.13, 183, None, 6, 'f', 'lanthanide'),
    'Pr': ('Praseodymium', 59, 140.91, 1.13, 182, None, 6, 'f', 'lanthanide'),
    'Pu': ('Plutonium', 94, 244.0, 1.28, 159, None, 7, 'f', 'actinide'),
    'Sm': ('Samarium', 62, 150.36, 1.17, 180, None, 6, 'f', 'lanthanide'),
    'Tb': ('Terbium', 65, 158.93, 1.20, 177, None, 6, 'f', 'lanthanide'),
    'Tc': ('Technetium', 43, 98.0, 1.90, 136, 7, 5, 'd', 'transition metal'),
    'Th': ('Thorium', 90, 232.04, 1.30, 180, None, 7, 'f', 'actinide'),
    'Tl': ('Thallium', 81, 204.38, 1.62, 170, 13, 6, 'p', 'post-transition metal'),
    'Tm': ('Thulium', 69, 168.93, 1.25, 176, None, 6, 'f', 'lanthanide'),
    'U': ('Uranium', 92, 238.03, 1.38, 156, None, 7, 'f', 'actinide'),
    'Yb': ('Ytterbium', 70, 173.05, 1.10, 176, None, 6, 'f', 'lanthanide'),
}


def escape_sql(s):
    if s is None:
        return "NULL"
    return "'" + str(s).replace("'", "''") + "'"


def num_or_null(v):
    if v is None:
        return "NULL"
    return str(v)


def main():
    with open("db/pure_element_data.json") as f:
        data = json.load(f)

    gs = data["ground_states"]
    lines = []
    lines.append("-- Pure element ground-state data from OQMD (auto-generated)")
    lines.append(f"-- {data['_meta']['unique_elements']} elements, "
                 f"{data['_meta']['total_entries_fetched']} polymorphs total")
    lines.append(f"-- Generated: {data['_meta']['generated_at']}")
    lines.append("BEGIN;")
    lines.append("")

    # Add missing elements to element table
    lines.append("-- Missing elements (not in original 62-element insert_data.sql)")
    existing_max_id = 62
    for i, (sym, (name, z, mass, en, radius, group, period, block, cat)) in enumerate(
        sorted(ELEMENT_DATA.items(), key=lambda x: x[1][1])  # sort by atomic_number
    ):
        eid = existing_max_id + i + 1
        lines.append(
            f"INSERT INTO element (element_id, symbol, name, atomic_number, "
            f"atomic_mass, electronegativity, atomic_radius, "
            f"group_number, period_number, block, category) "
            f"VALUES ({eid}, {escape_sql(sym)}, {escape_sql(name)}, {z}, "
            f"{mass}, {num_or_null(en)}, {num_or_null(radius)}, "
            f"{num_or_null(group)}, {num_or_null(period)}, "
            f"{escape_sql(block)}, {escape_sql(cat)});"
        )
    lines.append("")

    # Create pure_element_reference table
    lines.append("-- Reference energies for formation enthalpy calculation")
    lines.append("CREATE TABLE IF NOT EXISTS pure_element_reference (")
    lines.append("    pure_ref_id SERIAL PRIMARY KEY,")
    lines.append("    element_symbol VARCHAR(5) NOT NULL UNIQUE "
                 "REFERENCES element(symbol),")
    lines.append("    oqmd_entry_id INTEGER,")
    lines.append("    ground_state_spacegroup VARCHAR(30),")
    lines.append("    energy_per_atom DOUBLE PRECISION,  "
                 "-- eV/atom (delta_e from OQMD)")
    lines.append("    volume_per_atom DOUBLE PRECISION,  -- Angstrom^3/atom")
    lines.append("    stability DOUBLE PRECISION,  -- eV/atom above hull")
    lines.append("    band_gap DOUBLE PRECISION,  -- eV")
    lines.append("    n_polymorphs INTEGER,")
    lines.append("    source TEXT DEFAULT 'OQMD'")
    lines.append(");")
    lines.append("")
    lines.append("CREATE INDEX IF NOT EXISTS idx_pure_ref_symbol "
                 "ON pure_element_reference(element_symbol);")
    lines.append("")

    # Insert data for each element
    for elem, info in sorted(gs.items()):
        entry_id = f"elem_{elem.lower()}_{info['oqmd_entry_id']}"
        calc_id = f"calc_elem_{elem.lower()}"
        struct_id = f"struct_elem_{elem.lower()}"
        stab_id = f"stab_elem_{elem.lower()}"
        src_id = f"oqmd_{info['oqmd_entry_id']}"

        delta_e = info["delta_e_per_atom"]
        vol = info["volume_per_atom"]
        stability = info["stability"]
        band_gap = info["band_gap"]
        sg = info["spacegroup"]
        n_poly = info["n_polymorphs"]

        # pure_element_reference
        lines.append(
            f"INSERT INTO pure_element_reference "
            f"(element_symbol, oqmd_entry_id, ground_state_spacegroup, "
            f"energy_per_atom, volume_per_atom, stability, band_gap, n_polymorphs) "
            f"VALUES ({escape_sql(elem)}, {info['oqmd_entry_id']}, {escape_sql(sg)}, "
            f"{num_or_null(delta_e)}, {num_or_null(vol)}, "
            f"{num_or_null(stability)}, {num_or_null(band_gap)}, {n_poly});"
        )

        # material_entry
        lines.append(
            f"INSERT INTO material_entry "
            f"(entry_id, source_db, source_material_id, formula, reduced_formula, "
            f"chemical_system, number_of_elements) "
            f"VALUES ({escape_sql(entry_id)}, 'OQMD', {escape_sql(src_id)}, "
            f"{escape_sql(elem)}, {escape_sql(elem)}, {escape_sql(elem)}, 1);"
        )

        # composition
        lines.append(
            f"INSERT INTO composition "
            f"(composition_id, entry_id, element, atomic_fraction) "
            f"VALUES ({escape_sql(f'comp_{entry_id}')}, "
            f"{escape_sql(entry_id)}, {escape_sql(elem)}, 1.0);"
        )

        # structure
        lines.append(
            f"INSERT INTO structure "
            f"(structure_id, entry_id, prototype, strukturbericht, "
            f"space_group, volume_per_atom) "
            f"VALUES ({escape_sql(struct_id)}, {escape_sql(entry_id)}, "
            f"{escape_sql(f'{elem}_gs')}, NULL, "
            f"{escape_sql(sg)}, {num_or_null(vol)});"
        )

        # phase_stability
        is_stable = "true" if stability is not None and stability < 0.001 else "false"
        lines.append(
            f"INSERT INTO phase_stability "
            f"(stability_id, entry_id, formation_energy_per_atom, "
            f"energy_above_hull, is_stable, band_gap) "
            f"VALUES ({escape_sql(stab_id)}, {escape_sql(entry_id)}, "
            f"{num_or_null(delta_e)}, {num_or_null(stability)}, "
            f"{is_stable}, {num_or_null(band_gap)});"
        )

        # calculation
        lines.append(
            f"INSERT INTO calculation "
            f"(calculation_id, entry_id, method, functional, calculation_type) "
            f"VALUES ({escape_sql(calc_id)}, {escape_sql(entry_id)}, "
            f"'DFT', 'PBE', 'ground_state');"
        )
        lines.append("")

    lines.append("COMMIT;")

    outpath = "db/insert_pure_elements.sql"
    with open(outpath, "w") as f:
        f.write("\n".join(lines))

    n_inserts = sum(1 for line in lines if line.startswith("INSERT"))
    print(f"Wrote {outpath} ({len(gs)} elements, {n_inserts} INSERTs, "
          f"{len(ELEMENT_DATA)} new element rows)")


if __name__ == "__main__":
    main()
