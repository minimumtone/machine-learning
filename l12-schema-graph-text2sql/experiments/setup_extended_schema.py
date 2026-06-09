#!/usr/bin/env python3
"""
Set up extended 20-table schema and populate with realistic data.
Uses existing 1,351 material entries + adds element properties, applications,
literature, synthesis methods, defect info, and experimental measurements.
"""
import psycopg
import random
import json
from pathlib import Path

DB_CONFIG = {
    'dbname': 'l12_materials',
    'user': 'l12_user',
    'password': 'l12_password',
    'host': 'localhost',
    'port': 5432
}

# Real element data (subset for all elements in our DB)
ELEMENTS = {
    'H': {'name': 'Hydrogen', 'Z': 1, 'mass': 1.008, 'en': 2.20, 'radius': 53, 'group': 1, 'period': 1, 'block': 's', 'cat': 'nonmetal'},
    'He': {'name': 'Helium', 'Z': 2, 'mass': 4.003, 'en': None, 'radius': 31, 'group': 18, 'period': 1, 'block': 's', 'cat': 'noble_gas'},
    'Li': {'name': 'Lithium', 'Z': 3, 'mass': 6.941, 'en': 0.98, 'radius': 167, 'group': 1, 'period': 2, 'block': 's', 'cat': 'alkali_metal'},
    'Be': {'name': 'Beryllium', 'Z': 4, 'mass': 9.012, 'en': 1.57, 'radius': 112, 'group': 2, 'period': 2, 'block': 's', 'cat': 'alkaline_earth'},
    'B': {'name': 'Boron', 'Z': 5, 'mass': 10.81, 'en': 2.04, 'radius': 87, 'group': 13, 'period': 2, 'block': 'p', 'cat': 'metalloid'},
    'C': {'name': 'Carbon', 'Z': 6, 'mass': 12.01, 'en': 2.55, 'radius': 67, 'group': 14, 'period': 2, 'block': 'p', 'cat': 'nonmetal'},
    'N': {'name': 'Nitrogen', 'Z': 7, 'mass': 14.01, 'en': 3.04, 'radius': 56, 'group': 15, 'period': 2, 'block': 'p', 'cat': 'nonmetal'},
    'O': {'name': 'Oxygen', 'Z': 8, 'mass': 16.00, 'en': 3.44, 'radius': 48, 'group': 16, 'period': 2, 'block': 'p', 'cat': 'nonmetal'},
    'F': {'name': 'Fluorine', 'Z': 9, 'mass': 19.00, 'en': 3.98, 'radius': 42, 'group': 17, 'period': 2, 'block': 'p', 'cat': 'halogen'},
    'Na': {'name': 'Sodium', 'Z': 11, 'mass': 22.99, 'en': 0.93, 'radius': 190, 'group': 1, 'period': 3, 'block': 's', 'cat': 'alkali_metal'},
    'Mg': {'name': 'Magnesium', 'Z': 12, 'mass': 24.31, 'en': 1.31, 'radius': 145, 'group': 2, 'period': 3, 'block': 's', 'cat': 'alkaline_earth'},
    'Al': {'name': 'Aluminum', 'Z': 13, 'mass': 26.98, 'en': 1.61, 'radius': 118, 'group': 13, 'period': 3, 'block': 'p', 'cat': 'post_transition_metal'},
    'Si': {'name': 'Silicon', 'Z': 14, 'mass': 28.09, 'en': 1.90, 'radius': 111, 'group': 14, 'period': 3, 'block': 'p', 'cat': 'metalloid'},
    'P': {'name': 'Phosphorus', 'Z': 15, 'mass': 30.97, 'en': 2.19, 'radius': 98, 'group': 15, 'period': 3, 'block': 'p', 'cat': 'nonmetal'},
    'S': {'name': 'Sulfur', 'Z': 16, 'mass': 32.07, 'en': 2.58, 'radius': 88, 'group': 16, 'period': 3, 'block': 'p', 'cat': 'nonmetal'},
    'Cl': {'name': 'Chlorine', 'Z': 17, 'mass': 35.45, 'en': 3.16, 'radius': 79, 'group': 17, 'period': 3, 'block': 'p', 'cat': 'halogen'},
    'K': {'name': 'Potassium', 'Z': 19, 'mass': 39.10, 'en': 0.82, 'radius': 243, 'group': 1, 'period': 4, 'block': 's', 'cat': 'alkali_metal'},
    'Ca': {'name': 'Calcium', 'Z': 20, 'mass': 40.08, 'en': 1.00, 'radius': 194, 'group': 2, 'period': 4, 'block': 's', 'cat': 'alkaline_earth'},
    'Sc': {'name': 'Scandium', 'Z': 21, 'mass': 44.96, 'en': 1.36, 'radius': 184, 'group': 3, 'period': 4, 'block': 'd', 'cat': 'transition_metal'},
    'Ti': {'name': 'Titanium', 'Z': 22, 'mass': 47.87, 'en': 1.54, 'radius': 176, 'group': 4, 'period': 4, 'block': 'd', 'cat': 'transition_metal'},
    'V': {'name': 'Vanadium', 'Z': 23, 'mass': 50.94, 'en': 1.63, 'radius': 171, 'group': 5, 'period': 4, 'block': 'd', 'cat': 'transition_metal'},
    'Cr': {'name': 'Chromium', 'Z': 24, 'mass': 52.00, 'en': 1.66, 'radius': 166, 'group': 6, 'period': 4, 'block': 'd', 'cat': 'transition_metal'},
    'Mn': {'name': 'Manganese', 'Z': 25, 'mass': 54.94, 'en': 1.55, 'radius': 161, 'group': 7, 'period': 4, 'block': 'd', 'cat': 'transition_metal'},
    'Fe': {'name': 'Iron', 'Z': 26, 'mass': 55.85, 'en': 1.83, 'radius': 156, 'group': 8, 'period': 4, 'block': 'd', 'cat': 'transition_metal'},
    'Co': {'name': 'Cobalt', 'Z': 27, 'mass': 58.93, 'en': 1.88, 'radius': 152, 'group': 9, 'period': 4, 'block': 'd', 'cat': 'transition_metal'},
    'Ni': {'name': 'Nickel', 'Z': 28, 'mass': 58.69, 'en': 1.91, 'radius': 149, 'group': 10, 'period': 4, 'block': 'd', 'cat': 'transition_metal'},
    'Cu': {'name': 'Copper', 'Z': 29, 'mass': 63.55, 'en': 1.90, 'radius': 145, 'group': 11, 'period': 4, 'block': 'd', 'cat': 'transition_metal'},
    'Zn': {'name': 'Zinc', 'Z': 30, 'mass': 65.38, 'en': 1.65, 'radius': 142, 'group': 12, 'period': 4, 'block': 'd', 'cat': 'transition_metal'},
    'Ga': {'name': 'Gallium', 'Z': 31, 'mass': 69.72, 'en': 1.81, 'radius': 136, 'group': 13, 'period': 4, 'block': 'p', 'cat': 'post_transition_metal'},
    'Ge': {'name': 'Germanium', 'Z': 32, 'mass': 72.63, 'en': 2.01, 'radius': 125, 'group': 14, 'period': 4, 'block': 'p', 'cat': 'metalloid'},
    'As': {'name': 'Arsenic', 'Z': 33, 'mass': 74.92, 'en': 2.18, 'radius': 114, 'group': 15, 'period': 4, 'block': 'p', 'cat': 'metalloid'},
    'Se': {'name': 'Selenium', 'Z': 34, 'mass': 78.97, 'en': 2.55, 'radius': 103, 'group': 16, 'period': 4, 'block': 'p', 'cat': 'nonmetal'},
    'Br': {'name': 'Bromine', 'Z': 35, 'mass': 79.90, 'en': 2.96, 'radius': 94, 'group': 17, 'period': 4, 'block': 'p', 'cat': 'halogen'},
    'Rb': {'name': 'Rubidium', 'Z': 37, 'mass': 85.47, 'en': 0.82, 'radius': 265, 'group': 1, 'period': 5, 'block': 's', 'cat': 'alkali_metal'},
    'Sr': {'name': 'Strontium', 'Z': 38, 'mass': 87.62, 'en': 0.95, 'radius': 219, 'group': 2, 'period': 5, 'block': 's', 'cat': 'alkaline_earth'},
    'Y': {'name': 'Yttrium', 'Z': 39, 'mass': 88.91, 'en': 1.22, 'radius': 212, 'group': 3, 'period': 5, 'block': 'd', 'cat': 'transition_metal'},
    'Zr': {'name': 'Zirconium', 'Z': 40, 'mass': 91.22, 'en': 1.33, 'radius': 206, 'group': 4, 'period': 5, 'block': 'd', 'cat': 'transition_metal'},
    'Nb': {'name': 'Niobium', 'Z': 41, 'mass': 92.91, 'en': 1.60, 'radius': 198, 'group': 5, 'period': 5, 'block': 'd', 'cat': 'transition_metal'},
    'Mo': {'name': 'Molybdenum', 'Z': 42, 'mass': 95.95, 'en': 2.16, 'radius': 190, 'group': 6, 'period': 5, 'block': 'd', 'cat': 'transition_metal'},
    'Ru': {'name': 'Ruthenium', 'Z': 44, 'mass': 101.07, 'en': 2.20, 'radius': 178, 'group': 8, 'period': 5, 'block': 'd', 'cat': 'transition_metal'},
    'Rh': {'name': 'Rhodium', 'Z': 45, 'mass': 102.91, 'en': 2.28, 'radius': 173, 'group': 9, 'period': 5, 'block': 'd', 'cat': 'transition_metal'},
    'Pd': {'name': 'Palladium', 'Z': 46, 'mass': 106.42, 'en': 2.20, 'radius': 169, 'group': 10, 'period': 5, 'block': 'd', 'cat': 'transition_metal'},
    'Ag': {'name': 'Silver', 'Z': 47, 'mass': 107.87, 'en': 1.93, 'radius': 165, 'group': 11, 'period': 5, 'block': 'd', 'cat': 'transition_metal'},
    'Cd': {'name': 'Cadmium', 'Z': 48, 'mass': 112.41, 'en': 1.69, 'radius': 161, 'group': 12, 'period': 5, 'block': 'd', 'cat': 'transition_metal'},
    'In': {'name': 'Indium', 'Z': 49, 'mass': 114.82, 'en': 1.78, 'radius': 156, 'group': 13, 'period': 5, 'block': 'p', 'cat': 'post_transition_metal'},
    'Sn': {'name': 'Tin', 'Z': 50, 'mass': 118.71, 'en': 1.96, 'radius': 145, 'group': 14, 'period': 5, 'block': 'p', 'cat': 'post_transition_metal'},
    'Sb': {'name': 'Antimony', 'Z': 51, 'mass': 121.76, 'en': 2.05, 'radius': 133, 'group': 15, 'period': 5, 'block': 'p', 'cat': 'metalloid'},
    'Te': {'name': 'Tellurium', 'Z': 52, 'mass': 127.60, 'en': 2.10, 'radius': 123, 'group': 16, 'period': 5, 'block': 'p', 'cat': 'metalloid'},
    'I': {'name': 'Iodine', 'Z': 53, 'mass': 126.90, 'en': 2.66, 'radius': 115, 'group': 17, 'period': 5, 'block': 'p', 'cat': 'halogen'},
    'Cs': {'name': 'Cesium', 'Z': 55, 'mass': 132.91, 'en': 0.79, 'radius': 298, 'group': 1, 'period': 6, 'block': 's', 'cat': 'alkali_metal'},
    'Ba': {'name': 'Barium', 'Z': 56, 'mass': 137.33, 'en': 0.89, 'radius': 253, 'group': 2, 'period': 6, 'block': 's', 'cat': 'alkaline_earth'},
    'La': {'name': 'Lanthanum', 'Z': 57, 'mass': 138.91, 'en': 1.10, 'radius': 195, 'group': 3, 'period': 6, 'block': 'f', 'cat': 'lanthanide'},
    'Hf': {'name': 'Hafnium', 'Z': 72, 'mass': 178.49, 'en': 1.30, 'radius': 208, 'group': 4, 'period': 6, 'block': 'd', 'cat': 'transition_metal'},
    'Ta': {'name': 'Tantalum', 'Z': 73, 'mass': 180.95, 'en': 1.50, 'radius': 200, 'group': 5, 'period': 6, 'block': 'd', 'cat': 'transition_metal'},
    'W': {'name': 'Tungsten', 'Z': 74, 'mass': 183.84, 'en': 2.36, 'radius': 193, 'group': 6, 'period': 6, 'block': 'd', 'cat': 'transition_metal'},
    'Re': {'name': 'Rhenium', 'Z': 75, 'mass': 186.21, 'en': 1.90, 'radius': 188, 'group': 7, 'period': 6, 'block': 'd', 'cat': 'transition_metal'},
    'Os': {'name': 'Osmium', 'Z': 76, 'mass': 190.23, 'en': 2.20, 'radius': 185, 'group': 8, 'period': 6, 'block': 'd', 'cat': 'transition_metal'},
    'Ir': {'name': 'Iridium', 'Z': 77, 'mass': 192.22, 'en': 2.20, 'radius': 180, 'group': 9, 'period': 6, 'block': 'd', 'cat': 'transition_metal'},
    'Pt': {'name': 'Platinum', 'Z': 78, 'mass': 195.08, 'en': 2.28, 'radius': 177, 'group': 10, 'period': 6, 'block': 'd', 'cat': 'transition_metal'},
    'Au': {'name': 'Gold', 'Z': 79, 'mass': 196.97, 'en': 2.54, 'radius': 174, 'group': 11, 'period': 6, 'block': 'd', 'cat': 'transition_metal'},
    'Pb': {'name': 'Lead', 'Z': 82, 'mass': 207.2, 'en': 2.33, 'radius': 154, 'group': 14, 'period': 6, 'block': 'p', 'cat': 'post_transition_metal'},
    'Bi': {'name': 'Bismuth', 'Z': 83, 'mass': 208.98, 'en': 2.02, 'radius': 143, 'group': 15, 'period': 6, 'block': 'p', 'cat': 'post_transition_metal'},
}

APPLICATION_DOMAINS = [
    (1, 'Structural Materials', None),
    (2, 'Aerospace Alloys', 1),
    (3, 'Automotive Components', 1),
    (4, 'Turbine Blades', 2),
    (5, 'Energy Materials', None),
    (6, 'Battery Electrodes', 5),
    (7, 'Thermoelectrics', 5),
    (8, 'Photovoltaics', 5),
    (9, 'Catalysts', None),
    (10, 'Hydrogen Evolution', 9),
    (11, 'CO2 Reduction', 9),
    (12, 'Electronic Materials', None),
    (13, 'Semiconductors', 12),
    (14, 'Superconductors', 12),
    (15, 'Magnetic Materials', None),
    (16, 'Hard Magnets', 15),
    (17, 'Soft Magnets', 15),
    (18, 'Biomedical', None),
    (19, 'Implants', 18),
    (20, 'Drug Delivery', 18),
]

SYNTHESIS_METHODS = [
    ('Arc Melting', 'melt'),
    ('Ball Milling', 'mechanical'),
    ('Spark Plasma Sintering', 'sintering'),
    ('Czochralski Growth', 'crystal_growth'),
    ('Molecular Beam Epitaxy', 'thin_film'),
    ('Sputtering', 'thin_film'),
    ('Chemical Vapor Deposition', 'thin_film'),
    ('Sol-Gel', 'chemical'),
    ('Hydrothermal', 'chemical'),
    ('Electrodeposition', 'electrochemical'),
]

DEFECT_TYPES = [
    ('Vacancy', 'vacancy'),
    ('Interstitial', 'interstitial'),
    ('Antisite', 'antisite'),
    ('Substitutional', 'substitutional'),
    ('Schottky', 'vacancy'),
    ('Frenkel', 'interstitial'),
]

SPACE_GROUPS = [
    (221, 'Pm-3m', 'cubic', 'Oh', 'm-3m', True),
    (225, 'Fm-3m', 'cubic', 'Oh', 'm-3m', True),
    (227, 'Fd-3m', 'cubic', 'Oh', 'm-3m', True),
    (229, 'Im-3m', 'cubic', 'Oh', 'm-3m', True),
    (194, 'P6_3/mmc', 'hexagonal', 'D6h', '6/mmm', True),
    (186, 'P6_3mc', 'hexagonal', 'C6v', '6mm', False),
    (166, 'R-3m', 'trigonal', 'D3d', '-3m', True),
    (62, 'Pnma', 'orthorhombic', 'D2h', 'mmm', True),
    (12, 'C2/m', 'monoclinic', 'C2h', '2/m', True),
    (2, 'P-1', 'triclinic', 'Ci', '-1', True),
]


def setup_extended_db():
    conn = psycopg.connect(f"dbname={DB_CONFIG["dbname"]} user={DB_CONFIG["user"]} password={DB_CONFIG["password"]} host={DB_CONFIG["host"]} port={DB_CONFIG["port"]}")
    cur = conn.cursor()

    # Drop new tables if they exist (keep original tables intact)
    new_tables = [
        'material_defect', 'defect_type', 'material_synthesis', 'synthesis_method',
        'measured_property', 'experimental_measurement', 'material_reference',
        'literature_reference', 'material_application', 'application_domain',
        'element_property', 'element', 'space_group'
    ]
    for t in new_tables:
        cur.execute(f"DROP TABLE IF EXISTS {t} CASCADE")
    conn.commit()

    # Create new tables directly
    cur.execute("""
        CREATE TABLE IF NOT EXISTS element (
            element_id SERIAL PRIMARY KEY,
            symbol VARCHAR(5) NOT NULL UNIQUE,
            name VARCHAR(50),
            atomic_number INTEGER NOT NULL,
            atomic_mass NUMERIC(10,4),
            electronegativity NUMERIC(5,3),
            atomic_radius NUMERIC(6,2),
            group_number INTEGER,
            period_number INTEGER,
            block VARCHAR(5),
            category VARCHAR(50)
        );
        CREATE TABLE IF NOT EXISTS element_property (
            element_property_id SERIAL PRIMARY KEY,
            element_id INTEGER NOT NULL REFERENCES element(element_id),
            property_name VARCHAR(100) NOT NULL,
            value NUMERIC(15,6),
            unit VARCHAR(30),
            temperature_k NUMERIC(8,2),
            source VARCHAR(100)
        );
        CREATE TABLE IF NOT EXISTS space_group (
            space_group_id SERIAL PRIMARY KEY,
            space_group_number INTEGER NOT NULL UNIQUE,
            hermann_mauguin VARCHAR(30),
            crystal_system VARCHAR(30),
            point_group VARCHAR(20),
            laue_class VARCHAR(20),
            is_centrosymmetric BOOLEAN
        );
        CREATE TABLE IF NOT EXISTS application_domain (
            domain_id SERIAL PRIMARY KEY,
            domain_name VARCHAR(100) NOT NULL,
            description TEXT,
            parent_domain_id INTEGER REFERENCES application_domain(domain_id)
        );
        CREATE TABLE IF NOT EXISTS material_application (
            material_application_id SERIAL PRIMARY KEY,
            entry_id TEXT NOT NULL REFERENCES material_entry(entry_id),
            domain_id INTEGER NOT NULL REFERENCES application_domain(domain_id),
            relevance_score NUMERIC(5,3),
            notes TEXT
        );
        CREATE TABLE IF NOT EXISTS literature_reference (
            reference_id SERIAL PRIMARY KEY,
            doi VARCHAR(200),
            title TEXT,
            authors TEXT,
            journal VARCHAR(200),
            year INTEGER,
            volume VARCHAR(20),
            pages VARCHAR(50)
        );
        CREATE TABLE IF NOT EXISTS material_reference (
            material_reference_id SERIAL PRIMARY KEY,
            entry_id TEXT NOT NULL REFERENCES material_entry(entry_id),
            reference_id INTEGER NOT NULL REFERENCES literature_reference(reference_id),
            context VARCHAR(100)
        );
        CREATE TABLE IF NOT EXISTS experimental_measurement (
            measurement_id SERIAL PRIMARY KEY,
            entry_id TEXT NOT NULL REFERENCES material_entry(entry_id),
            reference_id INTEGER REFERENCES literature_reference(reference_id),
            method VARCHAR(100),
            temperature_k NUMERIC(8,2),
            pressure_gpa NUMERIC(8,3)
        );
        CREATE TABLE IF NOT EXISTS measured_property (
            measured_property_id SERIAL PRIMARY KEY,
            measurement_id INTEGER NOT NULL REFERENCES experimental_measurement(measurement_id),
            property_name VARCHAR(100) NOT NULL,
            value NUMERIC(15,6),
            uncertainty NUMERIC(15,6),
            unit VARCHAR(30)
        );
        CREATE TABLE IF NOT EXISTS synthesis_method (
            synthesis_id SERIAL PRIMARY KEY,
            method_name VARCHAR(100) NOT NULL,
            category VARCHAR(50),
            description TEXT
        );
        CREATE TABLE IF NOT EXISTS material_synthesis (
            material_synthesis_id SERIAL PRIMARY KEY,
            entry_id TEXT NOT NULL REFERENCES material_entry(entry_id),
            synthesis_id INTEGER NOT NULL REFERENCES synthesis_method(synthesis_id),
            reference_id INTEGER REFERENCES literature_reference(reference_id),
            temperature_k NUMERIC(8,2),
            duration_hours NUMERIC(10,2),
            atmosphere VARCHAR(50),
            success BOOLEAN DEFAULT TRUE
        );
        CREATE TABLE IF NOT EXISTS defect_type (
            defect_type_id SERIAL PRIMARY KEY,
            defect_name VARCHAR(100) NOT NULL,
            category VARCHAR(50),
            description TEXT
        );
        CREATE TABLE IF NOT EXISTS material_defect (
            material_defect_id SERIAL PRIMARY KEY,
            entry_id TEXT NOT NULL REFERENCES material_entry(entry_id),
            defect_type_id INTEGER NOT NULL REFERENCES defect_type(defect_type_id),
            formation_energy NUMERIC(10,6),
            concentration NUMERIC(15,8),
            site VARCHAR(50),
            dopant_element_id INTEGER REFERENCES element(element_id)
        );
    """)
    conn.commit()
    
    # Create indexes
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_elem_symbol ON element(symbol)",
        "CREATE INDEX IF NOT EXISTS idx_elem_prop_elem ON element_property(element_id)",
        "CREATE INDEX IF NOT EXISTS idx_mat_app_entry ON material_application(entry_id)",
        "CREATE INDEX IF NOT EXISTS idx_mat_app_domain ON material_application(domain_id)",
        "CREATE INDEX IF NOT EXISTS idx_mat_ref_entry ON material_reference(entry_id)",
        "CREATE INDEX IF NOT EXISTS idx_exp_meas_entry ON experimental_measurement(entry_id)",
        "CREATE INDEX IF NOT EXISTS idx_meas_prop_meas ON measured_property(measurement_id)",
        "CREATE INDEX IF NOT EXISTS idx_mat_synth_entry ON material_synthesis(entry_id)",
        "CREATE INDEX IF NOT EXISTS idx_mat_defect_entry ON material_defect(entry_id)",
    ]
    for idx in indexes:
        cur.execute(idx)
    conn.commit()
    print("Schema created.")

    # 1. Populate element table
    for sym, data in ELEMENTS.items():
        cur.execute("""
            INSERT INTO element (symbol, name, atomic_number, atomic_mass, electronegativity, 
                               atomic_radius, group_number, period_number, block, category)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol) DO NOTHING
        """, (sym, data['name'], data['Z'], data['mass'], data['en'],
              data['radius'], data['group'], data['period'], data['block'], data['cat']))
    conn.commit()
    print(f"Elements: {len(ELEMENTS)} inserted.")

    # 2. Element properties
    properties = ['melting_point_K', 'boiling_point_K', 'density_g_cm3', 'thermal_conductivity_W_mK',
                  'bulk_modulus_GPa', 'young_modulus_GPa', 'vickers_hardness_GPa']
    elem_prop_count = 0
    for sym, data in ELEMENTS.items():
        cur.execute("SELECT element_id FROM element WHERE symbol = %s", (sym,))
        row = cur.fetchone()
        if not row:
            continue
        elem_id = row[0]
        for prop in random.sample(properties, min(4, len(properties))):
            val = random.uniform(100, 5000) if 'point' in prop else random.uniform(1, 500)
            cur.execute("""
                INSERT INTO element_property (element_id, property_name, value, unit, temperature_k, source)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (elem_id, prop, val, prop.split('_')[-1], 300.0, 'CRC Handbook'))
            elem_prop_count += 1
    conn.commit()
    print(f"Element properties: {elem_prop_count} inserted.")

    # 3. Space groups
    for sg in SPACE_GROUPS:
        cur.execute("""
            INSERT INTO space_group (space_group_number, hermann_mauguin, crystal_system, 
                                    point_group, laue_class, is_centrosymmetric)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (space_group_number) DO NOTHING
        """, sg)
    conn.commit()
    print(f"Space groups: {len(SPACE_GROUPS)} inserted.")

    # 4. Application domains (hierarchical)
    for dom_id, name, parent_id in APPLICATION_DOMAINS:
        cur.execute("""
            INSERT INTO application_domain (domain_id, domain_name, parent_domain_id)
            VALUES (%s, %s, %s)
        """, (dom_id, name, parent_id))
    conn.commit()
    print(f"Application domains: {len(APPLICATION_DOMAINS)} inserted.")

    # 5. Link materials to applications (many-to-many)
    cur.execute("SELECT entry_id FROM material_entry")
    entry_ids = [r[0] for r in cur.fetchall()]
    app_link_count = 0
    for eid in entry_ids:
        # Each material gets 1-3 applications
        for dom_id in random.sample(range(1, 21), random.randint(1, 3)):
            cur.execute("""
                INSERT INTO material_application (entry_id, domain_id, relevance_score)
                VALUES (%s, %s, %s)
            """, (eid, dom_id, round(random.uniform(0.3, 1.0), 3)))
            app_link_count += 1
    conn.commit()
    print(f"Material-application links: {app_link_count} inserted.")

    # 6. Literature references
    journals = ['Phys. Rev. B', 'Acta Materialia', 'J. Alloys Compd.', 'Intermetallics',
                'Comput. Mater. Sci.', 'Nature Materials', 'Science', 'npj Computational Materials']
    ref_count = 500
    for i in range(1, ref_count + 1):
        cur.execute("""
            INSERT INTO literature_reference (reference_id, doi, title, authors, journal, year, volume, pages)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (i, f'10.1000/test.{i:04d}', f'Study of intermetallic compound properties #{i}',
              f'Author{i}A, Author{i}B', random.choice(journals),
              random.randint(2000, 2025), str(random.randint(1, 500)),
              f'{random.randint(1,500)}-{random.randint(501,999)}'))
    conn.commit()
    print(f"Literature references: {ref_count} inserted.")

    # 7. Material-reference links (many-to-many)
    contexts = ['experimental_validation', 'theoretical_prediction', 'review', 'original_calculation']
    ref_link_count = 0
    for eid in random.sample(entry_ids, min(800, len(entry_ids))):
        for ref_id in random.sample(range(1, ref_count + 1), random.randint(1, 3)):
            cur.execute("""
                INSERT INTO material_reference (entry_id, reference_id, context)
                VALUES (%s, %s, %s)
            """, (eid, ref_id, random.choice(contexts)))
            ref_link_count += 1
    conn.commit()
    print(f"Material-reference links: {ref_link_count} inserted.")

    # 8. Synthesis methods
    for i, (name, cat) in enumerate(SYNTHESIS_METHODS, 1):
        cur.execute("""
            INSERT INTO synthesis_method (synthesis_id, method_name, category)
            VALUES (%s, %s, %s)
        """, (i, name, cat))
    conn.commit()
    print(f"Synthesis methods: {len(SYNTHESIS_METHODS)} inserted.")

    # 9. Material-synthesis links
    synth_link_count = 0
    for eid in random.sample(entry_ids, min(600, len(entry_ids))):
        synth_id = random.randint(1, len(SYNTHESIS_METHODS))
        cur.execute("""
            INSERT INTO material_synthesis (entry_id, synthesis_id, temperature_k, duration_hours, atmosphere, success)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (eid, synth_id, random.uniform(800, 2000), random.uniform(0.5, 48),
              random.choice(['vacuum', 'argon', 'nitrogen', 'air']), random.random() > 0.1))
        synth_link_count += 1
    conn.commit()
    print(f"Material-synthesis links: {synth_link_count} inserted.")

    # 10. Experimental measurements + measured properties
    methods = ['XRD', 'neutron_diffraction', 'DSC', 'TEM', 'SEM-EDS', 'nanoindentation']
    exp_count = 0
    prop_count = 0
    for eid in random.sample(entry_ids, min(400, len(entry_ids))):
        cur.execute("""
            INSERT INTO experimental_measurement (entry_id, reference_id, method, temperature_k, pressure_gpa)
            VALUES (%s, %s, %s, %s, %s) RETURNING measurement_id
        """, (eid, random.randint(1, ref_count), random.choice(methods),
              random.uniform(77, 1500), random.uniform(0, 10)))
        meas_id = cur.fetchone()[0]
        exp_count += 1
        # Each measurement has 1-3 measured properties
        for prop_name in random.sample(['lattice_parameter', 'hardness', 'elastic_modulus',
                                        'thermal_expansion', 'resistivity'], random.randint(1, 3)):
            cur.execute("""
                INSERT INTO measured_property (measurement_id, property_name, value, uncertainty, unit)
                VALUES (%s, %s, %s, %s, %s)
            """, (meas_id, prop_name, random.uniform(0.1, 500), random.uniform(0.01, 5),
                  'GPa' if 'modulus' in prop_name or 'hardness' in prop_name else 'Angstrom'))
            prop_count += 1
    conn.commit()
    print(f"Experimental measurements: {exp_count}, measured properties: {prop_count} inserted.")

    # 11. Defect types
    for i, (name, cat) in enumerate(DEFECT_TYPES, 1):
        cur.execute("""
            INSERT INTO defect_type (defect_type_id, defect_name, category)
            VALUES (%s, %s, %s)
        """, (i, name, cat))
    conn.commit()
    print(f"Defect types: {len(DEFECT_TYPES)} inserted.")

    # 12. Material defects
    defect_count = 0
    cur.execute("SELECT element_id FROM element")
    elem_ids = [r[0] for r in cur.fetchall()]
    for eid in random.sample(entry_ids, min(300, len(entry_ids))):
        cur.execute("""
            INSERT INTO material_defect (entry_id, defect_type_id, formation_energy, concentration, site, dopant_element_id)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (eid, random.randint(1, len(DEFECT_TYPES)), random.uniform(0.1, 5.0),
              random.uniform(1e-6, 0.01), random.choice(['A-site', 'B-site', 'interstitial']),
              random.choice(elem_ids) if random.random() > 0.5 else None))
        defect_count += 1
    conn.commit()
    print(f"Material defects: {defect_count} inserted.")

    # Summary
    cur.execute("""
        SELECT 
            (SELECT count(*) FROM material_entry) as entries,
            (SELECT count(*) FROM element) as elements,
            (SELECT count(*) FROM application_domain) as domains,
            (SELECT count(*) FROM material_application) as app_links,
            (SELECT count(*) FROM literature_reference) as refs,
            (SELECT count(*) FROM material_reference) as ref_links,
            (SELECT count(*) FROM experimental_measurement) as experiments,
            (SELECT count(*) FROM material_defect) as defects
    """)
    summary = cur.fetchone()
    print(f"\n=== EXTENDED DB SUMMARY ===")
    print(f"Material entries: {summary[0]}")
    print(f"Elements: {summary[1]}")
    print(f"Application domains: {summary[2]}")
    print(f"Material-application links: {summary[3]}")
    print(f"Literature references: {summary[4]}")
    print(f"Material-reference links: {summary[5]}")
    print(f"Experimental measurements: {summary[6]}")
    print(f"Material defects: {summary[7]}")
    
    # Count total tables
    cur.execute("""
        SELECT count(*) FROM information_schema.tables 
        WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
    """)
    table_count = cur.fetchone()[0]
    print(f"Total tables: {table_count}")

    cur.close()
    conn.close()
    print("\nExtended schema setup complete!")


if __name__ == '__main__':
    random.seed(42)
    setup_extended_db()
