#!/usr/bin/env python3
"""Generate gold SQL and expected results for all 100 evaluation queries."""
from __future__ import annotations

import json
import os
from pathlib import Path

import psycopg

PROJECT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT / "evaluation"
GOLD_DIR = EVAL_DIR / "gold_sql"
RESULTS_DIR = EVAL_DIR / "expected_results"

GOLD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

CONNINFO = (
    f"host={os.getenv('POSTGRES_HOST', 'localhost')} "
    f"port={os.getenv('POSTGRES_PORT', '5432')} "
    f"dbname={os.getenv('POSTGRES_DB', 'l12_materials')} "
    f"user={os.getenv('POSTGRES_USER', 'l12_user')} "
    f"password={os.getenv('POSTGRES_PASSWORD', 'l12_password')}"
)

# ── Gold SQL mapping: query_id -> SQL ──
GOLD_SQL: dict[str, str] = {
    # ───── EASY (20) ─────
    "q_easy_001": """\
SELECT m.entry_id, m.formula, s.prototype, s.strukturbericht
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE s.prototype = 'L12' OR s.strukturbericht = 'L12'
ORDER BY m.formula
LIMIT 10000;""",

    "q_easy_002": """\
SELECT DISTINCT m.entry_id, m.formula, s.prototype
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
WHERE c.element = 'Ni'
  AND (s.prototype = 'L12' OR s.strukturbericht = 'L12')
ORDER BY m.formula
LIMIT 10000;""",

    "q_easy_003": """\
SELECT m.entry_id, m.formula, s.formula_type
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE s.formula_type = 'A3B'
ORDER BY m.formula
LIMIT 10000;""",

    "q_easy_004": """\
SELECT m.formula, s.lattice_a, s.lattice_b, s.lattice_c
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE s.prototype = 'L12' OR s.strukturbericht = 'L12'
ORDER BY s.lattice_a
LIMIT 10000;""",

    "q_easy_005": """\
SELECT DISTINCT m.entry_id, m.formula
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
WHERE c.element = 'Co'
ORDER BY m.formula
LIMIT 10000;""",

    "q_easy_006": """\
SELECT m.entry_id, m.formula, s.crystal_system
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE s.crystal_system = 'cubic'
ORDER BY m.formula
LIMIT 10000;""",

    "q_easy_007": """\
SELECT m.entry_id, m.formula, s.prototype
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE s.prototype = 'L12'
ORDER BY m.formula
LIMIT 10000;""",

    "q_easy_008": """\
SELECT DISTINCT m.entry_id, m.formula
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
WHERE c.element = 'Al'
ORDER BY m.formula
LIMIT 10000;""",

    "q_easy_009": """\
SELECT m.entry_id, m.formula, s.space_group_number
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE s.space_group_number = 221
ORDER BY m.formula
LIMIT 10000;""",

    "q_easy_010": """\
SELECT DISTINCT m.entry_id, m.formula, s.prototype
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
WHERE c.element = 'Ti'
  AND (s.prototype = 'L12' OR s.strukturbericht = 'L12')
ORDER BY m.formula
LIMIT 10000;""",

    "q_easy_011": """\
SELECT m.entry_id, m.formula
FROM material_entry m
ORDER BY m.formula
LIMIT 10000;""",

    "q_easy_012": """\
SELECT COUNT(*) AS l12_count
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE s.prototype = 'L12'
LIMIT 10000;""",

    "q_easy_013": """\
SELECT DISTINCT m.entry_id, m.formula
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
WHERE c.element = 'Pt'
ORDER BY m.formula
LIMIT 10000;""",

    "q_easy_014": """\
SELECT DISTINCT m.entry_id, m.formula
FROM material_entry m
JOIN composition c1 ON c1.entry_id = m.entry_id
JOIN composition c2 ON c2.entry_id = m.entry_id
WHERE c1.element = 'Ni' AND c2.element = 'Al'
ORDER BY m.formula
LIMIT 10000;""",

    "q_easy_015": """\
SELECT DISTINCT m.entry_id, m.formula, s.prototype
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
WHERE c.element = 'Ga'
  AND (s.prototype = 'L12' OR s.strukturbericht = 'L12')
ORDER BY m.formula
LIMIT 10000;""",

    "q_easy_016": """\
SELECT DISTINCT m.entry_id, m.formula
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
WHERE c.element = 'Fe'
ORDER BY m.formula
LIMIT 10000;""",

    "q_easy_017": """\
SELECT DISTINCT c.element, c.site_label
FROM composition c
WHERE c.site_label = 'A'
ORDER BY c.element
LIMIT 10000;""",

    "q_easy_018": """\
SELECT DISTINCT m.chemical_system
FROM material_entry m
ORDER BY m.chemical_system
LIMIT 10000;""",

    "q_easy_019": """\
SELECT m.entry_id, m.formula, s.prototype, s.crystal_system
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE s.prototype = 'L12' AND s.crystal_system = 'cubic'
ORDER BY m.formula
LIMIT 10000;""",

    "q_easy_020": """\
SELECT DISTINCT m.entry_id, m.formula, s.prototype
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
WHERE c.element = 'W'
  AND (s.prototype = 'L12' OR s.strukturbericht = 'L12')
ORDER BY m.formula
LIMIT 10000;""",

    # ───── MEDIUM (30) ─────
    "q_medium_001": """\
SELECT DISTINCT m.entry_id, m.formula, ps.formation_energy_per_atom, ps.energy_above_hull
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE c.element = 'Ni'
  AND (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.energy_above_hull <= 0.001
ORDER BY ps.formation_energy_per_atom ASC
LIMIT 10000;""",

    "q_medium_002": """\
SELECT m.formula, s.prototype, ps.formation_energy_per_atom
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.formation_energy_per_atom < 0
ORDER BY ps.formation_energy_per_atom ASC
LIMIT 10000;""",

    "q_medium_003": """\
SELECT DISTINCT m.entry_id, m.formula, ps.formation_energy_per_atom, ps.energy_above_hull
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE c.element = 'Al'
  AND (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.energy_above_hull <= 0.001
ORDER BY ps.formation_energy_per_atom ASC
LIMIT 10000;""",

    "q_medium_004": """\
SELECT m.formula, s.lattice_a,
       ABS(s.lattice_a - 3.57) AS lattice_diff
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
ORDER BY ABS(s.lattice_a - 3.57) ASC
LIMIT 10000;""",

    "q_medium_005": """\
SELECT m.formula, ps.formation_energy_per_atom, ps.energy_above_hull, ps.is_stable
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.energy_above_hull <= 0.001
ORDER BY ps.formation_energy_per_atom ASC
LIMIT 10000;""",

    "q_medium_006": """\
SELECT m.formula, ps.energy_above_hull, ps.formation_energy_per_atom
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.energy_above_hull > 0.001
  AND ps.energy_above_hull <= 0.05
ORDER BY ps.energy_above_hull ASC
LIMIT 10000;""",

    "q_medium_007": """\
SELECT m.formula, ps.formation_energy_per_atom
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE s.prototype = 'L12' OR s.strukturbericht = 'L12'
ORDER BY ps.formation_energy_per_atom ASC
LIMIT 10000;""",

    "q_medium_008": """\
SELECT DISTINCT m.entry_id, m.formula, ps.energy_above_hull, ps.formation_energy_per_atom
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE c.element = 'Co'
  AND (s.prototype = 'L12' OR s.strukturbericht = 'L12')
ORDER BY ps.energy_above_hull ASC
LIMIT 10000;""",

    "q_medium_009": """\
SELECT m.formula, s.lattice_a
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND s.lattice_a >= 3.5
ORDER BY s.lattice_a ASC
LIMIT 10000;""",

    "q_medium_010": """\
SELECT DISTINCT m.entry_id, m.formula, s.prototype
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
WHERE (c.element = 'Ni' OR c.element = 'Co')
  AND (s.prototype = 'L12' OR s.strukturbericht = 'L12')
ORDER BY m.formula
LIMIT 10000;""",

    "q_medium_011": """\
SELECT m.formula, ps.formation_energy_per_atom
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.formation_energy_per_atom <= -0.4
ORDER BY ps.formation_energy_per_atom ASC
LIMIT 10000;""",

    "q_medium_012": """\
SELECT m.formula, ps.energy_above_hull
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.energy_above_hull <= 0.01
ORDER BY ps.energy_above_hull ASC
LIMIT 10000;""",

    "q_medium_013": """\
SELECT DISTINCT m.formula, s.lattice_a, ps.formation_energy_per_atom
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE c.element = 'Ti'
  AND (s.prototype = 'L12' OR s.strukturbericht = 'L12')
ORDER BY m.formula
LIMIT 10000;""",

    "q_medium_014": """\
SELECT m.formula, s.lattice_a
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE s.prototype = 'L12' OR s.strukturbericht = 'L12'
ORDER BY s.lattice_a ASC
LIMIT 10000;""",

    "q_medium_015": """\
SELECT m.formula, m.chemical_system, s.lattice_a
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE m.chemical_system = 'Al-Ni'
  AND (s.prototype = 'L12' OR s.strukturbericht = 'L12')
ORDER BY m.formula
LIMIT 10000;""",

    "q_medium_016": """\
SELECT m.formula, s.lattice_a, ps.energy_above_hull
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.energy_above_hull <= 0.001
  AND s.lattice_a < 3.6
ORDER BY s.lattice_a ASC
LIMIT 10000;""",

    "q_medium_017": """\
SELECT DISTINCT m.formula, ps.energy_above_hull, ps.formation_energy_per_atom, ps.is_stable
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE c.element = 'Nb'
  AND (s.prototype = 'L12' OR s.strukturbericht = 'L12')
ORDER BY ps.energy_above_hull ASC
LIMIT 10000;""",

    "q_medium_018": """\
SELECT DISTINCT m.formula, s.lattice_a
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
WHERE c.element = 'Sc'
  AND (s.prototype = 'L12' OR s.strukturbericht = 'L12')
ORDER BY m.formula
LIMIT 10000;""",

    "q_medium_019": """\
SELECT m.formula, ps.is_stable, ps.energy_above_hull
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.is_stable = true
ORDER BY m.formula
LIMIT 10000;""",

    "q_medium_020": """\
SELECT m.formula, m.chemical_system, s.lattice_a
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND m.chemical_system != 'Al-Ni'
ORDER BY m.formula
LIMIT 10000;""",

    "q_medium_021": """\
SELECT c.element AS a_site_element, COUNT(DISTINCT m.entry_id) AS compound_count
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND c.site_label = 'A'
GROUP BY c.element
ORDER BY compound_count DESC
LIMIT 10000;""",

    "q_medium_022": """\
SELECT m.formula, ps.formation_energy_per_atom
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE s.prototype = 'L12' OR s.strukturbericht = 'L12'
ORDER BY ps.formation_energy_per_atom ASC
LIMIT 5;""",

    "q_medium_023": """\
SELECT m.formula, s.lattice_a
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND s.lattice_a >= 4.0
ORDER BY s.lattice_a ASC
LIMIT 10000;""",

    "q_medium_024": """\
SELECT DISTINCT m.formula, ps.formation_energy_per_atom
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE c.element = 'Pd'
  AND (s.prototype = 'L12' OR s.strukturbericht = 'L12')
ORDER BY ps.formation_energy_per_atom ASC
LIMIT 10000;""",

    "q_medium_025": """\
SELECT DISTINCT m.formula, ps.energy_above_hull, ps.is_stable
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE c.element = 'Cu'
  AND (s.prototype = 'L12' OR s.strukturbericht = 'L12')
ORDER BY ps.energy_above_hull ASC
LIMIT 10000;""",

    "q_medium_026": """\
SELECT c.element AS b_site_element,
       AVG(ps.formation_energy_per_atom) AS avg_formation_energy
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND c.site_label = 'B'
GROUP BY c.element
ORDER BY avg_formation_energy ASC
LIMIT 10000;""",

    "q_medium_027": """\
SELECT m.formula, s.volume_per_atom
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND s.volume_per_atom <= 12.0
ORDER BY s.volume_per_atom ASC
LIMIT 10000;""",

    "q_medium_028": """\
SELECT DISTINCT m.entry_id, m.formula, s.lattice_a
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
WHERE c.element = 'Ir'
  AND (s.prototype = 'L12' OR s.strukturbericht = 'L12')
ORDER BY m.formula
LIMIT 10000;""",

    "q_medium_029": """\
SELECT m.chemical_system, COUNT(*) AS count
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND m.number_of_elements = 2
GROUP BY m.chemical_system
ORDER BY count DESC
LIMIT 10000;""",

    "q_medium_030": """\
SELECT m.formula, s.lattice_a, ps.energy_above_hull
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.energy_above_hull <= 0.001
ORDER BY s.lattice_a DESC
LIMIT 10000;""",

    # ───── HARD (30) ─────
    "q_hard_001": """\
SELECT DISTINCT m.entry_id, m.formula, ps.energy_above_hull, ps.formation_energy_per_atom, s.lattice_a
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.energy_above_hull <= 0.05
ORDER BY ps.energy_above_hull ASC, ps.formation_energy_per_atom ASC
LIMIT 10000;""",

    "q_hard_002": """\
SELECT ca.element AS a_site, cb.element AS b_site,
       AVG(ps.formation_energy_per_atom) AS avg_eform
FROM material_entry m
JOIN composition ca ON ca.entry_id = m.entry_id AND ca.site_label = 'A'
JOIN composition cb ON cb.entry_id = m.entry_id AND cb.site_label = 'B'
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE s.prototype = 'L12' OR s.strukturbericht = 'L12'
GROUP BY ca.element, cb.element
ORDER BY avg_eform ASC
LIMIT 10000;""",

    "q_hard_003": """\
SELECT m.formula, s.lattice_a, ps.formation_energy_per_atom,
       ABS(s.lattice_a - 3.57) AS lattice_diff
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ABS(s.lattice_a - 3.57) <= 0.1
ORDER BY ps.formation_energy_per_atom ASC
LIMIT 10000;""",

    "q_hard_004": """\
SELECT m.formula, cp.value AS bulk_modulus, ps.energy_above_hull
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp ON cp.calculation_id = calc.calculation_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.energy_above_hull <= 0.001
  AND cp.property_name = 'bulk_modulus'
ORDER BY cp.value DESC
LIMIT 10000;""",

    "q_hard_005": """\
SELECT DISTINCT m.entry_id, m.formula, ps.formation_energy_per_atom, ps.energy_above_hull
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE (c.element = 'Ni' OR c.element = 'Co')
  AND (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.energy_above_hull <= 0.001
ORDER BY ps.formation_energy_per_atom ASC
LIMIT 10000;""",

    "q_hard_006": """\
SELECT c.element AS a_site_element, AVG(s.lattice_a) AS avg_lattice_a
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND c.site_label = 'A'
GROUP BY c.element
ORDER BY avg_lattice_a ASC
LIMIT 10000;""",

    "q_hard_007": """\
SELECT m.formula, s.lattice_a, ps.formation_energy_per_atom
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.formation_energy_per_atom <= -0.3
  AND s.lattice_a BETWEEN 3.5 AND 3.7
ORDER BY ps.formation_energy_per_atom ASC
LIMIT 10000;""",

    "q_hard_008": """\
SELECT m.formula, cp.value AS bulk_modulus
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp ON cp.calculation_id = calc.calculation_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND cp.property_name = 'bulk_modulus'
  AND cp.value >= 180
ORDER BY cp.value DESC
LIMIT 10000;""",

    "q_hard_009": """\
SELECT DISTINCT m.formula, s.lattice_a, ps.energy_above_hull,
       ABS(s.lattice_a - 3.57) AS lattice_diff
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE c.element = 'Co'
  AND (s.prototype = 'L12' OR s.strukturbericht = 'L12')
ORDER BY ps.energy_above_hull ASC, ABS(s.lattice_a - 3.57) ASC
LIMIT 10000;""",

    "q_hard_010": """\
SELECT DISTINCT m.formula, ps.formation_energy_per_atom
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE c.site_label = 'B'
  AND (c.element = 'Al' OR c.element = 'Ti')
  AND (s.prototype = 'L12' OR s.strukturbericht = 'L12')
ORDER BY ps.formation_energy_per_atom ASC
LIMIT 10000;""",

    "q_hard_011": """\
SELECT m.formula, cp.value AS shear_modulus
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp ON cp.calculation_id = calc.calculation_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND cp.property_name = 'shear_modulus'
ORDER BY cp.value DESC
LIMIT 10000;""",

    "q_hard_012": """\
SELECT m.formula, ps.formation_energy_per_atom, ps.energy_above_hull
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.energy_above_hull <= 0.001
ORDER BY ps.formation_energy_per_atom ASC
LIMIT 10;""",

    "q_hard_013": """\
SELECT DISTINCT m.formula, cp.value AS bulk_modulus, ps.formation_energy_per_atom
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp ON cp.calculation_id = calc.calculation_id
WHERE c.element = 'Ni'
  AND (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND cp.property_name = 'bulk_modulus'
ORDER BY cp.value DESC
LIMIT 10000;""",

    "q_hard_014": """\
SELECT c.element AS a_site_element,
       AVG(ps.formation_energy_per_atom) AS avg_eform,
       COUNT(DISTINCT m.entry_id) AS count
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.energy_above_hull <= 0.001
  AND c.site_label = 'A'
GROUP BY c.element
ORDER BY avg_eform ASC
LIMIT 10000;""",

    "q_hard_015": """\
SELECT m.formula, s.lattice_a, cp.value AS bulk_modulus,
       ABS(s.lattice_a - 3.55) AS lattice_diff
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp ON cp.calculation_id = calc.calculation_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND cp.property_name = 'bulk_modulus'
  AND ABS(s.lattice_a - 3.55) <= 0.1
ORDER BY cp.value DESC
LIMIT 10000;""",

    "q_hard_016": """\
SELECT m.formula, s.lattice_a, ps.formation_energy_per_atom
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE s.prototype = 'L12' OR s.strukturbericht = 'L12'
ORDER BY s.lattice_a ASC
LIMIT 10000;""",

    "q_hard_017": """\
SELECT m.formula, s.lattice_a
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND m.formula IN ('Co3Ti', 'Ni3Al')
ORDER BY m.formula
LIMIT 10000;""",

    "q_hard_018": """\
SELECT
  CASE WHEN ps.energy_above_hull <= 0.001 THEN 'stable' ELSE 'not_stable' END AS stability,
  COUNT(*) AS count
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE c.element = 'Al'
  AND (s.prototype = 'L12' OR s.strukturbericht = 'L12')
GROUP BY CASE WHEN ps.energy_above_hull <= 0.001 THEN 'stable' ELSE 'not_stable' END
ORDER BY stability
LIMIT 10000;""",

    "q_hard_019": """\
SELECT m.formula, ps.energy_above_hull
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.energy_above_hull > 0.001
  AND ps.energy_above_hull <= 0.05
ORDER BY ps.energy_above_hull ASC
LIMIT 10000;""",

    "q_hard_020": """\
SELECT c.element AS b_site_element, COUNT(DISTINCT m.entry_id) AS stable_count
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.energy_above_hull <= 0.001
  AND c.site_label = 'B'
GROUP BY c.element
ORDER BY stable_count DESC
LIMIT 10000;""",

    "q_hard_021": """\
SELECT m.formula, cp.property_name, cp.value, cp.unit
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp ON cp.calculation_id = calc.calculation_id
WHERE s.prototype = 'L12' OR s.strukturbericht = 'L12'
ORDER BY m.formula, cp.property_name
LIMIT 10000;""",

    "q_hard_022": """\
SELECT m.formula, cp.value AS bulk_modulus
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp ON cp.calculation_id = calc.calculation_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND calc.functional = 'PBE'
  AND cp.property_name = 'bulk_modulus'
ORDER BY cp.value DESC
LIMIT 10000;""",

    "q_hard_023": """\
SELECT m.formula, s.lattice_a,
       ABS(s.lattice_a - 3.57) AS mismatch
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE s.prototype = 'L12' OR s.strukturbericht = 'L12'
ORDER BY ABS(s.lattice_a - 3.57) ASC
LIMIT 10000;""",

    "q_hard_024": """\
SELECT DISTINCT m.formula, ps.energy_above_hull, ps.is_stable,
       cp.property_name, cp.value
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp ON cp.calculation_id = calc.calculation_id
WHERE c.element = 'Fe'
  AND (s.prototype = 'L12' OR s.strukturbericht = 'L12')
ORDER BY m.formula, cp.property_name
LIMIT 10000;""",

    "q_hard_025": """\
SELECT m.formula, cp.value AS bulk_modulus, ps.energy_above_hull
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp ON cp.calculation_id = calc.calculation_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.energy_above_hull <= 0.001
  AND cp.property_name = 'bulk_modulus'
  AND cp.value >= 160
ORDER BY cp.value DESC
LIMIT 10000;""",

    "q_hard_026": """\
SELECT m.formula, s.lattice_a, ps.energy_above_hull,
       ABS(s.lattice_a - 3.57) AS lattice_diff
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.energy_above_hull <= 0.05
ORDER BY ABS(s.lattice_a - 3.57) ASC, ps.energy_above_hull ASC
LIMIT 10000;""",

    "q_hard_027": """\
SELECT DISTINCT m.formula, ca.element AS a_site, cb.element AS b_site
FROM material_entry m
JOIN composition ca ON ca.entry_id = m.entry_id AND ca.site_label = 'A'
JOIN composition cb ON cb.entry_id = m.entry_id AND cb.site_label = 'B'
JOIN structure s ON s.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND (ca.element = 'Ni' OR ca.element = 'Co')
  AND (cb.element = 'Al' OR cb.element = 'Ti')
ORDER BY m.formula
LIMIT 10000;""",

    "q_hard_028": """\
SELECT m.formula, s.volume_per_atom, s.lattice_a
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE s.prototype = 'L12' OR s.strukturbericht = 'L12'
ORDER BY s.volume_per_atom ASC
LIMIT 10000;""",

    "q_hard_029": """\
SELECT m.formula, ps.energy_above_hull, ps.formation_energy_per_atom
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.energy_above_hull = 0
  AND ps.formation_energy_per_atom <= -0.4
ORDER BY ps.formation_energy_per_atom ASC
LIMIT 10000;""",

    "q_hard_030": """\
SELECT DISTINCT m.entry_id, m.formula, ps.energy_above_hull
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.energy_above_hull <= 0.001
  AND c.element IN ('Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn',
                     'Zr','Nb','Mo','Ru','Rh','Pd','Ag',
                     'Hf','Ta','W','Re','Os','Ir','Pt','Au')
ORDER BY ps.energy_above_hull ASC
LIMIT 10000;""",

    # ───── VERY HARD (20) ─────
    "q_vhard_001": """\
SELECT m.formula, s.lattice_a, ps.energy_above_hull, ps.formation_energy_per_atom,
       cp_bm.value AS bulk_modulus,
       ABS(s.lattice_a - 3.57) AS lattice_mismatch
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp_bm ON cp_bm.calculation_id = calc.calculation_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.energy_above_hull <= 0.05
  AND cp_bm.property_name = 'bulk_modulus'
ORDER BY ps.energy_above_hull ASC, ABS(s.lattice_a - 3.57) ASC
LIMIT 10000;""",

    "q_vhard_002": """\
SELECT m.formula, s.lattice_a, ps.formation_energy_per_atom,
       ps.energy_above_hull, cp_bm.value AS bulk_modulus,
       ABS(s.lattice_a - 3.57) AS lattice_diff
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp_bm ON cp_bm.calculation_id = calc.calculation_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.energy_above_hull <= 0.05
  AND cp_bm.property_name = 'bulk_modulus'
ORDER BY ps.formation_energy_per_atom ASC, ABS(s.lattice_a - 3.57) ASC
LIMIT 10000;""",

    "q_vhard_003": """\
SELECT m.formula, s.lattice_a, ps.energy_above_hull,
       cp_bm.value AS bulk_modulus,
       (1.0 - ps.energy_above_hull / 0.05) * 0.4
       + (1.0 - ABS(s.lattice_a - 3.57) / 0.3) * 0.3
       + (cp_bm.value / 300.0) * 0.3 AS weighted_score
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp_bm ON cp_bm.calculation_id = calc.calculation_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.energy_above_hull <= 0.05
  AND cp_bm.property_name = 'bulk_modulus'
ORDER BY weighted_score DESC
LIMIT 10000;""",

    "q_vhard_004": """\
SELECT DISTINCT m.formula, cp_bm.value AS bulk_modulus, ps.energy_above_hull
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp_bm ON cp_bm.calculation_id = calc.calculation_id
WHERE (c.element = 'Ni' OR c.element = 'Co')
  AND (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.energy_above_hull <= 0.001
  AND cp_bm.property_name = 'bulk_modulus'
ORDER BY cp_bm.value DESC
LIMIT 10;""",

    "q_vhard_005": """\
SELECT ca.element AS a_site, cb.element AS b_site,
       COUNT(*) AS total,
       SUM(CASE WHEN ps.energy_above_hull <= 0.001 THEN 1 ELSE 0 END) AS stable_count,
       AVG(ps.formation_energy_per_atom) AS avg_eform
FROM material_entry m
JOIN composition ca ON ca.entry_id = m.entry_id AND ca.site_label = 'A'
JOIN composition cb ON cb.entry_id = m.entry_id AND cb.site_label = 'B'
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE s.prototype = 'L12' OR s.strukturbericht = 'L12'
GROUP BY ca.element, cb.element
ORDER BY avg_eform ASC
LIMIT 10000;""",

    "q_vhard_006": """\
SELECT m.formula, s.lattice_a, cp_bm.value AS bulk_modulus,
       ps.energy_above_hull,
       ABS(s.lattice_a - 3.57) AS lattice_mismatch,
       (1.0 - ps.energy_above_hull / 0.05) * 0.35
       + (1.0 - ABS(s.lattice_a - 3.57) / 0.3) * 0.35
       + (cp_bm.value / 300.0) * 0.30 AS composite_score
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp_bm ON cp_bm.calculation_id = calc.calculation_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.energy_above_hull <= 0.05
  AND cp_bm.property_name = 'bulk_modulus'
ORDER BY composite_score DESC
LIMIT 10000;""",

    "q_vhard_007": """\
SELECT m.formula, s.lattice_a, ps.energy_above_hull,
       cp_bm.value AS bulk_modulus,
       ABS(s.lattice_a - 3.57) AS mismatch,
       (1.0 - ps.energy_above_hull / 0.05) * 0.35
       + (1.0 - LEAST(ABS(s.lattice_a - 3.57), 0.3) / 0.3) * 0.35
       + (LEAST(cp_bm.value, 300) / 300.0) * 0.30 AS score
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp_bm ON cp_bm.calculation_id = calc.calculation_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.energy_above_hull <= 0.05
  AND cp_bm.property_name = 'bulk_modulus'
ORDER BY score DESC
LIMIT 10000;""",

    "q_vhard_008": """\
SELECT DISTINCT m.formula, s.lattice_a, ps.energy_above_hull,
       cp_bm.value AS bulk_modulus,
       ABS(s.lattice_a - 3.57) AS lattice_diff_ni3al
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp_bm ON cp_bm.calculation_id = calc.calculation_id
WHERE c.element = 'Co'
  AND (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.energy_above_hull <= 0.05
  AND cp_bm.property_name = 'bulk_modulus'
ORDER BY ps.energy_above_hull ASC, ABS(s.lattice_a - 3.57) ASC
LIMIT 10000;""",

    "q_vhard_009": """\
SELECT ca.element AS a_site, cb.element AS b_site,
       AVG(ps.formation_energy_per_atom) AS avg_eform,
       AVG(s.lattice_a) AS avg_lattice,
       AVG(cp_bm.value) AS avg_bulk_modulus,
       SUM(CASE WHEN ps.energy_above_hull <= 0.001 THEN 1 ELSE 0 END) AS stable_count
FROM material_entry m
JOIN composition ca ON ca.entry_id = m.entry_id AND ca.site_label = 'A'
JOIN composition cb ON cb.entry_id = m.entry_id AND cb.site_label = 'B'
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp_bm ON cp_bm.calculation_id = calc.calculation_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND cp_bm.property_name = 'bulk_modulus'
GROUP BY ca.element, cb.element
ORDER BY avg_eform ASC
LIMIT 10000;""",

    "q_vhard_010": """\
SELECT m.formula, s.lattice_a, ps.formation_energy_per_atom,
       cp_bm.value AS bulk_modulus, cp_sm.value AS shear_modulus,
       ps.energy_above_hull
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp_bm ON cp_bm.calculation_id = calc.calculation_id
  AND cp_bm.property_name = 'bulk_modulus'
JOIN calculated_property cp_sm ON cp_sm.calculation_id = calc.calculation_id
  AND cp_sm.property_name = 'shear_modulus'
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.energy_above_hull <= 0.05
ORDER BY ps.energy_above_hull ASC, cp_bm.value DESC
LIMIT 10000;""",

    "q_vhard_011": """\
SELECT m.formula, s.lattice_a, ps.energy_above_hull,
       cp_bm.value AS bulk_modulus,
       CASE WHEN ps.energy_above_hull <= 0.001 THEN 'stable'
            WHEN ps.energy_above_hull <= 0.05 THEN 'metastable'
            ELSE 'unstable' END AS stability_class
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp_bm ON cp_bm.calculation_id = calc.calculation_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND cp_bm.property_name = 'bulk_modulus'
ORDER BY ps.energy_above_hull ASC
LIMIT 10000;""",

    "q_vhard_012": """\
SELECT m.formula, s.lattice_a, ps.formation_energy_per_atom,
       ps.energy_above_hull, cp_bm.value AS bulk_modulus,
       ABS(s.lattice_a - 3.57) AS mismatch,
       (1.0 - LEAST(ps.energy_above_hull, 0.05) / 0.05) * 0.30
       + (1.0 - LEAST(ABS(s.lattice_a - 3.57), 0.3) / 0.3) * 0.25
       + (LEAST(cp_bm.value, 300) / 300.0) * 0.25
       + (CASE WHEN ps.formation_energy_per_atom < -0.3 THEN 0.20 ELSE 0.10 END) AS design_score
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp_bm ON cp_bm.calculation_id = calc.calculation_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.energy_above_hull <= 0.05
  AND cp_bm.property_name = 'bulk_modulus'
ORDER BY design_score DESC
LIMIT 10000;""",

    "q_vhard_013": """\
SELECT m.formula, ps.energy_above_hull, ps.formation_energy_per_atom,
       s.lattice_a, cp_bm.value AS bulk_modulus
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp_bm ON cp_bm.calculation_id = calc.calculation_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.energy_above_hull <= 0.001
  AND ps.formation_energy_per_atom <= -0.3
  AND cp_bm.property_name = 'bulk_modulus'
  AND cp_bm.value >= 150
ORDER BY ps.formation_energy_per_atom ASC
LIMIT 10000;""",

    "q_vhard_014": """\
SELECT ca.element AS a_site, cb.element AS b_site,
       m.formula, s.lattice_a, ps.energy_above_hull,
       cp_bm.value AS bulk_modulus
FROM material_entry m
JOIN composition ca ON ca.entry_id = m.entry_id AND ca.site_label = 'A'
JOIN composition cb ON cb.entry_id = m.entry_id AND cb.site_label = 'B'
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp_bm ON cp_bm.calculation_id = calc.calculation_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.energy_above_hull <= 0.001
  AND cp_bm.property_name = 'bulk_modulus'
ORDER BY cp_bm.value DESC
LIMIT 10000;""",

    "q_vhard_015": """\
SELECT m.formula,
       s.lattice_a,
       ps.energy_above_hull,
       cp_bm.value AS bulk_modulus,
       cp_sm.value AS shear_modulus,
       ABS(s.lattice_a - 3.57) AS mismatch
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp_bm ON cp_bm.calculation_id = calc.calculation_id
  AND cp_bm.property_name = 'bulk_modulus'
JOIN calculated_property cp_sm ON cp_sm.calculation_id = calc.calculation_id
  AND cp_sm.property_name = 'shear_modulus'
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.energy_above_hull <= 0.05
ORDER BY ps.energy_above_hull ASC, cp_bm.value DESC
LIMIT 10000;""",

    "q_vhard_016": """\
SELECT DISTINCT m.formula, s.lattice_a, ps.energy_above_hull,
       cp_bm.value AS bulk_modulus,
       ABS(s.lattice_a - 3.57) AS mismatch
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp_bm ON cp_bm.calculation_id = calc.calculation_id
WHERE c.element IN ('Ni','Co','Fe','Pt','Ir','Pd','Rh')
  AND (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.energy_above_hull <= 0.001
  AND cp_bm.property_name = 'bulk_modulus'
ORDER BY cp_bm.value DESC
LIMIT 10000;""",

    "q_vhard_017": """\
SELECT m.formula, s.lattice_a, ps.formation_energy_per_atom,
       ps.energy_above_hull, cp_bm.value AS bulk_modulus,
       ABS(s.lattice_a - 3.57) AS mismatch,
       (1.0 - LEAST(ps.energy_above_hull, 0.05) / 0.05) * 0.30
       + (1.0 - LEAST(ABS(s.lattice_a - 3.57), 0.3) / 0.3) * 0.30
       + (LEAST(cp_bm.value, 300) / 300.0) * 0.20
       + (-ps.formation_energy_per_atom / 1.0) * 0.20 AS final_score
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp_bm ON cp_bm.calculation_id = calc.calculation_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND cp_bm.property_name = 'bulk_modulus'
ORDER BY final_score DESC
LIMIT 20;""",

    "q_vhard_018": """\
SELECT m.formula, s.lattice_a, ps.energy_above_hull,
       ps.formation_energy_per_atom,
       cp_bm.value AS bulk_modulus
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp_bm ON cp_bm.calculation_id = calc.calculation_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.energy_above_hull <= 0.001
  AND cp_bm.property_name = 'bulk_modulus'
  AND m.formula != 'Ni3Al'
ORDER BY ABS(s.lattice_a - 3.57) ASC, cp_bm.value DESC
LIMIT 10000;""",

    "q_vhard_019": """\
SELECT m.formula, s.lattice_a, ps.energy_above_hull,
       cp_bm.value AS bulk_modulus
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp_bm ON cp_bm.calculation_id = calc.calculation_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.energy_above_hull <= 0.05
  AND cp_bm.property_name = 'bulk_modulus'
  AND cp_bm.value >= 100
  AND ABS(s.lattice_a - 3.57) <= 0.2
ORDER BY ps.energy_above_hull ASC, cp_bm.value DESC
LIMIT 10000;""",

    "q_vhard_020": """\
SELECT m.formula, s.lattice_a, ps.formation_energy_per_atom,
       ps.energy_above_hull, ps.is_stable,
       cp_bm.value AS bulk_modulus,
       cp_sm.value AS shear_modulus,
       ABS(s.lattice_a - 3.57) AS mismatch,
       (1.0 - LEAST(ps.energy_above_hull, 0.05) / 0.05) * 0.25
       + (1.0 - LEAST(ABS(s.lattice_a - 3.57), 0.3) / 0.3) * 0.25
       + (LEAST(cp_bm.value, 300) / 300.0) * 0.20
       + (LEAST(cp_sm.value, 150) / 150.0) * 0.15
       + (-LEAST(ps.formation_energy_per_atom, 0) / 1.0) * 0.15 AS total_score
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp_bm ON cp_bm.calculation_id = calc.calculation_id
  AND cp_bm.property_name = 'bulk_modulus'
JOIN calculated_property cp_sm ON cp_sm.calculation_id = calc.calculation_id
  AND cp_sm.property_name = 'shear_modulus'
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
ORDER BY total_score DESC
LIMIT 10000;""",
}


def main() -> None:
    conn = psycopg.connect(CONNINFO)
    success = 0
    fail = 0

    for qid, sql in sorted(GOLD_SQL.items()):
        # Write gold SQL
        sql_path = GOLD_DIR / f"{qid}.sql"
        sql_path.write_text(sql, encoding="utf-8")

        # Execute and write expected results
        try:
            with conn.cursor() as cur:
                cur.execute(sql)
                columns = [desc[0] for desc in cur.description]
                rows = cur.fetchall()
            result = {
                "query_id": qid,
                "columns": columns,
                "row_count": len(rows),
                "rows": [list(r) for r in rows],
            }
            # Convert non-serializable types
            for row in result["rows"]:
                for i, val in enumerate(row):
                    if isinstance(val, float) and val != val:  # NaN
                        row[i] = None
                    elif hasattr(val, "__float__"):
                        row[i] = float(val)
            success += 1
        except Exception as e:
            result = {
                "query_id": qid,
                "columns": [],
                "row_count": 0,
                "rows": [],
                "error": str(e),
            }
            fail += 1
            print(f"  ERROR {qid}: {e}")

        result_path = RESULTS_DIR / f"{qid}.json"
        result_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    conn.close()
    print(f"Done: {success} success, {fail} failures, {success + fail} total")


if __name__ == "__main__":
    main()
