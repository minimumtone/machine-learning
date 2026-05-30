-- Sample queries for L1_2 materials database

-- 1. List all L12-type compounds
SELECT m.formula, s.prototype, ps.formation_energy_per_atom, ps.energy_above_hull
FROM material_entry m
JOIN structure s ON m.entry_id = s.entry_id
JOIN phase_stability ps ON m.entry_id = ps.entry_id
WHERE s.prototype = 'L12'
ORDER BY m.formula
LIMIT 100;

-- 2. Stable L12 compounds (energy_above_hull <= 0.001)
SELECT m.formula, s.lattice_a, ps.formation_energy_per_atom
FROM material_entry m
JOIN structure s ON m.entry_id = s.entry_id
JOIN phase_stability ps ON m.entry_id = ps.entry_id
WHERE s.prototype = 'L12'
  AND ps.energy_above_hull <= 0.001
ORDER BY ps.formation_energy_per_atom ASC
LIMIT 100;

-- 3. L12 compounds containing Ni
SELECT DISTINCT m.formula, ps.formation_energy_per_atom, ps.energy_above_hull
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE c.element = 'Ni'
  AND (s.prototype = 'L12' OR s.strukturbericht = 'L12')
ORDER BY ps.formation_energy_per_atom ASC
LIMIT 100;

-- 4. Lattice constants near Ni3Al (3.572 +/- 0.03 A)
SELECT m.formula, s.lattice_a,
       ABS(s.lattice_a - 3.572) AS delta_a,
       ps.formation_energy_per_atom
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE s.prototype = 'L12'
  AND ABS(s.lattice_a - 3.572) <= 0.03
ORDER BY delta_a ASC
LIMIT 100;

-- 5. Metastable L12 compounds (energy_above_hull <= 0.05)
SELECT m.formula, ps.energy_above_hull, ps.formation_energy_per_atom
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE s.prototype = 'L12'
  AND ps.energy_above_hull <= 0.05
  AND ps.energy_above_hull > 0.001
ORDER BY ps.energy_above_hull ASC
LIMIT 100;

-- 6. Bulk modulus of L12 compounds
SELECT m.formula, cp.value AS bulk_modulus_GPa, ps.formation_energy_per_atom
FROM material_entry m
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp ON cp.calculation_id = calc.calculation_id
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE s.prototype = 'L12'
  AND cp.property_name = 'bulk_modulus'
ORDER BY cp.value DESC
LIMIT 100;

-- 7. A-site / B-site element distribution
SELECT c.site_label, c.element, COUNT(*) AS count
FROM composition c
JOIN structure s ON s.entry_id = c.entry_id
WHERE s.prototype = 'L12'
GROUP BY c.site_label, c.element
ORDER BY c.site_label, count DESC
LIMIT 100;
