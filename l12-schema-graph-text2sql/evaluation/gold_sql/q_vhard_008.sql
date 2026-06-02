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
LIMIT 100;