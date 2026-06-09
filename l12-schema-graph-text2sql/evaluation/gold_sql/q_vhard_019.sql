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
LIMIT 10000;