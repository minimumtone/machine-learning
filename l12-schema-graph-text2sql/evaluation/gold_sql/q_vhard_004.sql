SELECT m.formula, cp_bm.value AS bulk_modulus, ps.energy_above_hull
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id AND calc.calculation_type = 'relaxation'
JOIN calculated_property cp_bm ON cp_bm.calculation_id = calc.calculation_id
WHERE (c.element = 'Ni' OR c.element = 'Co')
  AND (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.energy_above_hull <= 0.001
  AND cp_bm.property_name = 'bulk_modulus'
ORDER BY cp_bm.value DESC
LIMIT 10;