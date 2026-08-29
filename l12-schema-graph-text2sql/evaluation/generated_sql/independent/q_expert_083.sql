SELECT DISTINCT m.formula, cp_bm.value AS bulk_modulus, tp.debye_temperature_k, ps.energy_above_hull
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN calculation calc_bm ON calc_bm.entry_id = m.entry_id
JOIN calculated_property cp_bm ON cp_bm.calculation_id = calc_bm.calculation_id
     AND cp_bm.property_name = 'bulk_modulus'
JOIN calculation calc_tp ON calc_tp.entry_id = m.entry_id
JOIN thermal_property tp ON tp.calculation_id = calc_tp.calculation_id
WHERE c.element = 'Co'
  AND (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.is_stable = TRUE
  AND cp_bm.value >= 180
  AND tp.debye_temperature_k >= 400
ORDER BY cp_bm.value DESC, tp.debye_temperature_k DESC
LIMIT 10000;
