SELECT m.formula, cp.value AS bulk_modulus, tp.debye_temperature_k, ps.energy_above_hull
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp ON cp.calculation_id = calc.calculation_id
JOIN thermal_property tp ON tp.calculation_id = calc.calculation_id
WHERE c.element = 'Co'
  AND (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.is_stable = TRUE
  AND cp.property_name = 'bulk_modulus'
  AND cp.value >= 180
  AND tp.debye_temperature_k >= 400
ORDER BY cp.value DESC, tp.debye_temperature_k DESC
LIMIT 10000;
