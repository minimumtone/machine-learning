SELECT m.entry_id, m.formula, tp.thermal_conductivity
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN calculation cal_tp ON cal_tp.entry_id = m.entry_id
JOIN thermal_property tp ON tp.calculation_id = cal_tp.calculation_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND tp.thermal_conductivity IS NOT NULL
ORDER BY tp.thermal_conductivity DESC
LIMIT 10000;
