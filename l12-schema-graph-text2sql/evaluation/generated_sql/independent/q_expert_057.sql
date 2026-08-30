SELECT m.formula, s.strukturbericht, tp.thermal_conductivity, tp.temperature_k
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN thermal_property tp ON tp.calculation_id = calc.calculation_id
WHERE s.strukturbericht = 'L12'
  AND tp.thermal_conductivity IS NOT NULL
ORDER BY m.formula
LIMIT 10000;
