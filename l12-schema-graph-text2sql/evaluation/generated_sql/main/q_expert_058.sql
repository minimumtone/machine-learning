SELECT m.formula, tp.gruneisen_parameter
FROM material_entry m
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN thermal_property tp ON tp.calculation_id = calc.calculation_id
WHERE tp.gruneisen_parameter >= 2
ORDER BY tp.gruneisen_parameter DESC
LIMIT 10000;
