SELECT m.formula, cp.property_name, cp.value, cp.unit
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id AND calc.calculation_type = 'relaxation'
JOIN calculated_property cp ON cp.calculation_id = calc.calculation_id
WHERE s.prototype = 'L12' OR s.strukturbericht = 'L12'
ORDER BY m.formula, cp.property_name
LIMIT 10000;