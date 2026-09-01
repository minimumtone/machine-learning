SELECT m.formula, cp.value AS shear_modulus, cp.unit
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp ON cp.calculation_id = calc.calculation_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND cp.property_name = 'shear_modulus'
ORDER BY m.formula
LIMIT 10000;
