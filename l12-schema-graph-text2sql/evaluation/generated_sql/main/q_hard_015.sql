SELECT m.formula, s.lattice_a, cp.value AS bulk_modulus
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp ON cp.calculation_id = calc.calculation_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ABS(s.lattice_a - 3.55) <= 0.1
  AND cp.property_name = 'bulk_modulus'
ORDER BY cp.value DESC, ABS(s.lattice_a - 3.55) ASC
LIMIT 10000;
