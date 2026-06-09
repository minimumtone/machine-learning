SELECT m.formula, s.lattice_a, cp.value AS bulk_modulus,
       ABS(s.lattice_a - 3.55) AS lattice_diff
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp ON cp.calculation_id = calc.calculation_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND cp.property_name = 'bulk_modulus'
  AND ABS(s.lattice_a - 3.55) <= 0.1
ORDER BY cp.value DESC
LIMIT 10000;