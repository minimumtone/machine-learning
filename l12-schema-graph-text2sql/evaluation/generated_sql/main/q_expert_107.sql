SELECT
  m.formula,
  s.lattice_a,
  ABS(s.lattice_a - 3.57) AS lattice_diff,
  cp.value AS bulk_modulus
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp ON cp.calculation_id = calc.calculation_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND EXISTS (
    SELECT 1
    FROM composition c
    WHERE c.entry_id = m.entry_id
      AND c.element = 'Ni'
  )
  AND EXISTS (
    SELECT 1
    FROM composition c
    WHERE c.entry_id = m.entry_id
      AND c.element = 'Al'
  )
  AND ABS(s.lattice_a - 3.57) <= 0.1
  AND cp.property_name = 'bulk_modulus'
  AND cp.value >= 100
ORDER BY ABS(s.lattice_a - 3.57) ASC, cp.value DESC
LIMIT 10000;
