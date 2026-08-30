SELECT m.formula, et.poisson_ratio
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN elastic_tensor et ON et.calculation_id = calc.calculation_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND et.poisson_ratio < 0.25
ORDER BY et.poisson_ratio ASC
LIMIT 10000;
