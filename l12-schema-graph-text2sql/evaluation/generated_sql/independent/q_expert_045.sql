SELECT m.formula, et.poisson_ratio
FROM material_entry m
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN elastic_tensor et ON et.calculation_id = calc.calculation_id
WHERE et.poisson_ratio >= 0.3
ORDER BY et.poisson_ratio DESC
LIMIT 10000;
