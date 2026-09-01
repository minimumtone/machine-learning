SELECT m.formula, et.poisson_ratio
FROM elastic_tensor et
JOIN calculation calc ON calc.calculation_id = et.calculation_id
JOIN material_entry m ON m.entry_id = calc.entry_id
WHERE et.poisson_ratio >= 0.3
ORDER BY et.poisson_ratio DESC
LIMIT 10000;
