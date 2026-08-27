SELECT m.entry_id, m.formula, et.poisson_ratio
FROM material_entry m
JOIN calculation cal_et ON cal_et.entry_id = m.entry_id AND cal_et.calculation_type = 'relaxation'
JOIN elastic_tensor et ON et.calculation_id = cal_et.calculation_id
WHERE et.poisson_ratio >= 0.3
ORDER BY et.poisson_ratio DESC
LIMIT 10000;
