SELECT m.entry_id, m.formula, et.poisson_ratio
FROM material_entry m
JOIN elastic_tensor et ON et.entry_id = m.entry_id
WHERE et.poisson_ratio >= 0.3
ORDER BY et.poisson_ratio DESC
LIMIT 10000;
