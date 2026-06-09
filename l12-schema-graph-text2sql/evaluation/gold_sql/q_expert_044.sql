SELECT m.entry_id, m.formula, et.youngs_modulus
FROM material_entry m
JOIN elastic_tensor et ON et.entry_id = m.entry_id
WHERE et.youngs_modulus IS NOT NULL
ORDER BY et.youngs_modulus DESC
LIMIT 10000;
