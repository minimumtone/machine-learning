SELECT m.entry_id, m.formula, et.youngs_modulus
FROM material_entry m
JOIN calculation cal_et ON cal_et.entry_id = m.entry_id AND cal_et.calculation_type = 'relaxation'
JOIN elastic_tensor et ON et.calculation_id = cal_et.calculation_id
WHERE et.youngs_modulus IS NOT NULL
ORDER BY et.youngs_modulus DESC, m.entry_id ASC
LIMIT 10000;
