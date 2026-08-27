SELECT m.entry_id, m.formula, et.bulk_modulus_vrh, et.shear_modulus_vrh
FROM material_entry m
JOIN calculation cal_et ON cal_et.entry_id = m.entry_id AND cal_et.calculation_type = 'relaxation'
JOIN elastic_tensor et ON et.calculation_id = cal_et.calculation_id
WHERE et.is_stable = FALSE
ORDER BY m.formula, m.entry_id ASC
LIMIT 10000;
