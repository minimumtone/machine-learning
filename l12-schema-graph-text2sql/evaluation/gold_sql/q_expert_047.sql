SELECT m.entry_id, m.formula, et.bulk_modulus_vrh, et.shear_modulus_vrh
FROM material_entry m
JOIN elastic_tensor et ON et.entry_id = m.entry_id
WHERE et.is_stable = FALSE
ORDER BY m.formula
LIMIT 10000;
