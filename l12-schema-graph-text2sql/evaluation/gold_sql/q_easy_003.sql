SELECT m.formula, s.formula_type
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE s.formula_type = 'A3B'
ORDER BY m.formula
LIMIT 100;