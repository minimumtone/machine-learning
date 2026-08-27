SELECT m.entry_id, m.formula
FROM material_entry m
ORDER BY m.formula, m.entry_id ASC
LIMIT 10000;