SELECT m.entry_id, m.formula
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
WHERE c.element = 'Fe'
ORDER BY m.formula, m.entry_id ASC
LIMIT 10000;