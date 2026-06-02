SELECT DISTINCT m.entry_id, m.formula
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
WHERE c.element = 'Fe'
ORDER BY m.formula
LIMIT 100;