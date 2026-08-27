SELECT m.entry_id, m.formula
FROM material_entry m
JOIN composition c1 ON c1.entry_id = m.entry_id
JOIN composition c2 ON c2.entry_id = m.entry_id
WHERE c1.element = 'Ni' AND c2.element = 'Al'
ORDER BY m.formula
LIMIT 10000;