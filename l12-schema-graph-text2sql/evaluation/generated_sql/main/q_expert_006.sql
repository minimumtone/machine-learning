SELECT m.entry_id, m.formula, m.source_db, m.source_material_id
FROM material_entry m
WHERE m.formula = 'Ni3Al'
ORDER BY m.entry_id
LIMIT 10000;
