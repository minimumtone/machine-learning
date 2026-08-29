SELECT DISTINCT m.formula, s.crystal_system
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE s.crystal_system = 'cubic'
ORDER BY m.formula
LIMIT 10000;
