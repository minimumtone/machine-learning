SELECT m.entry_id, m.formula, s.space_group, s.space_group_number
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE s.space_group = 'Pm-3m' OR s.space_group_number = 221
ORDER BY m.formula
LIMIT 10000;
