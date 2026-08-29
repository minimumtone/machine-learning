SELECT COUNT(*) AS space_group_225_count
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE s.space_group_number = 225
LIMIT 10000;
