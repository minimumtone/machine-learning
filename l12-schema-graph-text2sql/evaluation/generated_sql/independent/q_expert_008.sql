SELECT COUNT(*) AS oqmd_entry_count
FROM material_entry m
WHERE m.source_db = 'OQMD'
LIMIT 10000;
