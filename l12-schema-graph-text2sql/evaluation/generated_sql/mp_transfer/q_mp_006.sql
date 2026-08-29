SELECT crystal_system, COUNT(*) AS material_count
FROM mp_entries
GROUP BY crystal_system
ORDER BY material_count DESC
LIMIT 10000;
