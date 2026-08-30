SELECT m.chemical_system, COUNT(*) AS entry_count
FROM material_entry m
GROUP BY m.chemical_system
ORDER BY entry_count DESC
LIMIT 10;
