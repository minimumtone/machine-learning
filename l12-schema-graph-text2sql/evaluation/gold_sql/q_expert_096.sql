SELECT chemical_system, COUNT(*) AS cnt
FROM material_entry
GROUP BY chemical_system
ORDER BY cnt DESC
LIMIT 10;
