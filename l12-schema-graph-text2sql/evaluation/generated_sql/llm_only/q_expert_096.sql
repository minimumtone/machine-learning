SELECT
  chemical_system,
  COUNT(*) AS entry_count
FROM material_entry
GROUP BY chemical_system
ORDER BY entry_count DESC, chemical_system
LIMIT 10;
