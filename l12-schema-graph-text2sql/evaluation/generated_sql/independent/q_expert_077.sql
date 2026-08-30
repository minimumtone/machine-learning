SELECT DISTINCT m.formula, m.source_db
FROM material_entry m
JOIN material_synthesis ms ON ms.entry_id = m.entry_id
WHERE ms.success = TRUE
  AND m.source_db IS NOT NULL
ORDER BY m.formula
LIMIT 10000;
