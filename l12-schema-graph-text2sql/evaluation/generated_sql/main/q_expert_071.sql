SELECT DISTINCT m.formula
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN material_synthesis ms ON ms.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ms.success = TRUE
ORDER BY m.formula
LIMIT 10000;
