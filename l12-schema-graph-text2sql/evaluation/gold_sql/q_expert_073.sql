SELECT DISTINCT m.entry_id, m.formula, ms.temperature_k
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN material_synthesis ms ON ms.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ms.temperature_k >= 1000
ORDER BY ms.temperature_k DESC
LIMIT 10000;
