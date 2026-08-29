SELECT m.formula, ms.temperature_k, s.prototype, s.strukturbericht
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN material_synthesis ms ON ms.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ms.temperature_k >= 1000
  AND ms.success = TRUE
ORDER BY ms.temperature_k DESC, m.formula
LIMIT 10000;
