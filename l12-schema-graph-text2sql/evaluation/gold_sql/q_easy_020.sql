SELECT m.entry_id, m.formula, s.prototype
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
WHERE c.element = 'W'
  AND (s.prototype = 'L12' OR s.strukturbericht = 'L12')
ORDER BY m.formula, m.entry_id ASC
LIMIT 10000;