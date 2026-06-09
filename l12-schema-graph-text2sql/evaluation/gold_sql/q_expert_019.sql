SELECT DISTINCT m.entry_id, m.formula, c.element, e.atomic_number
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
JOIN element e ON e.symbol = c.element
WHERE e.atomic_number >= 40
  AND (s.prototype = 'L12' OR s.strukturbericht = 'L12')
ORDER BY m.formula
LIMIT 10000;
