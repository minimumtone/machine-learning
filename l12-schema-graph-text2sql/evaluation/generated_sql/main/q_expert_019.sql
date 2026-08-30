SELECT DISTINCT m.formula, c.element, e.atomic_number, c.atomic_fraction
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN composition c ON c.entry_id = m.entry_id
JOIN element e ON e.symbol = c.element
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND e.atomic_number >= 40
ORDER BY m.formula, e.atomic_number, c.element
LIMIT 10000;
