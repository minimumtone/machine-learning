SELECT m.formula, c.element, c.atomic_fraction
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN composition c ON c.entry_id = m.entry_id
WHERE c.element = 'Sc'
  AND (s.prototype = 'NaCl' OR s.prototype = 'B1' OR s.strukturbericht = 'B1')
ORDER BY m.formula
LIMIT 10000;
