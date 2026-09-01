SELECT DISTINCT m.formula, c.element, c.atomic_fraction, s.prototype
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN composition c ON c.entry_id = m.entry_id
WHERE c.element = 'Si'
  AND s.prototype = 'BiF3'
ORDER BY m.formula
LIMIT 10000;
