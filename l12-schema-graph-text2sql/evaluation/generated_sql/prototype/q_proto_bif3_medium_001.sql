SELECT DISTINCT m.formula, c.element, c.atomic_fraction
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
WHERE c.element = 'Si'
  AND (s.prototype = 'BiF3' OR s.formula_type = 'BiF3')
ORDER BY m.formula
LIMIT 10000;
