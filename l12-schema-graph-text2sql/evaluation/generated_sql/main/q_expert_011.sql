SELECT m.formula, c.element, c.atomic_fraction, s.prototype
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
WHERE c.element = 'Pt'
  AND c.atomic_fraction >= 0.25
  AND (s.prototype = 'L12' OR s.strukturbericht = 'L12')
ORDER BY m.formula
LIMIT 10000;
