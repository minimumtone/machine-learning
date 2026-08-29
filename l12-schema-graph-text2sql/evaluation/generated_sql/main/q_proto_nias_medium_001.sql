SELECT m.formula, c.element, c.atomic_fraction, s.prototype, s.strukturbericht
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
WHERE c.element = 'Ti'
  AND (s.prototype = 'NiAs' OR s.strukturbericht = 'B81')
ORDER BY m.formula
LIMIT 10000;
