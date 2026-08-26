SELECT m.entry_id, m.formula, s.lattice_a
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
WHERE c.element = 'Ir'
  AND (s.prototype = 'L12' OR s.strukturbericht = 'L12')
ORDER BY m.formula
LIMIT 10000;