SELECT m.formula, s.prototype, s.strukturbericht, s.crystal_system
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND s.crystal_system = 'cubic'
ORDER BY m.formula
LIMIT 10000;
