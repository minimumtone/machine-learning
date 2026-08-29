SELECT m.formula, s.crystal_system, s.prototype, s.strukturbericht
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE (s.prototype = 'B2' OR s.strukturbericht = 'B2')
  AND LOWER(s.crystal_system) <> 'cubic'
ORDER BY s.crystal_system, m.formula
LIMIT 10000;
