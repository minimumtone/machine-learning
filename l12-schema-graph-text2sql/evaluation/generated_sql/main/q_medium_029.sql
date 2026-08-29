SELECT m.chemical_system, m.formula
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND m.number_of_elements = 2
ORDER BY m.chemical_system, m.formula
LIMIT 10000;
