SELECT m.chemical_system, COUNT(*) AS count
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND m.number_of_elements = 2
GROUP BY m.chemical_system
ORDER BY count DESC, m.chemical_system
LIMIT 10000;