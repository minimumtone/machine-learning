SELECT m.entry_id, m.formula, s.lattice_a
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE s.lattice_a BETWEEN 3.50 AND 3.60
  AND (s.prototype = 'L12' OR s.strukturbericht = 'L12')
ORDER BY s.lattice_a, m.entry_id ASC
LIMIT 10000;
