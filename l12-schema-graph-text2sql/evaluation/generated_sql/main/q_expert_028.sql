SELECT m.formula, s.lattice_a
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE (s.prototype = 'B2' OR s.strukturbericht = 'B2')
  AND s.lattice_a >= 4.0
ORDER BY s.lattice_a DESC, m.formula
LIMIT 10000;
