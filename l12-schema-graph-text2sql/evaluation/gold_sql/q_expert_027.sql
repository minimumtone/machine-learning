SELECT m.entry_id, m.formula, s.lattice_c
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE (s.prototype = 'NiAs' OR s.strukturbericht = 'B81')
  AND s.crystal_system = 'hexagonal'
ORDER BY s.lattice_c DESC
LIMIT 1;
