SELECT m.formula, s.lattice_a, s.lattice_b, s.lattice_c
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE s.prototype = 'L12' OR s.strukturbericht = 'L12'
ORDER BY s.lattice_a
LIMIT 100;