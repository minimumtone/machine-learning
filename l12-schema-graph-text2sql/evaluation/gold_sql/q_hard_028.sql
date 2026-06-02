SELECT m.formula, s.volume_per_atom, s.lattice_a
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE s.prototype = 'L12' OR s.strukturbericht = 'L12'
ORDER BY s.volume_per_atom ASC
LIMIT 100;