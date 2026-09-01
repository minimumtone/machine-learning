SELECT m.formula, s.lattice_c
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE s.prototype = 'NiAs'
  AND s.crystal_system = 'hexagonal'
ORDER BY s.lattice_c DESC NULLS LAST
LIMIT 1;
