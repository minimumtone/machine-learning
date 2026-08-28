SELECT m.entry_id, m.formula, s.lattice_a, s.lattice_c, s.crystal_system
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE s.lattice_a IS NOT NULL AND s.lattice_c IS NOT NULL
  AND ABS(s.lattice_a - s.lattice_c) > 0.01
ORDER BY ABS(s.lattice_a - s.lattice_c) DESC, m.entry_id ASC
LIMIT 10000;
