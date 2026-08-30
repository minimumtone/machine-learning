SELECT m.formula, s.lattice_a, s.lattice_c, ABS(s.lattice_a - s.lattice_c) AS lattice_a_c_diff
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE ABS(s.lattice_a - s.lattice_c) > 0.01
ORDER BY ABS(s.lattice_a - s.lattice_c) DESC
LIMIT 10000;
