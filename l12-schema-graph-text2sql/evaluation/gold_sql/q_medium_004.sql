SELECT m.formula, s.lattice_a, ABS(s.lattice_a - 3.572) AS delta_a
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE s.prototype = 'L12'
  AND ABS(s.lattice_a - 3.572) <= 0.03
ORDER BY delta_a ASC
LIMIT 100;