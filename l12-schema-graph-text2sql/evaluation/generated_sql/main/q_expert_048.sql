SELECT
  m.formula,
  s.lattice_a,
  s.lattice_c,
  s.lattice_c / s.lattice_a AS c_over_a
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE s.lattice_a IS NOT NULL
  AND s.lattice_c IS NOT NULL
  AND s.lattice_a <> 0
ORDER BY s.lattice_c / s.lattice_a DESC
LIMIT 10000;
