SELECT m.entry_id, m.formula, s.lattice_a, s.lattice_c,
  ROUND((s.lattice_c / NULLIF(s.lattice_a, 0))::numeric, 4) AS c_over_a
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
WHERE s.lattice_a > 0 AND s.lattice_c IS NOT NULL
ORDER BY (s.lattice_c / NULLIF(s.lattice_a, 0)) DESC, m.entry_id ASC
LIMIT 10000;
