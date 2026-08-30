SELECT
  me.entry_id,
  me.formula,
  s.lattice_a,
  s.lattice_c,
  ABS(s.lattice_a - s.lattice_c) AS a_c_difference
FROM material_entry AS me
JOIN structure AS s
  ON me.entry_id = s.entry_id
WHERE s.lattice_a IS NOT NULL
  AND s.lattice_c IS NOT NULL
  AND ABS(s.lattice_a - s.lattice_c) > 0.01
ORDER BY a_c_difference DESC;
