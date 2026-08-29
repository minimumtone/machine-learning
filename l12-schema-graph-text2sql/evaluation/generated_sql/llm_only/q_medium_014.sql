SELECT
  me.entry_id,
  me.formula,
  s.lattice_a
FROM material_entry AS me
JOIN structure AS s
  ON s.entry_id = me.entry_id
WHERE s.strukturbericht IN ('L1_2', 'L12', 'L1₂')
  AND s.lattice_a IS NOT NULL
ORDER BY s.lattice_a ASC;
