SELECT
  me.entry_id,
  me.formula,
  me.reduced_formula,
  s.lattice_a,
  s.space_group,
  s.crystal_system
FROM material_entry AS me
JOIN structure AS s
  ON me.entry_id = s.entry_id
WHERE s.strukturbericht = 'B2'
  AND s.lattice_a >= 4.0
ORDER BY s.lattice_a DESC;
