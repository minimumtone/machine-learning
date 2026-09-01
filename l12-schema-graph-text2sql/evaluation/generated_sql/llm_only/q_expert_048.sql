SELECT
  me.entry_id,
  me.formula,
  me.reduced_formula,
  s.structure_id,
  s.space_group,
  s.crystal_system,
  s.lattice_a,
  s.lattice_c,
  s.lattice_c / NULLIF(s.lattice_a, 0) AS c_over_a
FROM structure AS s
JOIN material_entry AS me
  ON me.entry_id = s.entry_id
WHERE s.lattice_a IS NOT NULL
  AND s.lattice_c IS NOT NULL
  AND s.lattice_a <> 0
ORDER BY c_over_a DESC
LIMIT 50;
