SELECT DISTINCT
  me.entry_id,
  me.formula,
  me.reduced_formula,
  me.chemical_system,
  s.structure_id,
  s.prototype,
  s.strukturbericht,
  s.formula_type,
  s.space_group_number,
  s.space_group,
  s.crystal_system,
  s.lattice_a,
  s.lattice_b,
  s.lattice_c,
  s.volume_per_atom
FROM material_entry AS me
JOIN structure AS s
  ON s.entry_id = me.entry_id
JOIN (
  SELECT entry_id
  FROM composition
  WHERE element IN ('Rh', 'Al')
  GROUP BY entry_id
  HAVING COUNT(DISTINCT element) = 2
) AS ca
  ON ca.entry_id = me.entry_id
ORDER BY
  me.reduced_formula,
  me.entry_id,
  s.structure_id;
