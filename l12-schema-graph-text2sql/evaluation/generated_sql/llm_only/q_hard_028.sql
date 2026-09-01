SELECT
  me.entry_id,
  me.formula,
  s.strukturbericht,
  s.lattice_a,
  CASE
    WHEN s.volume_per_atom IS NOT NULL AND s.volume_per_atom <> 0
      THEN 1.0 / s.volume_per_atom
    WHEN s.lattice_a IS NOT NULL AND s.lattice_a <> 0
      THEN COALESCE(pd.conventional_cell_atoms, 4)::double precision / POWER(s.lattice_a, 3)
  END AS atoms_per_volume
FROM structure s
JOIN material_entry me
  ON me.entry_id = s.entry_id
LEFT JOIN prototype_definition pd
  ON pd.prototype_id = s.prototype
WHERE s.strukturbericht IN ('L1_2', 'L12', 'L1₂')
  AND s.lattice_a IS NOT NULL
ORDER BY s.lattice_a;
