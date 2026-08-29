SELECT
  me.entry_id,
  me.formula,
  me.reduced_formula,
  mp.magnetic_anisotropy_energy
FROM material_entry AS me
JOIN structure AS s
  ON s.entry_id = me.entry_id
LEFT JOIN prototype_definition AS pd
  ON pd.prototype_id = s.prototype
JOIN magnetic_property AS mp
  ON mp.entry_id = me.entry_id
WHERE COALESCE(s.strukturbericht, pd.strukturbericht) = 'L12'
  AND mp.magnetic_anisotropy_energy IS NOT NULL
ORDER BY mp.magnetic_anisotropy_energy DESC
LIMIT 1;
