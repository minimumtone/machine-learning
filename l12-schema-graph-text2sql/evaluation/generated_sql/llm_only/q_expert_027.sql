SELECT
  me.entry_id,
  me.formula,
  s.prototype,
  s.strukturbericht,
  s.crystal_system,
  s.lattice_c
FROM structure AS s
JOIN material_entry AS me
  ON me.entry_id = s.entry_id
LEFT JOIN prototype_definition AS pd
  ON pd.prototype_id = s.prototype
WHERE lower(s.crystal_system) = 'hexagonal'
  AND s.lattice_c IS NOT NULL
  AND (
    lower(s.prototype) LIKE '%nias%'
    OR lower(pd.prototype_name) LIKE '%nias%'
    OR s.strukturbericht IN ('B8_1', 'B81')
  )
ORDER BY s.lattice_c DESC
LIMIT 1;
