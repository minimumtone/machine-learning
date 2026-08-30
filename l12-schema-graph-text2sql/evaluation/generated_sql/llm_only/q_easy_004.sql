SELECT
  me.entry_id,
  me.formula,
  me.reduced_formula,
  s.prototype,
  s.strukturbericht,
  s.lattice_a,
  s.lattice_b,
  s.lattice_c
FROM material_entry AS me
JOIN structure AS s
  ON s.entry_id = me.entry_id
LEFT JOIN prototype_definition AS pd
  ON pd.prototype_id = s.prototype
WHERE
  UPPER(REGEXP_REPLACE(COALESCE(s.strukturbericht, pd.strukturbericht, ''), '[^A-Za-z0-9]', '', 'g')) = 'L12'
ORDER BY me.reduced_formula, me.entry_id;
