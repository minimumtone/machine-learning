SELECT DISTINCT
  me.entry_id,
  me.formula,
  me.reduced_formula,
  me.chemical_system,
  COALESCE(s.strukturbericht, pd.strukturbericht) AS strukturbericht,
  COALESCE(pd.prototype_name, s.prototype) AS prototype,
  s.space_group_number,
  s.space_group
FROM material_entry AS me
JOIN structure AS s
  ON s.entry_id = me.entry_id
LEFT JOIN prototype_definition AS pd
  ON pd.prototype_id = s.prototype
WHERE regexp_replace(upper(COALESCE(s.strukturbericht, pd.strukturbericht, pd.prototype_name, s.prototype, '')), '[^A-Z0-9]', '', 'g') LIKE '%L12%'
ORDER BY me.reduced_formula, me.entry_id;
