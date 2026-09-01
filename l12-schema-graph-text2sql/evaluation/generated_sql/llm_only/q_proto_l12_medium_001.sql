SELECT DISTINCT
  me.entry_id,
  me.formula,
  me.reduced_formula,
  me.chemical_system,
  s.prototype,
  COALESCE(s.strukturbericht, pd.strukturbericht) AS strukturbericht,
  s.space_group_number,
  s.crystal_system
FROM material_entry AS me
JOIN composition AS c
  ON c.entry_id = me.entry_id
JOIN structure AS s
  ON s.entry_id = me.entry_id
LEFT JOIN prototype_definition AS pd
  ON pd.prototype_id = s.prototype
WHERE c.element = 'Ni'
  AND UPPER(REPLACE(COALESCE(s.strukturbericht, pd.strukturbericht, ''), '_', '')) = 'L12';
