SELECT DISTINCT
  me.entry_id,
  me.formula,
  me.reduced_formula,
  me.chemical_system,
  s.prototype,
  COALESCE(s.strukturbericht, pd.strukturbericht) AS strukturbericht,
  s.space_group_number,
  s.space_group
FROM material_entry AS me
JOIN composition AS c
  ON c.entry_id = me.entry_id
JOIN structure AS s
  ON s.entry_id = me.entry_id
LEFT JOIN prototype_definition AS pd
  ON pd.prototype_id = s.prototype
WHERE c.element = 'Ti'
  AND (
    UPPER(REPLACE(REPLACE(REPLACE(COALESCE(s.strukturbericht, pd.strukturbericht, ''), '₂', '2'), '_', ''), '-', '')) = 'L12'
    OR UPPER(REPLACE(REPLACE(REPLACE(COALESCE(s.prototype, ''), '₂', '2'), '_', ''), '-', '')) LIKE '%L12%'
    OR UPPER(REPLACE(REPLACE(REPLACE(COALESCE(pd.prototype_name, ''), '₂', '2'), '_', ''), '-', '')) LIKE '%L12%'
  )
ORDER BY me.reduced_formula, me.entry_id;
