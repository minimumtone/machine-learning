SELECT DISTINCT
  me.entry_id,
  me.formula,
  me.reduced_formula,
  s.prototype,
  COALESCE(s.strukturbericht, pd.strukturbericht) AS strukturbericht,
  s.crystal_system,
  s.space_group_number
FROM material_entry AS me
JOIN structure AS s
  ON s.entry_id = me.entry_id
LEFT JOIN prototype_definition AS pd
  ON pd.prototype_id = s.prototype
WHERE LOWER(s.crystal_system) = 'cubic'
  AND (
    REPLACE(UPPER(COALESCE(s.strukturbericht, '')), '_', '') IN ('L12', 'L1₂')
    OR REPLACE(UPPER(COALESCE(pd.strukturbericht, '')), '_', '') IN ('L12', 'L1₂')
  )
ORDER BY me.reduced_formula, me.entry_id;
