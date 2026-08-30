SELECT DISTINCT
  me.entry_id,
  me.formula,
  me.reduced_formula,
  s.space_group_number,
  s.space_group
FROM material_entry AS me
JOIN structure AS s
  ON me.entry_id = s.entry_id
WHERE s.space_group_number = 221;
