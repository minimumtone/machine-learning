SELECT DISTINCT
  me.entry_id,
  me.formula,
  me.reduced_formula,
  me.chemical_system
FROM material_entry AS me
JOIN structure AS s
  ON s.entry_id = me.entry_id
LEFT JOIN prototype_definition AS pd
  ON pd.prototype_id = s.prototype
WHERE s.prototype ILIKE 'BiF3'
   OR pd.prototype_name ILIKE 'BiF3'
ORDER BY me.reduced_formula, me.entry_id;
