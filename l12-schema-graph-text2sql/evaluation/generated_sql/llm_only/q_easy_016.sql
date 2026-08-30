SELECT DISTINCT
  me.entry_id,
  me.formula,
  me.reduced_formula,
  me.chemical_system
FROM material_entry AS me
JOIN composition AS c
  ON c.entry_id = me.entry_id
WHERE c.element = 'Fe'
  AND me.number_of_elements > 1
ORDER BY me.reduced_formula, me.entry_id;
