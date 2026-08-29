SELECT DISTINCT me.entry_id, me.formula, me.reduced_formula
FROM material_entry AS me
JOIN composition AS c
  ON c.entry_id = me.entry_id
WHERE c.element = 'Fe'
ORDER BY me.formula, me.entry_id;
