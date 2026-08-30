SELECT DISTINCT
  me.entry_id,
  me.formula,
  me.reduced_formula,
  me.chemical_system
FROM material_entry AS me
JOIN structure AS s
  ON s.entry_id = me.entry_id
WHERE s.formula_type = 'A3B'
ORDER BY me.reduced_formula, me.entry_id;
