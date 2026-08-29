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
WHERE me.number_of_elements > 1
  AND (
      s.prototype = 'BCC_B2'
      OR pd.prototype_name = 'BCC_B2'
      OR s.strukturbericht = 'B2'
  )
ORDER BY me.reduced_formula, me.entry_id;
