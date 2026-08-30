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
WHERE s.strukturbericht = 'B2'
   OR pd.strukturbericht = 'B2'
ORDER BY me.reduced_formula, me.entry_id;
