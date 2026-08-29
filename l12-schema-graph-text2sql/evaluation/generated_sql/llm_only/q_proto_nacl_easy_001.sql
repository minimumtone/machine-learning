SELECT DISTINCT
    me.entry_id,
    me.formula,
    me.reduced_formula,
    me.chemical_system,
    s.space_group_number,
    s.space_group,
    s.prototype,
    s.strukturbericht
FROM material_entry AS me
JOIN structure AS s
    ON s.entry_id = me.entry_id
LEFT JOIN prototype_definition AS pd
    ON pd.prototype_id = s.prototype
WHERE
    s.strukturbericht = 'B1'
    OR pd.strukturbericht = 'B1'
    OR s.prototype ILIKE '%NaCl%'
    OR pd.prototype_name ILIKE '%NaCl%'
    OR pd.prototype_name ILIKE '%rock%salt%'
    OR pd.description ILIKE '%rock%salt%'
ORDER BY
    me.reduced_formula,
    me.entry_id;
