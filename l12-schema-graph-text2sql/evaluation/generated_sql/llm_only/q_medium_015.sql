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
JOIN structure AS s
    ON s.entry_id = me.entry_id
LEFT JOIN prototype_definition AS pd
    ON pd.prototype_id = s.prototype
WHERE me.chemical_system = 'Al-Ni'
  AND UPPER(REPLACE(TRANSLATE(COALESCE(s.strukturbericht, pd.strukturbericht, ''), '₂', '2'), '_', '')) = 'L12'
ORDER BY me.formula, me.entry_id;
