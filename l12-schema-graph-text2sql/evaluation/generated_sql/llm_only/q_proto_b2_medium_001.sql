SELECT DISTINCT
    me.entry_id,
    me.formula,
    me.reduced_formula,
    me.chemical_system,
    s.prototype,
    s.strukturbericht,
    s.space_group_number,
    s.space_group
FROM material_entry AS me
JOIN structure AS s
    ON s.entry_id = me.entry_id
JOIN composition AS c
    ON c.entry_id = me.entry_id
WHERE c.element = 'Zr'
  AND s.strukturbericht = 'B2'
ORDER BY me.reduced_formula, me.entry_id;
