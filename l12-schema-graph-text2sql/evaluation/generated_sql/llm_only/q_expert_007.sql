SELECT DISTINCT
    me.entry_id,
    me.formula,
    me.reduced_formula,
    s.space_group_number,
    COALESCE(sg.hermann_mauguin, s.space_group) AS space_group
FROM material_entry AS me
JOIN structure AS s
    ON s.entry_id = me.entry_id
LEFT JOIN space_group AS sg
    ON sg.space_group_number = s.space_group_number
WHERE sg.hermann_mauguin = 'Pm-3m'
   OR s.space_group = 'Pm-3m'
ORDER BY me.reduced_formula, me.entry_id;
