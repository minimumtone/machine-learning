SELECT DISTINCT
    me.entry_id,
    me.formula,
    me.reduced_formula,
    me.chemical_system,
    s.prototype,
    pd.prototype_name,
    COALESCE(s.strukturbericht, pd.strukturbericht) AS strukturbericht,
    s.space_group_number,
    s.space_group
FROM material_entry AS me
JOIN structure AS s
    ON s.entry_id = me.entry_id
LEFT JOIN prototype_definition AS pd
    ON pd.prototype_id = s.prototype
WHERE EXISTS (
    SELECT 1
    FROM composition AS c
    WHERE c.entry_id = me.entry_id
      AND c.element = 'Sc'
)
AND (
    s.prototype ILIKE '%NaCl%'
    OR pd.prototype_name ILIKE '%NaCl%'
    OR COALESCE(s.strukturbericht, pd.strukturbericht) = 'B1'
)
ORDER BY me.entry_id;
