SELECT DISTINCT
    me.entry_id,
    me.formula,
    me.reduced_formula,
    me.chemical_system,
    s.prototype,
    s.strukturbericht,
    s.space_group_number,
    s.crystal_system
FROM material_entry AS me
JOIN structure AS s
    ON s.entry_id = me.entry_id
LEFT JOIN prototype_definition AS pd
    ON pd.prototype_id = s.prototype
WHERE EXISTS (
    SELECT 1
    FROM composition AS c
    WHERE c.entry_id = me.entry_id
      AND c.element IN ('Ni', 'Co')
)
AND (
    UPPER(REPLACE(REPLACE(COALESCE(s.strukturbericht, ''), '_', ''), '₂', '2')) = 'L12'
    OR UPPER(REPLACE(REPLACE(COALESCE(pd.strukturbericht, ''), '_', ''), '₂', '2')) = 'L12'
    OR s.prototype ILIKE '%L1_2%'
    OR pd.prototype_name ILIKE '%L1_2%'
    OR pd.prototype_name ILIKE '%L1₂%'
)
ORDER BY me.reduced_formula, me.entry_id;
