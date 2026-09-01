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
WHERE EXISTS (
    SELECT 1
    FROM composition AS c
    WHERE c.entry_id = me.entry_id
      AND c.element = 'Ni'
)
AND regexp_replace(upper(replace(s.strukturbericht, '₂', '2')), '[^A-Z0-9]', '', 'g') = 'L12'
AND me.number_of_elements >= 2;
