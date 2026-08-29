SELECT DISTINCT me.entry_id, me.formula, me.reduced_formula, me.chemical_system
FROM material_entry AS me
WHERE EXISTS (
    SELECT 1
    FROM composition AS c
    WHERE c.entry_id = me.entry_id
      AND c.element = 'Ni'
)
AND EXISTS (
    SELECT 1
    FROM composition AS c
    WHERE c.entry_id = me.entry_id
      AND c.element = 'Al'
);
