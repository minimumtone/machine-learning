SELECT DISTINCT
    me.entry_id,
    me.formula,
    me.reduced_formula,
    me.chemical_system,
    COALESCE(s.prototype, fe.prototype) AS prototype,
    COALESCE(s.strukturbericht, pd.strukturbericht, fe.strukturbericht) AS strukturbericht,
    COALESCE(s.space_group, fe.space_group) AS space_group
FROM material_entry AS me
LEFT JOIN structure AS s
    ON s.entry_id = me.entry_id
LEFT JOIN prototype_definition AS pd
    ON pd.prototype_id = s.prototype
LEFT JOIN formation_enthalpy AS fe
    ON fe.entry_id = me.entry_id
WHERE EXISTS (
    SELECT 1
    FROM composition AS c
    WHERE c.entry_id = me.entry_id
      AND c.element = 'Ga'
)
AND (
    regexp_replace(upper(translate(COALESCE(s.strukturbericht, ''), '₂', '2')), '[^A-Z0-9]', '', 'g') = 'L12'
    OR regexp_replace(upper(translate(COALESCE(pd.strukturbericht, ''), '₂', '2')), '[^A-Z0-9]', '', 'g') = 'L12'
    OR regexp_replace(upper(translate(COALESCE(fe.strukturbericht, ''), '₂', '2')), '[^A-Z0-9]', '', 'g') = 'L12'
)
ORDER BY me.reduced_formula, me.entry_id;
