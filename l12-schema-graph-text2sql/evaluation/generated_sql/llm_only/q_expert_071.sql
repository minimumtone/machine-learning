SELECT DISTINCT
    me.entry_id,
    me.formula,
    me.reduced_formula
FROM material_entry AS me
JOIN structure AS s
    ON s.entry_id = me.entry_id
LEFT JOIN prototype_definition AS pd
    ON pd.prototype_id = s.prototype
WHERE EXISTS (
    SELECT 1
    FROM material_synthesis AS ms
    WHERE ms.entry_id = me.entry_id
      AND ms.success = TRUE
)
AND (
    regexp_replace(upper(coalesce(s.strukturbericht, '')), '[^A-Z0-9]', '', 'g') = 'L12'
    OR regexp_replace(upper(coalesce(s.prototype, '')), '[^A-Z0-9]', '', 'g') = 'L12'
    OR regexp_replace(upper(coalesce(pd.strukturbericht, '')), '[^A-Z0-9]', '', 'g') = 'L12'
    OR regexp_replace(upper(coalesce(pd.prototype_name, '')), '[^A-Z0-9]', '', 'g') = 'L12'
)
ORDER BY me.reduced_formula, me.entry_id;
