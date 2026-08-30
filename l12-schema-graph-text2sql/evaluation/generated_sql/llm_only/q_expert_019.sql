SELECT DISTINCT
    me.entry_id,
    me.formula,
    me.reduced_formula,
    me.chemical_system,
    s.prototype,
    s.strukturbericht,
    s.space_group
FROM material_entry AS me
JOIN structure AS s
  ON s.entry_id = me.entry_id
LEFT JOIN prototype_definition AS pd
  ON pd.prototype_id = s.prototype
WHERE (
    regexp_replace(upper(COALESCE(s.strukturbericht, '')), '[^A-Z0-9]', '', 'g') = 'L12'
    OR regexp_replace(upper(COALESCE(pd.strukturbericht, '')), '[^A-Z0-9]', '', 'g') = 'L12'
    OR upper(COALESCE(pd.prototype_name, '')) LIKE '%L12%'
)
AND EXISTS (
    SELECT 1
    FROM composition AS c
    JOIN element AS e
      ON e.symbol = c.element
    WHERE c.entry_id = me.entry_id
      AND e.atomic_number >= 40
)
ORDER BY me.formula;
