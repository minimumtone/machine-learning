SELECT DISTINCT
    me.entry_id,
    me.formula,
    me.reduced_formula,
    me.chemical_system,
    pd.prototype_name,
    COALESCE(s.strukturbericht, pd.strukturbericht) AS strukturbericht,
    s.space_group_number,
    s.space_group
FROM material_entry AS me
JOIN composition AS c
    ON c.entry_id = me.entry_id
JOIN structure AS s
    ON s.entry_id = me.entry_id
LEFT JOIN prototype_definition AS pd
    ON pd.prototype_id = s.prototype
WHERE c.element = 'Ti'
  AND (
      pd.prototype_name ILIKE '%NiAs%'
      OR pd.prototype_name ILIKE '%Nickel arsenide%'
      OR s.prototype ILIKE '%NiAs%'
      OR s.strukturbericht IN ('B81', 'B8_1')
      OR pd.strukturbericht IN ('B81', 'B8_1')
  )
ORDER BY me.reduced_formula, me.entry_id;
