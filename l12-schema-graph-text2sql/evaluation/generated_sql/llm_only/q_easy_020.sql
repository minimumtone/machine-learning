SELECT DISTINCT
    me.entry_id,
    me.formula,
    me.reduced_formula,
    me.chemical_system,
    s.prototype,
    s.strukturbericht
FROM material_entry AS me
JOIN composition AS c
    ON c.entry_id = me.entry_id
JOIN structure AS s
    ON s.entry_id = me.entry_id
LEFT JOIN prototype_definition AS pd
    ON pd.prototype_id = s.prototype
WHERE c.element = 'W'
  AND (
      s.strukturbericht IN ('L1_2', 'L12', 'L1₂')
      OR pd.strukturbericht IN ('L1_2', 'L12', 'L1₂')
      OR s.prototype ILIKE '%L1_2%'
      OR s.prototype ILIKE '%L12%'
      OR pd.prototype_name ILIKE '%L1_2%'
      OR pd.prototype_name ILIKE '%L12%'
  )
ORDER BY me.reduced_formula, me.entry_id;
