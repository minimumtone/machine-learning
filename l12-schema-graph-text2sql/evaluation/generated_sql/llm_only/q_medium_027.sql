SELECT DISTINCT
    me.entry_id,
    me.formula,
    me.reduced_formula,
    s.prototype,
    COALESCE(s.strukturbericht, pd.strukturbericht) AS strukturbericht,
    s.space_group_number,
    s.space_group,
    s.volume_per_atom
FROM material_entry AS me
JOIN structure AS s
    ON s.entry_id = me.entry_id
LEFT JOIN prototype_definition AS pd
    ON pd.prototype_id = s.prototype
WHERE s.volume_per_atom <= 12
  AND (
      COALESCE(s.strukturbericht, pd.strukturbericht) IN ('L1_2', 'L1₂', 'L12')
      OR s.prototype ILIKE '%L1_2%'
      OR s.prototype ILIKE '%L12%'
      OR pd.prototype_name ILIKE '%L1_2%'
      OR pd.prototype_name ILIKE '%L12%'
  )
ORDER BY s.volume_per_atom ASC;
