SELECT DISTINCT
    me.entry_id,
    me.formula,
    me.reduced_formula,
    s.strukturbericht,
    s.lattice_a,
    s.lattice_b,
    s.lattice_c
FROM material_entry AS me
JOIN composition AS c
    ON c.entry_id = me.entry_id
JOIN structure AS s
    ON s.entry_id = me.entry_id
LEFT JOIN prototype_definition AS pd
    ON pd.prototype_id = s.prototype
WHERE c.element = 'Sc'
  AND (
      upper(replace(COALESCE(s.strukturbericht, ''), '_', '')) IN ('L12', 'L1₂')
      OR upper(replace(COALESCE(pd.strukturbericht, ''), '_', '')) IN ('L12', 'L1₂')
  )
ORDER BY me.reduced_formula, me.entry_id;
