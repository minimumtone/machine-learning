SELECT DISTINCT
  me.entry_id,
  me.formula,
  me.reduced_formula,
  ps.energy_above_hull
FROM material_entry AS me
JOIN phase_stability AS ps
  ON ps.entry_id = me.entry_id
JOIN structure AS s
  ON s.entry_id = me.entry_id
LEFT JOIN prototype_definition AS pd
  ON pd.prototype_id = s.prototype
WHERE ps.energy_above_hull <= 0.01
  AND (
    regexp_replace(upper(COALESCE(s.strukturbericht, '')), '[^A-Z0-9₂]', '', 'g') IN ('L12', 'L1₂')
    OR regexp_replace(upper(COALESCE(s.prototype, '')), '[^A-Z0-9₂]', '', 'g') IN ('L12', 'L1₂')
    OR regexp_replace(upper(COALESCE(pd.strukturbericht, '')), '[^A-Z0-9₂]', '', 'g') IN ('L12', 'L1₂')
    OR regexp_replace(upper(COALESCE(pd.prototype_name, '')), '[^A-Z0-9₂]', '', 'g') IN ('L12', 'L1₂')
  )
ORDER BY ps.energy_above_hull, me.reduced_formula;
