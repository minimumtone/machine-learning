SELECT DISTINCT
  me.entry_id,
  me.formula,
  me.reduced_formula,
  ps.energy_above_hull,
  s.prototype,
  s.strukturbericht
FROM material_entry AS me
JOIN structure AS s
  ON s.entry_id = me.entry_id
LEFT JOIN prototype_definition AS pd
  ON pd.prototype_id = s.prototype
JOIN phase_stability AS ps
  ON ps.entry_id = me.entry_id
JOIN material_synthesis AS ms
  ON ms.entry_id = me.entry_id
WHERE ms.success = TRUE
  AND ps.is_stable = TRUE
  AND (
    regexp_replace(upper(replace(COALESCE(s.strukturbericht, ''), '₂', '2')), '[^A-Z0-9]', '', 'g') = 'L12'
    OR regexp_replace(upper(replace(COALESCE(pd.strukturbericht, ''), '₂', '2')), '[^A-Z0-9]', '', 'g') = 'L12'
  )
ORDER BY me.reduced_formula, me.entry_id;
