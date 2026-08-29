SELECT
  me.entry_id,
  me.formula,
  me.reduced_formula,
  ps.formation_energy_per_atom,
  ps.energy_above_hull,
  s.prototype,
  s.strukturbericht,
  s.space_group
FROM material_entry AS me
JOIN phase_stability AS ps
  ON ps.entry_id = me.entry_id
JOIN structure AS s
  ON s.entry_id = me.entry_id
LEFT JOIN prototype_definition AS pd
  ON pd.prototype_id = s.prototype
WHERE ps.is_stable = TRUE
  AND (
    pd.prototype_name ILIKE '%NiAs%'
    OR s.prototype ILIKE '%NiAs%'
    OR s.strukturbericht IN ('B8_1', 'B81')
  )
ORDER BY ps.formation_energy_per_atom ASC;
