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
WHERE ps.is_stable = TRUE
  AND s.strukturbericht = 'B2'
ORDER BY ps.formation_energy_per_atom ASC;
