SELECT
  fh.entry_id,
  fh.formula,
  fh.reduced_formula,
  fh.formation_enthalpy_ev_per_atom,
  fh.energy_above_hull,
  fh.prototype,
  fh.space_group
FROM formation_enthalpy AS fh
WHERE fh.is_stable = TRUE
  AND fh.prototype = 'BiF3'
ORDER BY fh.formation_enthalpy_ev_per_atom ASC;
