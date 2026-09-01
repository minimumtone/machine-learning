SELECT
  entry_id,
  formula,
  reduced_formula,
  formation_enthalpy_ev_per_atom,
  energy_above_hull,
  prototype,
  strukturbericht
FROM formation_enthalpy
WHERE strukturbericht = 'B2'
  AND is_stable = TRUE
  AND formation_enthalpy_ev_per_atom < 0
ORDER BY formation_enthalpy_ev_per_atom;
