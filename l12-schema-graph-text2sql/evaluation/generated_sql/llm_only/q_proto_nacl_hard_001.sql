SELECT
  entry_id,
  formula,
  reduced_formula,
  formation_enthalpy_ev_per_atom AS formation_energy_ev_per_atom,
  energy_above_hull,
  prototype,
  strukturbericht,
  space_group
FROM formation_enthalpy
WHERE is_stable = TRUE
  AND (
    strukturbericht = 'B1'
    OR prototype ILIKE '%NaCl%'
  )
ORDER BY formation_enthalpy_ev_per_atom ASC;
