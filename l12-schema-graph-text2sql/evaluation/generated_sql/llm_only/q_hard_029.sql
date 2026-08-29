SELECT
  entry_id,
  formula,
  reduced_formula,
  formation_enthalpy_ev_per_atom,
  energy_above_hull,
  prototype,
  strukturbericht,
  space_group
FROM formation_enthalpy
WHERE energy_above_hull = 0
  AND formation_enthalpy_ev_per_atom <= -0.4
  AND (
    strukturbericht IN ('L1_2', 'L12', 'L1₂')
    OR prototype ILIKE '%L1_2%'
    OR prototype ILIKE '%L12%'
    OR prototype ILIKE '%L1₂%'
  )
ORDER BY formation_enthalpy_ev_per_atom ASC;
