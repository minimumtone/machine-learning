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
WHERE is_stable = TRUE
  AND (
    strukturbericht IN ('L1_2', 'L12', 'L1₂')
    OR prototype ILIKE '%L1_2%'
    OR prototype ILIKE '%L12%'
    OR prototype ILIKE '%L1₂%'
  )
ORDER BY reduced_formula, energy_above_hull;
