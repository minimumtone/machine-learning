SELECT
  entry_id,
  formula,
  reduced_formula,
  formation_enthalpy_ev_per_atom,
  weighted_element_delta_e,
  formation_enthalpy_ev_per_atom - weighted_element_delta_e AS energy_below_weighted_pure_reference_ev_per_atom,
  prototype,
  strukturbericht
FROM formation_enthalpy
WHERE strukturbericht IN ('L1_2', 'L12', 'L1₂')
  AND formation_enthalpy_ev_per_atom <= weighted_element_delta_e - 0.1
ORDER BY energy_below_weighted_pure_reference_ev_per_atom ASC;
