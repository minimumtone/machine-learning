SELECT
  fe.entry_id,
  fe.formula,
  fe.reference_set,
  fe.formation_enthalpy_ev_per_atom - ((3.0 * co.delta_e + ti.delta_e) / 4.0) AS formation_enthalpy_from_pure_refs_ev_per_atom
FROM formation_enthalpy fe
JOIN pure_element_reference co
  ON co.reference_set = fe.reference_set
 AND co.element_symbol = 'Co'
JOIN pure_element_reference ti
  ON ti.reference_set = fe.reference_set
 AND ti.element_symbol = 'Ti'
WHERE fe.reduced_formula = 'Co3Ti'
ORDER BY fe.energy_above_hull ASC NULLS LAST,
         formation_enthalpy_from_pure_refs_ev_per_atom ASC
LIMIT 1;
