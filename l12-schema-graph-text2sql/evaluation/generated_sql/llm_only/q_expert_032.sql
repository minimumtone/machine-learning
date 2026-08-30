SELECT
  me.entry_id,
  me.formula,
  me.reduced_formula,
  ps.formation_energy_per_atom,
  ps.reference_set
FROM phase_stability AS ps
JOIN material_entry AS me
  ON me.entry_id = ps.entry_id
WHERE ps.formation_energy_per_atom > 0
  AND me.number_of_elements > 1
ORDER BY ps.formation_energy_per_atom DESC, me.entry_id;
