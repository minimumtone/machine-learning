SELECT
  me.entry_id,
  me.formula,
  me.reduced_formula,
  ps.formation_energy_per_atom,
  ps.reference_set
FROM material_entry AS me
JOIN phase_stability AS ps
  ON ps.entry_id = me.entry_id
WHERE ps.formation_energy_per_atom <= -0.5
  AND me.number_of_elements > 1
ORDER BY ps.formation_energy_per_atom ASC;
