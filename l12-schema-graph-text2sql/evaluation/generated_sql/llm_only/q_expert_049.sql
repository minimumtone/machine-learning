SELECT
  me.entry_id,
  me.formula,
  et.bulk_modulus_vrh AS bulk_modulus_gpa,
  ps.formation_energy_per_atom AS formation_energy_ev_per_atom
FROM material_entry AS me
JOIN calculation AS c
  ON c.entry_id = me.entry_id
JOIN elastic_tensor AS et
  ON et.calculation_id = c.calculation_id
LEFT JOIN phase_stability AS ps
  ON ps.entry_id = me.entry_id
WHERE me.formula IN ('Ni3Al', 'AlNi3')
   OR me.reduced_formula IN ('Ni3Al', 'AlNi3');
