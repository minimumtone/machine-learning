SELECT DISTINCT
    me.entry_id,
    me.formula,
    et.bulk_modulus_vrh AS bulk_modulus_gpa,
    s.lattice_a AS lattice_constant_a
FROM material_entry AS me
JOIN phase_stability AS ps
    ON ps.entry_id = me.entry_id
JOIN structure AS s
    ON s.entry_id = me.entry_id
JOIN calculation AS c
    ON c.entry_id = me.entry_id
JOIN elastic_tensor AS et
    ON et.calculation_id = c.calculation_id
WHERE ps.energy_above_hull <= 0.001
  AND ps.formation_energy_per_atom <= -0.3
  AND et.bulk_modulus_vrh >= 150
  AND s.strukturbericht = 'L1_2'
  AND s.lattice_a IS NOT NULL
  AND et.bulk_modulus_vrh IS NOT NULL
ORDER BY s.lattice_a, et.bulk_modulus_vrh;
