SELECT m.entry_id, m.formula, s.lattice_a, ps.energy_above_hull, et.bulk_modulus_vrh
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN calculation cal_et ON cal_et.entry_id = m.entry_id
JOIN elastic_tensor et ON et.calculation_id = cal_et.calculation_id
WHERE ABS(s.lattice_a - 3.572) < 0.05
  AND ps.energy_above_hull <= 0.001
  AND et.bulk_modulus_vrh >= 150
ORDER BY ABS(s.lattice_a - 3.572)
LIMIT 10000;
