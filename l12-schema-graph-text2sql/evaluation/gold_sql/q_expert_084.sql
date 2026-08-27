SELECT m.entry_id, m.formula, s.lattice_a, et.shear_modulus_vrh, ps.energy_above_hull
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN calculation cal_et ON cal_et.entry_id = m.entry_id
JOIN elastic_tensor et ON et.calculation_id = cal_et.calculation_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND s.lattice_a BETWEEN 3.53 AND 3.59
  AND et.shear_modulus_vrh >= 70
  AND ps.energy_above_hull <= 0.001
ORDER BY et.shear_modulus_vrh DESC
LIMIT 10000;
