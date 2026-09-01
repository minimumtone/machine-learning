SELECT
  m.formula,
  ps.is_stable,
  ps.energy_above_hull,
  s.lattice_a,
  ABS(s.lattice_a - 3.57) AS lattice_diff,
  cp.value AS bulk_modulus,
  (1.0 - ps.energy_above_hull / 0.05) * 0.35
    + (1.0 - ABS(s.lattice_a - 3.57) / 0.3) * 0.35
    + (cp.value / 300.0) * 0.30 AS overall_score
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp ON cp.calculation_id = calc.calculation_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.is_stable = TRUE
  AND cp.property_name = 'bulk_modulus'
ORDER BY overall_score DESC
LIMIT 10000;
