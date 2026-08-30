SELECT
  m.formula,
  s.lattice_a,
  ps.energy_above_hull,
  cp_bm.value AS bulk_modulus,
  cp_g.value AS shear_modulus,
  cp_bm.value / NULLIF(cp_g.value, 0) AS b_g_ratio
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp_bm ON cp_bm.calculation_id = calc.calculation_id
  AND cp_bm.property_name = 'bulk_modulus'
JOIN calculated_property cp_g ON cp_g.calculation_id = calc.calculation_id
  AND cp_g.property_name = 'shear_modulus'
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.is_stable = TRUE
  AND s.lattice_a <= 4.0
  AND cp_bm.value / NULLIF(cp_g.value, 0) >= 2.0
ORDER BY b_g_ratio DESC, ps.energy_above_hull ASC, cp_bm.value DESC
LIMIT 10000;
