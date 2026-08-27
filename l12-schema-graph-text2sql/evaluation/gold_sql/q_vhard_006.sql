SELECT m.formula, s.lattice_a, cp_bm.value AS bulk_modulus,
       ps.energy_above_hull,
       ABS(s.lattice_a - 3.57) AS lattice_mismatch,
       (1.0 - ps.energy_above_hull / 0.05) * 0.35
       + (1.0 - ABS(s.lattice_a - 3.57) / 0.3) * 0.35
       + (cp_bm.value / 300.0) * 0.30 AS composite_score
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id AND calc.calculation_type = 'relaxation'
JOIN calculated_property cp_bm ON cp_bm.calculation_id = calc.calculation_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.is_stable = TRUE
  AND cp_bm.property_name = 'bulk_modulus'
ORDER BY composite_score DESC
LIMIT 10000;