SELECT m.formula, s.lattice_a, ps.energy_above_hull,
       cp_bm.value AS bulk_modulus,
       ABS(s.lattice_a - 3.57) AS mismatch,
       (1.0 - ps.energy_above_hull / 0.05) * 0.35
       + (1.0 - LEAST(ABS(s.lattice_a - 3.57), 0.3) / 0.3) * 0.35
       + (LEAST(cp_bm.value, 300) / 300.0) * 0.30 AS score
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id AND calc.calculation_type = 'relaxation'
JOIN calculated_property cp_bm ON cp_bm.calculation_id = calc.calculation_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.energy_above_hull <= 0.05
  AND cp_bm.property_name = 'bulk_modulus'
ORDER BY score DESC, m.entry_id ASC
LIMIT 10000;