SELECT m.formula, s.lattice_a, ps.formation_energy_per_atom,
       ps.energy_above_hull, cp_bm.value AS bulk_modulus,
       ABS(s.lattice_a - 3.57) AS mismatch,
       (1.0 - LEAST(ps.energy_above_hull, 0.05) / 0.05) * 0.30
       + (1.0 - LEAST(ABS(s.lattice_a - 3.57), 0.3) / 0.3) * 0.25
       + (LEAST(cp_bm.value, 300) / 300.0) * 0.25
       + (CASE WHEN ps.formation_energy_per_atom < -0.3 THEN 0.20 ELSE 0.10 END) AS design_score
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id AND calc.calculation_type = 'relaxation'
JOIN calculated_property cp_bm ON cp_bm.calculation_id = calc.calculation_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND m.chemical_system <> 'Al-Ni'
  AND ps.is_stable = TRUE
  AND cp_bm.property_name = 'bulk_modulus'
ORDER BY design_score DESC
LIMIT 10000;