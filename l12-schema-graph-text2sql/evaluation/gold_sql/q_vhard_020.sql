SELECT m.formula, s.lattice_a, ps.formation_energy_per_atom,
       ps.energy_above_hull, ps.is_stable,
       cp_bm.value AS bulk_modulus,
       cp_sm.value AS shear_modulus,
       ABS(s.lattice_a - 3.57) AS mismatch,
       (1.0 - LEAST(ps.energy_above_hull, 0.05) / 0.05) * 0.25
       + (1.0 - LEAST(ABS(s.lattice_a - 3.57), 0.3) / 0.3) * 0.25
       + (LEAST(cp_bm.value, 300) / 300.0) * 0.20
       + (LEAST(cp_sm.value, 150) / 150.0) * 0.15
       + (-LEAST(ps.formation_energy_per_atom, 0) / 1.0) * 0.15 AS total_score
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp_bm ON cp_bm.calculation_id = calc.calculation_id
  AND cp_bm.property_name = 'bulk_modulus'
JOIN calculated_property cp_sm ON cp_sm.calculation_id = calc.calculation_id
  AND cp_sm.property_name = 'shear_modulus'
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
ORDER BY total_score DESC
LIMIT 10000;