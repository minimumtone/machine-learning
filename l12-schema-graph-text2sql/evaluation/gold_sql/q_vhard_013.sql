SELECT m.formula, ps.energy_above_hull, ps.formation_energy_per_atom,
       s.lattice_a, cp_bm.value AS bulk_modulus
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id AND calc.calculation_type = 'relaxation'
JOIN calculated_property cp_bm ON cp_bm.calculation_id = calc.calculation_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.energy_above_hull <= 0.001
  AND ps.formation_energy_per_atom <= -0.3
  AND cp_bm.property_name = 'bulk_modulus'
  AND cp_bm.value >= 150
ORDER BY ps.formation_energy_per_atom ASC
LIMIT 10000;