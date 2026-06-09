SELECT m.formula, s.lattice_a, ps.formation_energy_per_atom,
       ps.energy_above_hull, cp_bm.value AS bulk_modulus,
       ABS(s.lattice_a - 3.57) AS lattice_diff
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp_bm ON cp_bm.calculation_id = calc.calculation_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.energy_above_hull <= 0.05
  AND cp_bm.property_name = 'bulk_modulus'
ORDER BY ps.formation_energy_per_atom ASC, ABS(s.lattice_a - 3.57) ASC
LIMIT 10000;