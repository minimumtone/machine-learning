SELECT m.formula, s.lattice_a, cp.value AS bulk_modulus, ps.formation_energy_per_atom
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp ON cp.calculation_id = calc.calculation_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.is_stable = TRUE
  AND cp.property_name = 'bulk_modulus'
  AND cp.value >= 180
  AND s.lattice_a <= 3.9
ORDER BY cp.value DESC
LIMIT 10000;
