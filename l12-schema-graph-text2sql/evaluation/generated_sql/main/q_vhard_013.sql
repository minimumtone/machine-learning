SELECT
  m.formula,
  cp.value AS bulk_modulus,
  s.lattice_a
FROM material_entry m
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp ON cp.calculation_id = calc.calculation_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.energy_above_hull <= 0.001
  AND ps.formation_energy_per_atom <= -0.3
  AND cp.property_name = 'bulk_modulus'
  AND cp.unit = 'GPa'
  AND cp.value >= 150
ORDER BY s.lattice_a ASC, cp.value ASC
LIMIT 10000;
