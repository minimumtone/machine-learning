SELECT
  m.formula,
  cp.value AS bulk_modulus,
  ps.formation_energy_per_atom AS renormalized_formation_energy_per_atom
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp ON cp.calculation_id = calc.calculation_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND cp.property_name = 'bulk_modulus'
  AND cp.unit = 'GPa'
  AND cp.value >= 150
ORDER BY ps.formation_energy_per_atom ASC
LIMIT 10000;
