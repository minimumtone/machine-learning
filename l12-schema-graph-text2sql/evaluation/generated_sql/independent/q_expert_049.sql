SELECT
  m.formula,
  cp.value AS bulk_modulus,
  ps.formation_energy_per_atom
FROM material_entry m
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp ON cp.calculation_id = calc.calculation_id
WHERE m.formula = 'Ni3Al'
  AND cp.property_name = 'bulk_modulus'
ORDER BY m.formula
LIMIT 10000;
