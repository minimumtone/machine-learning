SELECT
  m.formula,
  tp.debye_temperature_k,
  ps.formation_energy_per_atom AS rebased_formation_energy_per_atom
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN thermal_property tp ON tp.calculation_id = calc.calculation_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND tp.debye_temperature_k IS NOT NULL
  AND ps.formation_energy_per_atom IS NOT NULL
ORDER BY tp.debye_temperature_k DESC
LIMIT 10;
