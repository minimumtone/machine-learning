SELECT DISTINCT m.entry_id, m.formula, et.bulk_modulus_vrh, tp.debye_temperature_k
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN elastic_tensor et ON et.entry_id = m.entry_id
JOIN thermal_property tp ON tp.entry_id = m.entry_id
WHERE c.element = 'Co'
  AND (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.energy_above_hull <= 0.001
  AND et.bulk_modulus_vrh >= 180
  AND tp.debye_temperature_k >= 400
ORDER BY et.bulk_modulus_vrh DESC
LIMIT 10000;
