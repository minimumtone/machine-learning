SELECT m.entry_id, m.formula, et.bulk_modulus_vrh, tp.debye_temperature_k
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN calculation cal_et ON cal_et.entry_id = m.entry_id AND cal_et.calculation_type = 'relaxation'
JOIN elastic_tensor et ON et.calculation_id = cal_et.calculation_id
JOIN calculation cal_tp ON cal_tp.entry_id = m.entry_id AND cal_tp.calculation_type = 'relaxation'
JOIN thermal_property tp ON tp.calculation_id = cal_tp.calculation_id
    AND tp.temperature_k = 300  -- benchmark convention: representative temperature
WHERE c.element = 'Co'
  AND (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.energy_above_hull <= 0.001
  AND et.bulk_modulus_vrh >= 180
  AND tp.debye_temperature_k >= 400
ORDER BY et.bulk_modulus_vrh DESC
LIMIT 10000;
