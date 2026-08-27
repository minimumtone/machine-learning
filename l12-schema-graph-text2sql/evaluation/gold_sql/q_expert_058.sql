SELECT m.entry_id, m.formula, tp.gruneisen_parameter
FROM material_entry m
JOIN calculation cal_tp ON cal_tp.entry_id = m.entry_id AND cal_tp.calculation_type = 'relaxation'
JOIN thermal_property tp ON tp.calculation_id = cal_tp.calculation_id
    AND tp.temperature_k = 300  -- benchmark convention: representative temperature
WHERE tp.gruneisen_parameter >= 2.0
ORDER BY tp.gruneisen_parameter DESC
LIMIT 10000;
