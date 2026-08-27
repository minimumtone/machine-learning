-- VH: Niを含むL1₂化合物で、バルクモジュラスが150GPa以上かつデバイ温度が400K以上の安定な化合物を教えて
-- Tables: material_entry, composition, structure, phase_stability, calculation, calculated_property, thermal_property (7)
SELECT m.formula, cp_bm.value AS bulk_modulus,
       tp.debye_temperature_k, ps.energy_above_hull
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id AND calc.calculation_type = 'relaxation'
JOIN calculated_property cp_bm ON cp_bm.calculation_id = calc.calculation_id
JOIN calculation cal_tp ON cal_tp.entry_id = m.entry_id AND cal_tp.calculation_type = 'relaxation'
JOIN thermal_property tp ON tp.calculation_id = cal_tp.calculation_id
    AND tp.temperature_k = 300  -- benchmark convention: representative temperature
WHERE c.element = 'Ni'
  AND (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.is_stable = TRUE
  AND cp_bm.property_name = 'bulk_modulus'
  AND cp_bm.value >= 150
  AND tp.debye_temperature_k >= 400
ORDER BY cp_bm.value DESC
LIMIT 10000;
