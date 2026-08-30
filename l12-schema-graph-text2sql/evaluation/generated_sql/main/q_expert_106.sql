SELECT m.formula,
       ps.energy_above_hull,
       cp_mag.value AS magnetization,
       cp_bm.value AS bulk_modulus
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp_mag ON cp_mag.calculation_id = calc.calculation_id
     AND cp_mag.property_name = 'magnetization'
JOIN calculated_property cp_bm ON cp_bm.calculation_id = calc.calculation_id
     AND cp_bm.property_name = 'bulk_modulus'
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.is_stable = TRUE
  AND ABS(cp_mag.value) > 0
ORDER BY cp_mag.value DESC
LIMIT 10000;
