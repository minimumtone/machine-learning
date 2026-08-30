SELECT m.formula,
       ps.energy_above_hull,
       ps.is_stable,
       cp.value AS bulk_modulus
FROM material_entry m
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp ON cp.calculation_id = calc.calculation_id
     AND cp.property_name = 'bulk_modulus'
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.is_stable = TRUE
  AND cp.value >= 160
ORDER BY cp.value DESC
LIMIT 10000;
