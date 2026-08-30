SELECT m.formula, a.system_name, cp.value AS bulk_modulus, ps.energy_above_hull, ps.is_stable
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp ON cp.calculation_id = calc.calculation_id
JOIN material_alloy_system mas ON mas.entry_id = m.entry_id
JOIN alloy_system a ON a.alloy_system_id = mas.alloy_system_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.is_stable = TRUE
  AND cp.property_name = 'bulk_modulus'
  AND a.system_name ILIKE '%Ni%'
ORDER BY cp.value DESC
LIMIT 10;
