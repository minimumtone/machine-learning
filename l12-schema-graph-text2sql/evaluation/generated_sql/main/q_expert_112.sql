SELECT m.formula,
       ps.energy_above_hull,
       ps.is_stable,
       cp.value AS bulk_modulus,
       sm.method_name,
       ms.temperature_k
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp ON cp.calculation_id = calc.calculation_id
JOIN material_synthesis ms ON ms.entry_id = m.entry_id
JOIN synthesis_method sm ON sm.synthesis_id = ms.synthesis_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.is_stable = TRUE
  AND cp.property_name = 'bulk_modulus'
  AND cp.unit = 'GPa'
  AND cp.value >= 150
  AND ms.success = TRUE
ORDER BY cp.value DESC, ps.energy_above_hull ASC
LIMIT 10000;
