SELECT m.formula,
       ps.energy_above_hull,
       ps.is_stable,
       se.miller_index,
       se.surface_energy_j_m2,
       cp.value AS bulk_modulus
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN surface_energy se ON se.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp ON cp.calculation_id = calc.calculation_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.is_stable = TRUE
  AND se.surface_energy_j_m2 <= 2.0
  AND cp.property_name = 'bulk_modulus'
  AND cp.unit = 'GPa'
  AND cp.value >= 180
ORDER BY cp.value DESC, se.surface_energy_j_m2 ASC
LIMIT 10000;
