SELECT m.formula,
       ps.energy_above_hull,
       ps.is_stable,
       s.lattice_a,
       cp.value AS bulk_modulus
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp ON cp.calculation_id = calc.calculation_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.is_stable = TRUE
  AND cp.property_name = 'bulk_modulus_vrh'
  AND cp.unit = 'GPa'
  AND cp.value >= 200
  AND s.lattice_a BETWEEN 3.5 AND 4.0
ORDER BY cp.value DESC, s.lattice_a ASC
LIMIT 10000;
