SELECT
  m.formula,
  pde.hull_distance AS energy_above_hull,
  s.lattice_a,
  cp.value AS bulk_modulus,
  ((1 - pde.hull_distance / 0.05) * 0.4
   + (1 - ABS(s.lattice_a - 3.57) / 0.3) * 0.3
   + (cp.value / 300) * 0.3) AS weighted_score
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_diagram_entry pde ON pde.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp ON cp.calculation_id = calc.calculation_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND pde.hull_distance <= 0.05
  AND cp.property_name = 'bulk_modulus'
  AND cp.unit = 'GPa'
ORDER BY weighted_score DESC
LIMIT 10000;
