SELECT
  m.formula,
  phase_diagram_entry.hull_distance AS e_hull,
  s.lattice_a,
  ABS(s.lattice_a - 3.57) AS lattice_diff,
  cp.value AS bulk_modulus,
  ((1 - phase_diagram_entry.hull_distance / 0.05) * 0.35
   + (1 - LEAST(ABS(s.lattice_a - 3.57), 0.3) / 0.3) * 0.35
   + (LEAST(cp.value, 300) / 300) * 0.30) AS score
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_diagram_entry ON phase_diagram_entry.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp ON cp.calculation_id = calc.calculation_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND phase_diagram_entry.hull_distance <= 0.05
  AND cp.property_name = 'bulk_modulus'
ORDER BY score DESC
LIMIT 10000;
