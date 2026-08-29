SELECT m.formula,
       s.lattice_a,
       ABS(s.lattice_a - 3.57) AS lattice_diff,
       ps.energy_above_hull,
       ps.is_stable,
       cp.value AS bulk_modulus
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp ON cp.calculation_id = calc.calculation_id
     AND cp.property_name = 'bulk_modulus'
WHERE ABS(s.lattice_a - 3.57) <= 0.05
  AND ps.is_stable = TRUE
  AND cp.value >= 150
ORDER BY lattice_diff ASC, cp.value DESC
LIMIT 10000;
