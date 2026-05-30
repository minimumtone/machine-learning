SELECT m.formula, s.lattice_a,
       ps.energy_above_hull, ps.formation_energy_per_atom,
       cp.value AS bulk_modulus_GPa,
       (1.0 - ps.energy_above_hull / 0.05) * 0.4
       + (1.0 - ABS(s.lattice_a - 3.572) / 0.5) * 0.3
       + cp.value / 250.0 * 0.3 AS composite_score
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp ON cp.calculation_id = calc.calculation_id
WHERE s.prototype = 'L12'
  AND ps.energy_above_hull <= 0.05
  AND cp.property_name = 'bulk_modulus'
ORDER BY composite_score DESC
LIMIT 100;