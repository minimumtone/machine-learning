SELECT m.formula, s.lattice_a, ABS(s.lattice_a - 3.572) AS lattice_mismatch,
       ps.formation_energy_per_atom, ps.energy_above_hull,
       cp.value AS bulk_modulus_GPa
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
JOIN calculation calc ON calc.entry_id = m.entry_id
JOIN calculated_property cp ON cp.calculation_id = calc.calculation_id
WHERE s.prototype = 'L12'
  AND ps.energy_above_hull <= 0.05
  AND cp.property_name = 'bulk_modulus'
ORDER BY ps.energy_above_hull ASC, ABS(s.lattice_a - 3.572) ASC
LIMIT 100;