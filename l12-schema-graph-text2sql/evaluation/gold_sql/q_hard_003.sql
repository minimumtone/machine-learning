SELECT m.formula, s.lattice_a, ABS(s.lattice_a - 3.572) AS delta_a,
       ps.formation_energy_per_atom
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE s.prototype = 'L12'
  AND ABS(s.lattice_a - 3.572) <= 0.05
ORDER BY ps.formation_energy_per_atom ASC
LIMIT 100;