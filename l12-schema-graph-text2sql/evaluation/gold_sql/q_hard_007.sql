SELECT m.formula, s.lattice_a, ps.formation_energy_per_atom
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.formation_energy_per_atom <= -0.3
  AND s.lattice_a BETWEEN 3.5 AND 3.7
ORDER BY ps.formation_energy_per_atom ASC
LIMIT 100;