SELECT
  m.formula,
  ps.formation_energy_per_atom AS rebased_formation_energy_per_atom,
  s.lattice_a
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.is_stable = TRUE
  AND ps.formation_energy_per_atom < -0.3
ORDER BY ps.formation_energy_per_atom ASC
LIMIT 10000;
