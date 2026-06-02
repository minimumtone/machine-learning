SELECT DISTINCT m.entry_id, m.formula, ps.energy_above_hull, ps.formation_energy_per_atom, s.lattice_a
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE (s.prototype = 'L12' OR s.strukturbericht = 'L12')
  AND ps.energy_above_hull <= 0.05
ORDER BY ps.energy_above_hull ASC, ps.formation_energy_per_atom ASC
LIMIT 100;