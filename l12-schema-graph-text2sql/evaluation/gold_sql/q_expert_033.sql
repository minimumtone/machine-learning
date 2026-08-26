SELECT m.entry_id, m.formula, ps.formation_energy_per_atom, ps.energy_above_hull
FROM material_entry m
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE (s.prototype = 'B2' OR s.strukturbericht = 'B2')
  AND ps.energy_above_hull <= 0.001
  AND ps.formation_energy_per_atom < 0
ORDER BY ps.formation_energy_per_atom
LIMIT 10000;
