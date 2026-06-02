SELECT DISTINCT m.entry_id, m.formula, ps.energy_above_hull, ps.formation_energy_per_atom
FROM material_entry m
JOIN composition c ON c.entry_id = m.entry_id
JOIN structure s ON s.entry_id = m.entry_id
JOIN phase_stability ps ON ps.entry_id = m.entry_id
WHERE c.element = 'Co'
  AND (s.prototype = 'L12' OR s.strukturbericht = 'L12')
ORDER BY ps.energy_above_hull ASC
LIMIT 100;